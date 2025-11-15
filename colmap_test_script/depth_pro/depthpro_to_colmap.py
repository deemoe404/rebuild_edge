#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Depth Pro predictions on undistorted COLMAP images into COLMAP photometric depth bins.

Inputs (from image_undistorter):
  <dense_dir>/images/              # undistorted images
  <dense_dir>/sparse/cameras.txt   # undistorted intrinsics (fx, fy, cx, cy)
  <dense_dir>/sparse/images.txt    # image name -> camera id mapping

Outputs:
  <dense_dir>/stereo/depth_maps/<image_name>.photometric.bin

Notes:
  - Writes COLMAP depth bin format: header "w&h&1&" (ASCII) then row-major float32 depth [m].
  - Uses undistorted fx (pixels) as Depth Pro focal hint to preserve metric scale.
  - Requires the `depth_pro` package (Apple ml-depth-pro) and its checkpoint.
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import math


def compute_normals_from_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    # Expect depth HxW (meters). Returns HxWx3 float32 normals (unit length, camera frame),
    # using forward-difference cross product and padding borders.
    h, w = depth.shape
    # Pixel grids
    uu = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    vv = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)

    # Backproject to camera coordinates
    Z = depth.astype(np.float32)
    X = (uu - cx) * Z / (fx if fx != 0 else 1e-6)
    Y = (vv - cy) * Z / (fy if fy != 0 else 1e-6)

    # Forward differences (u-direction: axis=1, v-direction: axis=0)
    dXu = X[:, 1:] - X[:, :-1]
    dYu = Y[:, 1:] - Y[:, :-1]
    dZu = Z[:, 1:] - Z[:, :-1]
    dXv = X[1:, :] - X[:-1, :]
    dYv = Y[1:, :] - Y[:-1, :]
    dZv = Z[1:, :] - Z[:-1, :]

    # Align shapes to inner region (h-1, w-1)
    ax = np.stack([dXu[:-1, :], dYu[:-1, :], dZu[:-1, :]], axis=2)
    ay = np.stack([dXv[:, :-1], dYv[:, :-1], dZv[:, :-1]], axis=2)
    # Cross product order determines orientation. COLMAP PatchMatch normal maps
    # point toward the camera (negative Z in camera frame). Use ay x ax so that
    # Nz is predominantly negative; then explicitly flip any residual positive-Z
    # normals to ensure consistency with PatchMatch outputs.
    n = np.cross(ay, ax)

    # Normalize, avoid div-by-zero
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    n = n / norm

    # Pad to HxW by replicating edge values
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:-1, :-1, :] = n.astype(np.float32)
    normals[-1, :-1, :] = normals[-2, :-1, :]
    normals[:-1, -1, :] = normals[:-1, -2, :]
    normals[-1, -1, :] = normals[-2, -2, :]

    # Invalidate normals where depth is invalid or zero
    invalid = ~np.isfinite(Z) | (Z <= 0)
    normals[invalid] = 0.0

    # Enforce camera-facing convention (Nz <= 0 where valid)
    valid = ~invalid
    flip = valid & (normals[..., 2] > 0)
    normals[flip] *= -1.0
    return normals


def read_cameras_txt(path: Path) -> Dict[int, dict]:
    cams: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            # id, model, width, height, params...
            if len(toks) < 5:
                continue
            cam_id = int(toks[0])
            model = toks[1]
            w, h = map(int, toks[2:4])
            params = list(map(float, toks[4:]))
            if len(params) >= 4:
                fx, fy, cx, cy = params[:4]
            elif len(params) >= 3:
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
            else:
                # Fallback: keep zeros; Depth Pro can also estimate f when missing
                fx = fy = cx = cy = 0.0
            cams[cam_id] = dict(model=model, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def read_images_txt(path: Path) -> Dict[str, int]:
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    name2cam: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            # Expect first line of each image to end with non-numeric NAME.
            # Second line (2D-3D correspondences) is all numeric tokens; skip those.
            if len(toks) < 10 or is_number(toks[-1]):
                continue
            try:
                cam_id = int(toks[8])
                name = " ".join(toks[9:])
            except (ValueError, IndexError):
                continue
            if name:
                name2cam[name] = cam_id
    return name2cam


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    # COLMAP stores world->camera rotation as quaternion [qw, qx, qy, qz].
    w, x, y, z = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_images_with_points(path: Path) -> Dict[str, dict]:
    """
    Parse COLMAP images.txt including 2D-3D correspondences.

    Returns dict[name] = {
        'cam_id': int,
        'qvec': np.float64[4],
        'tvec': np.float64[3],
        'points2D': list[(x, y, point3D_id)]
    }
    """
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        i += 1
        if not header or header.startswith("#"):
            continue
        toks = header.split()
        if len(toks) < 10:
            continue
        try:
            q = np.array(list(map(float, toks[1:5])), dtype=np.float64)
            t = np.array(list(map(float, toks[5:8])), dtype=np.float64)
            cam_id = int(toks[8])
            name = " ".join(toks[9:])
        except Exception:
            continue

        pts: List[Tuple[float, float, int]] = []
        if i < len(lines):
            pts_line = lines[i].strip()
            i += 1
            if pts_line and not pts_line.startswith("#"):
                vals = pts_line.split()
                for j in range(0, len(vals), 3):
                    try:
                        x = float(vals[j]); y = float(vals[j + 1]); pid = int(vals[j + 2])
                        pts.append((x, y, pid))
                    except Exception:
                        break
        out[name] = dict(cam_id=cam_id, qvec=q, tvec=t, points2D=pts)
    return out


def read_points3D_txt(path: Path) -> Dict[int, np.ndarray]:
    pts: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 4:
                continue
            try:
                pid = int(toks[0])
                X = np.array(list(map(float, toks[1:4])), dtype=np.float64)
                pts[pid] = X
            except Exception:
                continue
    return pts


def geometric_depth_z(
    Xw: np.ndarray,
    qvec: np.ndarray,
    tvec: np.ndarray,
) -> Optional[float]:
    """Return z-depth (camera Z component) for world point Xw.

    p_cam = R * Xw + t, where R = world->cam from qvec, tvec.
    Depth is p_cam[2].
    """
    R = qvec2rotmat(qvec)
    p = R @ Xw.reshape(3) + tvec.reshape(3)
    z = float(p[2])
    return z if z > 0 else None


def robust_affine_for_image_z_or_ray(
    depth_np: np.ndarray,
    img_meta: dict,
    cams: Dict[int, dict],
    points3D: Dict[int, np.ndarray],
    max_samples: int = 800,
    q_low: float = 5.0,
    q_high: float = 95.0,
    clamp_min: float = 0.1,
    clamp_max: float = 10.0,
    shift_abs_max: float = 50.0,
) -> Tuple[str, float, float, int, float, float]:
    """Estimate affine (s, t) under two hypotheses and pick the better by MAD.

    Hypothesis A: network predicts z-depth directly.
    Hypothesis B: network predicts ray length; convert to z by dividing per-pixel ray norm.
    Returns (mode, s, t, num_pairs, madA, madB), where mode in {'z','raylen'}.
    """
    cam = cams.get(img_meta["cam_id"]) or {}
    fx = float(cam.get("fx", 0.0)); fy = float(cam.get("fy", 0.0))
    cx = float(cam.get("cx", 0.0)); cy = float(cam.get("cy", 0.0))
    H = int(cam.get("h", depth_np.shape[0])); W = int(cam.get("w", depth_np.shape[1]))

    # Collect samples of (d_pred, ray_norm, z_geom)
    dps: List[Tuple[float, float, float]] = []
    for (x, y, pid) in img_meta.get("points2D", [])[:max_samples]:
        if pid < 0:
            continue
        Xw = points3D.get(pid)
        if Xw is None:
            continue
        u = int(round(x)); v = int(round(y))
        if not (0 <= u < W and 0 <= v < H):
            continue
        d_pred = float(depth_np[v, u])
        if not math.isfinite(d_pred) or d_pred <= 0:
            continue
        z_geom = geometric_depth_z(Xw, img_meta["qvec"], img_meta["tvec"])
        if z_geom is None or not math.isfinite(z_geom) or z_geom <= 0:
            continue
        rx = (u - cx) / (fx if fx != 0 else 1e-6)
        ry = (v - cy) / (fy if fy != 0 else 1e-6)
        ray_n = float(math.sqrt(1.0 + rx * rx + ry * ry))
        dps.append((d_pred, ray_n, float(z_geom)))
        if len(dps) >= max_samples:
            break

    n = len(dps)
    if n < 20:
        return "z", 1.0, 0.0, n, float("inf"), float("inf")

    # Helper to robustly estimate affine (s, t) with IRLS (Huber)
    def estimate_affine_and_mad(pred_vals: np.ndarray, target_z: np.ndarray) -> Tuple[float, float, float]:
        A = np.stack([pred_vals, np.ones_like(pred_vals)], axis=1)
        x = np.array([1.0, 0.0], dtype=np.float64)
        for _ in range(5):
            r = A @ x - target_z
            mad = np.median(np.abs(r)) + 1e-6
            delta = 1.345 * mad
            w = np.ones_like(r)
            idx = np.abs(r) > delta
            w[idx] = delta / (np.abs(r[idx]) + 1e-6)
            W = np.diag(w)
            try:
                x = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ target_z, rcond=None)[0]
            except Exception:
                break
        s = float(x[0]); t = float(x[1])
        s = float(max(clamp_min, min(clamp_max, s)))
        t = float(max(-shift_abs_max, min(shift_abs_max, t)))
        res = np.abs(target_z - (s * pred_vals + t))
        mad = float(np.median(res))
        return s, t, mad

    arr = np.asarray(dps, dtype=np.float64)
    d_pred = arr[:, 0]
    ray_n = arr[:, 1]
    z_geom = arr[:, 2]

    # Hypothesis A: d_pred is z
    sA, tA, madA = estimate_affine_and_mad(d_pred, z_geom)

    # Hypothesis B: d_pred is ray length; convert to z via division by ray norm
    z_from_ray = d_pred / np.maximum(ray_n, 1e-6)
    sB, tB, madB = estimate_affine_and_mad(z_from_ray, z_geom)

    # Prefer B if significantly better
    if madB < 0.8 * madA:
        return "raylen", sB, tB, n, madA, madB
    else:
        return "z", sA, tA, n, madA, madB


def write_colmap_mat_bin(path: Path, arr: np.ndarray) -> None:
    # Accept HxW or HxWxC arrays (float32)
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    elif arr.ndim == 3:
        h, w, c = arr.shape
    else:
        raise ValueError("Array must be 2D or 3D")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"{w}&{h}&{c}&".encode("ascii"))
        arr.astype(np.float32).tofile(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Depth Pro on undistorted images and write COLMAP depth bins")
    ap.add_argument("--dense_dir", required=True, help="Path to <run>/dense from image_undistorter")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"], help="Numeric precision")
    ap.add_argument("--list", default=None, help="Optional image list to filter (relative paths ok)")
    ap.add_argument("--log", default=None, help="Optional log file")
    # Robust alignment clamps (for scale/shift) to control outliers/drift
    ap.add_argument("--scale_min", type=float, default=0.1, help="Lower clamp for per-image scale")
    ap.add_argument("--scale_max", type=float, default=10.0, help="Upper clamp for per-image scale")
    ap.add_argument("--shift_abs_max", type=float, default=50.0, help="Absolute clamp for per-image shift")
    args = ap.parse_args()

    dense = Path(args.dense_dir)
    undist_imgs = dense / "images"
    sparse_txt = dense / "sparse"
    cams = read_cameras_txt(sparse_txt / "cameras.txt")
    name2cam = read_images_txt(sparse_txt / "images.txt")
    # For per-image scale alignment
    imgs_meta = read_images_with_points(sparse_txt / "images.txt")
    pts3d = read_points3D_txt(sparse_txt / "points3D.txt")

    # Late imports to avoid requiring deps unless used
    import torch  # noqa: WPS433
    import depth_pro  # noqa: WPS433

    # Resolve device and precision
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    if args.precision == "auto":
        prec = torch.float16 if (dev.type == "cuda") else torch.float32
    else:
        prec = torch.float16 if args.precision == "fp16" else torch.float32

    model, transform = depth_pro.create_model_and_transforms(device=dev, precision=prec)
    model.eval()

    sel = None
    if args.list and Path(args.list).exists():
        with open(args.list, "r", encoding="utf-8") as f:
            sel = {Path(l.strip()).name for l in f if l.strip()}

    # Support nested directories under undistorted images (COLMAP may preserve subfolders)
    img_paths = [
        p for p in sorted(glob.glob(str(undist_imgs / "**" / "*"), recursive=True))
        if Path(p).is_file()
    ]
    out_dir = dense / "stereo" / "depth_maps"
    logf = open(args.log, "w", encoding="utf-8") if args.log else None
    # Optional CSV to summarize per-image scale/shift diagnostics for quick drift inspection
    scales_csv = dense / "stereo" / "depthpro_scales.csv"
    csv_fh = None
    try:
        # Append mode; create header if file did not exist
        new_file = not scales_csv.exists()
        csv_fh = open(scales_csv, "a", encoding="utf-8")
        if new_file:
            csv_fh.write("name,mode,scale,shift,num_pairs,mad_z,mad_ray,fx_px,f_est_px\n")
    except Exception:
        csv_fh = None

    def log(msg: str) -> None:
        print(msg)
        if logf:
            logf.write(msg + "\n")

    if not img_paths:
        log("[error] No images found under dense/images")
        return 2

    for p_str in img_paths:
        p = Path(p_str)
        # Image name as used by COLMAP in dense/sparse/images.txt (relative path under dense/images)
        try:
            name = str(p.relative_to(undist_imgs))
        except ValueError:
            # Fallback to basename if relative fails
            name = p.name
        if sel and name not in sel:
            continue
        if name not in name2cam:
            log(f"[warn] {name} not in dense/sparse/images.txt; skip")
            continue

        cam = cams.get(name2cam[name])
        if cam is None:
            log(f"[warn] camera missing for {name}; skip")
            continue

        # Load RGB; Depth Pro helper may also estimate f from EXIF, but we prefer undistorted fx
        image_np, _, _ = depth_pro.load_rgb(str(p))
        depth_in = transform(image_np).to(dev)

        fx = float(cam.get("fx", 0.0))
        f_px = None
        if fx > 0:
            # Depth Pro expects a torch tensor for f_px (it calls squeeze()).
            # Create on the right device and with the chosen precision.
            f_px = torch.tensor(fx, device=dev, dtype=prec)

        with torch.inference_mode():
            pred = model.infer(depth_in, f_px=f_px)
            depth = pred["depth"]

        # To numpy HxW
        if hasattr(depth, "detach"):
            depth_np = depth.detach().float().cpu().numpy()
        else:
            depth_np = np.asarray(depth)

        # Ensure exact size match with undistorted camera
        if depth_np.shape != (cam["h"], cam["w"]):
            # PIL resize expects HxW float32 via mode 'F'
            depth_np = np.array(
                Image.fromarray(depth_np.astype(np.float32), mode="F").resize(
                    (cam["w"], cam["h"]), resample=Image.BILINEAR
                )
            ).astype(np.float32)

        # Sanitize predicted depth
        depth_np = depth_np.astype(np.float32)
        depth_np[~np.isfinite(depth_np)] = 0.0
        depth_np[depth_np <= 0.0] = 0.0

        # Per-image robust scale alignment using sparse SfM correspondences
        meta = imgs_meta.get(name)
        if meta is not None and len(meta.get("points2D", [])) > 0:
            mode, s, t, n_pairs, madA, madB = robust_affine_for_image_z_or_ray(
                depth_np,
                meta,
                cams,
                pts3d,
                clamp_min=float(args.scale_min),
                clamp_max=float(args.scale_max),
                shift_abs_max=float(args.shift_abs_max),
            )
            if n_pairs >= 20:
                if mode == "raylen":
                    # Convert entire map from ray length to z-depth before affine correction
                    H, W = depth_np.shape
                    jj = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
                    ii = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
                    cam = cams[meta["cam_id"]]
                    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
                    rx = (jj - cx) / (fx if fx != 0 else 1e-6)
                    ry = (ii - cy) / (fy if fy != 0 else 1e-6)
                    ray_norm = np.sqrt(1.0 + rx * rx + ry * ry).astype(np.float32)
                    depth_np = depth_np / np.maximum(ray_norm, 1e-6)
                # Apply affine correction
                depth_np = (float(s) * depth_np + float(t)).astype(np.float32)
                log(f"[scale-z] {name}: mode={mode} s={s:.3f} t={t:.3f} pairs={n_pairs} madA={madA:.4f} madB={madB:.4f}")
                # Emit a CSV row for downstream diagnostics
                try:
                    f_est = pred.get("focallength_px", None)
                    f_est_val = float(f_est.item()) if hasattr(f_est, "item") else (float(f_est) if f_est is not None else float("nan"))
                except Exception:
                    f_est_val = float("nan")
                if csv_fh is not None:
                    fx_px = float(cam.get("fx", 0.0))
                    csv_fh.write(f"{name},{mode},{float(s):.6f},{float(t):.6f},{int(n_pairs)},{madA:.6f},{madB:.6f},{fx_px:.3f},{f_est_val:.3f}\n")

        # Safety clamp (scene-dependent; conservative bounds)
        depth_np = np.clip(depth_np, 1e-3, 1e6).astype(np.float32)

        # Write depth map
        out_path = out_dir / f"{name}.photometric.bin"
        write_colmap_mat_bin(out_path, depth_np.astype(np.float32))

        # Also write a normal map estimated from depth to help fusion filtering
        # Format matches COLMAP MatBin: header "w&h&3&" followed by HxWx3 float32.
        normal_dir = dense / "stereo" / "normal_maps"
        normals = compute_normals_from_depth(depth_np, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        write_colmap_mat_bin(normal_dir / f"{name}.photometric.bin", normals.astype(np.float32))
        f_est = pred.get("focallength_px", None)
        log(f"[ok] {name} → {out_path} fx_used={f_px} f_est={getattr(f_est, 'item', lambda: f_est)() if hasattr(f_est, 'item') else f_est}")

    # Marker file to signal depths are already scale-aligned
    try:
        (dense / "stereo" / ".depthpro_scaled").write_text("scaled=1\n", encoding="utf-8")
    except Exception:
        pass

    if logf:
        logf.close()
    if csv_fh:
        try:
            csv_fh.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
