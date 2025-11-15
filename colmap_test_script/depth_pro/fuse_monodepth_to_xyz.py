#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import math
import numpy as np
from PIL import Image


def read_cameras_txt(path: Path) -> Dict[int, dict]:
    cams: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
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
                cx, cy = params[1:3]
            else:
                fx = fy = cx = cy = 0.0
            cams[cam_id] = dict(model=model, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def read_images_extrinsics(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    name2pose: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 10 or is_number(toks[-1]):
                continue
            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz = map(float, toks[5:8])
            cam_id = int(toks[8])
            name = " ".join(toks[9:])
            # Rotation matrix (world->cam)
            R = quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            name2pose[name] = (R, t, cam_id)
    return name2pose


def read_points3d(path: Path) -> Dict[int, np.ndarray]:
    pts: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            toks = line.strip().split()
            if len(toks) < 8:
                continue
            try:
                pid = int(toks[0])
                X = float(toks[1]); Y = float(toks[2]); Z = float(toks[3])
                pts[pid] = np.array([X, Y, Z], dtype=np.float64)
            except Exception:
                continue
    return pts


def read_image_observations(path: Path) -> Dict[str, List[Tuple[float, float, int]]]:
    obs: Dict[str, List[Tuple[float, float, int]]] = {}
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        toks = line.split()
        if len(toks) < 10 or is_number(toks[-1]):
            continue
        # First line header
        try:
            name = " ".join(toks[9:])
        except Exception:
            continue
        # Second line: points2D
        if i < len(lines):
            line2 = lines[i].strip()
            i += 1
        else:
            break
        vals = line2.split()
        triples: List[Tuple[float, float, int]] = []
        for j in range(0, len(vals), 3):
            try:
                x = float(vals[j]); y = float(vals[j+1]); pid = int(vals[j+2])
            except Exception:
                continue
            if pid == -1:
                continue
            triples.append((x, y, pid))
        obs[name] = triples
    return obs


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return R


def read_mat_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        while True:
            ch = f.read(1)
            if not ch:
                break
            header += ch
            if ch == b"&":
                # We expect 3 &'s for w&h&c&
                if header.count(b"&") == 3:
                    break
        header_str = header.decode("ascii")
        # header like "2000&1452&1&" or "2000&1452&3&"
        w, h, c = map(int, header_str.strip("&").split("&"))
        count = w * h * c
        data = np.fromfile(f, dtype=np.float32, count=count)
        if c == 1:
            return data.reshape(h, w)
        else:
            return data.reshape(h, w, c)


def robust_median_scale(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    # Remove non-finite and extreme outliers
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-6
    keep = np.abs(arr - med) < 3.5 * mad
    arr = arr[keep]
    if arr.size == 0:
        return None
    s = float(np.median(arr))
    # Clamp to sane range to avoid catastrophes
    return float(max(0.1, min(5.0, s)))


def fuse(dense_dir: Path, out_txt: Path, stride: int, dmin: float, dmax: float, use_color: bool, scale_to_sparse: bool) -> int:
    sparse_txt = dense_dir / "sparse"
    cams = read_cameras_txt(sparse_txt / "cameras.txt")
    name2pose = read_images_extrinsics(sparse_txt / "images.txt")
    pts3d_path = sparse_txt / "points3D.txt"
    if scale_to_sparse and pts3d_path.exists():
        pts3d = read_points3d(pts3d_path)
        obs2d = read_image_observations(sparse_txt / "images.txt")
    else:
        pts3d = {}
        obs2d = {}
    img_dir = dense_dir / "images"
    depth_dir = dense_dir / "stereo" / "depth_maps"

    names = sorted([p.name for p in img_dir.glob("*")])
    if not names:
        print("No undistorted images found", flush=True)
        return 2

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as out:
        for name in names:
            if name not in name2pose:
                continue
            depth_path = depth_dir / f"{name}.photometric.bin"
            if not depth_path.exists():
                continue

            R, t, cam_id = name2pose[name]
            cam = cams.get(cam_id)
            if cam is None:
                continue

            depth = read_mat_bin(depth_path)
            # Optionally estimate a per-image scale factor using sparse correspondences
            scale = 1.0
            if scale_to_sparse and name in obs2d:
                R, t, _ = name2pose[name]
                H, W = depth.shape
                # Build quick bilinear sampler for monodepth
                def sample_depth(u: float, v: float) -> Optional[float]:
                    if u < 0 or v < 0 or u >= W-1 or v >= H-1:
                        return None
                    u0 = int(np.floor(u)); v0 = int(np.floor(v))
                    du = float(u - u0); dv = float(v - v0)
                    d00 = depth[v0, u0]; d10 = depth[v0, u0+1]; d01 = depth[v0+1, u0]; d11 = depth[v0+1, u0+1]
                    if not (np.isfinite(d00) and np.isfinite(d10) and np.isfinite(d01) and np.isfinite(d11)):
                        return None
                    return float(
                        (1-du)*(1-dv)*d00 + du*(1-dv)*d10 + (1-du)*dv*d01 + du*dv*d11
                    )
                scales: List[float] = []
                for (x, y, pid) in obs2d.get(name, [])[:2000]:
                    P = pts3d.get(pid)
                    if P is None:
                        continue
                    Pc = R @ P.reshape(3, 1) + t.reshape(3, 1)  # world->cam
                    z_gt = float(Pc[2, 0])
                    if z_gt <= 0:
                        continue
                    d_pred = sample_depth(x, y)
                    if d_pred is None or d_pred <= 0:
                        continue
                    r = z_gt / d_pred
                    if np.isfinite(r) and 0.01 < r < 100:
                        scales.append(float(r))
                s_est = robust_median_scale(scales)
                if s_est is not None:
                    scale = s_est
            if scale != 1.0:
                depth = depth * float(scale)
            h, w = depth.shape

            if use_color:
                try:
                    im = np.asarray(Image.open(img_dir / name).convert("RGB"))
                except Exception:
                    im = None
                    use_color = False
            else:
                im = None

            # Prepare grid indices
            ys = np.arange(0, h, stride)
            xs = np.arange(0, w, stride)
            uu, vv = np.meshgrid(xs, ys)
            d = depth[vv, uu]
            mask = np.isfinite(d) & (d > dmin) & (d < dmax)
            if not np.any(mask):
                continue

            uu = uu[mask].astype(np.float64)
            vv = vv[mask].astype(np.float64)
            d = d[mask].astype(np.float64)

            fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
            x = (uu - cx) * d / fx
            y = (vv - cy) * d / fy
            z = d
            Xc = np.stack([x, y, z], axis=1)

            # World coordinates: Xw = R^T * (Xc) - R^T * t
            Rt = R.T
            Xw = (Rt @ Xc.T).T - (Rt @ t.reshape(3, 1)).ravel()

            if im is not None:
                cols = im[vv.astype(np.int64), uu.astype(np.int64)]
            else:
                cols = np.zeros((Xw.shape[0], 3), dtype=np.uint8)

            for (X, Y, Z), (r, g, b) in zip(Xw, cols):
                out.write(f"{X} {Y} {Z} {int(r)} {int(g)} {int(b)}\n")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fuse DepthPro monocular depths into XYZRGB by backprojection")
    ap.add_argument("--dense_dir", required=True, help="Path to <run>/dense")
    ap.add_argument("--out", default=None, help="Output TXT path (xyzrgb)")
    ap.add_argument("--stride", type=int, default=12, help="Sampling stride in pixels")
    ap.add_argument("--min_depth", type=float, default=0.2, help="Min valid depth [m]")
    ap.add_argument("--max_depth", type=float, default=500.0, help="Max valid depth [m]")
    ap.add_argument("--color", action="store_true", help="Sample RGB from undistorted images")
    ap.add_argument("--scale_to_sparse", action="store_true", help="Estimate per-image scale from sparse 3D points to reduce multi-copy artifacts")
    args = ap.parse_args()

    dense_dir = Path(args.dense_dir)
    out_txt = (
        Path(args.out) if args.out else dense_dir / "fused_xyzrgb.txt"
    )
    return fuse(dense_dir, out_txt, args.stride, args.min_depth, args.max_depth, args.color, args.scale_to_sparse)


if __name__ == "__main__":
    raise SystemExit(main())
