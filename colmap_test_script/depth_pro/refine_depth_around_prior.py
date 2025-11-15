#!/usr/bin/env python3
"""
Local multi-view refinement around prior depth (DepthPro) using narrow search.

For each undistorted image (reference), and each pixel (optionally strided),
search a small depth range around the prior depth d0 (e.g., ±3%) and pick the
depth that minimizes a multi-view photometric cost (ZNCC) over a small window
across a handful of neighboring views.

Inputs under <run>/dense (from image_undistorter):
  - images/                      # undistorted images
  - sparse/{cameras.txt,images.txt,points3D.txt}
  - stereo/depth_maps/*.photometric.bin  # prior depth (DepthPro)

Outputs (in-place):
  - Overwrites *.photometric.bin with refined depth
  - Updates stereo/normal_maps/*.photometric.bin normals accordingly

Notes:
  - Neighbor view selection is geometric: pick K nearest camera centers.
  - Cost: average 1 - ZNCC(ref_patch, nbr_patch) across valid neighbors.
  - This is CPU-only and intended for small epsilon and small windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return R


def read_images_poses(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    out: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
            R = quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            out[name] = (R, t, cam_id)
    return out


def read_mat_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        amp = 0
        while True:
            b = f.read(1)
            if not b:
                break
            header += b
            if b == b"&":
                amp += 1
                if amp == 3:
                    break
        w, h, c = map(int, header.decode("ascii").strip("&").split("&"))
        data = np.fromfile(f, dtype=np.float32, count=w * h * c)
        if c == 1:
            return data.reshape(h, w)
        else:
            return data.reshape(h, w, c)


def write_mat_bin(path: Path, arr: np.ndarray) -> None:
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


def compute_normals_from_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth.shape
    uu = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    vv = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    Z = depth.astype(np.float32)
    X = (uu - cx) * Z / (fx if fx != 0 else 1e-6)
    Y = (vv - cy) * Z / (fy if fy != 0 else 1e-6)
    dXu = X[:, 1:] - X[:, :-1]
    dYu = Y[:, 1:] - Y[:, :-1]
    dZu = Z[:, 1:] - Z[:, :-1]
    dXv = X[1:, :] - X[:-1, :]
    dYv = Y[1:, :] - Y[:-1, :]
    dZv = Z[1:, :] - Z[:-1, :]
    ax = np.stack([dXu[:-1, :], dYu[:-1, :], dZu[:-1, :]], axis=2)
    ay = np.stack([dXv[:, :-1], dYv[:, :-1], dZv[:, :-1]], axis=2)
    # Match PatchMatch convention: camera-facing normals (Nz<=0)
    n = np.cross(ay, ax)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    n = n / norm
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:-1, :-1, :] = n.astype(np.float32)
    normals[-1, :-1, :] = normals[-2, :-1, :]
    normals[:-1, -1, :] = normals[:-1, -2, :]
    normals[-1, -1, :] = normals[-2, -2, :]
    invalid = ~np.isfinite(Z) | (Z <= 0)
    normals[invalid] = 0.0
    flip = (~invalid) & (normals[..., 2] > 0)
    normals[flip] *= -1.0
    return normals


def to_gray_float(im_path: Path) -> np.ndarray:
    im = Image.open(im_path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


def bilinear_at(img: np.ndarray, x: float, y: float) -> Optional[float]:
    h, w = img.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return None
    x0 = int(math.floor(x)); y0 = int(math.floor(y))
    dx = x - x0; dy = y - y0
    v00 = img[y0, x0]
    v10 = img[y0, x0 + 1]
    v01 = img[y0 + 1, x0]
    v11 = img[y0 + 1, x0 + 1]
    return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11


def bilinear_patch(img: np.ndarray, xc: float, yc: float, win: int) -> Optional[np.ndarray]:
    # Returns (2*win+1)^2 patch or None if any sample out of bounds
    sz = 2 * win + 1
    patch = np.empty((sz, sz), dtype=np.float32)
    for dv in range(-win, win + 1):
        for du in range(-win, win + 1):
            val = bilinear_at(img, xc + du, yc + dv)
            if val is None:
                return None
            patch[dv + win, du + win] = float(val)
    return patch


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    am = a - float(a.mean())
    bm = b - float(b.mean())
    da = float(np.sqrt(np.sum(am * am)) + 1e-6)
    db = float(np.sqrt(np.sum(bm * bm)) + 1e-6)
    return float(np.sum(am * bm) / (da * db))


def refine_image(
    name: str,
    dense_dir: Path,
    cams: Dict[int, dict],
    poses: Dict[str, Tuple[np.ndarray, np.ndarray, int]],
    neighbors: List[str],
    epsilon: float,
    win: int,
    iters: int,
    stride: int,
) -> int:
    img_dir = dense_dir / "images"
    depth_dir = dense_dir / "stereo" / "depth_maps"
    normal_dir = dense_dir / "stereo" / "normal_maps"

    if name not in poses:
        return 0
    Rr, tr, cam_id_r = poses[name]
    cam_r = cams.get(cam_id_r)
    if cam_r is None:
        return 0
    fxr, fyr, cxr, cyr = map(float, (cam_r["fx"], cam_r["fy"], cam_r["cx"], cam_r["cy"]))
    Hr, Wr = int(cam_r["h"]), int(cam_r["w"])

    depth_path = depth_dir / f"{name}.photometric.bin"
    if not depth_path.exists():
        return 0
    depth = read_mat_bin(depth_path).astype(np.float32)
    if depth.shape != (Hr, Wr):
        # Resize to expected shape
        depth = np.array(Image.fromarray(depth, mode="F").resize((Wr, Hr), resample=Image.BILINEAR), dtype=np.float32)

    ref_img = to_gray_float(img_dir / name)
    if ref_img.shape != (Hr, Wr):
        ref_img = np.array(Image.fromarray((ref_img * 255).astype(np.uint8), mode="L").resize((Wr, Hr), resample=Image.BILINEAR), dtype=np.uint8)
        ref_img = ref_img.astype(np.float32) / 255.0

    # Load neighbor images and poses
    nbr_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, int, int]] = []
    for nb in neighbors:
        if nb not in poses:
            continue
        Rn, tn, cam_id_n = poses[nb]
        cam_n = cams.get(cam_id_n)
        if cam_n is None:
            continue
        inn = to_gray_float(img_dir / nb)
        Hn, Wn = int(cam_n["h"]), int(cam_n["w"])
        if inn.shape != (Hn, Wn):
            inn = np.array(Image.fromarray((inn * 255).astype(np.uint8), mode="L").resize((Wn, Hn), resample=Image.BILINEAR), dtype=np.uint8)
            inn = inn.astype(np.float32) / 255.0
        fxn, fyn, cxn, cyn = map(float, (cam_n["fx"], cam_n["fy"], cam_n["cx"], cam_n["cy"]))
        nbr_data.append((inn, Rn, tn, fxn, fyn, cxn, cyn, Hn, Wn))
    if not nbr_data:
        return 0

    RrT = Rr.T
    refined = depth.copy()
    eps = float(epsilon)

    # Sampling depths: 5 hypotheses across ±eps around d0
    def depth_candidates(d0: float) -> np.ndarray:
        if not math.isfinite(d0) or d0 <= 0:
            return np.array([], dtype=np.float32)
        offs = np.linspace(-eps, eps, 5, dtype=np.float32)
        ds = d0 * (1.0 + offs)
        ds = ds[ds > 0]
        return ds.astype(np.float32)

    for it in range(int(iters)):
        # Iterate pixels with optional stride
        for v in range(0, Hr, int(stride)):
            if v % max(32, int(8 * stride)) == 0:
                # Lightweight progress ping
                pct = 100.0 * v / max(1, Hr - 1)
                print(f"[refine] {name}: iter {it+1}/{iters} row {v}/{Hr} ({pct:.1f}%)", flush=True)
            for u in range(0, Wr, int(stride)):
                d0 = float(refined[v, u])
                if not (math.isfinite(d0) and d0 > 0):
                    continue
                dcands = depth_candidates(d0)
                if dcands.size == 0:
                    continue
                # Reference patch
                ref_patch = bilinear_patch(ref_img, float(u), float(v), int(win))
                if ref_patch is None:
                    continue
                best_cost = float("inf")
                best_d = d0
                for d in dcands:
                    # 3D point in ref camera frame and world
                    Xcr = np.array([
                        (u - cxr) * d / (fxr if fxr != 0 else 1e-6),
                        (v - cyr) * d / (fyr if fyr != 0 else 1e-6),
                        d,
                    ], dtype=np.float64)
                    Xw = RrT @ (Xcr - tr.reshape(3))

                    costs: List[float] = []
                    for inn, Rn, tn, fxn, fyn, cxn, cyn, Hn, Wn in nbr_data:
                        Xcn = Rn @ Xw.reshape(3) + tn.reshape(3)
                        zn = float(Xcn[2])
                        if zn <= 0:
                            continue
                        un = fxn * Xcn[0] / zn + cxn
                        vn = fyn * Xcn[1] / zn + cyn
                        p_n = bilinear_patch(inn, un, vn, int(win))
                        if p_n is None:
                            continue
                        c = 1.0 - zncc(ref_patch, p_n)
                        costs.append(float(c))
                    if not costs:
                        continue
                    c_avg = float(np.mean(costs))
                    if c_avg < best_cost:
                        best_cost = c_avg
                        best_d = float(d)
                refined[v, u] = float(best_d)

    # Write back refined map and updated normals
    refined = np.clip(refined.astype(np.float32), 1e-3, 1e6)
    write_mat_bin(depth_path, refined)
    normals = compute_normals_from_depth(refined, fxr, fyr, cxr, cyr)
    write_mat_bin(normal_dir / f"{name}.photometric.bin", normals.astype(np.float32))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Local multi-view refinement around prior depth")
    ap.add_argument("--dense_dir", required=True, help="Path to <run>/dense")
    ap.add_argument("--epsilon", type=float, default=0.03, help="Fractional search range (±epsilon)")
    ap.add_argument("--neighbors", type=int, default=6, help="Number of neighbor views")
    ap.add_argument("--win", type=int, default=3, help="Patch half window")
    ap.add_argument("--iters", type=int, default=1, help="Refinement iterations")
    ap.add_argument("--stride", type=int, default=4, help="Pixel stride for refinement")
    args = ap.parse_args()

    dense = Path(args.dense_dir)
    img_dir = dense / "images"
    cams = read_cameras_txt(dense / "sparse" / "cameras.txt")
    poses = read_images_poses(dense / "sparse" / "images.txt")

    # List candidate images by presence in undistorted images dir
    names = sorted([p.name for p in img_dir.glob("**/*") if p.is_file()])
    if not names:
        print("[refine] No undistorted images found", flush=True)
        return 2

    # Precompute camera centers for neighbor selection
    centers: Dict[str, np.ndarray] = {}
    for nm, (R, t, _cid) in poses.items():
        C = -R.T @ t.reshape(3)
        centers[nm] = C.reshape(3)

    refined_count = 0
    for name in names:
        if name not in poses:
            continue
        # Select nearest neighbors by camera center distance (exclude self)
        Cr = centers.get(name)
        if Cr is None:
            continue
        dists: List[Tuple[float, str]] = []
        for nm, Cn in centers.items():
            if nm == name:
                continue
            dists.append((float(np.linalg.norm(Cn - Cr)), nm))
        dists.sort(key=lambda x: x[0])
        nbs = [nm for _, nm in dists[: max(0, int(args.neighbors))]]
        updated = refine_image(
            name,
            dense,
            cams,
            poses,
            nbs,
            float(args.epsilon),
            int(args.win),
            int(args.iters),
            int(args.stride),
        )
        refined_count += updated
        if updated:
            print(f"[refine] {name}: updated", flush=True)
        else:
            print(f"[refine] {name}: skipped", flush=True)

    print(f"[refine] Done. Updated {refined_count} image(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
