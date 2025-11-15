#!/usr/bin/env python3
"""
Scale COLMAP photometric depth bins (DepthPro outputs) per image so that the
predicted depths align to the sparse SfM model's metric scale.

Inputs (within <run>/dense):
  - sparse/cameras.txt
  - sparse/images.txt
  - sparse/points3D.txt
  - stereo/depth_maps/<name>.photometric.bin

Output (in-place):
  - Overwrites each depth bin with depth * per_image_scale (if a robust scale
    can be estimated). If no valid scale is found for an image, it is left as-is.

This reduces the classic "multi-copy/translated layers" artifact when fusing
monocular depth by enforcing a consistent per-view metric scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math
import numpy as np


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


def read_images_extrinsics_and_obs(path: Path) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, int]], Dict[str, List[Tuple[float, float, int]]]]:
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    name2pose: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    obs: Dict[str, List[Tuple[float, float, int]]] = {}
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
        try:
            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz = map(float, toks[5:8])
            cam_id = int(toks[8])
            name = " ".join(toks[9:])
        except Exception:
            continue
        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        name2pose[name] = (R, t, cam_id)

        # Points2D line
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
    return name2pose, obs


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


def read_mat_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        amp_count = 0
        while True:
            ch = f.read(1)
            if not ch:
                break
            header += ch
            if ch == b"&":
                amp_count += 1
                if amp_count == 3:
                    break
        header_str = header.decode("ascii")
        w, h, c = map(int, header_str.strip("&").split("&"))
        count = w * h * c
        data = np.fromfile(f, dtype=np.float32, count=count)
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


def robust_median_scale(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)
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
    return float(max(0.05, min(20.0, s)))


def estimate_scale_for_image(
    name: str,
    depth: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    obs2d: List[Tuple[float, float, int]],
    pts3d: Dict[int, np.ndarray],
    max_obs: int = 3000,
) -> Optional[float]:
    H, W = depth.shape

    def sample_depth(u: float, v: float) -> Optional[float]:
        if u < 0 or v < 0 or u >= W - 1 or v >= H - 1:
            return None
        u0 = int(np.floor(u)); v0 = int(np.floor(v))
        du = float(u - u0); dv = float(v - v0)
        d00 = depth[v0, u0]; d10 = depth[v0, u0 + 1]
        d01 = depth[v0 + 1, u0]; d11 = depth[v0 + 1, u0 + 1]
        if not (np.isfinite(d00) and np.isfinite(d10) and np.isfinite(d01) and np.isfinite(d11)):
            return None
        return float((1 - du) * (1 - dv) * d00 + du * (1 - dv) * d10 + (1 - du) * dv * d01 + du * dv * d11)

    samples: List[float] = []
    for (x, y, pid) in obs2d[:max_obs]:
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
            samples.append(float(r))
    return robust_median_scale(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-image scale alignment for COLMAP photometric depth bins using sparse SfM")
    ap.add_argument("--dense_dir", required=True, help="Path to <run>/dense")
    ap.add_argument("--dry_run", action="store_true", help="Only report scale factors without modifying files")
    args = ap.parse_args()

    dense = Path(args.dense_dir)
    sparse_txt = dense / "sparse"
    depth_dir = dense / "stereo" / "depth_maps"

    cams = read_cameras_txt(sparse_txt / "cameras.txt")
    name2pose, obs = read_images_extrinsics_and_obs(sparse_txt / "images.txt")
    pts3d = read_points3d(sparse_txt / "points3D.txt")

    depth_bins = sorted(depth_dir.glob("*.photometric.bin"))
    if not depth_bins:
        print("[warn] No photometric depth bins found; nothing to scale")
        return 0

    changed = 0
    for db_path in depth_bins:
        # COLMAP/DepthPro naming: <name>.photometric.bin where <name> matches images.txt NAME
        name = db_path.name
        img_name = name[:-len(".photometric.bin")] if name.endswith(".photometric.bin") else name
        if img_name not in name2pose:
            print(f"[skip] {name}: pose not found in sparse/images.txt")
            continue
        R, t, _ = name2pose[img_name]
        obs2d = obs.get(img_name, [])
        if not obs2d:
            print(f"[skip] {name}: no 2D-3D observations")
            continue
        depth_path = db_path
        try:
            depth = read_mat_bin(depth_path)
        except Exception as e:
            print(f"[skip] {name}: failed to read depth bin: {e}")
            continue
        s = estimate_scale_for_image(img_name, depth, R, t, obs2d, pts3d)
        if s is None:
            print(f"[no-scale] {name}: insufficient/unstable matches")
            continue
        if 0.98 <= s <= 1.02:
            print(f"[ok] {name}: scale≈1.00 (s={s:.3f})")
            continue
        print(f"[apply] {name}: scale {s:.3f}")
        if not args.dry_run:
            write_mat_bin(depth_path, (depth * s).astype(np.float32))
            changed += 1

    print(f"[done] Updated {changed} depth map(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
