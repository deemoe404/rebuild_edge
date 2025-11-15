#!/usr/bin/env python3
"""
Normal Visualization Utility: Convert COLMAP/NumPy normal maps to PNG.

Purpose
- Quickly preview surface normals produced by COLMAP or related pipelines by
  converting HxWx3 float32 normal maps into viewable RGB images.

Supported Inputs
- COLMAP MatBin (.photometric.bin): ASCII header "w&h&c&" then row‑major float32.
  Expects c==3 for normals (HxWx3). If c!=3, tries to adapt common layouts.
- NumPy .npy / .npz: arrays shaped HxWx3 or 3xHxW (first 3 channels used).

Output
- A single PNG saved to --out (default: <input>.normals.png) where normals in
  [-1, 1] are linearly mapped to [0, 255] per channel. Invalid vectors render
  as black.

Notes
- This is a visualization; it does not alter normals. For depth visualization,
  use scripts/visualize/depth_bin_to_png.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def read_mat_bin(path: Path) -> np.ndarray:
    """Read COLMAP MatBin array written as "w&h&c&" + float32 data.

    Returns np.ndarray with shape (H, W) if c==1, or (H, W, C) otherwise.
    """
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
        try:
            w, h, c = map(int, header.decode("ascii").strip("&").split("&"))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Invalid MatBin header in {path}: {header!r}") from e
        data = np.fromfile(f, dtype=np.float32, count=w * h * c)
        if c == 1:
            return data.reshape(h, w)
        else:
            return data.reshape(h, w, c)


def load_normals_any(path: Path) -> np.ndarray:
    """Load normal map from MatBin/NumPy and return HxWx3 float32 array.

    Accepts:
      - HxWx3
      - 3xHxW
      - HxWxC (C>=3) → take first 3 channels
      - CxHxW (C>=3) → take first 3 channels
    """
    suf = path.suffix.lower()
    if suf == ".bin":
        arr = read_mat_bin(path)
    elif suf == ".npy":
        arr = np.load(path)
    elif suf == ".npz":
        npz = np.load(path)
        # Prefer common keys
        for k in ("normals", "normal", "arr_0"):
            if k in npz:
                arr = npz[k]
                break
        else:
            keys = list(npz.keys())
            if not keys:
                raise RuntimeError(f"Empty npz file: {path}")
            arr = npz[keys[0]]
    else:
        # Default to MatBin
        arr = read_mat_bin(path)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        raise ValueError(f"Expected 3-channel normals; got 2D array {arr.shape}")
    if arr.ndim == 3:
        h, w, c = arr.shape
        # HxWxC or CxHxW
        if h in (1, 2, 3) and c not in (1, 2, 3):
            # Likely CHW; transpose to HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] < 3:
            raise ValueError(f"Array has <3 channels: {arr.shape}")
        return arr[..., :3]
    if arr.ndim == 4:
        # Handle NCHW/HWCN batches by taking first item
        if arr.shape[0] <= 8 and arr.shape[1] in (1, 2, 3):
            # NCHW -> take first and transpose
            arr = np.transpose(arr[0, :3, ...], (1, 2, 0))
            return arr
        if arr.shape[-1] in (1, 2, 3):
            # NHWC -> take first
            return arr[0, ..., :3]
    raise ValueError(f"Unsupported normals array shape: {arr.shape}")


def normalize_safe(n: np.ndarray) -> np.ndarray:
    """Ensure normals are unit length where valid; keep zeros for invalid.

    n: HxWx3 float32; returns same shape.
    """
    # Identify invalid as non-finite or near-zero norm
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    invalid = ~np.isfinite(n).all(axis=2, keepdims=True) | (norm < 1e-8)
    norm_safe = np.where(norm < 1e-8, 1.0, norm)
    n_unit = n / norm_safe
    n_unit[invalid.squeeze(axis=2)] = 0.0
    return n_unit


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    """Map normals in [-1,1] to uint8 RGB in [0,255]. Invalid → black.

    normals: HxWx3 float32
    """
    n = np.asarray(normals, dtype=np.float32)
    # Assume normals roughly in [-1,1]; clamp to be safe
    n = np.clip(n, -1.0, 1.0)
    rgb = ((n + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
    # Black-out invalid (zero vectors)
    invalid = ~np.isfinite(normals).any(axis=2) | (np.linalg.norm(normals, axis=2) < 1e-8)
    if np.any(invalid):
        rgb[invalid] = np.array([0, 0, 0], dtype=np.uint8)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert normal maps (.bin/.npy/.npz) to PNG")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input normal file (.photometric.bin, .npy, .npz)")
    ap.add_argument("--out", dest="out", default=None, help="Output image path (.png). Defaults to <in>.normals.png")
    ap.add_argument("--unit", action="store_true", help="Re-normalize vectors to unit length before coloring")
    args = ap.parse_args()

    in_path = Path(args.inp)
    normals = load_normals_any(in_path)
    if args.unit:
        normals = normalize_safe(normals)

    rgb = normals_to_rgb(normals)
    pil = Image.fromarray(rgb, mode="RGB")

    default_out = in_path.with_suffix(in_path.suffix + ".normals.png")
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    print(f"[ok] Wrote {out_path} from {in_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

