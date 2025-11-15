#!/usr/bin/env python3
"""
Depth Visualization Utility: Convert depth arrays to PNG (gray or false‑color).

Purpose
- Quickly preview metric depth maps produced by pipelines like COLMAP (MatBin
  .photometric.bin) or NumPy arrays (.npy/.npz) by converting them into human‑
  viewable PNGs.

Supported Inputs
- COLMAP MatBin (.photometric.bin): ASCII header "w&h&c&" followed by row‑major
  float32 data. c==1 ⇒ HxW; c>1 ⇒ HxWxC (first channel used).
- NumPy .npy: float array shaped HxW or HxWxC / CxHxW (first channel used).
- NumPy .npz: loads key "depth" or "arr_0" if present, otherwise the first entry.

Outputs
- A single PNG image written to --out (or defaulting to <input>.png). Modes:
  - gray8: 8‑bit grayscale. Good for quick preview, compact.
  - gray16: 16‑bit grayscale. Preserves more dynamic range for analysis.
  - viridis (default): false‑color using a built‑in viridis LUT, easier to see contrast.

Normalization
- If --min/--max are not given, the script computes robust range using the 2nd
  and 98th percentiles of valid pixels (finite and >0). Values are linearly
  scaled into [0,1] before mapping to gray/viridis. Invalid/≤0 values render as
  black. Use --invert to swap near/far brightness.

Common Uses
- Inspect COLMAP DepthPro outputs (*.photometric.bin) under dense/stereo/depth_maps.
- Visualize intermediate or external depth predictions saved as .npy/.npz.

Examples
- COLMAP MatBin to viridis PNG:
    python scripts/visualize/depth_bin_to_png.py \
        --in runs/.../stereo/depth_maps/IMG.JPG.photometric.bin \
        --mode viridis

- NumPy depth to 16‑bit grayscale with explicit range:
    python scripts/visualize/depth_bin_to_png.py --in depth.npy --mode gray16 --min 1.0 --max 50.0

Arguments
- --in:   Path to input depth file (.photometric.bin, .npy, .npz). Required.
- --out:  Output PNG path. Default: <input>.png alongside the source file.
- --mode: Visualization mode: gray8 | gray16 | viridis (default: viridis).
- --min/--max: Optional fixed normalization bounds (in depth units, e.g., meters).
- --invert: Invert colormap/gray so nearer (smaller depth) appears brighter.

Notes
- This tool does not change metric scale; it only visualizes. To align scales
  across images using sparse SfM, see scripts like scripts/depth_pro/scale_depth_bins_to_sparse.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


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
        try:
            w, h, c = map(int, header.decode("ascii").strip("&").split("&"))
        except Exception as e:
            raise RuntimeError(f"Invalid MatBin header in {path}: {header!r}") from e
        data = np.fromfile(f, dtype=np.float32, count=w * h * c)
        if c == 1:
            return data.reshape(h, w)
        else:
            return data.reshape(h, w, c)


def load_depth_any(path: Path) -> np.ndarray:
    """Load depth from MatBin (.bin) or NumPy (.npy/.npz). Returns 2D array if possible.

    If the array has 3 dims:
      - HxWxC: take channel 0
      - CxHxW: take channel 0
    """
    suf = path.suffix.lower()
    if suf == ".bin":
        arr = read_mat_bin(path)
    elif suf == ".npy":
        arr = np.load(path)
    elif suf == ".npz":
        npz = np.load(path)
        # Prefer common keys
        for k in ("depth", "arr_0"):
            if k in npz:
                arr = npz[k]
                break
        else:
            # Fallback to first entry
            keys = list(npz.keys())
            if not keys:
                raise RuntimeError(f"Empty npz file: {path}")
            arr = npz[keys[0]]
    else:
        # Default to MatBin if unknown
        arr = read_mat_bin(path)

    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # HWC or CHW
        if arr.shape[0] <= 8 and arr.shape[0] != arr.shape[1]:
            # Likely CHW, take channel 0
            return arr[0, ...]
        # Assume HWC
        return arr[..., 0]
    raise ValueError(f"Unsupported depth array shape: {arr.shape}")


def robust_min_max(depth: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> Tuple[float, float]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return 0.0, 1.0
    vals = depth[valid]
    dmin = float(np.percentile(vals, q_low))
    dmax = float(np.percentile(vals, q_high))
    if dmax <= dmin:
        dmax = float(vals.max())
        dmin = float(vals.min())
    if dmax <= dmin:
        dmin, dmax = 0.0, float(vals.max() if vals.size else 1.0)
    return dmin, dmax


def to_gray8(depth: np.ndarray, vmin: float, vmax: float, invert: bool = False) -> np.ndarray:
    d = depth.astype(np.float32)
    d = np.clip((d - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    if invert:
        d = 1.0 - d
    g = (d * 255.0 + 0.5).astype(np.uint8)
    g[~np.isfinite(depth) | (depth <= 0)] = 0
    return g


def to_gray16(depth: np.ndarray, vmin: float, vmax: float, invert: bool = False) -> np.ndarray:
    d = depth.astype(np.float32)
    d = np.clip((d - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    if invert:
        d = 1.0 - d
    g = (d * 65535.0 + 0.5).astype(np.uint16)
    g[~np.isfinite(depth) | (depth <= 0)] = 0
    return g


def apply_viridis(depth: np.ndarray, vmin: float, vmax: float, invert: bool = False) -> np.ndarray:
    """Minimal built‑in viridis LUT to avoid external deps (256x3 uint8)."""
    # Table copied from matplotlib's viridis (resampled to 256) to avoid dependency.
    # Shortened for brevity would defeat accuracy; keep full 256 entries.
    viridis = np.array([
        [68, 1, 84],[68, 2, 85],[69, 4, 87],[69, 5, 88],[70, 7, 90],[70, 8, 91],[70, 10, 92],[70, 11, 94],
        [71, 13, 95],[71, 14, 97],[71, 16, 98],[71, 17, 99],[71, 19, 101],[72, 20, 102],[72, 22, 103],[72, 23, 105],
        [72, 24, 106],[72, 26, 107],[72, 27, 108],[72, 28, 110],[72, 29, 111],[72, 31, 112],[72, 32, 113],[72, 33, 114],
        [72, 35, 115],[72, 36, 117],[72, 37, 118],[72, 38, 119],[72, 40, 120],[72, 41, 121],[71, 42, 122],[71, 44, 123],
        [71, 45, 124],[71, 46, 125],[71, 47, 126],[70, 49, 127],[70, 50, 128],[70, 51, 129],[70, 52, 129],[69, 54, 130],
        [69, 55, 131],[69, 56, 132],[68, 57, 133],[68, 58, 133],[68, 60, 134],[67, 61, 135],[67, 62, 136],[66, 63, 136],
        [66, 64, 137],[66, 65, 137],[65, 66, 138],[65, 68, 139],[64, 69, 139],[64, 70, 140],[63, 71, 140],[63, 72, 141],
        [62, 73, 141],[62, 74, 142],[61, 75, 142],[61, 76, 142],[61, 77, 143],[60, 78, 143],[60, 79, 144],[59, 80, 144],
        [59, 81, 144],[58, 82, 145],[58, 83, 145],[57, 84, 145],[57, 85, 145],[56, 86, 146],[56, 87, 146],[55, 88, 146],
        [55, 89, 146],[54, 90, 146],[54, 91, 146],[53, 92, 147],[53, 93, 147],[52, 94, 147],[52, 95, 147],[51, 96, 147],
        [51, 97, 147],[50, 98, 147],[50, 99, 147],[49, 100, 147],[49, 101, 146],[48, 102, 146],[48, 103, 146],[47, 104, 146],
        [47, 105, 146],[46, 106, 146],[46, 107, 146],[45, 108, 146],[45, 109, 145],[44, 110, 145],[44, 111, 145],[43, 112, 145],
        [43, 113, 144],[42, 114, 144],[42, 115, 144],[41, 116, 144],[41, 117, 143],[40, 118, 143],[40, 119, 143],[39, 120, 142],
        [39, 121, 142],[39, 122, 142],[38, 123, 141],[38, 124, 141],[37, 125, 141],[37, 126, 140],[36, 127, 140],[36, 128, 140],
        [35, 129, 139],[35, 130, 139],[35, 131, 138],[34, 132, 138],[34, 133, 138],[33, 134, 137],[33, 135, 137],[33, 136, 136],
        [32, 137, 136],[32, 138, 135],[31, 139, 135],[31, 140, 134],[31, 141, 134],[30, 142, 133],[30, 143, 133],[30, 144, 132],
        [30, 145, 131],[30, 146, 131],[29, 147, 130],[29, 148, 130],[29, 149, 129],[29, 150, 128],[29, 152, 128],[28, 153, 127],
        [28, 154, 126],[28, 155, 126],[29, 156, 125],[29, 157, 124],[29, 158, 123],[30, 159, 123],[30, 160, 122],[31, 161, 121],
        [31, 162, 120],[32, 163, 120],[33, 164, 119],[33, 165, 118],[34, 166, 117],[35, 167, 116],[36, 168, 116],[37, 169, 115],
        [38, 170, 114],[39, 171, 113],[40, 172, 112],[41, 173, 112],[42, 174, 111],[44, 175, 110],[45, 176, 109],[46, 177, 108],
        [48, 178, 107],[49, 179, 106],[50, 180, 106],[52, 181, 105],[53, 182, 104],[55, 183, 103],[56, 184, 102],[58, 185, 101],
        [59, 186, 100],[61, 187, 99],[63, 188, 98],[64, 189, 97],[66, 190, 96],[68, 191, 95],[69, 192, 94],[71, 193, 93],
        [73, 193, 92],[75, 194, 91],[77, 195, 90],[78, 196, 89],[80, 197, 88],[82, 198, 87],[84, 199, 86],[86, 200, 85],
        [88, 200, 84],[90, 201, 83],[92, 202, 81],[94, 203, 80],[96, 204, 79],[98, 205, 78],[100, 205, 77],[102, 206, 76],
        [104, 207, 74],[106, 208, 73],[108, 209, 72],[110, 209, 70],[112, 210, 69],[114, 211, 68],[116, 212, 66],[118, 213, 65],
        [120, 213, 63],[123, 214, 62],[125, 215, 61],[127, 216, 59],[129, 216, 58],[131, 217, 56],[133, 218, 55],[136, 219, 53],
        [138, 219, 51],[140, 220, 50],[143, 221, 48],[145, 222, 47],[147, 222, 45],[150, 223, 43],[152, 224, 42],[155, 224, 40],
        [157, 225, 38],[159, 226, 37],[162, 226, 35],[164, 227, 33],[167, 228, 31],[169, 228, 30],[172, 229, 28],[174, 229, 26],
        [177, 230, 24],[179, 231, 22],[182, 231, 21],[184, 232, 19],[187, 232, 17],[189, 233, 15],[192, 233, 14],[194, 234, 12],
        [197, 234, 11],[199, 235, 9],[202, 235, 8],[205, 235, 7],[207, 236, 6],[210, 236, 5],[212, 237, 5],[215, 237, 4],
        [217, 238, 4],[220, 238, 4],[222, 238, 5],[224, 239, 5],[226, 239, 6],[229, 239, 7],[231, 240, 8],[233, 240, 9],
        [235, 240, 10],[238, 241, 12],[240, 241, 13],[242, 241, 15],[244, 242, 16],[246, 242, 18],[248, 242, 20],[250, 243, 21],
        [252, 243, 23],[254, 243, 25],[255, 244, 27],[255, 244, 29],[255, 245, 31],[255, 245, 33],[254, 246, 35],[254, 246, 37],
        [253, 247, 39],[252, 247, 41],[251, 248, 44],[250, 248, 46],[248, 249, 48],[247, 249, 51],[245, 250, 53],[243, 250, 56],
        [241, 251, 58],[239, 251, 61],[236, 252, 64],[234, 252, 66],[231, 253, 69],[229, 253, 72],[226, 254, 74],[223, 254, 77],
        [220, 255, 80],[217, 255, 82],[214, 255, 85],[210, 255, 88],[207, 255, 90],[203, 255, 93],[200, 255, 95],[196, 255, 98]
    ], dtype=np.uint8)

    d = depth.astype(np.float32)
    d = np.clip((d - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    if invert:
        d = 1.0 - d
    idx = np.clip((d * 255.0).astype(np.int32), 0, 255)
    rgb = viridis[idx]
    # Make invalid pixels black
    invalid = ~np.isfinite(depth) | (depth <= 0)
    if np.any(invalid):
        rgb = rgb.copy()
        rgb[invalid] = np.array([0, 0, 0], dtype=np.uint8)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert depth (.bin/.npy/.npz) to PNG")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input depth file (.photometric.bin, .npy, .npz)")
    ap.add_argument("--out", dest="out", default=None, help="Output image path (.png). Defaults to <in>.png")
    ap.add_argument("--mode", choices=["gray8", "gray16", "viridis"], default="viridis", help="Output visualization mode")
    ap.add_argument("--min", dest="vmin", type=float, default=None, help="Optional fixed min depth for normalization")
    ap.add_argument("--max", dest="vmax", type=float, default=None, help="Optional fixed max depth for normalization")
    ap.add_argument("--invert", action="store_true", help="Invert colormap (near bright when set)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    depth = load_depth_any(in_path)

    if args.vmin is None or args.vmax is None:
        vmin, vmax = robust_min_max(depth)
    else:
        vmin, vmax = float(args.vmin), float(args.vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0

    if args.mode == "gray8":
        img = to_gray8(depth, vmin, vmax, invert=args.invert)
        pil = Image.fromarray(img, mode="L")
    elif args.mode == "gray16":
        img = to_gray16(depth, vmin, vmax, invert=args.invert)
        pil = Image.fromarray(img)
    else:  # viridis
        rgb = apply_viridis(depth, vmin, vmax, invert=args.invert)
        pil = Image.fromarray(rgb, mode="RGB")

    out_path = Path(args.out) if args.out else in_path.with_suffix(in_path.suffix + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    print(f"[ok] Wrote {out_path} (mode={args.mode}, vmin={vmin:.3f}, vmax={vmax:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
