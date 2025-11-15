#!/usr/bin/env python3
"""
Depth Difference Visualization: Compare two depth maps (metric domain).

What it does
- Loads two depth maps A and B, computes the per‑pixel difference (A−B),
  and writes two visualizations plus summary statistics. Depths are treated as
  metric values (e.g., meters). This avoids errors from comparing colorized PNGs
  by resolving the original depth sources when possible.

Accepted Inputs
- Direct sources: .photometric.bin (COLMAP MatBin), .npy, .npz
- Convenience: If a provided path ends with .png and a corresponding source file
  exists (e.g., *.photometric.bin.png → *.photometric.bin, depth.npy.png → depth.npy),
  the script loads the source depth instead of the PNG preview.

Preprocessing
- If A and B have different resolutions, B is bilinearly resized to match A.
- Invalid pixels (non‑finite or ≤0) in either map are excluded from statistics
  and rendered as black in the output images.

Outputs (written to --out_dir)
- diff_abs.png:   |A − B| visualized with viridis; emphasizes magnitude of error.
- diff_signed.png: A − B visualized with a symmetric blue‑white‑red map, where
  white=0, red>0 (A deeper than B), blue<0 (A shallower than B). The symmetric
  range is based on the 98th percentile of |A−B| to reduce outlier dominance.

Reported Statistics (A−B over valid overlap)
- mean:   Average signed error (bias). Negative → A < B on average.
- median: Median signed error; robust central tendency.
- rmse:   Root‑mean‑square error; combines magnitude and spread.
- min/max: Extremes of signed error within valid pixels.

Usage
  python scripts/visualize/visualize_depth_diff.py \
      --a runs/.../stereo/depth_maps/IMG_A.JPG.photometric.bin.png \
      --b runs/.../stereo/depth_maps/IMG_B.JPG.photometric.bin.png \
      --out_dir outputs/depth_diff

Arguments
- --a: First depth (or its PNG preview). Used as reference resolution.
- --b: Second depth (or its PNG preview). Resized to A if shapes differ.
- --out_dir: Output directory to save diff_abs.png and diff_signed.png.

Notes
- Depth semantics: This compares numeric depth values as stored (typically
  z‑depth in meters for COLMAP formats). If your data are ray lengths or have a
  different convention, interpret signs accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def resolve_source_from_png(p: Path) -> Path:
    if p.suffix.lower() != ".png":
        return p
    # Try stripping the .png and see if original exists
    src = p.with_suffix("")
    if src.exists():
        return src
    # Common cases
    s = str(p)
    if s.endswith(".photometric.bin.png"):
        cand = Path(s[:-4])  # drop '.png'
        if cand.exists():
            return cand
    if s.endswith(".npy.png"):
        cand = Path(s[:-4])
        if cand.exists():
            return cand
    if s.endswith(".npz.png"):
        cand = Path(s[:-4])
        if cand.exists():
            return cand
    # Fall back to original
    return p


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
    suf = path.suffix.lower()
    if suf == ".bin":
        arr = read_mat_bin(path)
    elif suf == ".npy":
        arr = np.load(path)
    elif suf == ".npz":
        npz = np.load(path)
        for k in ("depth", "arr_0"):
            if k in npz:
                arr = npz[k]
                break
        else:
            keys = list(npz.keys())
            if not keys:
                raise RuntimeError(f"Empty npz file: {path}")
            arr = npz[keys[0]]
    else:
        # As a last resort, try reading as image and normalize to 0..1; warn that it's not metric
        im = Image.open(path).convert("F")
        arr = np.asarray(im, dtype=np.float32)

    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # HWC or CHW
        if arr.shape[0] <= 8 and arr.shape[0] != arr.shape[1]:
            return arr[0, ...]
        return arr[..., 0]
    raise ValueError(f"Unsupported depth array shape: {arr.shape}")


def robust_min_max(vals: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> Tuple[float, float]:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, q_low))
    hi = float(np.percentile(v, q_high))
    if hi <= lo:
        return float(v.min()), float(v.max())
    return lo, hi


def to_viridis(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
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
    x = np.clip((x - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    idx = np.clip((x * 255.0).astype(np.int32), 0, 255)
    return viridis[idx]


def to_bwr(delta: np.ndarray, T: float) -> np.ndarray:
    """Blue-White-Red map for signed values in [-T, T]; 0 maps to white."""
    x = np.clip(delta / max(T, 1e-6), -1.0, 1.0)
    # Build RGB
    # Negative: blue (0, 0, 255) to white (255,255,255) as x goes -1->0
    # Positive: white to red (255, 0, 0) as x goes 0->1
    r = np.empty_like(x, dtype=np.float32)
    g = np.empty_like(x, dtype=np.float32)
    b = np.empty_like(x, dtype=np.float32)
    neg = x < 0
    pos = ~neg
    # Negative branch
    xn = -x[neg]  # 0..1
    r[neg] = 255.0 * (1.0 - xn)
    g[neg] = 255.0 * (1.0 - xn)
    b[neg] = 255.0
    # Positive branch
    xp = x[pos]
    r[pos] = 255.0
    g[pos] = 255.0 * (1.0 - xp)
    b[pos] = 255.0 * (1.0 - xp)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize difference between two depth maps")
    ap.add_argument("--a", required=True, help="Path to first depth (.bin/.npy/.npz or the derived .png)")
    ap.add_argument("--b", required=True, help="Path to second depth (.bin/.npy/.npz or the derived .png)")
    ap.add_argument("--out_dir", default="outputs/depth_diff", help="Directory to write outputs")
    args = ap.parse_args()

    pA = resolve_source_from_png(Path(args.a))
    pB = resolve_source_from_png(Path(args.b))

    A = load_depth_any(pA).astype(np.float32)
    B = load_depth_any(pB).astype(np.float32)

    # Resize B to A's shape if needed
    if B.shape != A.shape:
        # Use PIL bilinear on float32 via mode 'F'
        B_img = Image.fromarray(B, mode="F").resize((A.shape[1], A.shape[0]), resample=Image.BILINEAR)
        B = np.array(B_img, dtype=np.float32)

    # Mask invalid (<=0 or non-finite) from either
    valid = np.isfinite(A) & np.isfinite(B) & (A > 0) & (B > 0)
    delta = np.full_like(A, np.nan, dtype=np.float32)
    delta[valid] = A[valid] - B[valid]
    absd = np.abs(delta)

    # Stats
    v = delta[np.isfinite(delta)]
    if v.size == 0:
        print("[error] No overlapping valid depth pixels; cannot compute difference")
        return 2
    mean = float(np.mean(v))
    med = float(np.median(v))
    rmse = float(np.sqrt(np.mean(v * v)))
    vmin, vmax = float(np.min(v)), float(np.max(v))
    print(f"[stats] delta=A-B: mean={mean:.6f}, median={med:.6f}, rmse={rmse:.6f}, min={vmin:.6f}, max={vmax:.6f}")

    # Visualize |delta|
    lo, hi = robust_min_max(absd[np.isfinite(absd)])
    abs_vis = to_viridis(np.nan_to_num(absd, nan=0.0), lo, hi)
    abs_vis[~valid] = np.array([0, 0, 0], dtype=np.uint8)

    # Visualize signed delta with symmetric range
    T = float(np.percentile(np.abs(v), 98))
    signed_vis = to_bwr(np.nan_to_num(delta, nan=0.0), T)
    signed_vis[~valid] = np.array([0, 0, 0], dtype=np.uint8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_path = out_dir / "diff_abs.png"
    sgn_path = out_dir / "diff_signed.png"
    Image.fromarray(abs_vis, mode="RGB").save(abs_path)
    Image.fromarray(signed_vis, mode="RGB").save(sgn_path)
    print(f"[ok] Wrote {abs_path} (|A-B|, viridis) and {sgn_path} (A-B, BWR; white=0, blue<0, red>0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
