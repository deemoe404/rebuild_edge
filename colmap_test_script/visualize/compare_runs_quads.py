#!/usr/bin/env python3
"""
Compare two runs: generate 2x2 quad images per photo for depth and/or normals.

Layout (per image and per modality):
- Top-left: RGB undistorted image
- Top-right: Signed difference between Run A and Run B
- Bottom-left: Run A visualization
- Bottom-right: Run B visualization

Inputs
- --run_a, --run_b: paths to run working dirs (each contains `dense/...`).
- The script looks under `<run>/dense/stereo/depth_maps` and `.../normal_maps`.
- RGB is taken from `<run_a>/dense/images/<name>` (fallback to run_b if missing).

Outputs
- By default saved under `<run_a>/compare_<basename(run_b)>/depth_quads` and
  `.../normal_quads` mirroring the input relative paths. Can override with
  `--out_dir` (then subfolders `depth_quads`/`normal_quads` are created there).

Notes
- Depth visual uses a shared robust range computed from both runs (2nd–98th pct).
- Depth diff uses blue‑white‑red (BWR) centered at 0 with symmetric range from
  the 98th percentile of |A−B|.
- Normal visual maps [-1,1] to [0,255] per channel.
- Normal diff supports modes:
  - rgb (default): per‑channel signed diff mapped with 0 centered at 128.
  - dot: dot(nA, nB) ∈ [-1,1] mapped with BWR (white=0, red>0, blue<0).
  - angle: acos(dot)/deg ∈ [0,180] visualized with viridis (unsigned).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


# -------------------- I/O helpers --------------------
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
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Invalid MatBin header in {path}: {header!r}") from e
        data = np.fromfile(f, dtype=np.float32, count=w * h * c)
        if c == 1:
            return data.reshape(h, w)
        else:
            return data.reshape(h, w, c)


def load_depth_bin(path: Path) -> np.ndarray:
    arr = read_mat_bin(path)
    if arr.ndim == 3:
        # Take first channel
        arr = arr[..., 0]
    return np.asarray(arr, dtype=np.float32)


def load_normal_bin(path: Path) -> np.ndarray:
    arr = read_mat_bin(path).astype(np.float32)
    if arr.ndim == 2:
        raise ValueError(f"Expected HxWx3 normals at {path}, got 2D array {arr.shape}")
    if arr.ndim == 3:
        if arr.shape[2] < 3:
            raise ValueError(f"Array has <3 channels for normals at {path}: {arr.shape}")
        return arr[..., :3]
    raise ValueError(f"Unsupported normals array shape at {path}: {arr.shape}")


def normalize_normals_unit(n: np.ndarray) -> np.ndarray:
    """Return a unit-length normal map; keep invalid as zeros.

    n: HxWx3 float32 in [-1,1] range (not necessarily unit).
    """
    n = np.asarray(n, dtype=np.float32)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    # Identify invalid (nan/inf/zero)
    invalid = ~np.isfinite(n).all(axis=2, keepdims=True) | (norm < 1e-8)
    norm_safe = np.where(norm < 1e-8, 1.0, norm)
    out = n / norm_safe
    out[invalid.squeeze(axis=2)] = 0.0
    return out


def face_camera(n: np.ndarray) -> np.ndarray:
    """Flip normals to make Nz <= 0 (camera-facing) where valid."""
    out = np.array(n, dtype=np.float32, copy=True)
    valid = np.isfinite(out).all(axis=2) & (np.linalg.norm(out, axis=2) > 1e-8)
    flip = valid & (out[..., 2] > 0)
    out[flip] *= -1.0
    return out


def try_load_rgb(paths: Iterable[Path], size: Tuple[int, int]) -> Image.Image:
    for p in paths:
        if p.exists():
            try:
                im = Image.open(p).convert("RGB")
                if im.size != size:
                    im = im.resize(size, resample=Image.BILINEAR)
                return im
            except Exception:
                continue
    # Fallback: blank image
    return Image.new("RGB", size, (0, 0, 0))


def robust_range(values: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> Tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, q_low))
    hi = float(np.percentile(v, q_high))
    if hi <= lo:
        return float(v.min()), float(v.max())
    return lo, hi


# -------------------- Color maps --------------------
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
    x = np.clip(delta / max(T, 1e-6), -1.0, 1.0)
    r = np.empty_like(x, dtype=np.float32)
    g = np.empty_like(x, dtype=np.float32)
    b = np.empty_like(x, dtype=np.float32)
    neg = x < 0
    pos = ~neg
    xn = -x[neg]
    r[neg] = 255.0 * (1.0 - xn)
    g[neg] = 255.0 * (1.0 - xn)
    b[neg] = 255.0
    xp = x[pos]
    r[pos] = 255.0
    g[pos] = 255.0 * (1.0 - xp)
    b[pos] = 255.0 * (1.0 - xp)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def normals_to_rgb(n: np.ndarray) -> np.ndarray:
    n = np.clip(n.astype(np.float32), -1.0, 1.0)
    rgb = ((n + 1.0) * 0.5 * 255.0).round().astype(np.uint8)
    invalid = ~np.isfinite(n).any(axis=2) | (np.linalg.norm(n, axis=2) < 1e-8)
    if np.any(invalid):
        rgb[invalid] = np.array([0, 0, 0], dtype=np.uint8)
    return rgb


def resize_float32(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if img.ndim == 2:
        pil = Image.fromarray(img.astype(np.float32), mode="F").resize((w, h), resample=Image.BILINEAR)
        return np.array(pil, dtype=np.float32)
    elif img.ndim == 3:
        # Resize per channel
        chs = []
        for c in range(img.shape[2]):
            pil = Image.fromarray(img[..., c].astype(np.float32), mode="F").resize((w, h), resample=Image.BILINEAR)
            chs.append(np.array(pil, dtype=np.float32))
        return np.stack(chs, axis=2)
    else:
        raise ValueError("Unsupported ndim for resize")


# -------------------- Quad builders --------------------
def build_depth_quad(depth_a: np.ndarray, depth_b: np.ndarray, rgb_im: Image.Image) -> Image.Image:
    Ha, Wa = depth_a.shape
    Hb, Wb = depth_b.shape
    if (Ha, Wa) != (Hb, Wb):
        depth_b = resize_float32(depth_b, (Ha, Wa))
    H, W = Ha, Wa

    # Valid mask
    valid_a = np.isfinite(depth_a) & (depth_a > 0)
    valid_b = np.isfinite(depth_b) & (depth_b > 0)
    valid = valid_a & valid_b

    # Visualization range from both maps
    vals = np.concatenate([depth_a[valid_a].ravel(), depth_b[valid_b].ravel()]) if valid.any() else depth_a[valid_a].ravel()
    vmin, vmax = robust_range(vals) if vals.size > 0 else (0.0, 1.0)
    vis_a = to_viridis(depth_a, vmin, vmax)
    vis_b = to_viridis(depth_b, vmin, vmax)
    vis_a[~valid_a] = 0
    vis_b[~valid_b] = 0

    # Signed diff A-B
    delta = (depth_a - depth_b).astype(np.float32)
    abd = np.abs(delta)[valid]
    T = float(np.percentile(abd, 98.0)) if abd.size > 0 else 1.0
    diff = to_bwr(delta, T)
    diff[~valid] = 0

    # Prepare panels
    rgb_panel = rgb_im.resize((W, H), resample=Image.BILINEAR)
    top_left = rgb_panel
    top_right = Image.fromarray(diff, mode="RGB")
    bot_left = Image.fromarray(vis_a, mode="RGB")
    bot_right = Image.fromarray(vis_b, mode="RGB")

    # Compose canvas 2W x 2H
    canvas = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
    canvas.paste(top_left, (0, 0))
    canvas.paste(top_right, (W, 0))
    canvas.paste(bot_left, (0, H))
    canvas.paste(bot_right, (W, H))
    return canvas


def build_normal_quad(
    n_a: np.ndarray,
    n_b: np.ndarray,
    rgb_im: Image.Image,
    diff_mode: str = "rgb",
    unit: bool = True,
    face_cam: bool = True,
) -> Image.Image:
    Ha, Wa, _ = n_a.shape
    Hb, Wb, _ = n_b.shape
    if (Ha, Wa) != (Hb, Wb):
        n_b = resize_float32(n_b, (Ha, Wa))
    H, W = Ha, Wa

    # Valid masks
    va = np.isfinite(n_a).all(axis=2) & (np.linalg.norm(n_a, axis=2) > 1e-8)
    vb = np.isfinite(n_b).all(axis=2) & (np.linalg.norm(n_b, axis=2) > 1e-8)
    valid = va & vb

    # Optional normalization/sign unification for fair visualization/diff
    if unit:
        n_a = normalize_normals_unit(n_a)
        n_b = normalize_normals_unit(n_b)
    if face_cam:
        n_a = face_camera(n_a)
        n_b = face_camera(n_b)

    # Bottom visuals
    vis_a = normals_to_rgb(n_a)
    vis_b = normals_to_rgb(n_b)

    # Diff panel
    if diff_mode == "dot":
        dp = np.sum(n_a * n_b, axis=2)
        dp = np.clip(dp, -1.0, 1.0)
        diff = to_bwr(dp, 1.0)
    elif diff_mode == "angle":
        dp = np.sum(n_a * n_b, axis=2)
        dp = np.clip(dp, -1.0, 1.0)
        ang = np.degrees(np.arccos(dp))  # 0..180
        vmin, vmax = 0.0, float(np.percentile(ang[valid], 98.0)) if valid.any() else 30.0
        diff = to_viridis(ang, vmin, vmax)
    else:  # rgb per‑channel signed diff, centered at 128
        d = (n_a - n_b).astype(np.float32)
        # Robust per-channel scale
        ch_T = []
        for c in range(3):
            v = np.abs(d[..., c][valid])
            T = float(np.percentile(v, 98.0)) if v.size > 0 else 1.0
            ch_T.append(max(T, 1e-6))
        Tvec = np.array(ch_T, dtype=np.float32).reshape(1, 1, 3)
        x = np.clip(d / Tvec, -1.0, 1.0)
        diff = ((x * 0.5 + 0.5) * 255.0).round().astype(np.uint8)
    diff[~valid] = np.array([0, 0, 0], dtype=np.uint8)

    # Compose
    rgb_panel = rgb_im.resize((W, H), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
    canvas.paste(rgb_panel, (0, 0))
    canvas.paste(Image.fromarray(diff, mode="RGB"), (W, 0))
    canvas.paste(Image.fromarray(vis_a, mode="RGB"), (0, H))
    canvas.paste(Image.fromarray(vis_b, mode="RGB"), (W, H))
    return canvas


# -------------------- Main flow --------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two runs with 2x2 quads for depth/normal")
    ap.add_argument("--run_a", required=True, help="Path to run A working dir (contains dense)")
    ap.add_argument("--run_b", required=True, help="Path to run B working dir (contains dense)")
    ap.add_argument("--type", choices=["depth", "normal", "both"], default="both", help="Which modality to export")
    ap.add_argument("--out_dir", default=None, help="Output root; defaults under run A as compare_<run_b>")
    ap.add_argument("--normal_diff", choices=["rgb", "dot", "angle"], default="rgb", help="Diff visualization for normals")
    ap.add_argument("--normal_keep_magnitude", action="store_true", help="Do not re-normalize normals to unit length before visualization/diff")
    ap.add_argument("--normal_keep_sign", action="store_true", help="Do not flip normals to be camera-facing (Nz<=0)")
    ap.add_argument("--max_images", type=int, default=0, help="Optional limit for number of images processed per modality")
    args = ap.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    dense_a = run_a / "dense"
    dense_b = run_b / "dense"
    depth_a_dir = dense_a / "stereo" / "depth_maps"
    depth_b_dir = dense_b / "stereo" / "depth_maps"
    normal_a_dir = dense_a / "stereo" / "normal_maps"
    normal_b_dir = dense_b / "stereo" / "normal_maps"
    rgb_a_dir = dense_a / "images"
    rgb_b_dir = dense_b / "images"

    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        out_root = run_a / f"compare_{run_b.name}"
    out_depth = out_root / "depth_quads"
    out_normal = out_root / "normal_quads"

    processed_any = False

    if args.type in ("depth", "both"):
        if not depth_a_dir.exists() or not depth_b_dir.exists():
            print(f"[warn] depth_maps missing: {depth_a_dir} or {depth_b_dir}")
        else:
            bins_a = sorted(depth_a_dir.rglob("*.photometric.bin"))
            count = 0
            for pa in bins_a:
                rel = pa.relative_to(depth_a_dir)
                pb = depth_b_dir / rel
                if not pb.exists():
                    print(f"[skip] Run B missing depth for {rel}")
                    continue
                try:
                    da = load_depth_bin(pa)
                    db = load_depth_bin(pb)
                except Exception as e:  # noqa: BLE001
                    print(f"[error] Failed loading depth: {rel}: {e}")
                    continue
                H, W = da.shape
                # RGB path candidates
                name_with_ext = rel.name[:-len(".photometric.bin")] if str(rel).endswith(".photometric.bin") else rel.name
                rgb_paths = [rgb_a_dir / rel.with_name(name_with_ext), rgb_b_dir / rel.with_name(name_with_ext)]
                rgb_im = try_load_rgb(rgb_paths, size=(W, H))

                quad = build_depth_quad(da, db, rgb_im)
                out_path = out_depth / rel.with_suffix(".depth_quad.png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                quad.save(out_path)
                print(f"[ok] depth quad → {out_path}")
                processed_any = True
                count += 1
                if args.max_images and count >= args.max_images:
                    break

    if args.type in ("normal", "both"):
        if not normal_a_dir.exists() or not normal_b_dir.exists():
            print(f"[warn] normal_maps missing: {normal_a_dir} or {normal_b_dir}")
        else:
            bins_a = sorted(normal_a_dir.rglob("*.photometric.bin"))
            count = 0
            for pa in bins_a:
                rel = pa.relative_to(normal_a_dir)
                pb = normal_b_dir / rel
                if not pb.exists():
                    print(f"[skip] Run B missing normals for {rel}")
                    continue
                try:
                    na = load_normal_bin(pa)
                    nb = load_normal_bin(pb)
                except Exception as e:  # noqa: BLE001
                    print(f"[error] Failed loading normals: {rel}: {e}")
                    continue
                H, W, _ = na.shape
                name_with_ext = rel.name[:-len(".photometric.bin")] if str(rel).endswith(".photometric.bin") else rel.name
                rgb_paths = [rgb_a_dir / rel.with_name(name_with_ext), rgb_b_dir / rel.with_name(name_with_ext)]
                rgb_im = try_load_rgb(rgb_paths, size=(W, H))

                quad = build_normal_quad(
                    na,
                    nb,
                    rgb_im,
                    diff_mode=args.normal_diff,
                    unit=not args.normal_keep_magnitude,
                    face_cam=not args.normal_keep_sign,
                )
                out_path = out_normal / rel.with_suffix(".normal_quad.png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                quad.save(out_path)
                print(f"[ok] normal quad → {out_path}")
                processed_any = True
                count += 1
                if args.max_images and count >= args.max_images:
                    break

    if not processed_any:
        print("[warn] Nothing processed. Check run paths and contents.")
        return 2
    print(f"[done] Quads written under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
