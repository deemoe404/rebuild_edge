#!/usr/bin/env python3
"""
Generate sparse depth PNGs (16-bit) per image from a COLMAP SfM model.

What this does
- Parses COLMAP sparse model (TXT or BIN via model_converter).
- For each image, places depth at observed 2D-3D correspondences and writes a
  16-bit PNG aligned to that image's resolution.
- Depth values are z-depth in meters (camera frame), scaled by --scale and
  rounded to uint16 (0 for invalid).

Why
- To produce sparse depth inputs compatible with single-image PSD+DepthPro
  inference, which expects KITTI-style 16-bit PNG sparse depth and a matching
  RGB image. Use the same --scale here and --sparse_scale there (defaults 256).

Inputs
- --sparse_dir: Path to COLMAP sparse model directory containing images.(txt|bin),
  points3D.(txt|bin), cameras.(txt|bin). If only BIN exists, we convert to TXT.

Outputs
- --out_dir: Directory where per-image sparse depth PNGs are written. The file
  names mirror COLMAP image names with .png extension, preserving subfolders.
- Optional --export_intrinsics_csv writes per-image fx,fy,cx,cy for convenience.

Notes
- PNGs are sized using camera width/height from cameras.txt.
- Depth uses camera Z after world->camera transform. If multiple observations
  map to the same pixel, the nearest (smallest positive z) wins.
- Pixels are integer-rounded from COLMAP 2D observations.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image


# -------------------------
# COLMAP TXT parsers
# -------------------------
def read_cameras_txt(path: Path) -> Dict[int, dict]:
    cams: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
            # Extract fx,fy,cx,cy when available per model
            if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE") and len(params) >= 4:
                fx, fy, cx, cy = params[:4]
            elif model in ("SIMPLE_PINHOLE",) and len(params) >= 3:
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            elif model in ("SIMPLE_RADIAL", "RADIAL", "FOV") and len(params) >= 3:
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            else:
                fx = fy = cx = cy = 0.0
            cams[cam_id] = dict(model=model, w=w, h=h, fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
    return cams


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
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


def geometric_depth_z(Xw: np.ndarray, qvec: np.ndarray, tvec: np.ndarray) -> Optional[float]:
    R = qvec2rotmat(qvec)
    p = R @ Xw.reshape(3) + tvec.reshape(3)
    z = float(p[2])
    if not np.isfinite(z) or z <= 0:
        return None
    return z


# -------------------------
# Utilities
# -------------------------
def ensure_txt_model(sparse_dir: Path) -> Path:
    """Return a directory that contains cameras.txt/images.txt/points3D.txt.

    If sparse_dir already has TXT files, return it. If it has BIN files,
    use colmap model_converter to a sibling '<sparse_dir>_txt' and return that.
    """
    cam_txt = sparse_dir / "cameras.txt"
    img_txt = sparse_dir / "images.txt"
    pts_txt = sparse_dir / "points3D.txt"
    if cam_txt.exists() and img_txt.exists() and pts_txt.exists():
        return sparse_dir

    cam_bin = sparse_dir / "cameras.bin"
    img_bin = sparse_dir / "images.bin"
    pts_bin = sparse_dir / "points3D.bin"
    if cam_bin.exists() and img_bin.exists() and pts_bin.exists():
        out_dir = sparse_dir.parent / f"{sparse_dir.name}_txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "colmap",
                    "model_converter",
                    "--input_path",
                    str(sparse_dir),
                    "--output_path",
                    str(out_dir),
                    "--output_type",
                    "TXT",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return out_dir
        except FileNotFoundError:
            raise SystemExit("Error: 'colmap' not found in PATH for BIN->TXT conversion")
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"Error: colmap model_converter failed (status {e.returncode}). stderr=\n{e.stderr.decode(errors='ignore')}"
            )

    raise SystemExit(
        f"Error: No COLMAP TXT or BIN found in {sparse_dir}. Expected cameras.(txt|bin), images.(txt|bin), points3D.(txt|bin)."
    )


def safe_uint16_from_meters(depth_m: float, scale: float) -> int:
    if not np.isfinite(depth_m) or depth_m <= 0:
        return 0
    v = int(round(depth_m * scale))
    if v < 1:
        return 1  # preserve tiny positive depths
    if v > 65535:
        return 65535
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Export COLMAP SfM sparse depths as 16-bit PNGs")
    ap.add_argument("--sparse_dir", required=True, help="Path to COLMAP sparse model dir")
    ap.add_argument("--out_dir", required=True, help="Output directory for sparse depth PNGs")
    ap.add_argument("--scale", type=float, default=256.0, help="Meters -> uint16 scale (KITTI uses 256.0)")
    ap.add_argument(
        "--export_intrinsics_csv",
        default=None,
        help="Optional path to write per-image fx,fy,cx,cy CSV for convenience",
    )
    args = ap.parse_args()

    sparse_dir = Path(args.sparse_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we have TXT model
    txt_dir = ensure_txt_model(sparse_dir)
    cams = read_cameras_txt(txt_dir / "cameras.txt")
    imgs = read_images_with_points(txt_dir / "images.txt")
    pts3d = read_points3D_txt(txt_dir / "points3D.txt")

    # Optional intrinsics CSV per image
    csv_fh = None
    if args.export_intrinsics_csv:
        csv_path = Path(args.export_intrinsics_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = open(csv_path, "w", encoding="utf-8")
        csv_fh.write("image,fx,fy,cx,cy,width,height\n")

    # Process each image entry in images.txt
    count = 0
    for name, meta in imgs.items():
        cam = cams.get(meta["cam_id"]) or {}
        W = int(cam.get("w", 0))
        H = int(cam.get("h", 0))
        if W <= 0 or H <= 0:
            print(f"[warn] skip {name}: invalid camera size {W}x{H}")
            continue

        # float depth buffer to decide nearest when collisions occur; 0 means invalid
        depth_buf = np.zeros((H, W), dtype=np.float32)

        # Populate from 2D-3D correspondences
        for (x, y, pid) in meta.get("points2D", []):
            if pid < 0:
                continue
            Xw = pts3d.get(pid)
            if Xw is None:
                continue
            z = geometric_depth_z(Xw, meta["qvec"], meta["tvec"])
            if z is None:
                continue
            u = int(round(x)); v = int(round(y))
            if not (0 <= u < W and 0 <= v < H):
                continue
            # Keep the nearest positive depth when collisions happen
            if depth_buf[v, u] == 0 or z < depth_buf[v, u]:
                depth_buf[v, u] = float(z)

        # Convert to uint16 PNG (KITTI style scaling)
        if np.count_nonzero(depth_buf > 0) == 0:
            print(f"[info] {name}: no valid sparse depth; writing all-zero PNG")
        depth_u16 = np.zeros_like(depth_buf, dtype=np.uint16)
        if args.scale <= 0:
            raise SystemExit("--scale must be > 0")
        # Vectorized conversion with clipping
        scaled = np.rint(depth_buf * float(args.scale))
        scaled[scaled < 1] = 0  # reserve 0 for invalid
        scaled[scaled > 65535] = 65535
        depth_u16[:, :] = scaled.astype(np.uint16)

        # Output path mirrors relative name under out_dir, replacing extension with .png
        rel = Path(name)
        out_path = out_dir / rel
        out_path = out_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(depth_u16).save(out_path)
        count += 1

        if csv_fh is not None:
            csv_fh.write(
                f"{name},{cam.get('fx', 0.0)},{cam.get('fy', 0.0)},{cam.get('cx', 0.0)},{cam.get('cy', 0.0)},{W},{H}\n"
            )

    if csv_fh is not None:
        csv_fh.close()

    print(
        f"[done] Wrote {count} sparse depth PNGs to {out_dir}. Use --sparse_scale {args.scale} when calling your single-image inference."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

