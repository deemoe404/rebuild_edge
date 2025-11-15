#!/usr/bin/env python3
import argparse, sys, os

def parse_points3d_txt(path):
    xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 8: continue
            try:
                xs.append(float(parts[1]))
                ys.append(float(parts[2]))
                zs.append(float(parts[3]))
                rs.append(int(parts[4]))
                gs.append(int(parts[5]))
                bs.append(int(parts[6]))
            except Exception:
                continue
    return xs, ys, zs, rs, gs, bs

def parse_xyzrgb_txt(path):
    xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line: continue
            parts = line.strip().split()
            if len(parts) < 3: continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                xs.append(x); ys.append(y); zs.append(z)
                if len(parts) >= 6:
                    r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
                else:
                    r = g = b = 0
                rs.append(max(0, min(255, r)))
                gs.append(max(0, min(255, g)))
                bs.append(max(0, min(255, b)))
            except Exception:
                continue
    return xs, ys, zs, rs, gs, bs

def write_las(out_path, xs, ys, zs, rs, gs, bs, srs=None, scale=0.001):
    try:
        import numpy as np
        import laspy
    except Exception as e:
        print(f"Error: required modules not available (numpy/laspy): {e}", file=sys.stderr)
        return 2
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    if xs.size == 0:
        print("Error: no points to write", file=sys.stderr)
        return 3
    rs = np.asarray(rs, dtype=np.uint16)
    gs = np.asarray(gs, dtype=np.uint16)
    bs = np.asarray(bs, dtype=np.uint16)
    r16 = (rs.astype(np.uint16) * 257)  # 8-bit -> 16-bit
    g16 = (gs.astype(np.uint16) * 257)
    b16 = (bs.astype(np.uint16) * 257)

    ox, oy, oz = float(xs.mean()), float(ys.mean()), float(zs.mean())
    sx = sy = sz = float(scale)

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = [ox, oy, oz]
    header.scales = [sx, sy, sz]
    try:
        from laspy import crs as _crs
        if srs:
            try:
                if srs.upper().startswith('EPSG:'):
                    code = int(srs.split(':')[1])
                    header.add_crs(_crs.CRS.from_epsg(code))
                else:
                    header.add_crs(_crs.CRS.from_string(srs))
            except Exception:
                pass
    except Exception:
        pass

    las = laspy.LasData(header)
    las.x = xs
    las.y = ys
    las.z = zs
    if r16.size == xs.size:
        las.red = r16
        las.green = g16
        las.blue = b16
    las.write(out_path)
    return 0

def main():
    ap = argparse.ArgumentParser(description='Convert COLMAP TXT or XYZRGB TXT to LAS/LAZ (ECEF-friendly).')
    ap.add_argument('--colmap_points3D', help='Path to COLMAP points3D.txt')
    ap.add_argument('--xyzrgb_txt', help='Path to TXT with x y z [r g b] ... per line')
    ap.add_argument('--out', required=True, help='Output LAS/LAZ path (*.las or *.laz)')
    ap.add_argument('--srs', help='CRS to embed, e.g., EPSG:4978')
    ap.add_argument('--scale', type=float, default=0.001, help='Quantization scale in meters (default 0.001)')
    args = ap.parse_args()

    mode = 'colmap' if args.colmap_points3D else 'xyzrgb' if args.xyzrgb_txt else None
    if mode is None:
        print('Error: specify --colmap_points3D or --xyzrgb_txt', file=sys.stderr)
        sys.exit(1)

    if mode == 'colmap':
        xs, ys, zs, rs, gs, bs = parse_points3d_txt(args.colmap_points3D)
    else:
        xs, ys, zs, rs, gs, bs = parse_xyzrgb_txt(args.xyzrgb_txt)

    code = write_las(args.out, xs, ys, zs, rs, gs, bs, srs=args.srs, scale=args.scale)
    sys.exit(code)

if __name__ == '__main__':
    main()

