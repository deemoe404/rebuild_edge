#!/usr/bin/env python3
# Dump all metadata from image files (EXIF/XMP/IPTC/MakerNotes) using exiftool if available.
# Falls back to Pillow for basic EXIF and image info. Can also emit the raw embedded XMP packet.
# Usage:
#   python scripts/helper/dump_jpg_metadata.py <file_or_dir> [more files...] [--xmp-raw] [-r] [-o out.jsonl]
# Examples:
#   python scripts/helper/dump_jpg_metadata.py DJI_0001.JPG --xmp-raw
#   python scripts/helper/dump_jpg_metadata.py ./photos -r -o meta.jsonl
import argparse, sys, subprocess, json, shutil, os, re
from typing import Any, Dict

# Regex to extract the raw XMP packet if present
XMP_RE = re.compile(br'<x:xmpmeta[^>]*>.*?</x:xmpmeta>', re.DOTALL)

def extract_raw_xmp(path: str):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        m = XMP_RE.search(data)
        if not m:
            return None
        try:
            return m.group(0).decode('utf-8', errors='replace')
        except Exception:
            return m.group(0).decode('latin-1', errors='replace')
    except Exception as e:
        return f"<XMP extraction error: {e}>"

def run_exiftool(path: str):
    # -a (duplicates), -G1 (group names at level 1), -s (short tag names), -n (numeric), -struct (structured output)
    cmd = ['exiftool','-a','-G1','-s','-n','-struct','-api','largefilesupport=1','-json', path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode('utf-8', errors='ignore') or 'exiftool failed')
    txt = proc.stdout.decode('utf-8', errors='ignore')
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # exiftool sometimes emits UTF-8 with stray BOM or warnings; try to sanitize
        txt = txt.strip()
        return json.loads(txt)

def fallback_pillow(path: str):
    out = {}
    try:
        from PIL import Image, ExifTags
    except Exception as e:
        return {"error": f"Pillow fallback unavailable: {e}"}
    try:
        with Image.open(path) as im:
            out['format'] = im.format
            out['mode'] = im.mode
            out['size'] = im.size  # (width, height)
            # raw info dict from PIL (may contain icc_profile, dpi, etc.)
            raw_info = {}
            for k, v in im.info.items():
                if isinstance(v, (bytes, bytearray)):
                    try:
                        raw_info[k] = v.decode('utf-8', 'replace')
                    except Exception:
                        raw_info[k] = f"<{len(v)} bytes>"
                else:
                    raw_info[k] = v
            if raw_info:
                out['PIL_info'] = raw_info
            # EXIF via Pillow (basic; MakerNotes/extended tags may be opaque)
            exif_data = {}
            exif = im.getexif()
            if exif:
                for tag_id, val in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
                    try:
                        # Convert bytes to safe text
                        if isinstance(val, (bytes, bytearray)):
                            exif_data[tag_name] = val.decode('utf-8', 'replace')
                        else:
                            exif_data[tag_name] = val
                    except Exception:
                        exif_data[tag_name] = f"<unserializable type {type(val).__name__}>"
                out['EXIF'] = exif_data
    except Exception as e:
        out['error'] = str(e)
    return out

def process(path: str, include_xmp_raw=False, use_exiftool=True):
    record: Dict[str, Any] = {"SourceFile": os.path.abspath(path)}
    used_exiftool = False
    if use_exiftool and shutil.which('exiftool'):
        try:
            data = run_exiftool(path)
            # exiftool -json returns a list with one dict
            record['ExifTool'] = data[0] if isinstance(data, list) and data else data
            used_exiftool = True
        except Exception as e:
            record['ExifToolError'] = str(e)
    else:
        record['ExifToolAvailable'] = False
    if not used_exiftool:
        record['Fallback'] = fallback_pillow(path)
    if include_xmp_raw:
        record['XMP_Raw'] = extract_raw_xmp(path)
    return record

def iter_targets(inputs, recursive=False):
    exts = ('.jpg', '.jpeg', '.dng', '.tif', '.tiff', '.heic', '.heif', '.png')
    for p in inputs:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(exts):
                        yield os.path.join(root, f)
                if not recursive:
                    break
        else:
            yield p

def main():
    ap = argparse.ArgumentParser(
        description='Dump all metadata from image files as JSONL. Prefers exiftool; falls back to Pillow.')
    ap.add_argument('inputs', nargs='+', help='Image file(s) or directory(ies)')
    ap.add_argument('-r','--recursive', action='store_true', help='Recurse into directories')
    ap.add_argument('--no-exiftool', action='store_true', help='Disable exiftool even if present')
    ap.add_argument('--xmp-raw', action='store_true', help='Include raw XMP XML packet')
    ap.add_argument('-o','--output', help='Write to file (JSON Lines). Defaults to stdout')
    args = ap.parse_args()

    out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
    for path in iter_targets(args.inputs, args.recursive):
        rec = process(path, include_xmp_raw=args.xmp_raw, use_exiftool=not args.no_exiftool)
        json.dump(rec, out, ensure_ascii=False)
        out.write('\n')
        out.flush()
    if args.output:
        out.close()

if __name__ == '__main__':
    main()
