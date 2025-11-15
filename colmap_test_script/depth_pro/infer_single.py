"""
Single-Image Inference (Depth Pro backend)
==========================================

What it does
- Runs PSD with Depth Pro as the monocular depth (MDE) backend on a single RGB image + sparse depth, and saves a depth visualization and optionally raw depth.

Dependencies
- PyTorch + torchvision (install appropriate build for your CUDA/CPU)
- Python packages: timm==0.4.12, einops, scikit-image, numba, tensorboardX, omegaconf, opencv-python, matplotlib, Pillow, numpy
- Example install:
  pip install timm==0.4.12 einops scikit-image numba tensorboardX omegaconf opencv-python matplotlib pillow numpy

Weights
- Depth Pro weights (.pt): e.g., checkpoints/depth_pro.pt
- PSD pretrained weights (.pth/.ckpt, e.g., PSD-NK-DPr): e.g., checkpoints/PSD-NK-DPr.pth
- This script takes both paths as CLI args; putting them under ./checkpoints is recommended.

Defaults
- --psd_ckpt defaults to PSD_NK_DPr_checkpoint.ckpt (repo root)
- --depthpro_ckpt defaults to third_party/ml-depth-pro/checkpoints/depth_pro.pt
  Override these on the CLI if your paths differ.

Inputs
- --rgb: Path to the RGB image (any resolution). A 384x384 copy is internally used for Depth Pro.
- --sparse: Path to a sparse depth 16-bit PNG aligned with the RGB (KITTI-style, divided by 256.0 to meters by default). Use --sparse_scale if your scale differs.
- Camera intrinsics (optional but strongly recommended): --fx --fy --cx --cy in pixels.
  - In this repository, the FOV head of the integrated Depth Pro is commented out, so if intrinsics are not provided, the script falls back to fx=fy=W/2 and cx=W/2, cy=H/2 which is only an approximation.
- --data: Dataset profile affecting depth range alignment and diffusion parameters, e.g., NYUv2, KITTI, VOID1500.
  About --data
  - Choose a profile matching your scene, not the checkpoint name.
  - Indoor: NYUv2 (typical depth 0.1–10 m)
  - Driving/outdoor: KITTI (typical depth 0.5–90 m)
  - Other domains: VOID1500, Cityscape, DrivingStereo, etc.
  - Do not use "NK" here. Although the released weights may be trained on a NYU+KITTI mix ("NK"), this single-image script does not pass the per-image domain flag required by the NK branch. Pick the closest single domain instead (usually NYUv2 or KITTI).
  - --data controls internal range alignment, KNN/propagation hyperparameters, and post-processing; it does not change the sparse depth unit.

Outputs
- --out: Depth visualization PNG (0–255 linear normalization; for viewing only)
- --out_npy: Optional, raw depth as .npy (float, meters)

Examples
1) Minimal usage (approx intrinsics; not recommended for strict quantitative use):
   python scripts/depth_pro/infer_single.py \
     --psd_ckpt checkpoints/PSD-NK-DPr.pth \
     --depthpro_ckpt checkpoints/depth_pro.pt \
     --rgb /path/to/rgb.png \
     --sparse /path/to/sparse.png \
     --data NYUv2 \
     --out out.png

2) With camera intrinsics (recommended):
   python scripts/depth_pro/infer_single.py \
     --psd_ckpt checkoints/PSD-NK-DPr.pth \
     --depthpro_ckpt checkpoints/depth_pro.pt \
     --rgb /path/to/rgb.png \
     --sparse /path/to/sparse.png \
     --data KITTI \
     --fx 721.5 --fy 721.5 --cx 609.6 --cy 172.8 \
     --out out.png --out_npy depth.npy

3) Sparse depth uses a different scale (e.g., divide by 1000 to meters):
   python scripts/depth_pro/infer_single.py ... --sparse_scale 1000.0

Arguments
- --cfg: Base YAML (default configs/cfg.yml). The script switches MDEBranch to depthpro and injects the depthpro weights in memory; dataset paths in the original cfg are not used for single-image inference.
- --psd_ckpt: PSD pretrained weights (.pth or .ckpt).
- --depthpro_ckpt: Depth Pro .pt weight path.
- --rgb: Path to RGB image.
- --sparse: Path to sparse depth 16-bit PNG.
- --sparse_scale: Divide sparse depth by this to get meters (default 256.0).
- --data: Dataset name/profile (e.g., NYUv2, KITTI) affecting alignment and ranges.
- --fx --fy --cx --cy: Camera intrinsics in pixels. If omitted, defaults to fx=fy=W/2 and cx=W/2, cy=H/2.
- --out: Output visualization PNG (default output_depth.png).
- --out_npy: Optional, save raw depth as .npy.

Notes
- GPU is strongly recommended (ViT-Large encoders are memory heavy).
- Without real intrinsics, scale can be unstable; provide fx/fy/cx/cy whenever possible.
- Sparse depth must be aligned with the RGB (same viewpoint and resolution) for best results.
- The PNG output is visualization only (linearly stretched). Use .npy for metric depth.
- python scripts/depth_pro/infer_single.py --psd_ckpt PSD_NK_DPr_checkpoint.ckpt --depthpro_ckpt third_party/ml-depth-pro/checkpoints/depth_pro.pt --rgb DJI_20250330151738_0001_V.JPG --sparse DJI_20250330151738_0001_V.png --fx 1326.7540200491298 --fy 1319.9252949777992 --cx 1000.0 --cy 726.5 --data KITTI --out out.png --out_npy depth.npy
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf

# Ensure third_party/PSD is on sys.path so that `network.*` and
# `datasets.*` can be imported when this script is executed from
# the repository root (e.g., `python scripts/depth_pro/infer_single.py`).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PSD_ROOT = os.path.join(REPO_ROOT, "third_party", "PSD")
for p in (PSD_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from network.ipde_5c_l1_rezero_un_3 import DCNet

# Avoid importing the entire PSD `datasets` package (its __init__ imports
# many optional datasets). Load only `ip_basic.py` directly.
import importlib.util as _importlib_util
_ip_basic_path = os.path.join(PSD_ROOT, "datasets", "ip_basic.py")
_spec = _importlib_util.spec_from_file_location("psd_ip_basic", _ip_basic_path)
_ip_basic = _importlib_util.module_from_spec(_spec)
assert _spec and _spec.loader is not None
_spec.loader.exec_module(_ip_basic)
ip_fill = _ip_basic.fill_in_fast


def load_sparse_depth(path: str, target_hw: Optional[tuple[int, int]] = None, scale_divisor: float = 256.0) -> torch.Tensor:
    """Load a KITTI/16-bit PNG style sparse depth as float32 meters.

    Args:
        path: path to 16-bit depth PNG
        target_hw: optional (H, W) to resize with nearest
        scale_divisor: divisor to convert to meters (default 256.0 for KITTI)
    Returns:
        Tensor [1, 1, H, W]
    """
    img = Image.open(path)
    arr = np.array(img, dtype=np.int32)
    depth = arr.astype(np.float32) / float(scale_divisor)
    depth[depth <= 0] = 0.0
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    if target_hw is not None:
        depth = F.interpolate(depth, size=target_hw, mode="nearest")
    return depth


def load_rgb_as_tensor(path: str) -> torch.Tensor:
    """Load RGB image to tensor in [0,1], shape [1,3,H,W]."""
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ten


def to_normalized(rgb: torch.Tensor) -> torch.Tensor:
    """Normalize with mean/std = 0.5 for Depth Pro pipeline."""
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def build_cfg(cfg_path: str, depthpro_ckpt: str) -> OmegaConf:
    # Load YAML without struct/readonly so we can add keys.
    cfg = OmegaConf.load(cfg_path)
    # Switch backend to Depth Pro and add the required block.
    cfg.MDEBranch.backbone = "depthpro"
    cfg.MDEBranch["depthpro"] = {
        "pretrained": depthpro_ckpt,
        "resize_h": 384,
        "resize_w": 384,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    # For single-image inference, backbone initializations from external
    # "pretrained" files are unnecessary (weights come from PSD checkpoint).
    # Avoid FileNotFoundError from training-time absolute paths.
    if "RegressBranch" in cfg and "resnet" in cfg.RegressBranch:
        cfg.RegressBranch.resnet.pretrained = None
    if "CSLBranch" in cfg and "resnet" in cfg.CSLBranch:
        cfg.CSLBranch.resnet.pretrained = None
    return cfg


def robust_load_state_dict(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load PSD checkpoint that may be either a full training ckpt or raw state_dict.

    Handles PyTorch>=2.6 weights_only semantics with a safe fallback.
    """
    try:
        # Prefer safe loading when possible (PyTorch 2.6+ default behavior)
        state = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        # Fallback: allow full pickle if the file contains non-tensor objects
        # such as OmegaConf configs saved in training checkpoints.
        import warnings
        warnings.warn(f"Safe load failed ({e}); retrying with weights_only=False.")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state

    # strip potential 'module.' prefixes
    new_state = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if unexpected:
        print(f"[warn] Unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"[warn] Missing keys when loading checkpoint: {missing}")


def main():
    parser = argparse.ArgumentParser("Single-image inference with PSD (Depth Pro backend)")
    default_cfg = os.path.join(PSD_ROOT, "configs", "cfg.yml")
    parser.add_argument("--cfg", type=str, default=default_cfg, help="Path to base config YAML")
    parser.add_argument(
        "--psd_ckpt",
        type=str,
        default="PSD_NK_DPr_checkpoint.ckpt",
        help="Path to PSD checkpoint (.ckpt/.pth). Default: PSD_NK_DPr_checkpoint.ckpt",
    )
    parser.add_argument(
        "--depthpro_ckpt",
        type=str,
        default="third_party/ml-depth-pro/checkpoints/depth_pro.pt",
        help="Path to Depth Pro weights (.pt). Default: third_party/ml-depth-pro/checkpoints/depth_pro.pt",
    )
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--sparse", type=str, required=True, help="Path to sparse depth PNG (16-bit)")
    parser.add_argument("--sparse_scale", type=float, default=256.0, help="Divisor for sparse depth to meters")
    parser.add_argument(
        "--data",
        type=str,
        default="NYUv2",
        help="Dataset profile (e.g., NYUv2, KITTI, VOID1500). Choose to match your scene; do not use 'NK' here.",
    )
    parser.add_argument("--fx", type=float, default=None, help="fx in pixels (defaults to W/2 if None)")
    parser.add_argument("--fy", type=float, default=None, help="fy in pixels (defaults to H/2 if None)")
    parser.add_argument("--cx", type=float, default=None, help="cx in pixels (defaults to W/2)")
    parser.add_argument("--cy", type=float, default=None, help="cy in pixels (defaults to H/2)")
    parser.add_argument("--out", type=str, default="output_depth.png", help="Output depth visualization (PNG)")
    parser.add_argument("--out_npy", type=str, default=None, help="Optional raw depth as NumPy .npy path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and model
    cfg = build_cfg(args.cfg, args.depthpro_ckpt)
    # Reduce KNN update rate for single-image inference to avoid OOM on large images
    ds_name = args.data
    if ds_name in cfg and hasattr(cfg[ds_name], "knn_rate"):
        try:
            cfg[ds_name].knn_rate = float(min(float(cfg[ds_name].knn_rate), 0.05))
        except Exception:
            cfg[ds_name].knn_rate = 0.05
    net = DCNet(cfg).to(device).eval()
    robust_load_state_dict(net, args.psd_ckpt)

    # RGB and sparse
    rgb = load_rgb_as_tensor(args.rgb)
    H, W = rgb.shape[2], rgb.shape[3]
    rgb = rgb.to(device)

    # normalized copy used by network (inputs['rgb'])
    rgb_norm = to_normalized(rgb)

    sparse = load_sparse_depth(args.sparse, target_hw=(H, W), scale_divisor=args.sparse_scale).to(device)

    # generate ip (fast fill) used in the pipeline
    # Convert to HxW for the fast fill-in utility
    sparse_np = sparse.detach().cpu().numpy().squeeze()  # 1,1,H,W -> H,W
    ip_np = ip_fill(sparse_np, max_depth=float(np.max(sparse_np)))
    ip_tensor = torch.from_numpy(ip_np).unsqueeze(0).unsqueeze(0).to(device)  # 1,1,H,W

    # Depth Pro expects 384x384 normalized input
    rgb_for_mde = torch.nn.functional.interpolate(rgb, size=(384, 384), mode="bilinear", align_corners=False)
    rgb_for_mde = to_normalized(rgb_for_mde)

    # intrinsics
    fx = args.fx if args.fx is not None else W / 2.0
    fy = args.fy if args.fy is not None else H / 2.0
    cx = args.cx if args.cx is not None else W / 2.0
    cy = args.cy if args.cy is not None else H / 2.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device).unsqueeze(0)

    inputs = {
        "rgb": rgb_norm,
        "rgb_s": rgb,  # not normalized, seldom used downstream
        "rgb_mde": rgb_for_mde,
        "sparse": sparse,
        "ip": ip_tensor,
        "K": K,
    }

    with torch.no_grad():
        outputs = net(inputs, epoch=42, is_test=True, data=args.data)
        depth_list = outputs["depth"]
        # pick post-processed prediction
        pred_depth = depth_list[-1]  # [1,1,H,W]

    # Save visualization (normalized colormap similar to training utils)
    d = pred_depth.detach().cpu()
    d_min = torch.clamp(d.min(), min=1e-8)
    d_max = d.max()
    d_vis = (d - d_min) / (d_max - d_min + 1e-8)
    d_vis = (d_vis * 255.0).byte().squeeze(0).squeeze(0).numpy()
    Image.fromarray(d_vis).save(args.out)
    print(f"Saved visualization to: {args.out}")

    if args.out_npy is not None:
        np.save(args.out_npy, d.squeeze(0).squeeze(0).numpy())
        print(f"Saved raw depth .npy to: {args.out_npy}")


if __name__ == "__main__":
    main()
