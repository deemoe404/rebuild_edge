#!/usr/bin/env bash
set -euo pipefail

# COLMAP end-to-end reconstruction script for DJI drone datasets
#
# Steps:
#  1) Build image list from dataset root (recursive)
#  2) Feature extraction (SIFT w/ GPU if available)
#  3) Matching: sequential + spatial (uses EXIF GPS if present)
#  4) Sparse reconstruction (mapper)
#  5) Dense reconstruction (undistort → PatchMatch/DepthPro → fusion)
#
# Usage:
#  scripts/colmap_reconstruct.sh \
#    --dataset_root ../datasets/uestc_campus \
#    --workspace outputs/uestc_campus \
#    [--camera_model OPENCV] [--single_camera] [--overlap 8] [--max_gps_neighbor_dist 120] \
#    [--max_image_size 4000] [--dense_max_image_size 2000] [--threads -1] \\
#    [--align_gps] [--alignment_max_error 20] [--alignment_type ecef] [--no_dense] \\
#    [--mvs patchmatch]  # {patchmatch, depthpro}
#    [--pm_depth_min <m>] [--pm_depth_max <m>] \\
#    [--fusion_max_reproj_error 4] [--fusion_max_depth_error 0.02] [--fusion_max_normal_error 20]
#    Mapper speed knobs: \\
#    [--mapper_ba_local_max_iters 15] [--mapper_ba_global_max_iters 25] \\
#    [--mapper_ba_global_images_ratio 1.6] [--mapper_ba_global_points_ratio 1.6]
#
# After completion, artifacts are written to:
#   Sparse (COLMAP format): <run_dir>/sparse/0
#   Sparse (LAS/LAZ, unaligned model): <run_dir>/sparse_unaligned.(laz|las)
#   Sparse aligned (LAS/LAZ when --align_gps succeeds): <run_dir>/sparse_aligned_ecef.(laz|las)
#   Dense fused (LAS/LAZ when dense enabled): <run_dir>/dense/fused_ecef.(laz|las)
#
# Notes:
# - Default camera model is OPENCV to allow radial+tangential distortion.
# - EXIF focal length and GPS priors are imported automatically by COLMAP.
# - For large drone sets, sequential+spatial is typically faster and robust.


########################################
# Parse arguments
########################################
DATASET_ROOT="../datasets/uestc_campus"
WORKSPACE="outputs/uestc_campus"
CAMERA_MODEL="OPENCV"         # {SIMPLE_RADIAL, RADIAL, OPENCV, PINHOLE, ...}
SINGLE_CAMERA=1                # treat all images as same camera intrinsics
SEQUENTIAL_OVERLAP=8           # neighbor overlap for sequential matching
MAX_GPS_NEIGHBOR_DIST=120      # meters for spatial matcher (uses EXIF GPS)
RUN_DENSE=1
ALIGN_GPS=0
ALIGN_MAX_ERROR=20   # meters; must be > 0 for COLMAP >=3.12
# Alignment frame for model_aligner when --align_gps is on.
# Default changed to global ECEF so independently processed flights land in the same world frame.
ALIGNMENT_TYPE="ecef"          # {plane, ecef, enu, enu-plane, enu-plane-unscaled, custom}
# Global image size cap (applied to SIFT extraction and undistortion)
MAX_IMAGE_SIZE=4000
# Global CPU thread cap (-1 uses all cores where supported)
THREADS=-1
# Dense-specific defaults
DENSE_MAX_IMAGE_SIZE=2000      # cap MVS resolution for stability/speed
# Optional global override depth range for PatchMatch stereo (empty to keep auto)
PM_DEPTH_MIN=""                # e.g. 3   (meters)
PM_DEPTH_MAX=""                # e.g. 500 (meters)
# PatchMatch options
PM_ENABLE_GEOM=0               # 0: off (faster, default), 1: on (more robust)
PM_NUM_ITERATIONS=5            # reduce to 3 for speed if needed
PM_WINDOW_RADIUS=5             # reduce to 4/3 for speed (more noise)
PM_NUM_SAMPLES=15              # reduce to 10-12 for speed
PM_CACHE_SIZE=32               # increase if GPU memory allows (e.g., 64)
PM_NUM_SRC=16                  # number of source images per ref (was 20). Applies via patch-match.cfg rewrite
# Fusion thresholds (slightly relaxed to reduce over-filtering)
FUSION_MAX_REPROJ_ERROR=4
FUSION_MAX_DEPTH_ERROR=0.02
FUSION_MAX_NORMAL_ERROR=20
FUSION_CHECK_NUM_IMAGES=50     # reduce to ~30 for speed if needed

# MVS implementation selector
MVS_IMPL="patchmatch"          # {patchmatch, depthpro}
# PSD dataset profile for single-image inference (used when --mvs depthpro)
PSD_DATA="KITTI"
# Default checkpoints (override via CLI)
PSD_CKPT="checkpoints/PSD_NK_DPr_checkpoint.ckpt"
DEPTHPRO_CKPT="checkpoints/depth_pro.pt"

# DepthPro-specific helper PatchMatch (to populate consistency graphs)
DEPTHPRO_PRIME_PATCHMATCH=1
DEPTHPRO_PM_NUM_ITERATIONS=1
DEPTHPRO_PM_WINDOW_RADIUS=3
DEPTHPRO_PM_NUM_SAMPLES=8
DEPTHPRO_PM_CACHE_SIZE=8

# DepthPro scaling and fusion fallback controls
DEPTHPRO_SCALE_MIN=0.05       # lower clamp for per-image scale
DEPTHPRO_SCALE_MAX=20.0       # upper clamp for per-image scale
DEPTHPRO_MIN_FUSED_POINTS=150000  # trigger backprojection fallback if fused points below this
DEPTHPRO_FORCE_BACKPROJECT=0  # force monodepth backprojection output even if fusion produced points
DEPTHPRO_STRICT_FUSION=0      # use tighter fusion thresholds for external depths
DEPTHPRO_TWO_STAGE=1          # try strict fusion first, then relax thresholds before backprojection

# DepthPro local refinement (optional): narrow multi-view search around prior depth
DEPTHPRO_REFINE_LOCAL=0
DEPTHPRO_REFINE_EPSILON=0.03      # +/- search range as fraction of prior depth (e.g., 0.03 = ±3%)
DEPTHPRO_REFINE_NEIGHBORS=6       # number of neighbor views to use
DEPTHPRO_REFINE_WIN=3             # patch half window (3 => 7x7)
DEPTHPRO_REFINE_ITERS=1           # refinement iterations
DEPTHPRO_REFINE_STRIDE=4          # pixel stride for refinement (reduce for speed)

# Mapper (SfM) speed/quality knobs
MAPPER_BA_LOCAL_MAX_ITERS=15
MAPPER_BA_GLOBAL_MAX_ITERS=25
MAPPER_BA_GLOBAL_FRAMES_RATIO=1.6
MAPPER_BA_GLOBAL_POINTS_RATIO=1.6
MAPPER_BA_GLOBAL_MAX_REFINEMENTS=3
MAPPER_BA_LOCAL_MAX_REFINEMENTS=1
MAPPER_BA_USE_GPU=1   # try GPU BA when GPU present

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="$2"; shift 2;;
    --workspace)
      WORKSPACE="$2"; shift 2;;
    --camera_model)
      CAMERA_MODEL="$2"; shift 2;;
    --single_camera)
      SINGLE_CAMERA=1; shift 1;;
    --per_image_camera)
      SINGLE_CAMERA=0; shift 1;;
    --overlap)
      SEQUENTIAL_OVERLAP="$2"; shift 2;;
    --max_gps_neighbor_dist)
      MAX_GPS_NEIGHBOR_DIST="$2"; shift 2;;
    --no_dense)
      RUN_DENSE=0; shift 1;;
    --align_gps)
      ALIGN_GPS=1; shift 1;;
    --alignment_max_error)
      ALIGN_MAX_ERROR="$2"; shift 2;;
    --alignment_type)
      ALIGNMENT_TYPE="$2"; shift 2;;
    --max_image_size)
      MAX_IMAGE_SIZE="$2"; shift 2;;
    --dense_max_image_size)
      DENSE_MAX_IMAGE_SIZE="$2"; shift 2;;
    --pm_depth_min)
      PM_DEPTH_MIN="$2"; shift 2;;
    --pm_depth_max)
      PM_DEPTH_MAX="$2"; shift 2;;
    --enable_geom_consistency)
      PM_ENABLE_GEOM=1; shift 1;;
    --pm_num_iterations)
      PM_NUM_ITERATIONS="$2"; shift 2;;
    --pm_window_radius)
      PM_WINDOW_RADIUS="$2"; shift 2;;
    --pm_num_samples)
      PM_NUM_SAMPLES="$2"; shift 2;;
    --pm_cache_size)
      PM_CACHE_SIZE="$2"; shift 2;;
    --pm_num_src)
      PM_NUM_SRC="$2"; shift 2;;
    --fusion_max_reproj_error)
      FUSION_MAX_REPROJ_ERROR="$2"; shift 2;;
    --fusion_max_depth_error)
      FUSION_MAX_DEPTH_ERROR="$2"; shift 2;;
    --fusion_max_normal_error)
      FUSION_MAX_NORMAL_ERROR="$2"; shift 2;;
    --fusion_check_num_images)
      FUSION_CHECK_NUM_IMAGES="$2"; shift 2;;
    --mvs)
      MVS_IMPL="$2"; shift 2;;
    --mapper_ba_local_max_iters)
      MAPPER_BA_LOCAL_MAX_ITERS="$2"; shift 2;;
    --mapper_ba_global_max_iters)
      MAPPER_BA_GLOBAL_MAX_ITERS="$2"; shift 2;;
    --mapper_ba_global_images_ratio)
      MAPPER_BA_GLOBAL_FRAMES_RATIO="$2"; shift 2;;
    --mapper_ba_global_points_ratio)
      MAPPER_BA_GLOBAL_POINTS_RATIO="$2"; shift 2;;
    --mapper_ba_global_max_refinements)
      MAPPER_BA_GLOBAL_MAX_REFINEMENTS="$2"; shift 2;;
    --mapper_ba_local_max_refinements)
      MAPPER_BA_LOCAL_MAX_REFINEMENTS="$2"; shift 2;;
    --mapper_ba_use_gpu)
      MAPPER_BA_USE_GPU="$2"; shift 2;;
    --threads|--num_threads)
      THREADS="$2"; shift 2;;
    --depthpro_skip_consistency)
      DEPTHPRO_PRIME_PATCHMATCH=0; shift 1;;
    --depthpro_consistency_num_iterations)
      DEPTHPRO_PM_NUM_ITERATIONS="$2"; shift 2;;
    --depthpro_consistency_window_radius)
      DEPTHPRO_PM_WINDOW_RADIUS="$2"; shift 2;;
    --depthpro_consistency_num_samples)
      DEPTHPRO_PM_NUM_SAMPLES="$2"; shift 2;;
    --depthpro_consistency_cache_size)
      DEPTHPRO_PM_CACHE_SIZE="$2"; shift 2;;
    --depthpro_scale_min)
      DEPTHPRO_SCALE_MIN="$2"; shift 2;;
    --depthpro_scale_max)
      DEPTHPRO_SCALE_MAX="$2"; shift 2;;
    --depthpro_min_fused_points)
      DEPTHPRO_MIN_FUSED_POINTS="$2"; shift 2;;
    --depthpro_force_backproject)
      DEPTHPRO_FORCE_BACKPROJECT=1; shift 1;;
    --depthpro_strict_fusion)
      DEPTHPRO_STRICT_FUSION=1; shift 1;;
    --depthpro_two_stage)
      DEPTHPRO_TWO_STAGE=1; shift 1;;
    --depthpro_one_stage)
      DEPTHPRO_TWO_STAGE=0; shift 1;;
    --depthpro_refine_local)
      DEPTHPRO_REFINE_LOCAL=1; shift 1;;
    --depthpro_refine_epsilon)
      DEPTHPRO_REFINE_EPSILON="$2"; shift 2;;
    --depthpro_refine_neighbors)
      DEPTHPRO_REFINE_NEIGHBORS="$2"; shift 2;;
    --depthpro_refine_win)
      DEPTHPRO_REFINE_WIN="$2"; shift 2;;
    --depthpro_refine_iters)
      DEPTHPRO_REFINE_ITERS="$2"; shift 2;;
    --depthpro_refine_stride)
      DEPTHPRO_REFINE_STRIDE="$2"; shift 2;;
    --psd_data)
      PSD_DATA="$2"; shift 2;;
    --psd_ckpt)
      PSD_CKPT="$2"; shift 2;;
    --depthpro_ckpt)
      DEPTHPRO_CKPT="$2"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *)
      echo "Unknown option: $1"; exit 1;;
  esac
done


########################################
# Helpers
########################################
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH" >&2
    exit 1
  fi
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

find_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo 1
  else
    echo 0
  fi
}

pick_sparse_model_dir() {
  # Prefer '0' if present (COLMAP default for largest/first model).
  local base="$1"
  if [[ -d "$base/0" ]]; then
    echo "$base/0"
    return 0
  fi
  # Otherwise pick the first subdir available.
  local first_dir
  first_dir=$(find "$base" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)
  if [[ -n "$first_dir" ]]; then
    echo "$first_dir"
    return 0
  fi
  return 1
}


# Pretty print helpers (colors + emoji)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer helper/pretty.sh; fall back to legacy path
if [[ -f "$SCRIPT_DIR/helper/pretty.sh" ]]; then
  # shellcheck disable=SC1091
  . "$SCRIPT_DIR/helper/pretty.sh"
  setup_colors || true
elif [[ -f "$SCRIPT_DIR/pretty.sh" ]]; then
  # shellcheck disable=SC1091
  . "$SCRIPT_DIR/pretty.sh"
  setup_colors || true
else
  setup_colors() { :; }
  kv() { printf "%s : %s\n" "$1" "$2"; }
  step() { printf "[%s/%s] %s\n" "$1" "$2" "$3"; }
  info() { echo "$*"; }
  ok() { echo "$*"; }
  warn() { echo "$*"; }
  err() { echo "$*"; }
fi


########################################
# Preflight
########################################
require_cmd colmap
mkdir -p "$WORKSPACE"

# Build a unique run directory to keep outputs clean per run
RUN_DIR="$WORKSPACE/run_$(timestamp)"
# Normalize run dir to absolute path for robust redirections
RUN_DIR_ABS=$(realpath -m "$RUN_DIR")
mkdir -p "$RUN_DIR_ABS/logs" "$RUN_DIR_ABS/sparse" "$RUN_DIR_ABS/dense"

# Normalize dataset root to an absolute path
DATASET_ROOT_ABS=$(realpath -m "$DATASET_ROOT")
if [[ ! -d "$DATASET_ROOT_ABS" ]]; then
  echo "Error: dataset_root does not exist: $DATASET_ROOT_ABS" >&2
  exit 1
fi

title "$(emoji "🧭")" "COLMAP Reconstruction"
kv "Dataset root" "$DATASET_ROOT_ABS"
kv "Run dir"       "$RUN_DIR"
timer_start total_run

DB_PATH="$RUN_DIR_ABS/database.db"
IMG_LIST="$RUN_DIR_ABS/image_list.txt"

# Create a relative image list (relative to DATASET_ROOT_ABS)
(
  cd "$DATASET_ROOT_ABS"
  # Find JPG/JPEG files recursively and output relative paths (no leading ./)
  find . -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.JPG' -o -iname '*.JPEG' \) \
    | sed 's#^\./##' \
    | sort \
    > "$IMG_LIST"
)

NUM_IMAGES=$(wc -l < "$IMG_LIST" | tr -d ' ')
if [[ "$NUM_IMAGES" -lt 2 ]]; then
  echo "Error: found only $NUM_IMAGES images under $DATASET_ROOT_ABS" >&2
  exit 1
fi
info "Found $NUM_IMAGES images"

USE_GPU=$(find_gpu)
kv "SIFT use_gpu" "$USE_GPU"
kv "Max image size" "$MAX_IMAGE_SIZE"
kv "Threads" "$THREADS"


########################################
# 1) Feature extraction
########################################
step 1 5 "$(emoji "🔍") Extracting features..." "$RUN_DIR/logs/01_feature_extractor.log"
timer_start step_feature
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$DATASET_ROOT_ABS" \
  --image_list_path "$IMG_LIST" \
  --ImageReader.camera_model "$CAMERA_MODEL" \
  --ImageReader.single_camera "$SINGLE_CAMERA" \
  --SiftExtraction.use_gpu "$USE_GPU" \
  --SiftExtraction.max_image_size "$MAX_IMAGE_SIZE" \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.num_threads "$THREADS" \
  >> "$RUN_DIR_ABS/logs/01_feature_extractor.log" 2>&1
ok "Feature extraction done in $(timer_end step_feature)"


########################################
# 2) Matching: sequential + spatial (GPS)
########################################
step 2 5 "$(emoji "🔗") Matching (sequential)..." "$RUN_DIR/logs/02_sequential_matcher.log"
timer_start step_match_seq
colmap sequential_matcher \
  --database_path "$DB_PATH" \
  --SiftMatching.use_gpu "$USE_GPU" \
  --SequentialMatching.overlap "$SEQUENTIAL_OVERLAP" \
  --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 0 \
  --SiftMatching.num_threads "$THREADS" \
  >> "$RUN_DIR_ABS/logs/02_sequential_matcher.log" 2>&1
ok "Sequential matching done in $(timer_end step_match_seq)"

step 3 5 "$(emoji "📍") Matching (spatial via EXIF GPS)..." "$RUN_DIR/logs/03_spatial_matcher.log"
timer_start step_match_spatial
colmap spatial_matcher \
  --database_path "$DB_PATH" \
  --SiftMatching.use_gpu "$USE_GPU" \
  --SpatialMatching.max_num_neighbors 50 \
  --SpatialMatching.max_distance "$MAX_GPS_NEIGHBOR_DIST" \
  --SiftMatching.num_threads "$THREADS" \
  >> "$RUN_DIR_ABS/logs/03_spatial_matcher.log" 2>&1
ok "Spatial matching done in $(timer_end step_match_spatial)"


########################################
# 3) Sparse reconstruction (SfM)
########################################
step 4 5 "$(emoji "🗺️") Running mapper (sparse reconstruction)..." "$RUN_DIR/logs/04_mapper.log"
timer_start step_mapper
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$DATASET_ROOT_ABS" \
  --output_path "$RUN_DIR_ABS/sparse" \
  --Mapper.multiple_models 1 \
  --Mapper.min_model_size 10 \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 \
  --Mapper.num_threads "$THREADS" \
  --Mapper.ba_use_gpu $([[ "$USE_GPU" -eq 1 && "$MAPPER_BA_USE_GPU" -eq 1 ]] && echo 1 || echo 0) \
  --Mapper.ba_gpu_index $([[ "$USE_GPU" -eq 1 && "$MAPPER_BA_USE_GPU" -eq 1 ]] && echo 0 || echo -1) \
  --Mapper.ba_local_max_num_iterations "$MAPPER_BA_LOCAL_MAX_ITERS" \
  --Mapper.ba_global_max_num_iterations "$MAPPER_BA_GLOBAL_MAX_ITERS" \
  --Mapper.ba_global_images_ratio "$MAPPER_BA_GLOBAL_FRAMES_RATIO" \
  --Mapper.ba_global_points_ratio "$MAPPER_BA_GLOBAL_POINTS_RATIO" \
  --Mapper.ba_global_max_refinements "$MAPPER_BA_GLOBAL_MAX_REFINEMENTS" \
  --Mapper.ba_local_max_refinements "$MAPPER_BA_LOCAL_MAX_REFINEMENTS" \
  >> "$RUN_DIR_ABS/logs/04_mapper.log" 2>&1
ok "Mapper finished in $(timer_end step_mapper)"

SPARSE_MODEL_DIR=$(pick_sparse_model_dir "$RUN_DIR_ABS/sparse" || true)
if [[ -z "$SPARSE_MODEL_DIR" || ! -d "$SPARSE_MODEL_DIR" ]]; then
  err "No sparse model produced in $RUN_DIR/sparse"
  echo "Check logs under $RUN_DIR/logs for details." >&2
  exit 2
fi
info "Selected sparse model: $SPARSE_MODEL_DIR"


########################################
# 4) Dense reconstruction (optional)
########################################
if [[ "$ALIGN_GPS" -eq 1 ]]; then
  info "Aligning sparse model to GPS priors (EXIF)... (log: $RUN_DIR/logs/04b_model_aligner.log)"
  ALIGNED_DIR="$RUN_DIR_ABS/sparse_aligned"
  mkdir -p "$ALIGNED_DIR"
  set +e
  timer_start step_align
  colmap model_aligner \
    --input_path "$SPARSE_MODEL_DIR" \
    --output_path "$ALIGNED_DIR" \
    --database_path "$DB_PATH" \
    --ref_is_gps 1 \
    --alignment_type "$ALIGNMENT_TYPE" \
    --alignment_max_error "$ALIGN_MAX_ERROR" \
    >> "$RUN_DIR_ABS/logs/04b_model_aligner.log" 2>&1
  ALIGN_STATUS=$?
  set -e
  if [[ $ALIGN_STATUS -eq 0 && -f "$ALIGNED_DIR/images.bin" ]]; then
    SPARSE_MODEL_DIR="$ALIGNED_DIR"
    ok "Aligned sparse model: $SPARSE_MODEL_DIR (in $(timer_end step_align))"
  else
    warn "GPS alignment failed or incomplete (in $(timer_end step_align)). Continuing with unaligned model."
  fi
fi

if [[ "$RUN_DENSE" -eq 1 ]]; then
  step 5 5 "$(emoji "🧩") Dense: undistort images..." "$RUN_DIR/logs/05_image_undistorter.log"
  timer_start step_undistort
  colmap image_undistorter \
    --image_path "$DATASET_ROOT_ABS" \
    --input_path "$SPARSE_MODEL_DIR" \
    --output_path "$RUN_DIR_ABS/dense" \
    --output_type COLMAP \
    --max_image_size "$DENSE_MAX_IMAGE_SIZE" \
    >> "$RUN_DIR_ABS/logs/05_image_undistorter.log" 2>&1
  ok "Undistortion done in $(timer_end step_undistort)"

  # Ensure undistorted model exists in TXT format for downstream tools (DepthPro expects TXT)
  if [[ -d "$RUN_DIR_ABS/dense/sparse" ]]; then
    if [[ ! -f "$RUN_DIR_ABS/dense/sparse/cameras.txt" || ! -f "$RUN_DIR_ABS/dense/sparse/images.txt" ]]; then
      echo "Converting undistorted sparse model to TXT for DepthPro..." >> "$RUN_DIR_ABS/logs/05_image_undistorter.log" 2>&1
      colmap model_converter \
        --input_path "$RUN_DIR_ABS/dense/sparse" \
        --output_path "$RUN_DIR_ABS/dense/sparse" \
        --output_type TXT \
        >> "$RUN_DIR_ABS/logs/05_image_undistorter.log" 2>&1 || true
    fi
  fi

  # Choose MVS implementation: PSD-refined DepthPro (photometric bins) or built-in PatchMatch
  if [[ "$MVS_IMPL" == "depthpro" ]]; then
    # 1) Generate sparse depth PNGs + intrinsics CSV from undistorted sparse model
    info "$(emoji "🧪") Dense: Export sparse depth (PNG) for PSD guidance (log: $RUN_DIR/logs/06a_psd_sparse.log)"
    timer_start step_psd_sparse
    if ! python3 "$SCRIPT_DIR/depth_pro/colmap_sparse_to_sparse_depth.py" \
      --sparse_dir "$RUN_DIR_ABS/dense/sparse" \
      --out_dir "$RUN_DIR_ABS/dense/psd_sparse_depth" \
      --export_intrinsics_csv "$RUN_DIR_ABS/dense/psd_sparse_depth/intrinsics.csv" >> "$RUN_DIR_ABS/logs/06a_psd_sparse.log" 2>&1; then
      err "Exporting PSD sparse depth failed"; exit 3
    fi
    ok "PSD sparse depth ready in $(timer_end step_psd_sparse)"

    # 2) Run PSD (with Depth Pro backbone) to write COLMAP photometric bins
    info "$(emoji "🧊") Dense: PSD inference → COLMAP depth bins (log: $RUN_DIR/logs/06_psd.log)"
    timer_start step_psd
    if ! python3 "$SCRIPT_DIR/depth_pro/psd_to_colmap.py" \
      --dense_dir "$RUN_DIR_ABS/dense" \
      --sparse_png_dir "$RUN_DIR_ABS/dense/psd_sparse_depth" \
      --intrinsics_csv "$RUN_DIR_ABS/dense/psd_sparse_depth/intrinsics.csv" \
      --psd_ckpt "$PSD_CKPT" \
      --depthpro_ckpt "$DEPTHPRO_CKPT" \
      --list "$IMG_LIST" \
      --data "$PSD_DATA" >> "$RUN_DIR_ABS/logs/06_psd.log" 2>&1; then
      err "PSD stage failed"; exit 3
    fi
    ok "PSD done in $(timer_end step_psd)"
    FUSION_INPUT_TYPE="photometric"
    # Fusion thresholds for external single‑view depths. Use stricter gates when
    # requested; otherwise keep permissive defaults to maximize recall.
    if [[ "${DEPTHPRO_STRICT_FUSION}" -eq 1 ]]; then
      # More conservative fusion to suppress multi‑layer shells and noisy regions
      FUSION_CHECK_NUM_IMAGES=${FUSION_CHECK_NUM_IMAGES:-2}
      FUSION_MAX_REPROJ_ERROR=${FUSION_MAX_REPROJ_ERROR:-4}
      FUSION_MAX_DEPTH_ERROR=${FUSION_MAX_DEPTH_ERROR:-0.05}
      FUSION_MAX_NORMAL_ERROR=${FUSION_MAX_NORMAL_ERROR:-30}
    else
      FUSION_CHECK_NUM_IMAGES=1
      FUSION_MAX_REPROJ_ERROR=8
      FUSION_MAX_DEPTH_ERROR=0.40
      FUSION_MAX_NORMAL_ERROR=180
    fi
  else
    info "$(emoji "🧊") Dense: PatchMatch stereo... (log: $RUN_DIR/logs/06_patch_match_stereo.log)"
    timer_start step_patchmatch
    # Reduce source images per reference view by rewriting patch-match.cfg, if requested
    if [[ -f "$RUN_DIR_ABS/dense/stereo/patch-match.cfg" && -n "$PM_NUM_SRC" ]]; then
      if [[ "$PM_NUM_SRC" =~ ^[0-9]+$ && "$PM_NUM_SRC" -gt 0 ]]; then
        sed -i -E "s/__auto__, [0-9]+$/__auto__, ${PM_NUM_SRC}/" "$RUN_DIR_ABS/dense/stereo/patch-match.cfg" || true
      fi
    fi
    # Optional depth override args
    PM_EXTRA_ARGS=()
    if [[ -n "${PM_DEPTH_MIN}" ]]; then
      PM_EXTRA_ARGS+=(--PatchMatchStereo.depth_min "${PM_DEPTH_MIN}")
    fi
    if [[ -n "${PM_DEPTH_MAX}" ]]; then
      PM_EXTRA_ARGS+=(--PatchMatchStereo.depth_max "${PM_DEPTH_MAX}")
    fi
    colmap patch_match_stereo \
      --workspace_path "$RUN_DIR_ABS/dense" \
      --workspace_format COLMAP \
      --PatchMatchStereo.gpu_index $([[ "$USE_GPU" -eq 1 ]] && echo 0 || echo -1) \
      --PatchMatchStereo.max_image_size "$DENSE_MAX_IMAGE_SIZE" \
      --PatchMatchStereo.geom_consistency $([[ "$PM_ENABLE_GEOM" -eq 1 ]] && echo true || echo false) \
      --PatchMatchStereo.num_iterations "$PM_NUM_ITERATIONS" \
      --PatchMatchStereo.window_radius "$PM_WINDOW_RADIUS" \
      --PatchMatchStereo.num_samples "$PM_NUM_SAMPLES" \
      --PatchMatchStereo.cache_size "$PM_CACHE_SIZE" \
      --PatchMatchStereo.filter 1 \
      "${PM_EXTRA_ARGS[@]}" \
      >> "$RUN_DIR_ABS/logs/06_patch_match_stereo.log" 2>&1
    ok "PatchMatch done in $(timer_end step_patchmatch)"
    # Choose fusion input type to match PatchMatch output
    FUSION_INPUT_TYPE=$([[ "$PM_ENABLE_GEOM" -eq 1 ]] && echo geometric || echo photometric)
  fi

  info "$(emoji "🔄") Dense: stereo fusion (TXT model output for high precision)... (log: $RUN_DIR/logs/07_stereo_fusion.log)"
  timer_start step_fusion
  FUSED_TXT_DIR="$RUN_DIR_ABS/dense/fused_txt"
  mkdir -p "$FUSED_TXT_DIR"
  colmap stereo_fusion \
    --workspace_path "$RUN_DIR_ABS/dense" \
    --workspace_format COLMAP \
    --input_type "$FUSION_INPUT_TYPE" \
    --output_type TXT \
    --output_path "$FUSED_TXT_DIR" \
    --StereoFusion.num_threads "$THREADS" \
    --StereoFusion.max_reproj_error "$FUSION_MAX_REPROJ_ERROR" \
    --StereoFusion.max_depth_error "$FUSION_MAX_DEPTH_ERROR" \
    --StereoFusion.max_normal_error "$FUSION_MAX_NORMAL_ERROR" \
    --StereoFusion.check_num_images "$FUSION_CHECK_NUM_IMAGES" \
    >> "$RUN_DIR_ABS/logs/07_stereo_fusion.log" 2>&1
  ok "Stereo fusion done in $(timer_end step_fusion)"

  # Prepare fusion output reference
  DENSE_PC_TXT_MODE="colmap"
  DENSE_PC_TXT_PATH="$FUSED_TXT_DIR/points3D.txt"

  # Convert to LAS/LAZ (prefer LAZ if lazrs available). This avoids large-magnitude float32 precision loss of PLY in ECEF.
  if command -v python3 >/dev/null 2>&1; then
    info "Converting dense fused points to LA(S/Z)..."
    timer_start step_dense_convert
    if [[ "$DENSE_PC_TXT_MODE" == "colmap" ]]; then
      CONVERT_ARGS=(--colmap_points3D "$DENSE_PC_TXT_PATH")
    else
      CONVERT_ARGS=(--xyzrgb_txt "$DENSE_PC_TXT_PATH")
    fi
    if python3 "$SCRIPT_DIR/helper/pointcloud_to_las.py" "${CONVERT_ARGS[@]}" --out "$RUN_DIR_ABS/dense/fused_ecef.laz" --srs EPSG:4978 >> "$RUN_DIR_ABS/logs/07_stereo_fusion.log" 2>&1; then
      ok "Dense point cloud (LAZ): $RUN_DIR/dense/fused_ecef.laz (in $(timer_end step_dense_convert))"
      rm -rf "$FUSED_TXT_DIR" || true
    else
      warn "LAZ conversion failed; trying uncompressed LAS..."
      if python3 "$SCRIPT_DIR/helper/pointcloud_to_las.py" "${CONVERT_ARGS[@]}" --out "$RUN_DIR_ABS/dense/fused_ecef.las" --srs EPSG:4978 >> "$RUN_DIR_ABS/logs/07_stereo_fusion.log" 2>&1; then
        ok "Dense point cloud (LAS): $RUN_DIR/dense/fused_ecef.las (in $(timer_end step_dense_convert))"
        rm -rf "$FUSED_TXT_DIR" || true
      else
        warn "Failed to produce LAS/LAZ from dense fused data (in $(timer_end step_dense_convert)). Leaving TXT."
      fi
    fi
  else
    warn "python3 not found; skipping LAS/LAZ conversion for dense output."
  fi
else
  warn "Dense reconstruction disabled (--no_dense)."
fi

#
########################################
# 5) Export sparse model(s) to LAS/LAZ (no PLY)
########################################
info "$(emoji "📦") Exporting sparse model(s) to LAS/LAZ... (log: $RUN_DIR/logs/08_model_export_las.log)"
timer_start step_export
{
  # Prefer aligned model for global ECEF export
  if [[ -f "$RUN_DIR_ABS/sparse_aligned/points3D.bin" ]]; then
    echo "  -> Aligned sparse (TXT -> LAZ/LAS)"
    mkdir -p "$RUN_DIR_ABS/sparse_aligned_txt"
    colmap model_converter \
      --input_path "$RUN_DIR_ABS/sparse_aligned" \
      --output_path "$RUN_DIR_ABS/sparse_aligned_txt" \
      --output_type TXT || echo "  Warn: failed to export aligned sparse to TXT"
    if [[ -f "$RUN_DIR_ABS/sparse_aligned_txt/points3D.txt" ]]; then
      if python3 "$SCRIPT_DIR/helper/pointcloud_to_las.py" --colmap_points3D "$RUN_DIR_ABS/sparse_aligned_txt/points3D.txt" --out "$RUN_DIR_ABS/sparse_aligned_ecef.laz" --srs EPSG:4978; then
        echo "  -> Wrote $RUN_DIR_ABS/sparse_aligned_ecef.laz"
        rm -rf "$RUN_DIR_ABS/sparse_aligned_txt" || true
      elif python3 "$SCRIPT_DIR/helper/pointcloud_to_las.py" --colmap_points3D "$RUN_DIR_ABS/sparse_aligned_txt/points3D.txt" --out "$RUN_DIR_ABS/sparse_aligned_ecef.las" --srs EPSG:4978; then
        echo "  -> Wrote $RUN_DIR_ABS/sparse_aligned_ecef.las"
        rm -rf "$RUN_DIR_ABS/sparse_aligned_txt" || true
      else
        echo "  Warn: failed to produce LAS/LAZ for aligned sparse; leaving TXT" >&2
      fi
    fi
  else
    echo "  Note: no aligned sparse model found; skipping aligned LAS/LAZ export"
  fi

  # Optionally export the largest unaligned model as well (in its own frame)
  UNALIGNED_DIR=$(pick_sparse_model_dir "$RUN_DIR_ABS/sparse" || true)
  if [[ -n "$UNALIGNED_DIR" && -f "$UNALIGNED_DIR/points3D.bin" ]]; then
    echo "  -> Unaligned sparse (TXT -> LAZ/LAS)"
    mkdir -p "$RUN_DIR_ABS/sparse_txt"
    colmap model_converter \
      --input_path "$UNALIGNED_DIR" \
      --output_path "$RUN_DIR_ABS/sparse_txt" \
      --output_type TXT || echo "  Warn: failed to export unaligned sparse to TXT"
    if [[ -f "$RUN_DIR_ABS/sparse_txt/points3D.txt" ]]; then
      if python3 "$SCRIPT_DIR/helper/pointcloud_to_las.py" --colmap_points3D "$RUN_DIR_ABS/sparse_txt/points3D.txt" --out "$RUN_DIR_ABS/sparse_unaligned.laz"; then
        echo "  -> Wrote $RUN_DIR_ABS/sparse_unaligned.laz"
        rm -rf "$RUN_DIR_ABS/sparse_txt" || true
      elif python3 "$SCRIPT_DIR/helper/pointcloud_to_las.py" --colmap_points3D "$RUN_DIR_ABS/sparse_txt/points3D.txt" --out "$RUN_DIR_ABS/sparse_unaligned.las"; then
        echo "  -> Wrote $RUN_DIR_ABS/sparse_unaligned.las"
        rm -rf "$RUN_DIR_ABS/sparse_txt" || true
      else
        echo "  Warn: failed to produce LAS/LAZ for unaligned sparse; leaving TXT" >&2
      fi
    fi
  else
    echo "  Skip unaligned export: model not found"
  fi
} >> "$RUN_DIR_ABS/logs/08_model_export_las.log" 2>&1
ok "Sparse model export done in $(timer_end step_export)"

info "Sparse model path: $SPARSE_MODEL_DIR"
ok "COLMAP run finished at: $RUN_DIR (in $(timer_end total_run)) (LAS/LAZ outputs where possible)"
