#!/usr/bin/env bash
set -euo pipefail

# Run COLMAP reconstruction per immediate subfolder under a dataset root.
# This produces one reconstruction (sparse and optional dense) per flight folder.
#
# Example:
#  scripts/colmap_reconstruct_by_folder.sh \
#    --dataset_root ../datasets/uestc_campus \
#    --workspace outputs/uestc_campus \
#    --no_dense --align_gps

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found" >&2
    exit 1
  fi
}

# Pretty print helpers (colors + emoji)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
  title() { echo "$*"; }
  info() { echo "$*"; }
  ok() { echo "$*"; }
  warn() { echo "$*"; }
fi

DATASET_ROOT=""
WORKSPACE=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="$2"; shift 2;;
    --workspace)
      WORKSPACE="$2"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *)
      FORWARD_ARGS+=("$1"); shift 1;;
  esac
done

if [[ -z "${DATASET_ROOT}" || -z "${WORKSPACE}" ]]; then
  echo "Usage: $0 --dataset_root <dir> --workspace <dir> [colmap_reconstruct.sh flags...]" >&2
  exit 1
fi

require_cmd colmap

ROOT_ABS=$(realpath -m "$DATASET_ROOT")
WORKSPACE_ABS=$(realpath -m "$WORKSPACE")
mkdir -p "$WORKSPACE_ABS"

kv "Dataset root" "$ROOT_ABS"
kv "Workspace"    "$WORKSPACE_ABS"

shopt -s nullglob
mapfile -t SUBDIRS < <(find "$ROOT_ABS" -mindepth 1 -maxdepth 1 -type d | sort)
shopt -u nullglob

if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
  echo "No subdirectories found under $ROOT_ABS" >&2
  exit 1
fi

has_images() {
  local d="$1"
  local n
  n=$(find "$d" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.JPG' -o -iname '*.JPEG' \) | wc -l | tr -d ' ')
  [[ "$n" -ge 2 ]]
}

for d in "${SUBDIRS[@]}"; do
  base=$(basename "$d")
  if ! has_images "$d"; then
    warn "Skip '$base': no images found"
    continue
  fi

  OUT_DIR="$WORKSPACE_ABS/$base"
  mkdir -p "$OUT_DIR"
  title "$(emoji "✈️")" "Running flight '$base' → workspace: $OUT_DIR"
  timer_start flight_"$base"
  scripts/colmap_reconstruct.sh \
    --dataset_root "$d" \
    --workspace "$OUT_DIR" \
    "${FORWARD_ARGS[@]}"
  ok "Flight '$base' finished in $(timer_end flight_"$base")"
done

ok "All flights processed. Outputs under: $WORKSPACE_ABS/<flight>/run_<timestamp>"
