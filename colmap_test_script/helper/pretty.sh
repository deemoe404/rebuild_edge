#!/usr/bin/env bash
# Lightweight pretty-print helpers for colorful CLI output with optional emoji.

# Prevent multiple inclusions
if [[ -n "${PRETTY_SH_INCLUDED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
PRETTY_SH_INCLUDED=1

setup_colors() {
  # Enable ANSI colors if stdout is a TTY and NO_COLOR is not set
  if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    RST="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
    RED="\033[31m"; GRN="\033[32m"; YLW="\033[33m"; BLU="\033[34m"; MGN="\033[35m"; CYN="\033[36m"; WHT="\033[37m"
  else
    RST=""; BOLD=""; DIM=""; RED=""; GRN=""; YLW=""; BLU=""; MGN=""; CYN=""; WHT=""
  fi
}

emoji() {
  # Disable emoji if NO_EMOJI is set
  if [[ -n "${NO_EMOJI:-}" ]]; then
    printf ""
  else
    printf "%s" "$1"
  fi
}

hr() {
  # Horizontal rule: hr "-" 60
  local char="${1:--}"; local n="${2:-60}"
  local line=""
  for _ in $(seq 1 "$n"); do line+="$char"; done
  printf "%b%s%b\n" "$DIM" "$line" "$RST"
}

kv() {
  # Key-value pretty print: kv "Key" "Value"
  local key="$1"; shift
  local val="$*"
  printf "%b%-14s%b : %s\n" "$BOLD" "$key" "$RST" "$val"
}

tag() {
  # tag <color> <label> <message>
  local color="$1"; local label="$2"; shift 2
  printf "%b[%s]%b %s\n" "$color" "$label" "$RST" "$*"
}

info()  { tag "$BLU"  "$(emoji "ℹ️ ") INFO"  "$*"; }
ok()    { tag "$GRN"  "$(emoji "✅") DONE"  "$*"; }
warn()  { tag "$YLW"  "$(emoji "⚠️ ") WARN"  "$*"; }
err()   { tag "$RED"  "$(emoji "❌") ERROR" "$*"; }

step() {
  # step <idx> <total> <icon> <message> [log_path]
  local idx="$1"; local total="$2"; local icon="$3"; local msg="$4"; local log="${5:-}"
  printf "%b[%s/%s]%b %s %s" "$BOLD" "$idx" "$total" "$RST" "$icon" "$msg"
  if [[ -n "$log" ]]; then
    printf " %b(log: %s)%b" "$DIM" "$log" "$RST"
  fi
  printf "\n"
}

title() {
  # title <emoji> <text>
  local icon="$1"; shift
  printf "%s %b%s%b\n" "$icon" "$BOLD" "$*" "$RST"
}

# --- Timing helpers ---------------------------------------------------------
_timer_key() {
  # sanitize to a safe env var suffix
  local k="${1//[^A-Za-z0-9_]/_}"
  printf "TIMER_%s" "$k"
}

now_secs() { date +%s; }

timer_start() {
  # timer_start <name>
  local var; var=$(_timer_key "$1")
  local t; t=$(now_secs)
  printf -v "$var" '%s' "$t"
}

fmt_duration() {
  # fmt_duration <seconds>
  local s=$1
  if [[ -z "$s" || "$s" -lt 0 ]]; then echo "?"; return; fi
  local h=$((s/3600)); local m=$(((s%3600)/60)); local ss=$((s%60))
  if (( h > 0 )); then
    printf "%dh%02dm%02ds" "$h" "$m" "$ss"
  elif (( m > 0 )); then
    printf "%dm%02ds" "$m" "$ss"
  else
    printf "%ds" "$ss"
  fi
}

timer_end() {
  # timer_end <name> -> prints formatted duration, unsets timer
  local var; var=$(_timer_key "$1")
  local start=${!var:-}
  if [[ -z "$start" ]]; then echo "?"; return; fi
  local end; end=$(now_secs)
  local dt=$((end - start))
  unset "$var"
  fmt_duration "$dt"
}
