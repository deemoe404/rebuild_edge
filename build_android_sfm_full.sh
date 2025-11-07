#!/usr/bin/env bash
# build_android_sfm_full.sh
# Purpose: From an Android Studio project root, fetch & cross-compile
#          gflags, glog, Ceres (static), and OpenCV+contrib (with sfm)
#          for Android arm64-v8a, then stage .so/.a and headers for use
#          in an Android Library module (AAR) or app.
# Host: macOS (Apple Silicon or Intel)
# Toolchain: Android NDK r27.0+ (prefers 27.0.12077973)
#
# Usage:
#   chmod +x ./build_android_sfm_full.sh
#   ./build_android_sfm_full.sh
#
# Optional env vars to override defaults:
#   ANDROID_SDK, ANDROID_NDK_HOME, NDK_VERSION
#   EIGEN3_CMAKE_DIR  (if you already have Eigen3Config.cmake)
#   ENABLE_TBB        (default ON; set OFF to skip TBB backend)
#   TARGET_MODULE     (Android module to receive jniLibs; default: autodetect 'sfmfull' or 'app')
#   MAX_LONG_EDGE     (image scale hint for downstream; unused here but exported)
#
set -euo pipefail
IFS=$'\n\t'

### Pretty logging -----------------------------------------------------------
# Print to both the log (default stdout after redirection) and a preserved
# console FD (3) when available, to avoid flooding the CLI while keeping
# full logs on disk.
cecho() {
  printf "\033[1;36m%s\033[0m\n" "$*" >&3 2>/dev/null || true
  printf "\033[1;36m%s\033[0m\n" "$*"
} # cyan
wecho() {
  printf "\033[1;33m%s\033[0m\n" "$*" >&3 2>/dev/null || true
  printf "\033[1;33m%s\033[0m\n" "$*"
} # yellow
recho() {
  printf "\033[1;31m%s\033[0m\n" "$*" >&3 2>/dev/null || true
  printf "\033[1;31m%s\033[0m\n" "$*"
} # red
gecho() {
  printf "\033[1;32m%s\033[0m\n" "$*" >&3 2>/dev/null || true
  printf "\033[1;32m%s\033[0m\n" "$*"
} # green

### Paths & layout -----------------------------------------------------------
ROOT_DIR="$(pwd)"
TP_DIR="$ROOT_DIR/third_party"         # sources
BUILD_DIR="$ROOT_DIR/android-build"     # build trees
INSTALL_DIR="$ROOT_DIR/android-install" # cmake install prefixes
OUT_DIR="$ROOT_DIR/out/android"         # staging for libs/headers
OUT_LIB_DIR="$OUT_DIR/jniLibs/arm64-v8a"
OUT_INC_DIR="$OUT_DIR/include"

mkdir -p "$TP_DIR" "$BUILD_DIR" "$INSTALL_DIR" "$OUT_LIB_DIR" "$OUT_INC_DIR"

### Centralized logging ------------------------------------------------------
# All command output goes to a timestamped log file under out/logs.
# Console prints only brief progress/summaries via FD 3.
LOG_DIR="$ROOT_DIR/out/logs"
mkdir -p "$LOG_DIR"
# Preserve original stdout/stderr for concise console messages
exec 3>&1 4>&2
LOG_FILE="$LOG_DIR/build_android_sfm_full_$(date +%Y%m%d_%H%M%S).log"
printf "Logging to %s (console shows brief progress)\n" "$LOG_FILE" >&3
exec >"$LOG_FILE" 2>&1
# Update latest symlink for convenience
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log" 2>/dev/null || true

# On exit, summarize to console and, on failure, show the tail of the log
trap 'code=$?; if [[ $code -ne 0 ]]; then \
  printf "\n[!] Build failed. See log: %s\n" "$LOG_FILE" >&3; \
  tail -n 50 "$LOG_FILE" >&3 2>/dev/null || true; \
else \
  printf "\n[\xE2\x9C\x93] Build complete. Log saved: %s\n" "$LOG_FILE" >&3; \
fi' EXIT

### Android SDK/NDK discovery -----------------------------------------------
: "${ANDROID_SDK:=$HOME/Library/Android/sdk}"
if [[ ! -d "$ANDROID_SDK" ]]; then
  recho "Android SDK not found at $ANDROID_SDK"
  recho "Install via Android Studio → SDK Manager, or export ANDROID_SDK before running."
  exit 1
fi

# Prefer a specific NDK (27.0) to avoid ABI/STL mismatches
: "${NDK_VERSION:=27.0.12077973}"
if [[ -z "${ANDROID_NDK_HOME:-}" || ! -d "$ANDROID_NDK_HOME" ]]; then
  # Try preferred 27.0, else last side-by-side NDK
  if [[ -d "$ANDROID_SDK/ndk/$NDK_VERSION" ]]; then
    export ANDROID_NDK_HOME="$ANDROID_SDK/ndk/$NDK_VERSION"
  else
    # pick the newest installed NDK
    CANDIDATES=($(ls -1 "$ANDROID_SDK/ndk" 2>/dev/null || true))
    if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
      recho "No NDK found under $ANDROID_SDK/ndk. Install 'NDK (Side by side)' in SDK Manager."
      exit 1
    fi
    # pick highest version
    mapfile -t SORTED < <(printf '%s\n' "${CANDIDATES[@]}" | sort -V)
    export ANDROID_NDK_HOME="$ANDROID_SDK/ndk/${SORTED[-1]}"
    wecho "Using NDK: $ANDROID_NDK_HOME (override with ANDROID_NDK_HOME or NDK_VERSION)"
  fi
fi
export NDK="$ANDROID_NDK_HOME"

TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
  recho "Missing android.toolchain.cmake at $TOOLCHAIN"
  exit 1
fi

# Tools check
need_tool() { command -v "$1" >/dev/null 2>&1 || { recho "Missing tool: $1"; MISSING=1; }; }
MISSING=0
need_tool git; need_tool cmake; need_tool ninja
if [[ ${MISSING} -eq 1 ]]; then
  recho "Please install missing tools (brew install cmake ninja git)."
  exit 1
fi

### Eigen3 discovery/installation -------------------------------------------
if [[ -z "${EIGEN3_CMAKE_DIR:-}" ]]; then
  # Try Homebrew eigen@3 first
  if [[ -f "/opt/homebrew/opt/eigen@3/share/eigen3/cmake/Eigen3Config.cmake" ]]; then
    export EIGEN3_CMAKE_DIR="/opt/homebrew/opt/eigen@3/share/eigen3/cmake"
    gecho "Using Homebrew eigen@3 at $EIGEN3_CMAKE_DIR"
  elif [[ -f "/usr/local/opt/eigen@3/share/eigen3/cmake/Eigen3Config.cmake" ]]; then
    export EIGEN3_CMAKE_DIR="/usr/local/opt/eigen@3/share/eigen3/cmake"
    gecho "Using Homebrew eigen@3 (Intel) at $EIGEN3_CMAKE_DIR"
  else
    # Fallback: download & install locally
    cecho "Installing Eigen 3.4.0 locally..."
    EIGEN_SRC="$TP_DIR/eigen"
    EIGEN_BUILD="$BUILD_DIR/eigen"
    EIGEN_INST="$INSTALL_DIR/eigen"
    if [[ ! -d "$EIGEN_SRC" ]]; then
      curl -L -o "$TP_DIR/eigen-3.4.0.tar.gz" https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      mkdir -p "$EIGEN_SRC"
      tar -xzf "$TP_DIR/eigen-3.4.0.tar.gz" --strip-components=1 -C "$EIGEN_SRC"
    fi
    cmake -S "$EIGEN_SRC" -B "$EIGEN_BUILD" -DCMAKE_INSTALL_PREFIX="$EIGEN_INST"
    cmake --build "$EIGEN_BUILD" --target install -j
    export EIGEN3_CMAKE_DIR="$EIGEN_INST/share/eigen3/cmake"
    gecho "Eigen installed to $EIGEN3_CMAKE_DIR"
  fi
fi
[[ -f "$EIGEN3_CMAKE_DIR/Eigen3Config.cmake" ]] || { recho "Eigen3Config.cmake not found in $EIGEN3_CMAKE_DIR"; exit 1; }

### Common CMake flags -------------------------------------------------------
ANDROID_CMAKE_FLAGS=(
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN"
  -DANDROID_ABI=arm64-v8a
  -DANDROID_PLATFORM=android-24
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)

### gflags (shared) ----------------------------------------------------------
cecho "Building gflags (Android) ..."
GFLAGS_SRC="$TP_DIR/gflags"
GFLAGS_BUILD="$BUILD_DIR/gflags"
GFLAGS_INST="$INSTALL_DIR/gflags"
[[ -d "$GFLAGS_SRC" ]] || git clone --depth=1 -b v2.2.2 https://github.com/gflags/gflags.git "$GFLAGS_SRC"
cmake -G Ninja -S "$GFLAGS_SRC" -B "$GFLAGS_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$GFLAGS_INST"
cmake --build "$GFLAGS_BUILD" --target install -j
GFLAGS_CMAKE_DIR="$GFLAGS_INST/lib/cmake/gflags"

### glog (shared, with gflags) ----------------------------------------------
cecho "Building glog (Android) ..."
GLOG_SRC="$TP_DIR/glog"
GLOG_BUILD="$BUILD_DIR/glog"
GLOG_INST="$INSTALL_DIR/glog"
[[ -d "$GLOG_SRC" ]] || git clone --depth=1 -b v0.7.0 https://github.com/google/glog.git "$GLOG_SRC"
cmake -G Ninja -S "$GLOG_SRC" -B "$GLOG_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=ON -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$GLOG_INST" \
  -Dgflags_DIR="$GFLAGS_CMAKE_DIR"
cmake --build "$GLOG_BUILD" --target install -j
GLOG_CMAKE_DIR="$GLOG_INST/lib/cmake/glog"

### Ceres (static) -----------------------------------------------------------
cecho "Building Ceres (Android, static) ..."
CERES_SRC="$TP_DIR/ceres-solver"
CERES_BUILD="$BUILD_DIR/ceres"
CERES_INST="$INSTALL_DIR/ceres"
[[ -d "$CERES_SRC" ]] || git clone --depth=1 -b 2.2.0 https://github.com/ceres-solver/ceres-solver.git "$CERES_SRC"
cmake -G Ninja -S "$CERES_SRC" -B "$CERES_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DMINIGLOG=ON -DSUITESPARSE=OFF -DLAPACK=OFF \
  -DEigen3_DIR="$EIGEN3_CMAKE_DIR" \
  -DCMAKE_INSTALL_PREFIX="$CERES_INST"
cmake --build "$CERES_BUILD" --target install -j
CERES_CMAKE_DIR="$CERES_INST/lib/cmake/Ceres"

### TBB backend toggle ------------------------------------------------------
# Default ON. If enabled, let OpenCV download/build oneTBB internally.
: "${ENABLE_TBB:=ON}"

### OpenCV + contrib/sfm (shared) -------------------------------------------
# If TBB is enabled, require C++17 for oneTBB headers.
CXX_STD=14
if [[ "$ENABLE_TBB" == "ON" ]]; then CXX_STD=17; fi
cecho "Building OpenCV + contrib (Android, with sfm) ..."
OPENCV_SRC="$TP_DIR/opencv"
OPENCV_CONTRIB_SRC="$TP_DIR/opencv_contrib"
OPENCV_BUILD="$BUILD_DIR/opencv"
OPENCV_INST="$INSTALL_DIR/opencv"
[[ -d "$OPENCV_SRC" ]] || git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv.git "$OPENCV_SRC"
[[ -d "$OPENCV_CONTRIB_SRC" ]] || git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv_contrib.git "$OPENCV_CONTRIB_SRC"

cmake -G Ninja -S "$OPENCV_SRC" -B "$OPENCV_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -USFM_GLOG_GFLAGS_TEST -USFM_GLOG_GFLAGS_TEST_CACHE_KEY \
  -DBUILD_SHARED_LIBS=ON \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DOPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB_SRC/modules" \
  -DBUILD_opencv_sfm=ON \
  -DWITH_EIGEN=ON -DEigen3_DIR="$EIGEN3_CMAKE_DIR" \
  -DCeres_DIR="$CERES_CMAKE_DIR" \
  -Dgflags_DIR="$GFLAGS_CMAKE_DIR" \
  -Dglog_DIR="$GLOG_CMAKE_DIR" \
  -DGFLAGS_INCLUDE_DIRS="$GFLAGS_INST/include" \
  -DGLOG_INCLUDE_DIRS="$GLOG_INST/include" \
  -DGFLAGS_LIBRARIES="$GFLAGS_INST/lib/libgflags.so" \
  -DGLOG_LIBRARIES="$GLOG_INST/lib/libglog.so" \
  -DCMAKE_CXX_STANDARD=$CXX_STD -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-} -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS -DGLOG_NO_ABBREVIATED_SEVERITIES" \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_ANDROID_PROJECTS=OFF \
  -DBUILD_LIST=core,imgproc,features2d,flann,calib3d,imgcodecs,video,xfeatures2d,sfm \
  -DCMAKE_INSTALL_PREFIX="$OPENCV_INST" \
  $([[ "$ENABLE_TBB" == "ON" ]] && printf '%s' "-DWITH_TBB=ON -DBUILD_TBB=ON -DTBB_TEST=OFF -DTBB_BUILD_SHARED=OFF")
cmake --build "$OPENCV_BUILD" --target install -j

### Stage outputs ------------------------------------------------------------
cecho "Staging libraries and headers ..."
# OpenCV .so
find "$OPENCV_INST" -name 'libopencv_*.so' -path '*/arm64-v8a/*' -print -exec cp -v {} "$OUT_LIB_DIR" \;
# If OpenCV built with TBB dynamically, its sdk/native/libs may contain libtbb.so
if [[ -f "$OPENCV_INST/sdk/native/libs/arm64-v8a/libtbb.so" ]]; then
  cp -v "$OPENCV_INST/sdk/native/libs/arm64-v8a/libtbb.so" "$OUT_LIB_DIR" || true
fi
# Dependencies .so
cp -v "$GFLAGS_INST/lib/libgflags.so" "$OUT_LIB_DIR" || true
cp -v "$GLOG_INST/lib/libglog.so"   "$OUT_LIB_DIR" || true
# No need to copy TBB .so when statically linked into OpenCV.
# Ceres static
mkdir -p "$OUT_DIR/libstatic"
cp -v "$CERES_INST/lib/libceres.a" "$OUT_DIR/libstatic/" || true

# Headers (for your JNI builds)
# OpenCV headers live under include/opencv4
mkdir -p "$OUT_INC_DIR/opencv4" "$OUT_INC_DIR/ceres" "$OUT_INC_DIR/eigen3"
rsync -a "$OPENCV_INST/sdk/native/jni/include/" "$OUT_INC_DIR/opencv4/"
rsync -a "$CERES_INST/include/" "$OUT_INC_DIR/ceres/"
# Eigen: prefer Homebrew path if used; else local install
if [[ -d "$INSTALL_DIR/eigen/include/eigen3" ]]; then
  rsync -a "$INSTALL_DIR/eigen/include/eigen3/" "$OUT_INC_DIR/eigen3/"
else
  # Homebrew keg-only include path
  if [[ -d "/opt/homebrew/opt/eigen@3/include/eigen3" ]]; then
    rsync -a "/opt/homebrew/opt/eigen@3/include/eigen3/" "$OUT_INC_DIR/eigen3/"
  elif [[ -d "/usr/local/opt/eigen@3/include/eigen3" ]]; then
    rsync -a "/usr/local/opt/eigen@3/include/eigen3/" "$OUT_INC_DIR/eigen3/"
  fi
fi

### Optional: copy into an Android module's jniLibs -------------------------
TARGET_MODULE=${TARGET_MODULE:-}
if [[ -z "$TARGET_MODULE" ]]; then
  if [[ -d "$ROOT_DIR/sfmfull" ]]; then TARGET_MODULE=sfmfull; 
  elif [[ -d "$ROOT_DIR/app" ]]; then TARGET_MODULE=app; 
  else TARGET_MODULE=""; fi
fi
if [[ -n "$TARGET_MODULE" && -d "$ROOT_DIR/$TARGET_MODULE" ]]; then
  MOD_JNI="$ROOT_DIR/$TARGET_MODULE/src/main/jniLibs/arm64-v8a"
  mkdir -p "$MOD_JNI"
  cecho "Copying .so into $TARGET_MODULE module jniLibs ..."
  rsync -av "$OUT_LIB_DIR/" "$MOD_JNI/"
  gecho "jniLibs staged at: $MOD_JNI"
else
  wecho "No TARGET_MODULE chosen or module not found. Skipping jniLibs copy."
  wecho "You can export TARGET_MODULE=yourlib and rerun to auto-copy."
fi

### Summary -----------------------------------------------------------------
gecho "\n✅ Build complete. Key artifacts:"
printf "%s %s\n" "- OpenCV (with sfm) .so:     " "$OUT_LIB_DIR/libopencv_*.so"
printf "%s %s\n" "- gflags/glog .so:           " "$OUT_LIB_DIR/libgflags.so, libglog.so"
printf "%s %s\n" "- Ceres static .a:           " "$OUT_DIR/libstatic/libceres.a"
printf "%s %s\n" "- Headers staged under:      " "$OUT_INC_DIR/{opencv4,ceres,eigen3}"

wecho "\nNext steps:"
cat << 'EOF'
1) In your Android library module (e.g., sfmfull), import these libs in CMake:
   - Add imported SHARED libs for opencv_*, glog, gflags (from src/main/jniLibs/arm64-v8a)
   - Link STATIC libceres.a from out/android/libstatic
   - include_directories(out/android/include/{opencv4,ceres,eigen3})
2) Ensure app & library modules restrict ABI to arm64-v8a.
3) Load your JNI library first (System.loadLibrary("sfmfull"));
   Android will resolve dependent .so from the same ABI directory.
EOF
