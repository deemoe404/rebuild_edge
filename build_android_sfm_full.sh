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

# Resolve NDK host tag for prebuilt toolchains (darwin-x86_64, darwin-arm64, etc.)
NDK_PREBUILT_DIR="$NDK/toolchains/llvm/prebuilt"
if [[ -d "$NDK_PREBUILT_DIR/darwin-arm64" ]]; then
  export NDK_HOST_TAG="darwin-arm64"
elif [[ -d "$NDK_PREBUILT_DIR/darwin-x86_64" ]]; then
  export NDK_HOST_TAG="darwin-x86_64"
else
  # Fallback: first matching directory for current OS
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  match=$(ls -1d "$NDK_PREBUILT_DIR/$os-"* 2>/dev/null | head -n1 || true)
  if [[ -n "$match" ]]; then
    export NDK_HOST_TAG="$(basename "$match")"
  else
    recho "Could not determine NDK host tag under $NDK_PREBUILT_DIR"
    recho "Ensure a prebuilt toolchain directory like darwin-x86_64 exists."
    exit 1
  fi
fi

# Tools check
need_tool() { command -v "$1" >/dev/null 2>&1 || { recho "Missing tool: $1"; MISSING=1; }; }
MISSING=0
need_tool git; need_tool cmake; need_tool ninja; need_tool curl; need_tool rsync; need_tool unzip
if [[ ${MISSING} -eq 1 ]]; then
  recho "Please install missing tools (brew install cmake ninja git curl rsync unzip)."
  exit 1
fi

# Parallel jobs for non-CMake builds (e.g., Boost)
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)

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

# Unify Release configuration and PIC for all CMake-based deps
ANDROID_CMAKE_FLAGS+=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
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
[[ -d "$GLOG_SRC" ]] || git clone --depth=1 -b v0.6.0 https://github.com/google/glog.git "$GLOG_SRC"
cmake -G Ninja -S "$GLOG_SRC" -B "$GLOG_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=ON -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$GLOG_INST" \
  -Dgflags_DIR="$GFLAGS_CMAKE_DIR"
cmake --build "$GLOG_BUILD" --target install -j
GLOG_CMAKE_DIR="$GLOG_INST/lib/cmake/glog"

### Ceres (static) -----------------------------------------------------------
# cecho "Building Ceres (Android, static) ..."
# CERES_SRC="$TP_DIR/ceres-solver"
# CERES_BUILD="$BUILD_DIR/ceres"
# CERES_INST="$INSTALL_DIR/ceres"
# [[ -d "$CERES_SRC" ]] || git clone --depth=1 -b 2.2.0 https://github.com/ceres-solver/ceres-solver.git "$CERES_SRC"
# cmake -G Ninja -S "$CERES_SRC" -B "$CERES_BUILD" \
#   "${ANDROID_CMAKE_FLAGS[@]}" \
#   -DBUILD_SHARED_LIBS=OFF \
#   -DMINIGLOG=ON -DSUITESPARSE=OFF -DLAPACK=OFF \
#   -DEigen3_DIR="$EIGEN3_CMAKE_DIR" \
#   -DCMAKE_INSTALL_PREFIX="$CERES_INST"
# cmake --build "$CERES_BUILD" --target install -j
# CERES_CMAKE_DIR="$CERES_INST/lib/cmake/Ceres"

cecho "Building Ceres (Android, static) ..."
CERES_SRC="$TP_DIR/ceres-solver"
CERES_BUILD="$BUILD_DIR/ceres"
CERES_INST="$INSTALL_DIR/ceres"
[[ -d "$CERES_SRC" ]] || git clone --depth=1 -b 2.2.0 https://github.com/ceres-solver/ceres-solver.git "$CERES_SRC"

cmake -G Ninja -S "$CERES_SRC" -B "$CERES_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DMINIGLOG=OFF -DGFLAGS=ON \
  -DSUITESPARSE=OFF -DLAPACK=OFF \
  -DEigen3_DIR="$EIGEN3_CMAKE_DIR" \
  -Dglog_DIR="$GLOG_CMAKE_DIR" \
  -Dgflags_DIR="$GFLAGS_CMAKE_DIR" \
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

### Boost (static, minimal) ---------------------------------------------------
cecho "Building Boost (Android, static, via CMake) ..."
BOOST_VER_D=1.89.0
BOOST_SRC="$TP_DIR/boost-src"
BOOST_BLD="$BUILD_DIR/boost"
BOOST_INST="$INSTALL_DIR/boost"

if [[ ! -d "$BOOST_SRC/.git" ]]; then
  git clone --recursive --depth=1 -b "boost-${BOOST_VER_D}" https://github.com/boostorg/boost "$BOOST_SRC"
else
  git -C "$BOOST_SRC" fetch --tags origin "boost-${BOOST_VER_D}"
  git -C "$BOOST_SRC" checkout "boost-${BOOST_VER_D}"
  git -C "$BOOST_SRC" submodule update --init --recursive
fi

cmake -G Ninja -S "$BOOST_SRC" -B "$BOOST_BLD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DCMAKE_CXX_STANDARD=17 -DBUILD_SHARED_LIBS=OFF \
  -DBOOST_ENABLE_CMAKE=ON \
  -DBoost_USE_STATIC_LIBS=ON \
  -DBOOST_INCLUDE_LIBRARIES="headers;heap;dynamic_bitset" \
  -DCMAKE_INSTALL_PREFIX="$BOOST_INST"

cmake --build "$BOOST_BLD" --target install -j

# --- 安装后自检：Boost 总配置 + headers 子包配置必备 ---
boost_headers_dirA="$BOOST_INST/lib/cmake/boost_headers/boost_headers-config.cmake"
boost_headers_dirB="$BOOST_INST/lib/cmake/boost_headers-${BOOST_VER_D}/boost_headers-config.cmake"

[[ -f "$BOOST_INST/lib/cmake/Boost-${BOOST_VER_D}/BoostConfig.cmake" ]] || {
  recho "BoostConfig.cmake missing"; exit 1; }

if [[ ! -f "$boost_headers_dirA" && ! -f "$boost_headers_dirB" ]]; then
  wecho "boost_headers-config.cmake not found after install; building target boost_headers explicitly..."
  cmake --build "$BOOST_BLD" --target boost_headers -j || {
    recho "Failed to build boost_headers target"; exit 1; }
  cmake --build "$BOOST_BLD" --target install -j
fi

if [[ ! -f "$boost_headers_dirA" && ! -f "$boost_headers_dirB" ]]; then
  recho "boost_headers-config.cmake still missing (check that headers component was installed)"
  recho "Looked at:"
  recho "  $boost_headers_dirA"
  recho "  $boost_headers_dirB"
  exit 1
fi

### SQLite3 (static) ----------------------------------------------------------
cecho "Building SQLite3 (Android, static) ..."
SQLITE_SRC="$TP_DIR/sqlite"
SQLITE_BUILD="$BUILD_DIR/sqlite"
SQLITE_INST="$INSTALL_DIR/sqlite"
if [[ ! -f "$SQLITE_SRC/sqlite3.c" || ! -f "$SQLITE_SRC/sqlite3.h" ]]; then
  mkdir -p "$SQLITE_SRC"
  # Note: update version if needed; this is 3.45.2 (3450200)
  SQLITE_VER_NUM=3450200
  SQLITE_ZIP="$TP_DIR/sqlite-amalgamation.zip"
  rm -f "$SQLITE_ZIP" 2>/dev/null || true
  # The sqlite.org path uses the release year; sometimes versions live under the previous year.
  YEARS=(2025 2024 2023)
  DL_OK=0
  for Y in "${YEARS[@]}"; do
    URL="https://www.sqlite.org/${Y}/sqlite-amalgamation-${SQLITE_VER_NUM}.zip"
    wecho "Fetching SQLite amalgamation from $URL"
    if curl -fL --retry 3 --retry-delay 2 -o "$SQLITE_ZIP" "$URL"; then
      # Validate it's actually a zip archive
      if unzip -tq "$SQLITE_ZIP" >/dev/null 2>&1; then
        DL_OK=1
        break
      else
        wecho "Downloaded file is not a valid ZIP; trying next year path..."
      fi
    else
      wecho "Download failed from $URL; trying next year path..."
    fi
  done
  if [[ "$DL_OK" -ne 1 ]]; then
    recho "Failed to download a valid SQLite amalgamation archive."
    recho "Tried years: ${YEARS[*]} for version ${SQLITE_VER_NUM}"
    recho "Last file saved at: $SQLITE_ZIP"
    exit 1
  fi
  unzip -q "$SQLITE_ZIP" -d "$SQLITE_SRC"
  mv "$SQLITE_SRC"/sqlite-amalgamation-*/{sqlite3.c,sqlite3.h} "$SQLITE_SRC"/
fi
mkdir -p "$SQLITE_BUILD"
cat > "$SQLITE_BUILD/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(sqlite3 C)
if(NOT DEFINED SQLITE_SRC_DIR)
  message(FATAL_ERROR "SQLITE_SRC_DIR not defined; pass -DSQLITE_SRC_DIR=/path/to/sqlite")
endif()
add_library(sqlite3 STATIC ${SQLITE_SRC_DIR}/sqlite3.c)
target_compile_definitions(sqlite3 PRIVATE
  SQLITE_OMIT_LOAD_EXTENSION
  SQLITE_THREADSAFE=1
  SQLITE_TEMP_STORE=2
  SQLITE_DEFAULT_MEMSTATUS=0
  SQLITE_OMIT_DEPRECATED
)
target_include_directories(sqlite3 PUBLIC ${SQLITE_SRC_DIR})
install(TARGETS sqlite3 ARCHIVE DESTINATION lib)
install(FILES ${SQLITE_SRC_DIR}/sqlite3.h DESTINATION include)
EOF
cmake -G Ninja -S "$SQLITE_BUILD" -B "$SQLITE_BUILD/_b" \
  "${ANDROID_CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$SQLITE_INST" \
  -DSQLITE_SRC_DIR="$SQLITE_SRC"
cmake --build "$SQLITE_BUILD/_b" --target install -j

### FreeImage (static) --------------------------------------------------------
cecho "Building FreeImage (Android, static) ..."
FREEIMG_SRC="$TP_DIR/freeimage"
FREEIMG_BUILD="$BUILD_DIR/freeimage"
FREEIMG_INST="$INSTALL_DIR/freeimage"
: "${FREEIMG_REPO:=https://github.com/danoli3/FreeImage}"
: "${FREEIMG_BRANCH:=3.18.0_cpp17}"
DESIRED_REMOTE="${FREEIMG_REPO%.git}"
# Ensure existing checkout (if any) matches the requested fork; otherwise re-clone.
if [[ -d "$FREEIMG_SRC/.git" ]]; then
  CURRENT_REMOTE="$(git -C "$FREEIMG_SRC" config --get remote.origin.url 2>/dev/null || true)"
  CURRENT_REMOTE="${CURRENT_REMOTE%.git}"
  if [[ -z "$CURRENT_REMOTE" || "$CURRENT_REMOTE" != "$DESIRED_REMOTE" ]]; then
    wecho "Existing FreeImage repo differs from ${FREEIMG_REPO}; re-cloning."
    rm -rf "$FREEIMG_SRC"
  fi
fi
if [[ ! -d "$FREEIMG_SRC/.git" ]]; then
  wecho "Cloning FreeImage from ${FREEIMG_REPO} (${FREEIMG_BRANCH})"
  rm -rf "$FREEIMG_SRC" 2>/dev/null || true
  if ! git clone --depth=1 -b "$FREEIMG_BRANCH" "$FREEIMG_REPO" "$FREEIMG_SRC"; then
    recho "Git clone of FreeImage fork failed."
    recho "Tried ${FREEIMG_REPO} branch ${FREEIMG_BRANCH}"
    exit 1
  fi
else
  wecho "Updating FreeImage checkout at $FREEIMG_SRC to ${FREEIMG_BRANCH}"
  if ! git -C "$FREEIMG_SRC" fetch origin "$FREEIMG_BRANCH"; then
    recho "Failed to fetch ${FREEIMG_BRANCH} from ${FREEIMG_REPO}"
    exit 1
  fi
  if ! git -C "$FREEIMG_SRC" checkout "$FREEIMG_BRANCH"; then
    recho "Failed to checkout ${FREEIMG_BRANCH} in $FREEIMG_SRC"
    exit 1
  fi
  if ! git -C "$FREEIMG_SRC" pull --ff-only origin "$FREEIMG_BRANCH"; then
    recho "Failed to fast-forward FreeImage to origin/${FREEIMG_BRANCH}"
    exit 1
  fi
fi
if [[ ! -d "$FREEIMG_SRC/Source" ]]; then
  recho "FreeImage sources missing under $FREEIMG_SRC even after syncing."
  exit 1
fi

# On Android we don't need Windows Media Photo / JPEG XR (LibJXR). It also
# introduces headers like windowsmediaphoto.h which are irrelevant on Android.
# We patch the internal plugin registry to skip JXR on Android, and we exclude
# LibJXR and PluginJXR sources from our CMake wrapper below.
FREEIMG_PLUGIN_CPP="$FREEIMG_SRC/Source/FreeImage/Plugin.cpp"
if [[ -f "$FREEIMG_PLUGIN_CPP" ]]; then
  wecho "Patching FreeImage Plugin.cpp to disable JXR/JP2 on Android ..."
  awk '
    BEGIN {patched_jxr=0; patched_jp2=0}
    {
      line = $0
      if (line ~ /s_plugins->AddNode\((InitJXR)\);/ && !patched_jxr) {
        print "#if !defined(__ANDROID__) // patched by build_android_sfm_full.sh"
        print line
        print "#endif"
        patched_jxr=1
        next
      }
      if (line ~ /s_plugins->AddNode\((InitJP2|InitJ2K|InitJPC)\);/ && !patched_jp2) {
        print "#if !defined(__ANDROID__) // patched by build_android_sfm_full.sh"
        print line
        print "#endif"
        patched_jp2=1
        next
      }
      print line
    }
  ' "$FREEIMG_PLUGIN_CPP" >"$FREEIMG_PLUGIN_CPP.patched" && mv "$FREEIMG_PLUGIN_CPP.patched" "$FREEIMG_PLUGIN_CPP"
fi

# --- Patch LibTIFF config macros so Android 走 POSIX ---
TIFF_CFG="$FREEIMG_SRC/Source/LibTIFF4/tif_config.h"
if [[ -f "$TIFF_CFG" ]]; then
  awk '
    BEGIN { }
    {
      # 删除/替换任何对 HAVE_IO_H 的定义 → 彻底变为未定义
      if ($0 ~ /^#define[ \t]+HAVE_IO_H[ \t]+[01]/) { print "/* #undef HAVE_IO_H */"; next }

      # 确保 HAVE_UNISTD_H = 1
      if ($0 ~ /^#define[ \t]+HAVE_UNISTD_H[ \t]+0/) { print "#define HAVE_UNISTD_H 1"; next }

      print
    }
  ' "$TIFF_CFG" > "$TIFF_CFG.patched" && mv "$TIFF_CFG.patched" "$TIFF_CFG"
fi

mkdir -p "$FREEIMG_BUILD"
cat > "$FREEIMG_BUILD/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(freeimage C CXX)

# -------- options to slim down on Android --------
option(FREEIMG_WITH_WEBP "Enable WebP codec" OFF)           # Android: 默认关
option(FREEIMG_USE_INTERNAL_ZLIB "Use bundled zlib" OFF)    # 用 NDK 自带 zlib

if(NOT DEFINED FREEIMG_SRC_DIR)
  message(FATAL_ERROR "FREEIMG_SRC_DIR not defined; pass -DFREEIMG_SRC_DIR=/path/to/freeimage")
endif()

file(GLOB_RECURSE SRC_CPP
    "${FREEIMG_SRC_DIR}/Source/*.cpp"
    "${FREEIMG_SRC_DIR}/Source/*/*.cpp")
file(GLOB_RECURSE SRC_C
    "${FREEIMG_SRC_DIR}/Source/*.c"
    "${FREEIMG_SRC_DIR}/Source/*/*.c")

# ---- prune codecs we明确不需要/会在Android上添麻烦的 ----
# 0) Drop LibTIFF generator programs (host tools; K&R C not valid in C17)
list(FILTER SRC_C EXCLUDE REGEX ".*/LibTIFF4/mk.*\\.[cC]$")

# 1) 全量去掉 JXR/JP2(OpenJPEG) 树与其插件
list(FILTER SRC_CPP EXCLUDE REGEX ".*/FreeImage/PluginJXR\\.cpp$")
list(FILTER SRC_CPP EXCLUDE REGEX ".*/FreeImage/PluginJP2\\.cpp$")
list(FILTER SRC_CPP EXCLUDE REGEX ".*/LibOpenJPEG/.*")
list(FILTER SRC_C   EXCLUDE REGEX ".*/LibOpenJPEG/.*")
list(FILTER SRC_CPP EXCLUDE REGEX ".*/LibJXR/.*")
list(FILTER SRC_C   EXCLUDE REGEX ".*/LibJXR/.*")

# 2) 仅保留 Unix 版的 LibTIFF 平台层
list(FILTER SRC_C EXCLUDE REGEX ".*/LibTIFF4/tif_win32\\.c$")
list(FILTER SRC_C EXCLUDE REGEX ".*/LibTIFF4/tif_wince\\.c$")
list(FILTER SRC_C EXCLUDE REGEX ".*/LibTIFF4/tif_vms\\.c$")

# 3) 可选关闭 WebP（默认OFF，最省事）
if(NOT FREEIMG_WITH_WEBP)
  list(FILTER SRC_C   EXCLUDE REGEX ".*/LibWebP/.*")
  list(FILTER SRC_CPP EXCLUDE REGEX ".*/FreeImage/PluginWebP\\.cpp$")
endif()

# 4) 用系统 zlib（默认OFF，不编译内置ZLib源码）
if(NOT FREEIMG_USE_INTERNAL_ZLIB)
  list(FILTER SRC_C EXCLUDE REGEX ".*/Source/ZLib/.*")
endif()

# 5) 可选：去掉 LibJPEG 示例/老平台实现（安全瘦身）
list(FILTER SRC_C EXCLUDE REGEX ".*/LibJPEG/example\\.c$")
list(FILTER SRC_C EXCLUDE REGEX ".*/LibJPEG/jmemdos\\.c$")
list(FILTER SRC_C EXCLUDE REGEX ".*/LibJPEG/jmemmac\\.c$")
list(FILTER SRC_C EXCLUDE REGEX ".*/LibJPEG/jmemname\\.c$")

# 6) LibPNG 的示例/测试文件（库不需要）
list(FILTER SRC_C EXCLUDE REGEX ".*/LibPNG/example\\.c$")
list(FILTER SRC_C EXCLUDE REGEX ".*/LibPNG/pngtest\\.c$")

# 7) 禁用老旧/易不兼容的编码器实现
#    - OJPEG 非常老 + 依赖路径复杂
#    - JPEG 12-bit 与内置 8-bit libjpeg 常冲突
#    - LZMA / ZSTD 如无明确需求亦剔除，省去后续未定义符号风险
list(FILTER SRC_C   EXCLUDE REGEX ".*/LibTIFF4/tif_ojpeg\\.c$")
list(FILTER SRC_C   EXCLUDE REGEX ".*/LibTIFF4/tif_jpeg_12\\.c$")
list(FILTER SRC_C   EXCLUDE REGEX ".*/LibTIFF4/tif_lzma\\.c$")
list(FILTER SRC_C   EXCLUDE REGEX ".*/LibTIFF4/tif_zstd\\.c$")

add_library(FreeImage STATIC ${SRC_CPP} ${SRC_C})

target_compile_options(FreeImage PRIVATE -UHAVE_IO_H)

# ---- headers for all bundled libs we keep ----
# root
target_include_directories(FreeImage PUBLIC
  "${FREEIMG_SRC_DIR}/Source"
  "${FREEIMG_SRC_DIR}/Source/LibJPEG"
  "${FREEIMG_SRC_DIR}/Source/LibPNG"
  "${FREEIMG_SRC_DIR}/Source/LibTIFF4"
)

# WebP头路径（仅当启用时）
if(FREEIMG_WITH_WEBP)
  target_include_directories(FreeImage PUBLIC
    "${FREEIMG_SRC_DIR}/Source/LibWebP"
    "${FREEIMG_SRC_DIR}/Source/LibWebP/src"
  )
endif()

# 内置 Zlib 的头（仅当启用时）
if(FREEIMG_USE_INTERNAL_ZLIB)
  target_include_directories(FreeImage PUBLIC
    "${FREEIMG_SRC_DIR}/Source/ZLib"
  )
else()
  # 用系统 zlib，给下游透出 -lz
  target_link_libraries(FreeImage INTERFACE z)
endif()

# 解决 tif_unix.c 在非 Windows 误 include <io.h> 的边角
target_compile_definitions(FreeImage PRIVATE
  FREEIMAGE_LIB
  HAVE_UNISTD_H=1
)

target_compile_options(FreeImage PRIVATE
  -Wno-deprecated-non-prototype
)
set_property(TARGET FreeImage PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS FreeImage ARCHIVE DESTINATION lib)
install(DIRECTORY ${FREEIMG_SRC_DIR}/Source/ DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*/.git" EXCLUDE)

EOF
cmake -G Ninja -S "$FREEIMG_BUILD" -B "$FREEIMG_BUILD/_b" \
  "${ANDROID_CMAKE_FLAGS[@]}" -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX="$FREEIMG_INST" \
  -DFREEIMG_SRC_DIR="$FREEIMG_SRC"
cmake --build "$FREEIMG_BUILD/_b" --target install -j

# LZ4
LZ4_SRC="$TP_DIR/lz4"
LZ4_INST="$INSTALL_DIR/lz4"
[[ -d "$LZ4_SRC" ]] || git clone --depth=1 https://github.com/lz4/lz4 "$LZ4_SRC"
mkdir -p "$LZ4_INST/include"
cp -r "$LZ4_SRC/lib"/*.h "$LZ4_INST/include/"

# --- FLANN (static) -----------------------------------------------------------
cecho "Building FLANN (Android, static) ..."
FLANN_SRC="$TP_DIR/flann"
FLANN_BUILD="$BUILD_DIR/flann"
FLANN_INST="$INSTALL_DIR/flann"
[[ -d "$FLANN_SRC" ]] || git clone --depth=1 -b 1.9.2 https://github.com/flann-lib/flann "$FLANN_SRC"

cmake -G Ninja -S "$FLANN_SRC" -B "$FLANN_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
  -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_PYTHON_BINDINGS=OFF \
  -DUSE_OPENMP=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_LZ4=ON -DUSE_LZ4=OFF \
  -DCMAKE_INSTALL_PREFIX="$FLANN_INST"

# 只编译静态目标（避免触发 flann_cpp.so）
cmake --build "$FLANN_BUILD" --target flann_cpp_s flann_s -j

# 手动“安装”到前缀（头 + 静态库）
mkdir -p "$FLANN_INST/lib" "$FLANN_INST/include"
cp -v "$FLANN_BUILD/lib/libflann_cpp_s.a" "$FLANN_INST/lib/"
cp -v "$FLANN_BUILD/lib/libflann_s.a"     "$FLANN_INST/lib/"
rsync -a "$FLANN_SRC/src/cpp/flann/" "$FLANN_INST/include/flann/"

### COLMAP (library only; no GUI/CUDA) ---------------------------------------
cecho "Building COLMAP (Android, library only) ..."
COLMAP_TAG=${COLMAP_TAG:-3.11.1}
COLMAP_SRC="$TP_DIR/colmap"
COLMAP_BUILD="$BUILD_DIR/colmap"
COLMAP_INST="$INSTALL_DIR/colmap"
[[ -d "$COLMAP_SRC" ]] || git clone --depth=1 -b "$COLMAP_TAG" https://github.com/colmap/colmap "$COLMAP_SRC"

# --- Patch COLMAP FindDependencies.cmake: skip desktop OpenGL/GLEW/CGAL on Android ---
cecho "Patching COLMAP FindDependencies.cmake for Android (skip desktop OpenGL/GLEW/CGAL) ..."
FD="$COLMAP_SRC/cmake/FindDependencies.cmake"
if [[ -f "$FD" ]]; then
  cp -f "$FD" "$FD.bak"

  awk '
    BEGIN{
      patched_ogl=0; opened_guard=0; patched_cgal=0;
      prev1=""; prev2="";
    }
    {
      line=$0

      # 如果已经有我们加过的 if 守卫，标记已补丁，直接原样输出
      if (line ~ /^[ \t]*if\(OPENGL_ENABLED AND GUI_ENABLED AND NOT ANDROID\)/) {
        patched_ogl=1
      }

      # ---- 包裹 OpenGL/GLEW 的查找 ----
      # 以 set(OpenGL_GL_PREFERENCE GLVND) 作为起始锚点
      if (!patched_ogl && line ~ /^[ \t]*set\(OpenGL_GL_PREFERENCE[ \t]+GLVND\)/) {
        print "if(OPENGL_ENABLED AND GUI_ENABLED AND NOT ANDROID)"
        print line
        opened_guard=1
        next
      }
      # 守卫内转发 find_package(OpenGL ...)
      if (opened_guard && line ~ /^[ \t]*find_package\(OpenGL/) {
        print line
        next
      }
      # 守卫内遇到 find_package(Glew ...) 后收尾 endif()
      if (opened_guard && line ~ /^[ \t]*find_package\(Glew/) {
        print line
        print "endif()"
        opened_guard=0
        patched_ogl=1
        next
      }

      # ---- 仅非 Android 查找 CGAL ----
      if (line ~ /^[ \t]*find_package\(CGAL[ \t]/) {
        # 避免重复包裹：若上一两行已有 if(NOT ANDROID) 则保持原状
        if (prev1 ~ /^[ \t]*if\(NOT ANDROID\)/ || prev2 ~ /^[ \t]*if\(NOT ANDROID\)/) {
          print line
        } else {
          print "if(NOT ANDROID)"
          print "  " line
          print "endif()"
          patched_cgal=1
        }
        prev2=prev1; prev1=line
        next
      }

      # 默认输出
      print line
      prev2=prev1; prev1=line
    }
    END{
      # 若意外到文件末尾仍未闭合，补一个 endif()
      if (opened_guard) { print "endif()" }
    }
  ' "$FD.bak" > "$FD"

  # 简要校验（打印关键行到控制台）
  grep -nE \
    'if\(OPENGL_ENABLED AND GUI_ENABLED AND NOT ANDROID\)|find_package\(OpenGL|find_package\(Glew|if\(NOT ANDROID\)|find_package\(CGAL' \
    "$FD" >&3 || true
else
  wecho "Warning: $FD not found; skip patch."
fi

# --- Skip heavy/desktop-only deps on Android -------------------------------
FD="$COLMAP_SRC/cmake/FindDependencies.cmake"
[[ -f "$FD" ]] || { recho "FindDependencies.cmake not found"; exit 1; }

# 若已打过补丁就跳过
if grep -qE 'NOT ANDROID AND RETRIEVAL_ENABLED.*find_package\(FLANN' "$FD"; then
  wecho "FLANN/LZ4/Metis Android guards already present; skipping patch."
else
  cp -f "$FD" "$FD.bak"

  awk '
    # 小工具：按给定条件输出包裹后的 find_package()
    function emit_guarded(pkg, cond) {
      print "if(" cond ")"
      print "  find_package(" pkg " ${COLMAP_FIND_TYPE})"
      print "endif()"
    }
    {
      line=$0
      # 1) LZ4：Android 上通常不必；如需要可自行构建后删除本补丁
      if (line ~ /^[ \t]*find_package\(LZ4[ \t]/) {
        emit_guarded("LZ4","NOT ANDROID")
        next
      }
      # 2) Metis：同理，仅非 Android 查找
      if (line ~ /^[ \t]*find_package\(Metis[ \t]/) {
        emit_guarded("Metis","NOT ANDROID")
        next
      }
      # 其他行照抄
      print line
    }
  ' "$FD.bak" > "$FD"

  # 简短校验输出到控制台
  grep -nE \
    'find_package\(FLANN|find_package\(LZ4|find_package\(Metis|NOT ANDROID AND RETRIEVAL_ENABLED' \
    "$FD" >&3 || true
fi

# --- Drop METIS-dependent sources from colmap_math on Android ---
MATH_CMAKE="$COLMAP_SRC/src/colmap/math/CMakeLists.txt"
if [[ -f "$MATH_CMAKE" ]]; then
  cp -f "$MATH_CMAKE" "$MATH_CMAKE.bak"

  # 1) 过滤掉列表里出现的 graph_cut.cc / graph_cut.h（无论是否带相对路径）
  awk '
    {
      line = $0
      gsub(/(^|[ \t])([A-Za-z0-9_\/.-]*graph_cut\.cc)([ \t\)]|$)/, " ", line)
      gsub(/(^|[ \t])([A-Za-z0-9_\/.-]*graph_cut\.h)([ \t\)]|$)/, " ", line)
      print line
    }
  ' "$MATH_CMAKE.bak" > "$MATH_CMAKE.tmp" && mv "$MATH_CMAKE.tmp" "$MATH_CMAKE"

  # 2) 若存在 colmap_math 目标，则强制定义 COLMAP_HAS_METIS=0
  #   （在 add_library(colmap_math ...) 之后追加一行）
  if grep -qE 'add_library\(\s*colmap_math\b' "$MATH_CMAKE"; then
    awk '
      BEGIN{added=0}
      {
        print
        if (!added && $0 ~ /add_library\(\s*colmap_math\b/) {
          print "target_compile_definitions(colmap_math PRIVATE COLMAP_HAS_METIS=0)"
          added=1
        }
      }
    ' "$MATH_CMAKE" > "$MATH_CMAKE.patched" && mv "$MATH_CMAKE.patched" "$MATH_CMAKE"
  fi
else
  echo "[warn] $MATH_CMAKE not found; skip METIS source filter." >&3
fi

# --- Patch PoissonRecon/MyTime.h for Android (NDK 没有 <sys/timeb.h>) ---
POISSON_MYTIME="$COLMAP_SRC/src/thirdparty/PoissonRecon/MyTime.h"
if [[ -f "$POISSON_MYTIME" ]]; then
  cat > "$POISSON_MYTIME" <<'HPP'
#ifndef MY_TIME_INCLUDED
#define MY_TIME_INCLUDED

#if defined(_WIN32)
#  include <windows.h>
static inline double Time(){
  LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
  return double(c.QuadPart)/double(f.QuadPart);
}
#elif defined(__ANDROID__)
#  include <sys/time.h>
static inline double Time(){
  struct timeval tv; gettimeofday(&tv, nullptr);
  return double(tv.tv_sec) + 1e-6*double(tv.tv_usec);
}
#else
#  include <sys/timeb.h>
static inline double Time(){
  struct timeb tb; ftime(&tb);
  return double(tb.time) + 1e-3*double(tb.millitm);
}
#endif

#endif // MY_TIME_INCLUDED
HPP
fi

EXE_CMAKE="$COLMAP_SRC/src/colmap/exe/CMakeLists.txt"
if ! grep -q "Skip building colmap CLI on Android" "$EXE_CMAKE"; then
  printf '%s\n' 'if(ANDROID)
    message(STATUS "Skip building colmap CLI on Android")
    return()
  endif()' | cat - "$EXE_CMAKE" > "$EXE_CMAKE.new" && mv "$EXE_CMAKE.new" "$EXE_CMAKE"
fi

TOP_CMAKE="$COLMAP_SRC/CMakeLists.txt"
cp -f "$TOP_CMAKE" "$TOP_CMAKE.bak"
awk '
  BEGIN{
    printed_guard=0;
    in_export_set=0;
  }
  {
    line=$0

    # Detect start of COLMAP_EXPORT_LIBS set(...)
    if (line ~ /^set\(COLMAP_EXPORT_LIBS/) {
      in_export_set=1
    }

    # While inside that set(...), drop the colmap_exe entry (any spacing)
    if (in_export_set) {
      if (line ~ /^[ \t]*colmap_exe[ \t]*$/) {
        next
      }
      # End of the set(...) block
      if (line ~ /\)/) {
        in_export_set=0
      }
    }

    print line

    # Mark that we should append the guard once (we do it in END)
    if (!printed_guard && line ~ /^set\(COLMAP_EXPORT_LIBS/) {
      printed_guard=1
    }
  }
  END{
    print ""
    print "# --- Added by build script: export CLI only if it exists ---"
    print "if (TARGET colmap_exe)"
    print "  list(APPEND COLMAP_EXPORT_LIBS colmap_exe)"
    print "endif()"
  }
' "$TOP_CMAKE.bak" > "$TOP_CMAKE"


cmake -G Ninja -S "$COLMAP_SRC" -B "$COLMAP_BUILD" \
  "${ANDROID_CMAKE_FLAGS[@]}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF -DTESTS_ENABLED=OFF \
  -DGUI_ENABLED=OFF -DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DRETRIEVAL_ENABLED=OFF \
  -DCOLMAP_MIN_DEPS=ON \
  -DGRAPH_CUT_ENABLED=OFF \
  -DMETIS_ENABLED=OFF \
  -DCMAKE_FIND_ROOT_PATH="$NDK;$BOOST_INST;$GFLAGS_INST;$GLOG_INST;$CERES_INST;$FREEIMG_INST;$SQLITE_INST;$FLANN_INST" \
  -DCMAKE_PREFIX_PATH="$BOOST_INST;$GFLAGS_INST;$GLOG_INST;$CERES_INST;$FREEIMG_INST;$SQLITE_INST;$FLANN_INST" \
  -DEigen3_DIR="$EIGEN3_CMAKE_DIR" \
  -DCeres_DIR="$CERES_CMAKE_DIR" \
  -Dgflags_DIR="$GFLAGS_CMAKE_DIR" \
  -Dglog_DIR="$GLOG_CMAKE_DIR" \
  -DBoost_DIR="$BOOST_INST/lib/cmake/Boost-${BOOST_VER_D}" \
  -DBoost_USE_STATIC_LIBS=ON \
  -DBOOST_ROOT="$BOOST_INST" \
  -DBOOST_INCLUDEDIR="$BOOST_INST/include" \
  -DBOOST_LIBRARYDIR="$BOOST_INST/lib" \
  -DBoost_INCLUDE_DIR="$BOOST_INST/include" \
  -DBoost_INCLUDE_DIRS="$BOOST_INST/include" \
  -DBoost_LIBRARY_DIRS="$BOOST_INST/lib" \
  -DFLANN_INCLUDE_DIR="$FLANN_INST/include" \
  -DFLANN_LIBRARY="$FLANN_INST/lib/libflann_cpp_s.a" \
  -DFLANN_INCLUDE_DIR="$FLANN_INST/include" \
  -DFLANN_LIBRARY="$FLANN_INST/lib/libflann_cpp_s.a" \
  -DFLANN_LIBRARIES="$FLANN_INST/lib/libflann_cpp_s.a;$FLANN_INST/lib/libflann_s.a" \
  -DFreeImage_INCLUDE_DIRS="$FREEIMG_INST/include" \
  -DFreeImage_LIBRARIES="$FREEIMG_INST/lib/libFreeImage.a" \
  -DSQLite3_INCLUDE_DIRS="$SQLITE_INST/include" \
  -DSQLite3_LIBRARIES="$SQLITE_INST/lib/libsqlite3.a" \
  -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS:-} -DFREEIMAGE_LIB -isystem \"$BOOST_INST/include\" -I\"$LZ4_INST/include\"" \
  -DCMAKE_INSTALL_PREFIX="$COLMAP_INST"
cmake --build "$COLMAP_BUILD" --target install -j

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
# Ensure libc++_shared.so is also present to satisfy C++ runtime at load time
CXXSHARED=$(find "$NDK" -name libc++_shared.so \( -path "*/libs/arm64-v8a/*" -o -path "*/sysroot/usr/lib/*/libc++_shared.so" -o -path "*/sysroot/usr/lib/aarch64-linux-android*/libc++_shared.so" \) -print -quit 2>/dev/null || true)
if [[ -n "$CXXSHARED" ]]; then
  cp -v "$CXXSHARED" "$OUT_LIB_DIR" || true
fi
# No need to copy TBB .so when statically linked into OpenCV.
# Ceres static
mkdir -p "$OUT_DIR/libstatic"
cp -v "$CERES_INST/lib/libceres.a" "$OUT_DIR/libstatic/" || true
cp -v "$SQLITE_INST/lib/libsqlite3.a" "$OUT_DIR/libstatic/" || true
cp -v "$FREEIMG_INST/lib/libFreeImage.a" "$OUT_DIR/libstatic/" || true

OMP_SO=$(find "$NDK" -name libomp.so -path "*/lib/*/libomp.so" -print -quit 2>/dev/null || true)
if [[ -n "$OMP_SO" ]]; then
  cp -v "$OMP_SO" "$OUT_LIB_DIR" || true
fi

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

### Stage COLMAP outputs -----------------------------------------------------
cecho "Staging COLMAP libs/headers ..."
mkdir -p "$OUT_DIR/libstatic/colmap" "$OUT_INC_DIR/colmap" "$OUT_DIR/colmap_cmake"
# Static libraries live under lib/colmap
if [[ -d "$COLMAP_INST/lib/colmap" ]]; then
  rsync -a "$COLMAP_INST/lib/colmap/" "$OUT_DIR/libstatic/colmap/"
fi
# Public headers
if [[ -d "$COLMAP_INST/include/colmap" ]]; then
  rsync -a "$COLMAP_INST/include/colmap/" "$OUT_INC_DIR/colmap/"
fi
# CMake package config for consumer find_package(colmap)
if [[ -d "$COLMAP_INST/share/colmap" ]]; then
  rsync -a "$COLMAP_INST/share/colmap/" "$OUT_DIR/colmap_cmake/"
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
printf "%s %s\n" "- COLMAP static .a:          " "$OUT_DIR/libstatic/colmap/libcolmap_*.a"
printf "%s %s\n" "- Headers staged under:      " "$OUT_INC_DIR/{opencv4,ceres,eigen3,colmap}"
printf "%s %s\n" "- CMake config (colmap):     " "$OUT_DIR/colmap_cmake/"

wecho "\nNext steps:"
cat << 'EOF'
1) In your Android library module (e.g., sfmfull), import these libs in CMake:
   - Add imported SHARED libs for opencv_*, glog, gflags (from src/main/jniLibs/arm64-v8a)
   - Link STATIC libceres.a from out/android/libstatic
   - Link STATIC libcolmap_*.a from out/android/libstatic/colmap
   - Link STATIC libsqlite3.a and libFreeImage.a from out/android/libstatic (or installed dirs)
   - include_directories(out/android/include/{opencv4,ceres,eigen3,colmap})
   - Optionally: add CMAKE_PREFIX_PATH to out/android/colmap_cmake and use find_package(colmap 3.12 REQUIRED)
2) Ensure app & library modules restrict ABI to arm64-v8a.
3) Load your JNI library first (System.loadLibrary("sfmfull"));
   Android will resolve dependent .so from the same ABI directory.
EOF
