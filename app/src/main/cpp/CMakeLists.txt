cmake_minimum_required(VERSION 3.22.1)

project(sfm_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sfm_native SHARED
    native_sfm.cpp
)

# Use staged OpenCV headers produced by build_android_sfm_full.sh
# This contains the installed layout (opencv4/opencv2/...) expected by includes like <opencv2/core.hpp>
# Compute repo root from this CMakeLists.txt location to avoid brittle relative segments.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." REALPATH)
set(OPENCV_INCLUDE_DIR "${REPO_ROOT}/out/android/include/opencv4")
set(OPENCV_CONTRIB_SFM_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../third_party/opencv_contrib/modules/sfm/include")
set(CERES_INCLUDE_DIR "${REPO_ROOT}/out/android/include/ceres")
set(CERES_MINIGLOG_INCLUDE_DIR "${REPO_ROOT}/out/android/include/ceres/ceres/internal/miniglog")
set(EIGEN_INCLUDE_DIR "${REPO_ROOT}/out/android/include/eigen3")
set(COLMAP_INCLUDE_DIR "${REPO_ROOT}/out/android/include/colmap")
set(COLMAP_SOURCE_INCLUDE_DIR "${REPO_ROOT}/third_party/colmap/src")
set(BOOST_INCLUDE_DIR "${REPO_ROOT}/third_party/boost-src")
set(BOOST_STAGE_INCLUDE_DIR "${REPO_ROOT}/android-install/boost/include")
set(BOOST_PREPROCESSOR_INCLUDE_DIR "${REPO_ROOT}/third_party/boost-src/libs/preprocessor/include")
set(BOOST_PROGRAM_OPTIONS_INCLUDE_DIR "${REPO_ROOT}/third_party/boost-src/libs/program_options/include")
set(BOOST_HEAP_INCLUDE_DIR "${REPO_ROOT}/third_party/boost-src/libs/heap/include")
set(GLOG_INCLUDE_DIR "${REPO_ROOT}/android-install/glog/include")
set(GFLAGS_INCLUDE_DIR "${REPO_ROOT}/android-install/gflags/include")
set(SQLITE_INCLUDE_DIR "${REPO_ROOT}/android-install/sqlite/include")
set(COLMAP_SOURCE_DIR "${REPO_ROOT}/third_party/colmap/src")
set(FLANN_INCLUDE_DIR "${REPO_ROOT}/third_party/flann/src/cpp")
set(LZ4_INCLUDE_DIR "${REPO_ROOT}/third_party/lz4/lib")
set(BOOST_PROGRAM_OPTIONS_LIB "${REPO_ROOT}/android-install/boost/lib/libboost_program_options.a")
set(LZ4_STATIC_LIB "${REPO_ROOT}/android-install/lz4/lib/liblz4.a")

target_include_directories(sfm_native PRIVATE
    ${OPENCV_INCLUDE_DIR}
    ${OPENCV_CONTRIB_SFM_INCLUDE_DIR}
    ${CERES_INCLUDE_DIR}
    ${CERES_MINIGLOG_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${COLMAP_INCLUDE_DIR}
    ${COLMAP_SOURCE_INCLUDE_DIR}
    ${BOOST_INCLUDE_DIR}
    ${BOOST_STAGE_INCLUDE_DIR}
    ${BOOST_PREPROCESSOR_INCLUDE_DIR}
    ${BOOST_PROGRAM_OPTIONS_INCLUDE_DIR}
    ${BOOST_HEAP_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${GFLAGS_INCLUDE_DIR}
    ${SQLITE_INCLUDE_DIR}
    ${FLANN_INCLUDE_DIR}
    ${LZ4_INCLUDE_DIR}
)

target_sources(sfm_native PRIVATE
    ${COLMAP_SOURCE_DIR}/colmap/exe/feature.cc
    ${COLMAP_SOURCE_DIR}/colmap/exe/sfm.cc
    ${COLMAP_SOURCE_DIR}/colmap/exe/model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/freeimage_stub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_cut_stub.cpp
)

# Ensure SFM APIs (e.g., reconstruct) are visible in headers
target_compile_definitions(sfm_native PRIVATE CERES_FOUND)

# Prebuilt shared libraries in jniLibs
set(JNILIBS_DIR "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

add_library(gflags SHARED IMPORTED)
set_target_properties(gflags PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libgflags.so")

add_library(glog SHARED IMPORTED)
set_target_properties(glog PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libglog.so")

add_library(opencv_core SHARED IMPORTED)
set_target_properties(opencv_core PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_core.so")

add_library(opencv_imgproc SHARED IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_imgproc.so")

add_library(opencv_imgcodecs SHARED IMPORTED)
set_target_properties(opencv_imgcodecs PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_imgcodecs.so")

add_library(opencv_flann SHARED IMPORTED)
set_target_properties(opencv_flann PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_flann.so")

add_library(opencv_features2d SHARED IMPORTED)
set_target_properties(opencv_features2d PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_features2d.so")

add_library(opencv_calib3d SHARED IMPORTED)
set_target_properties(opencv_calib3d PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_calib3d.so")

add_library(opencv_video SHARED IMPORTED)
set_target_properties(opencv_video PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_video.so")

add_library(opencv_xfeatures2d SHARED IMPORTED)
set_target_properties(opencv_xfeatures2d PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_xfeatures2d.so")

add_library(opencv_sfm SHARED IMPORTED)
set_target_properties(opencv_sfm PROPERTIES IMPORTED_LOCATION "${JNILIBS_DIR}/libopencv_sfm.so")

# Link Ceres static library produced by build_android_sfm_full.sh
set(CERES_STATIC_LIB "${REPO_ROOT}/out/android/libstatic/libceres.a")
if (EXISTS "${CERES_STATIC_LIB}")
  add_library(ceres STATIC IMPORTED)
  set_target_properties(ceres PROPERTIES IMPORTED_LOCATION "${CERES_STATIC_LIB}")
else()
  message(WARNING "Ceres static lib not found at ${CERES_STATIC_LIB}. Build with build_android_sfm_full.sh first.")
endif()

# COLMAP and other static libraries produced by build_android_sfm_full.sh
set(STATIC_LIB_DIR "${REPO_ROOT}/out/android/libstatic")
set(COLMAP_STATIC_LIB_DIR_OUT "${STATIC_LIB_DIR}/colmap")
set(COLMAP_STATIC_LIB_DIR_INSTALL "${REPO_ROOT}/android-install/colmap/lib")

file(GLOB COLMAP_STATIC_LIBS
    "${COLMAP_STATIC_LIB_DIR_OUT}/libcolmap_*.a"
    "${COLMAP_STATIC_LIB_DIR_OUT}/libPoseLib.a"
    "${COLMAP_STATIC_LIB_DIR_INSTALL}/libcolmap_*.a"
    "${COLMAP_STATIC_LIB_DIR_INSTALL}/libPoseLib.a"
)
list(FILTER COLMAP_STATIC_LIBS EXCLUDE REGEX "libcolmap_exe\\.a$")
list(REMOVE_DUPLICATES COLMAP_STATIC_LIBS)
if (NOT COLMAP_STATIC_LIBS)
  message(WARNING "No COLMAP static libs found in ${COLMAP_STATIC_LIB_DIR_OUT} or ${COLMAP_STATIC_LIB_DIR_INSTALL}. Build with build_android_sfm_full.sh first.")
endif()

set(SQLITE3_STATIC_LIB "${STATIC_LIB_DIR}/libsqlite3.a")
if (NOT EXISTS "${SQLITE3_STATIC_LIB}")
  message(WARNING "SQLite3 static lib not found at ${SQLITE3_STATIC_LIB}")
endif()

set(FREEIMAGE_STATIC_LIB "${STATIC_LIB_DIR}/libFreeImage.a")
if (NOT EXISTS "${FREEIMAGE_STATIC_LIB}")
  message(WARNING "FreeImage static lib not found at ${FREEIMAGE_STATIC_LIB}")
endif()

find_library(log-lib log)
find_library(android-lib android)
find_library(z-lib z)

set(ADDITIONAL_LINK_LIBS)

# Link against the NDK's OpenMP runtime so prebuilt COLMAP objects that
# were compiled with -fopenmp can resolve their symbols.
set(LIBOMP_ARCH "")
if (ANDROID_ABI STREQUAL "armeabi-v7a")
  set(LIBOMP_ARCH "arm")
elseif(ANDROID_ABI STREQUAL "arm64-v8a")
  set(LIBOMP_ARCH "aarch64")
elseif(ANDROID_ABI STREQUAL "x86")
  set(LIBOMP_ARCH "i386")
elseif(ANDROID_ABI STREQUAL "x86_64")
  set(LIBOMP_ARCH "x86_64")
elseif(ANDROID_ABI STREQUAL "riscv64")
  set(LIBOMP_ARCH "riscv64")
endif()

if (LIBOMP_ARCH)
  string(REGEX MATCH "^[0-9]+" CLANG_VERSION_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
  if (NOT CLANG_VERSION_MAJOR)
    set(CLANG_VERSION_MAJOR 18)
  endif()

  get_filename_component(LLVM_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
  get_filename_component(LLVM_PREBUILT_DIR "${LLVM_BIN_DIR}" DIRECTORY)
  set(LIBOMP_PATH "${LLVM_PREBUILT_DIR}/lib/clang/${CLANG_VERSION_MAJOR}/lib/linux/${LIBOMP_ARCH}/libomp.a")
  if (EXISTS "${LIBOMP_PATH}")
    list(APPEND ADDITIONAL_LINK_LIBS "${LIBOMP_PATH}")
  else()
    message(WARNING "libomp not found at ${LIBOMP_PATH}")
  endif()
endif()

if (z-lib)
  list(APPEND ADDITIONAL_LINK_LIBS ${z-lib})
endif()

target_link_libraries(
    sfm_native
    android
    ${log-lib}
    gflags
    glog
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_flann
    opencv_features2d
    opencv_calib3d
    opencv_video
    opencv_xfeatures2d
    opencv_sfm
    ceres
    ${COLMAP_STATIC_LIBS}
    ${SQLITE3_STATIC_LIB}
    ${FREEIMAGE_STATIC_LIB}
    ${BOOST_PROGRAM_OPTIONS_LIB}
    ${LZ4_STATIC_LIB}
    ${ADDITIONAL_LINK_LIBS}
)
