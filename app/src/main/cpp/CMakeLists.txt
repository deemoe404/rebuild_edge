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

target_include_directories(sfm_native PRIVATE
    ${OPENCV_INCLUDE_DIR}
    ${OPENCV_CONTRIB_SFM_INCLUDE_DIR}
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

find_library(log-lib log)
find_library(android-lib android)

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
)
