package com.example.rebuild_edge

object SfmNative {
    init {
        // Load dependencies first
        try {
            System.loadLibrary("gflags")
        } catch (_: Throwable) {}
        try {
            System.loadLibrary("glog")
        } catch (_: Throwable) {}
        // oneTBB (optional) if OpenCV is built with TBB backend
        try {
            System.loadLibrary("tbb")
        } catch (_: Throwable) {}
        System.loadLibrary("opencv_core")
        System.loadLibrary("opencv_imgproc")
        System.loadLibrary("opencv_imgcodecs")
        System.loadLibrary("opencv_flann")
        System.loadLibrary("opencv_features2d")
        System.loadLibrary("opencv_calib3d")
        System.loadLibrary("opencv_video")
        System.loadLibrary("opencv_xfeatures2d")
        System.loadLibrary("opencv_sfm")
        System.loadLibrary("sfm_native")
    }

    external fun runSfm(
        imagePaths: Array<String>,
        outDir: String,
        alignUsingGps: Boolean,
        maxLongEdge: Int,
        mode: Int,
        stride: Int,
        window: Int,
        kNeighbors: Int
    ): String

    external fun runColmapSfm(
        datasetPath: String,
        runDir: String,
        imageListPath: String,
        logPath: String,
        cameraModel: String,
        singleCamera: Boolean,
        sequentialOverlap: Int,
        maxGpsNeighborDist: Int,
        maxImageSize: Int,
        threads: Int,
        alignGps: Boolean,
        alignmentType: String,
        alignmentMaxError: Double
    ): String

    external fun runColmapPatchMatch(
        datasetPath: String,
        runDir: String,
        workspaceDir: String,
        logPath: String,
        maxImageSize: Int,
        geomConsistency: Boolean,
        depthMin: Double,
        depthMax: Double,
        numIterations: Int,
        windowRadius: Int,
        numSamples: Int,
        cacheSize: Int,
        threads: Int,
        fusionMaxReprojError: Double,
        fusionMaxDepthError: Double,
        fusionMaxNormalError: Double,
        fusionMinNumConsistent: Int
    ): String
}
