package com.example.rebuild_edge.ui.psd

object PsdNative {
    init {
        System.loadLibrary("psd_native")
    }

    external fun depthToPoint(
        depth: FloatArray,
        intrinsics: FloatArray,
        batch: Int,
        height: Int,
        width: Int
    ): FloatArray

    external fun cspn(
        sparse: FloatArray,
        dense: FloatArray,
        feature: FloatArray,
        batch: Int,
        channels: Int,
        height: Int,
        width: Int,
        iteration: Int
    ): FloatArray

    external fun fillSparseFast(
        sparse: FloatArray,
        batch: Int,
        height: Int,
        width: Int,
        iterations: Int
    ): FloatArray

    /**
     * KNN propagation: given point cloud [B,3,H,W], feature [B,C,H,W], current depth [B,1,H,W],
     * and sparse depth [B,1,H,W], update depth using knn nearest non-zero sparse points.
     * candidateLimit>0 caps the number of sparse points considered (speed vs. quality).
     */
    external fun knnPropagate(
        point: FloatArray,
        feature: FloatArray,
        depth: FloatArray,
        sparse: FloatArray,
        batch: Int,
        channels: Int,
        height: Int,
        width: Int,
        knn: Int,
        candidateLimit: Int
    ): FloatArray
}
