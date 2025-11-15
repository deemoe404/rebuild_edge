package com.example.rebuild_edge.data

data class TaskRecord(
    val id: String,
    val startedAt: Long,
    val durationMs: Long,
    val outDir: String,
    val logPath: String,
    val inputCount: Int,
    val alignUsingGps: Boolean,
    val maxLongEdge: Int,
    val mode: Int,
    val stride: Int,
    val window: Int,
    val kNeighbors: Int,
    val resultSummary: String,
    val success: Boolean,
    val taskType: String = TYPE_OPENCV,
    val extraJson: String = ""
) {
    companion object {
        const val TYPE_OPENCV = "opencv"
        const val TYPE_COLMAP = "colmap"
    }
}
