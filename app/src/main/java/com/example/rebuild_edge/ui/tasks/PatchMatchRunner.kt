package com.example.rebuild_edge.ui.tasks

import com.example.rebuild_edge.SfmNative
import com.example.rebuild_edge.data.TaskRecord
import java.io.File
import java.io.FileInputStream
import java.util.Locale
import java.util.UUID
import org.json.JSONObject

class PatchMatchRunner(
    private val record: TaskRecord,
    private val extra: JSONObject?
) {

    data class Options(
        val maxImageSize: Int,
        val threads: Int,
        val geomConsistency: Boolean,
        val depthMin: Double?,
        val depthMax: Double?,
        val numIterations: Int,
        val windowRadius: Int,
        val numSamples: Int,
        val cacheSize: Int,
        val fusionMaxReprojError: Double,
        val fusionMaxDepthError: Double,
        val fusionMaxNormalError: Double,
        val fusionMinConsistent: Int
    )

    data class Result(
        val runId: String,
        val workspaceDir: File,
        val fusedFile: File?,
        val fusedPoints: Long,
        val depthCount: Int,
        val normalCount: Int,
        val logFile: File,
        val options: Options,
        val runtimeMs: Long
    )

    fun run(options: Options): Result {
        val datasetDir = extra?.optString("datasetDir")
            ?.takeIf { it.isNotBlank() }
            ?.let { File(it) }
            ?: throw IllegalStateException("COLMAP 任务缺少 datasetDir 信息")
        if (!datasetDir.isDirectory) {
            throw IllegalStateException("数据集目录不存在: ${datasetDir.absolutePath}")
        }
        val runDir = File(record.outDir)
        if (!runDir.isDirectory) {
            throw IllegalStateException("任务输出目录不存在: ${runDir.absolutePath}")
        }

        val runId = UUID.randomUUID().toString()
        val workspaceDir = File(runDir, "patchmatch_runs/$runId")
        if (workspaceDir.exists()) {
            workspaceDir.deleteRecursively()
        }
        workspaceDir.mkdirs()
        if (!workspaceDir.exists()) {
            throw IllegalStateException("无法创建 PatchMatch 输出目录: ${workspaceDir.absolutePath}")
        }
        val logFile = File(workspaceDir, "patchmatch.log")

        val start = System.currentTimeMillis()
        val result = SfmNative.runColmapPatchMatch(
            datasetDir.absolutePath,
            runDir.absolutePath,
            workspaceDir.absolutePath,
            logFile.absolutePath,
            options.maxImageSize,
            options.geomConsistency,
            options.depthMin ?: -1.0,
            options.depthMax ?: -1.0,
            options.numIterations,
            options.windowRadius,
            options.numSamples,
            options.cacheSize,
            options.threads,
            options.fusionMaxReprojError,
            options.fusionMaxDepthError,
            options.fusionMaxNormalError,
            options.fusionMinConsistent
        )
        val runtime = System.currentTimeMillis() - start
        if (!result.startsWith("OK")) {
            throw IllegalStateException(result)
        }
        val fusedFile = File(workspaceDir, "fused.ply").takeIf { it.exists() }
        val fusedPoints = fusedFile?.let { readPlyVertexCount(it) } ?: 0L
        val depthDir = File(workspaceDir, "stereo/depth_maps")
        val normalDir = File(workspaceDir, "stereo/normal_maps")
        val depthCount = depthDir.listFiles()?.count { it.isFile } ?: 0
        val normalCount = normalDir.listFiles()?.count { it.isFile } ?: 0

        return Result(
            runId = runId,
            workspaceDir = workspaceDir,
            fusedFile = fusedFile,
            fusedPoints = fusedPoints,
            depthCount = depthCount,
            normalCount = normalCount,
            logFile = logFile,
            options = options,
            runtimeMs = runtime
        )
    }

    private fun readPlyVertexCount(file: File): Long {
        return runCatching {
            FileInputStream(file).bufferedReader().useLines { seq ->
                val line = seq.firstOrNull { it.lowercase(Locale.getDefault()).startsWith("element vertex") }
                line?.split(Regex("\\s+"))?.getOrNull(2)?.toLongOrNull() ?: 0L
            }
        }.getOrDefault(0L)
    }

    companion object {
        fun hasArtifacts(record: TaskRecord, extra: JSONObject?): Boolean {
            if (record.taskType != TaskRecord.TYPE_COLMAP) return false
            val runDir = File(record.outDir)
            val sparseDir = File(runDir, "sparse")
            val alignedDir = File(runDir, "sparse_aligned")
            val datasetDir = extra?.optString("datasetDir")
                ?.takeIf { it.isNotBlank() }
                ?.let { File(it) }
            val hasSparse = (alignedDir.exists() && alignedDir.isDirectory) ||
                (sparseDir.exists() && sparseDir.isDirectory)
            return hasSparse && datasetDir?.isDirectory == true
        }
    }
}
