package com.example.rebuild_edge.ui.tasks

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import com.example.rebuild_edge.R
import com.example.rebuild_edge.data.TaskRecord
import com.example.rebuild_edge.data.TaskStore
import com.example.rebuild_edge.databinding.FragmentTaskDetailBinding
import com.example.rebuild_edge.databinding.ItemFileBinding
import com.example.rebuild_edge.databinding.ItemPatchmatchRunBinding
import com.example.rebuild_edge.databinding.ItemSparseDepthBinding
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

class TaskDetailFragment : Fragment() {
    private var _binding: FragmentTaskDetailBinding? = null
    private val binding get() = _binding!!
    private var record: TaskRecord? = null
    private var fileItems: List<Pair<File, String>> = emptyList()
    private var sparseProgressStartMs: Long = 0L

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentTaskDetailBinding.inflate(inflater, container, false)
        val root = binding.root

        val id = arguments?.getString("taskId")
        record = id?.let { TaskStore.getById(requireContext(), it) }

        bindUi()
        return root
    }

    private fun setupPatchMatchSection(rec: TaskRecord, extra: JSONObject?) {
        val isColmap = rec.taskType == TaskRecord.TYPE_COLMAP
        binding.groupPatchMatch.isVisible = isColmap
        if (!isColmap) {
            binding.txtPatchMatchRunsTitle.isVisible = false
            binding.containerPatchMatchRuns.removeAllViews()
            return
        }
        val ready = PatchMatchRunner.hasArtifacts(rec, extra)
        binding.btnRunPatchMatch.isEnabled = ready
        binding.progressPatchMatch.isVisible = false
        binding.txtPatchMatchProgress.isVisible = false
        if (!ready) {
            binding.txtPatchMatchStatus.text = "当前任务缺少稠密所需的 datasetDir 或 sparse 模型，无法运行 PatchMatch"
            return
        }
        binding.txtPatchMatchStatus.text = "使用 CPU PatchMatch 生成稠密深度与点云"

        fun setDefault(edit: android.widget.EditText, value: String) {
            if (edit.text.isNullOrBlank()) {
                edit.setText(value)
            }
        }
        setDefault(
            binding.editPatchMatchMaxSize,
            (extra?.optInt("denseMaxImageSize", 2000) ?: 2000).toString()
        )
        val defaultThreads = extra?.optInt("threads")?.takeIf { it > 0 }
            ?: Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        setDefault(binding.editPatchMatchThreads, defaultThreads.toString())
        setDefault(binding.editPatchMatchIterations, "5")
        setDefault(binding.editPatchMatchWindow, "5")
        setDefault(binding.editPatchMatchSamples, "15")
        setDefault(binding.editPatchMatchCache, "32")
        setDefault(binding.editFusionReproj, "4.0")
        setDefault(binding.editFusionDepth, "0.02")
        setDefault(binding.editFusionNormal, "20.0")
        setDefault(binding.editFusionMinConsistent, "2")
        // depth min/max default empty so that COLMAP auto range applies

        binding.btnRunPatchMatch.setOnClickListener {
            val options = collectPatchMatchOptions() ?: return@setOnClickListener
            runPatchMatchGeneration(rec, extra, options)
        }
    }

    private fun collectPatchMatchOptions(): PatchMatchRunner.Options? {
        val maxSize = binding.editPatchMatchMaxSize.text.toString().trim().toIntOrNull()
        if (maxSize == null || maxSize <= 0) {
            Toast.makeText(requireContext(), "请输入合法的 Max image size", Toast.LENGTH_SHORT).show()
            return null
        }
        val threads = binding.editPatchMatchThreads.text.toString().trim().toIntOrNull()
            ?: Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        val depthMin = binding.editPatchMatchDepthMin.text.toString().trim().toDoubleOrNull()
        val depthMax = binding.editPatchMatchDepthMax.text.toString().trim().toDoubleOrNull()
        if (depthMin != null && depthMax != null && depthMin >= depthMax) {
            Toast.makeText(requireContext(), "深度范围不合法 (min < max)", Toast.LENGTH_SHORT).show()
            return null
        }
        val iterations = binding.editPatchMatchIterations.text.toString().trim().toIntOrNull() ?: 5
        val window = binding.editPatchMatchWindow.text.toString().trim().toIntOrNull() ?: 5
        val samples = binding.editPatchMatchSamples.text.toString().trim().toIntOrNull() ?: 15
        val cacheGb = binding.editPatchMatchCache.text.toString().trim().toIntOrNull() ?: 32
        val fusionReproj = binding.editFusionReproj.text.toString().trim().toDoubleOrNull() ?: 4.0
        val fusionDepth = binding.editFusionDepth.text.toString().trim().toDoubleOrNull() ?: 0.02
        val fusionNormal = binding.editFusionNormal.text.toString().trim().toDoubleOrNull() ?: 20.0
        val fusionMin = binding.editFusionMinConsistent.text.toString().trim().toIntOrNull() ?: 2
        return PatchMatchRunner.Options(
            maxImageSize = maxSize,
            threads = threads.coerceAtLeast(1),
            geomConsistency = binding.checkPatchMatchGeom.isChecked,
            depthMin = depthMin,
            depthMax = depthMax,
            numIterations = iterations.coerceAtLeast(1),
            windowRadius = window.coerceAtLeast(1),
            numSamples = samples.coerceAtLeast(1),
            cacheSize = cacheGb.coerceAtLeast(1),
            fusionMaxReprojError = fusionReproj,
            fusionMaxDepthError = fusionDepth,
            fusionMaxNormalError = fusionNormal,
            fusionMinConsistent = fusionMin.coerceAtLeast(1)
        )
    }

    private fun runPatchMatchGeneration(rec: TaskRecord, extra: JSONObject?, options: PatchMatchRunner.Options) {
        binding.btnRunPatchMatch.isEnabled = false
        binding.progressPatchMatch.isVisible = true
        binding.txtPatchMatchProgress.isVisible = true
        binding.txtPatchMatchProgress.text = "运行中..."
        viewLifecycleOwner.lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { PatchMatchRunner(rec, extra).run(options) }
            }
            binding.progressPatchMatch.isVisible = false
            binding.txtPatchMatchProgress.isVisible = false
            binding.btnRunPatchMatch.isEnabled = true
            if (result.isSuccess) {
                val run = result.getOrThrow()
                val newExtra = rec.extraJson.takeIf { it.isNotBlank() }
                    ?.let { runCatching { JSONObject(it) }.getOrNull() }
                    ?: JSONObject()
                val runJson = JSONObject().apply {
                    put("runId", run.runId)
                    put("workspaceDir", run.workspaceDir.absolutePath)
                    put("logPath", run.logFile.absolutePath)
                    put("generatedAt", System.currentTimeMillis())
                    put("maxImageSize", options.maxImageSize)
                    put("threads", options.threads)
                    put("geomConsistency", options.geomConsistency)
                    put("numIterations", options.numIterations)
                    put("windowRadius", options.windowRadius)
                    put("numSamples", options.numSamples)
                    put("cacheSize", options.cacheSize)
                    put("fusionMaxReprojError", options.fusionMaxReprojError)
                    put("fusionMaxDepthError", options.fusionMaxDepthError)
                    put("fusionMaxNormalError", options.fusionMaxNormalError)
                    put("fusionMinConsistent", options.fusionMinConsistent)
                    put("fusedFile", run.fusedFile?.absolutePath ?: "")
                    put("fusedPoints", run.fusedPoints)
                    put("depthCount", run.depthCount)
                    put("normalCount", run.normalCount)
                    put("runtimeMs", run.runtimeMs)
                }
                options.depthMin?.let { runJson.put("depthMin", it) }
                options.depthMax?.let { runJson.put("depthMax", it) }
                val arr = newExtra.optJSONArray("patchmatchRuns")
                    ?: JSONArray().also { newExtra.put("patchmatchRuns", it) }
                arr.put(runJson)
                ensureTag(newExtra, "dense")
                ensureTag(newExtra, "patchmatch")
                updateRecordExtra(newExtra)
                Toast.makeText(requireContext(), "PatchMatch 已完成", Toast.LENGTH_SHORT).show()
                bindUi()
            } else {
                val msg = result.exceptionOrNull()?.message ?: "未知错误"
                binding.txtPatchMatchStatus.text = "PatchMatch 失败: $msg"
                Toast.makeText(requireContext(), "PatchMatch 失败: $msg", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun setupSparseDepthSection(rec: TaskRecord, extra: JSONObject?) {
        val hasArtifacts = SparseDepthGenerator.hasArtifacts(rec)
        binding.groupSparseDepth.isVisible = hasArtifacts
        if (!hasArtifacts) {
            binding.txtSparseDepthStatus.text = "当前任务缺少 camera_poses.json 或 PLY，无法生成稀疏深度"
            binding.btnGenerateSparseDepth.isEnabled = false
            binding.progressSparseDepth.isVisible = false
            binding.txtSparseDepthProgress.isVisible = false
            return
        }
        binding.btnGenerateSparseDepth.isEnabled = true
        binding.progressSparseDepth.isVisible = false
        binding.progressSparseDepth.isIndeterminate = true
        binding.progressSparseDepth.progress = 0
        binding.txtSparseDepthProgress.isVisible = false
        val runs = parseSparseDepthRuns(extra)
        val sparseObj = runs.lastOrNull() ?: extra?.optJSONObject("sparseDepth")
        val statusText = sparseObj?.let { obj ->
            val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
            val time = obj.optLong("generatedAt", 0L)
            val tsText = if (time > 0) fmt.format(Date(time)) else "?"
            val width = obj.optInt("width")
            val height = obj.optInt("height")
            val points = obj.optLong("points2d")
            val runId = obj.optString("runId").takeIf { it.isNotBlank() } ?: "--"
            "上次生成: ${width}x${height} · ${points} 像素 · ${tsText} · ID=$runId"
        } ?: "尚未生成稀疏深度"
        binding.txtSparseDepthStatus.text = statusText

        val baseSize = runs.lastOrNull()?.let { it.optInt("width") to it.optInt("height") }
            ?.takeIf { it.first > 0 && it.second > 0 }
            ?: SparseDepthGenerator.estimateSourceResolution(File(rec.outDir), extra)
        val defaultWidth = sparseObj?.optInt("width")?.takeIf { it > 0 }
            ?: baseSize?.first
        val defaultHeight = sparseObj?.optInt("height")?.takeIf { it > 0 }
            ?: baseSize?.second
        if (binding.editSparseWidth.text.isNullOrBlank()) {
            defaultWidth?.let { binding.editSparseWidth.setText(it.toString()) }
        }
        if (binding.editSparseHeight.text.isNullOrBlank()) {
            defaultHeight?.let { binding.editSparseHeight.setText(it.toString()) }
        }

        binding.btnGenerateSparseDepth.setOnClickListener {
            val width = binding.editSparseWidth.text.toString().trim().toIntOrNull()
            val height = binding.editSparseHeight.text.toString().trim().toIntOrNull()
            if (width == null || width <= 0 || height == null || height <= 0) {
                Toast.makeText(requireContext(), "请输入合法的宽和高", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            runSparseDepthGeneration(rec, extra, width, height)
        }
    }

    private fun runSparseDepthGeneration(rec: TaskRecord, extra: JSONObject?, width: Int, height: Int) {
        android.util.Log.i("TaskDetailFragment", "Generate sparse depth task=${rec.id} size=${width}x$height")
        sparseProgressStartMs = System.currentTimeMillis()
        binding.btnGenerateSparseDepth.isEnabled = false
        binding.progressSparseDepth.isVisible = true
        binding.progressSparseDepth.isIndeterminate = true
        binding.progressSparseDepth.progress = 0
        binding.txtSparseDepthProgress.isVisible = true
        binding.txtSparseDepthProgress.text = "准备中..."
        binding.txtSparseDepthStatus.text = "正在生成 ${width}x${height} ..."
        viewLifecycleOwner.lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    SparseDepthGenerator(rec, extra).generate(width, height) { current, total ->
                        if (total <= 0) return@generate
                        val elapsed = System.currentTimeMillis() - sparseProgressStartMs
                        val percent = ((current.toDouble() / total.toDouble()) * 100).toInt().coerceIn(0, 100)
                        val etaMs = if (current > 0) ((elapsed / current) * (total - current)) else -1
                        viewLifecycleOwner.lifecycleScope.launch(Dispatchers.Main) {
                            binding.progressSparseDepth.isIndeterminate = false
                            binding.progressSparseDepth.progress = percent
                            binding.txtSparseDepthProgress.text = buildString {
                                append("进度 $current/$total ($percent%)")
                                append(" · ")
                                append(
                                    if (etaMs > 0) formatEta(etaMs)
                                    else "预计剩余 --"
                                )
                            }
                        }
                    }
                }
            }
            binding.progressSparseDepth.isVisible = false
            binding.txtSparseDepthProgress.isVisible = false
            binding.btnGenerateSparseDepth.isEnabled = true
            if (result.isSuccess) {
                val gen = result.getOrThrow()
                android.util.Log.i("TaskDetailFragment", "Sparse depth success dir=${gen.outputDir} files=${gen.files.size}")
                val newExtra = rec.extraJson.takeIf { it.isNotBlank() }
                    ?.let { runCatching { JSONObject(it) }.getOrNull() }
                    ?: JSONObject()
                val sparseJson = JSONObject().apply {
                    put("dir", gen.outputDir.absolutePath)
                    put("width", gen.width)
                    put("height", gen.height)
                    put("sourceWidth", gen.sourceWidth)
                    put("sourceHeight", gen.sourceHeight)
                    put("points2d", gen.pointsWritten)
                    put("files", gen.files.size)
                    put("generatedAt", System.currentTimeMillis())
                    put("runId", gen.runId)
                    put("metadataPath", gen.metadataFile.absolutePath)
                }
                val runsArr = newExtra.optJSONArray("sparseDepthRuns") ?: JSONArray().also { newExtra.put("sparseDepthRuns", it) }
                runsArr.put(sparseJson)
                newExtra.put("sparseDepth", sparseJson)
                ensureTag(newExtra, "sfm")
                ensureTag(newExtra, "sparse_depth")
                updateRecordExtra(newExtra)
                Toast.makeText(requireContext(), "稀疏深度已生成", Toast.LENGTH_SHORT).show()
                bindUi()
            } else {
                val msg = result.exceptionOrNull()?.message ?: "未知错误"
                android.util.Log.e("TaskDetailFragment", "Sparse depth failed: $msg", result.exceptionOrNull())
                binding.txtSparseDepthStatus.text = "生成失败: $msg"
                binding.txtSparseDepthProgress.isVisible = false
                Toast.makeText(requireContext(), "生成失败: $msg", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun formatEta(ms: Long): String {
        val totalSeconds = (ms / 1000).coerceAtLeast(0)
        val hours = (totalSeconds / 3600).toInt()
        val minutes = ((totalSeconds % 3600) / 60).toInt()
        val seconds = (totalSeconds % 60).toInt()
        return if (hours > 0) {
            "预计剩余 ${hours}小时${minutes}分"
        } else {
            "预计剩余 ${minutes}分${seconds}秒"
        }
    }

    private fun updateRecordExtra(newExtra: JSONObject) {
        val rec = record ?: return
        val updated = rec.copy(extraJson = newExtra.toString())
        TaskStore.update(requireContext(), updated)
        record = updated
    }

    private fun ensureTag(extra: JSONObject, tag: String) {
        val arr = extra.optJSONArray("tags") ?: JSONArray().also { extra.put("tags", it) }
        for (i in 0 until arr.length()) {
            if (arr.optString(i) == tag) return
        }
        arr.put(tag)
    }

    private fun parseSparseDepthRuns(extra: JSONObject?): List<JSONObject> {
        if (extra == null) return emptyList()
        val arr = extra.optJSONArray("sparseDepthRuns")
        val list = mutableListOf<JSONObject>()
        if (arr != null) {
            for (i in 0 until arr.length()) {
                val obj = arr.optJSONObject(i) ?: continue
                list.add(obj)
            }
        }
        if (list.isEmpty()) {
            extra.optJSONObject("sparseDepth")?.let { list.add(it) }
        }
        return list
    }

    private fun parsePatchMatchRuns(extra: JSONObject?): List<JSONObject> {
        if (extra == null) return emptyList()
        val arr = extra.optJSONArray("patchmatchRuns") ?: return emptyList()
        val list = mutableListOf<JSONObject>()
        for (i in 0 until arr.length()) {
            val obj = arr.optJSONObject(i) ?: continue
            list.add(obj)
        }
        return list
    }

    private fun bindSparseDepthRuns(extra: JSONObject?) {
        val container = binding.containerSparseRuns
        container.removeAllViews()
        val runs = parseSparseDepthRuns(extra)
        if (runs.isEmpty()) {
            val emptyView = LayoutInflater.from(requireContext()).inflate(android.R.layout.simple_list_item_1, container, false) as android.widget.TextView
            emptyView.text = "暂无稀疏深度输出"
            container.addView(emptyView)
            return
        }
        runs.forEach { obj ->
            val bindingItem = ItemSparseDepthBinding.inflate(LayoutInflater.from(requireContext()), container, false)
            val runId = obj.optString("runId").takeIf { it.isNotBlank() } ?: "--"
            val width = obj.optInt("width")
            val height = obj.optInt("height")
            val dir = obj.optString("dir")
            val meta = obj.optString("metadataPath", File(dir, "metadata.json").absolutePath)
            val points = obj.optLong("points2d")
            val ts = obj.optLong("generatedAt")
            val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
            val tsText = if (ts > 0) fmt.format(Date(ts)) else "--"
            bindingItem.txtRunTitle.text = "ID: $runId"
            bindingItem.txtRunDetail.text = "${width}x${height} · 像素=$points · 时间=$tsText\nmetadata: $meta"
            bindingItem.btnPreviewSparse.setOnClickListener {
                val args = Bundle().apply {
                    putString("runDir", dir)
                    putString("runId", runId)
                }
                findNavController().navigate(R.id.action_task_detail_to_sparse_preview, args)
            }
            bindingItem.btnShareSparse.setOnClickListener {
                shareSparseDepth(dir, runId)
            }
            container.addView(bindingItem.root)
        }
    }

    private fun bindPatchMatchRuns(extra: JSONObject?) {
        val container = binding.containerPatchMatchRuns
        container.removeAllViews()
        val runs = parsePatchMatchRuns(extra)
        val showRuns = runs.isNotEmpty() && binding.groupPatchMatch.isVisible
        binding.txtPatchMatchRunsTitle.isVisible = showRuns
        if (!binding.groupPatchMatch.isVisible) return
        if (runs.isEmpty()) {
            val emptyView = LayoutInflater.from(requireContext())
                .inflate(android.R.layout.simple_list_item_1, container, false) as android.widget.TextView
            emptyView.text = "暂未运行 PatchMatch"
            container.addView(emptyView)
            return
        }
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
        runs.forEach { obj ->
            val itemBinding = ItemPatchmatchRunBinding.inflate(LayoutInflater.from(requireContext()), container, false)
            val runId = obj.optString("runId")
            val fusedPoints = obj.optLong("fusedPoints")
            val ts = obj.optLong("generatedAt")
            val runtimeMs = obj.optLong("runtimeMs")
            val depthCount = obj.optInt("depthCount")
            val normalCount = obj.optInt("normalCount")
            val workspace = obj.optString("workspaceDir")
            val fusedText = if (fusedPoints > 0) {
                String.format(Locale.getDefault(), "%,d", fusedPoints)
            } else {
                "--"
            }
            val runtimeText = if (runtimeMs > 0) {
                String.format(Locale.getDefault(), "%.1f 秒", runtimeMs / 1000.0)
            } else {
                "--"
            }
            val tsText = if (ts > 0) fmt.format(Date(ts)) else "--"
            val maxSize = obj.optInt("maxImageSize")
            val threads = obj.optInt("threads")
            val geom = obj.optBoolean("geomConsistency", false)
            val iterations = obj.optInt("numIterations")
            val window = obj.optInt("windowRadius")
            itemBinding.txtPatchRunTitle.text = "ID: $runId · $tsText · 点=${fusedText}"
            itemBinding.txtPatchRunDetail.text = buildString {
                append("深度=${depthCount} · 法线=${normalCount} · 用时=$runtimeText\n")
                append("maxSize=$maxSize, 线程=$threads, geom=$geom, iter=$iterations, window=$window")
            }
            itemBinding.btnPatchRunShare.setOnClickListener {
                sharePatchMatchRun(workspace, runId)
            }
            itemBinding.btnPatchRunDetail.setOnClickListener {
                val args = Bundle().apply {
                    putString("runDir", workspace)
                    putString("runInfo", obj.toString())
                    putString("runId", runId)
                }
                findNavController().navigate(R.id.action_task_detail_to_patchmatch_detail, args)
            }
            container.addView(itemBinding.root)
        }
    }

    private fun shareSparseDepth(dirPath: String, runId: String) {
        val dir = File(dirPath)
        if (!dir.isDirectory) {
            Toast.makeText(requireContext(), "稀疏深度目录不存在", Toast.LENGTH_SHORT).show()
            return
        }
        val zipFile = zipDirectory(dir, runId, prefix = "sparse")
        if (zipFile == null || !zipFile.exists()) {
            Toast.makeText(requireContext(), "打包ZIP失败", Toast.LENGTH_SHORT).show()
            return
        }
        shareUris(listOf(zipFile))
    }

    private fun sharePatchMatchRun(dirPath: String?, runId: String) {
        val dir = dirPath?.let { File(it) }
        if (dir == null || !dir.isDirectory) {
            Toast.makeText(requireContext(), "PatchMatch 目录不存在", Toast.LENGTH_SHORT).show()
            return
        }
        val zipFile = zipDirectory(dir, runId, prefix = "patchmatch")
        if (zipFile == null || !zipFile.exists()) {
            Toast.makeText(requireContext(), "打包ZIP失败", Toast.LENGTH_SHORT).show()
            return
        }
        shareUris(listOf(zipFile))
    }

    private fun zipDirectory(dir: File, runId: String, prefix: String = "sparse"): File? {
        return runCatching {
            val zipName = "${prefix}_${runId.ifBlank { System.currentTimeMillis().toString() }}.zip"
            val outFile = File(requireContext().cacheDir, zipName)
            ZipOutputStream(FileOutputStream(outFile)).use { zos ->
                dir.walkTopDown().forEach { file ->
                    if (file.isDirectory) return@forEach
                    val rel = dir.toURI().relativize(file.toURI()).path
                    zos.putNextEntry(ZipEntry(rel))
                    FileInputStream(file).use { fis ->
                        fis.copyTo(zos)
                    }
                    zos.closeEntry()
                }
            }
            outFile
        }.getOrNull()
    }

    private fun renderFileList(files: List<File>) {
        val container = binding.containerFiles
        container.removeAllViews()
        if (files.isEmpty()) {
            val emptyView = LayoutInflater.from(requireContext()).inflate(android.R.layout.simple_list_item_1, container, false) as android.widget.TextView
            emptyView.text = "无输出文件"
            container.addView(emptyView)
            return
        }
        files.forEach { file ->
            val itemBinding = ItemFileBinding.inflate(LayoutInflater.from(requireContext()), container, false)
            itemBinding.txtFileName.text = file.name
            itemBinding.txtFileSize.text = formatSize(file.length())
            itemBinding.btnShare.setOnClickListener { shareUris(listOf(file)) }
            container.addView(itemBinding.root)
        }
    }

    private fun formatSize(bytes: Long): String {
        if (bytes <= 0) return "0 B"
        val units = arrayOf("B", "KB", "MB", "GB", "TB")
        var value = bytes.toDouble()
        var unitIndex = 0
        while (value >= 1024 && unitIndex < units.lastIndex) {
            value /= 1024
            unitIndex++
        }
        return String.format(Locale.getDefault(), "%.1f %s", value, units[unitIndex])
    }

    private fun bindUi() {
        val rec = record ?: return
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        val ts = fmt.format(Date(rec.startedAt))
        val mins = (rec.durationMs / 1000) / 60
        val secs = (rec.durationMs / 1000) % 60
        val typeLabel = when (rec.taskType) {
            TaskRecord.TYPE_COLMAP -> "COLMAP"
            else -> "OpenCV"
        }
        val extra = rec.extraJson.takeIf { it.isNotBlank() }?.let { runCatching { JSONObject(it) }.getOrNull() }
        binding.txtSummary.text = buildString {
            append(if (rec.success) "✅ 成功" else "⚠️ 可能失败")
            append(" · 类型 ").append(typeLabel)
            append(" · 开始时间 ").append(ts)
            append(" · 用时 ").append(mins).append("分").append(secs).append("秒")
            if (rec.taskType == TaskRecord.TYPE_COLMAP) {
                append("\nCOLMAP 参数: camera=")
                    .append(extra?.optString("cameraModel") ?: "--")
                append(", overlap=")
                    .append(extra?.optInt("sequentialOverlap", rec.mode) ?: rec.mode)
                append(", gps=")
                    .append(extra?.optInt("maxGpsNeighborDist", rec.stride) ?: rec.stride)
                append(", threads=")
                    .append(extra?.optInt("threads", rec.window) ?: rec.window)
                append(", alignType=")
                    .append(extra?.optString("alignmentType") ?: "ecef")
                append(", 对齐=").append(rec.alignUsingGps)
            } else {
                append("\n参数: 对齐=").append(rec.alignUsingGps)
                append(", maxEdge=").append(rec.maxLongEdge)
                append(", 模式=").append(rec.mode)
                append(", stride=").append(rec.stride)
                append(", window=").append(rec.window)
                append(", k=").append(rec.kNeighbors)
            }
        }

        setupPatchMatchSection(rec, extra)
        bindPatchMatchRuns(extra)
        setupSparseDepthSection(rec, extra)
        bindSparseDepthRuns(extra)

        val outDir = File(rec.outDir)
        val files = if (outDir.exists()) {
            outDir.listFiles()?.filter { it.isFile }
                ?.sortedBy { it.name.lowercase(Locale.getDefault()) }
                ?: emptyList()
        } else {
            emptyList()
        }
        fileItems = files.map { it to it.name }
        renderFileList(files)

        binding.btnShareLog.setOnClickListener {
            shareUris(listOf(File(rec.logPath)))
        }
        binding.btnShareAll.setOnClickListener {
            shareUris(fileItems.map { it.first })
        }
        val logFile = File(rec.logPath)
        binding.btnViewLog.isEnabled = logFile.exists()
        binding.btnViewLog.text = if (logFile.exists()) "查看日志" else "日志文件不存在"
        binding.btnViewLog.setOnClickListener {
            if (!logFile.exists()) {
                Toast.makeText(requireContext(), "找不到日志文件", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val args = Bundle().apply { putString("taskId", rec.id) }
            findNavController().navigate(R.id.action_task_detail_to_log, args)
        }
    }

    private fun shareSingle(file: File) {
        shareUris(listOf(file))
    }

    private fun shareUris(files: List<File>) {
        if (files.isEmpty()) return
        val ctx = requireContext()
        val auth = ctx.packageName + ".fileprovider"
        val uris = files.filter { it.exists() }.map { f ->
            FileProvider.getUriForFile(ctx, auth, f)
        }
        if (uris.isEmpty()) return
        val intent = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).apply {
                type = "*/*"
                putExtra(Intent.EXTRA_STREAM, uris.first())
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                type = "*/*"
                putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList<Uri>(uris))
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }
        startActivity(Intent.createChooser(intent, "分享输出文件"))
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

class FilesAdapter(
    private val files: List<Pair<File, String>>,
    private val onShare: (File) -> Unit
) : androidx.recyclerview.widget.RecyclerView.Adapter<FilesAdapter.VH>() {
    class VH(val binding: ItemFileBinding) : androidx.recyclerview.widget.RecyclerView.ViewHolder(binding.root)
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val binding = ItemFileBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return VH(binding)
    }
    override fun getItemCount(): Int = files.size
    override fun onBindViewHolder(holder: VH, position: Int) {
        val (file, label) = files[position]
        holder.binding.txtFileName.text = label
        holder.binding.btnShare.setOnClickListener { onShare(file) }
        holder.binding.root.setOnClickListener { onShare(file) }
    }
}
