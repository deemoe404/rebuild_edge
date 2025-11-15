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
import com.example.rebuild_edge.databinding.FragmentPatchmatchRunBinding
import com.example.rebuild_edge.databinding.ItemFileBinding
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.ArrayList
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.json.JSONObject

class PatchmatchRunDetailFragment : Fragment() {

    private var _binding: FragmentPatchmatchRunBinding? = null
    private val binding get() = _binding!!
    private var runDir: File? = null
    private var runInfo: JSONObject? = null
    private var runId: String? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentPatchmatchRunBinding.inflate(inflater, container, false)
        runDir = arguments?.getString("runDir")?.let { File(it) }
        runInfo = arguments?.getString("runInfo")?.let { runCatching { JSONObject(it) }.getOrNull() }
        runId = arguments?.getString("runId")
        bindUi()
        return binding.root
    }

    private fun bindUi() {
        val dir = runDir
        val info = runInfo
        if (dir == null || info == null) {
            binding.txtPatchRunSummary.text = "运行数据缺失"
            binding.btnSharePatchWorkspace.isEnabled = false
            binding.btnSharePatchLog.isEnabled = false
            binding.containerPatchRunFiles.removeAllViews()
            return
        }
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
        val ts = info.optLong("generatedAt")
        val tsText = if (ts > 0) fmt.format(Date(ts)) else "--"
        val fusedPts = info.optLong("fusedPoints")
        val fusedText = if (fusedPts > 0) String.format(Locale.getDefault(), "%,d", fusedPts) else "--"
        val runtimeMs = info.optLong("runtimeMs")
        val runtimeText = if (runtimeMs > 0) String.format(Locale.getDefault(), "%.1f 秒", runtimeMs / 1000.0) else "--"
        val depthCount = info.optInt("depthCount")
        val normalCount = info.optInt("normalCount")
        val maxSize = info.optInt("maxImageSize")
        val threads = info.optInt("threads")
        val geom = info.optBoolean("geomConsistency", false)
        val iterations = info.optInt("numIterations")
        val window = info.optInt("windowRadius")
        val samples = info.optInt("numSamples")
        val cache = info.optInt("cacheSize")
        val fusionReproj = info.optDouble("fusionMaxReprojError", 4.0)
        val fusionDepth = info.optDouble("fusionMaxDepthError", 0.02)
        val fusionNormal = info.optDouble("fusionMaxNormalError", 20.0)
        val fusionMin = info.optInt("fusionMinConsistent")
        binding.txtPatchRunSummary.text = buildString {
            append("ID=${runId ?: "--"} · 时间=$tsText · 点=${fusedText}\n")
            append("深度=${depthCount} · 法线=${normalCount} · 用时=$runtimeText\n")
            append("maxSize=$maxSize, 线程=$threads, geom=$geom, iter=$iterations, window=$window, samples=$samples, cache=${cache}GB\n")
            append("融合阈值: reproj=$fusionReproj, depth=$fusionDepth, normal=$fusionNormal, minConsistent=$fusionMin")
        }
        binding.btnSharePatchWorkspace.isEnabled = true
        binding.btnSharePatchWorkspace.setOnClickListener {
            val zip = zipDirectory(dir, "patchmatch_run_${runId ?: ts.toString()}")
            if (zip == null) {
                Toast.makeText(requireContext(), "打包目录失败", Toast.LENGTH_SHORT).show()
            } else {
                shareFiles(listOf(zip))
            }
        }
        val logPath = info.optString("logPath")
        binding.btnSharePatchLog.isEnabled = logPath.isNotBlank()
        binding.btnSharePatchLog.setOnClickListener {
            val logFile = File(logPath)
            if (!logFile.exists()) {
                Toast.makeText(requireContext(), "找不到日志文件", Toast.LENGTH_SHORT).show()
            } else {
                shareFiles(listOf(logFile))
            }
        }

        val entries = mutableListOf<Pair<File, String>>()
        info.optString("fusedFile").takeIf { it.isNotBlank() }?.let { path ->
            val file = File(path)
            if (file.exists()) entries.add(file to "稠密点云 (fused.ply)")
        }
        val depthDir = File(dir, "stereo/depth_maps")
        if (depthDir.exists()) {
            entries.add(depthDir to "Depth maps (${depthCount} 文件)")
        }
        val normalDir = File(dir, "stereo/normal_maps")
        if (normalDir.exists()) {
            entries.add(normalDir to "Normal maps (${normalCount} 文件)")
        }
        val cfg = File(dir, "stereo/patch-match.cfg")
        if (cfg.exists()) {
            entries.add(cfg to "Patch-match 配置")
        }
        bindFileEntries(entries)
    }

    private fun bindFileEntries(entries: List<Pair<File, String>>) {
        val container = binding.containerPatchRunFiles
        container.removeAllViews()
        if (entries.isEmpty()) {
            val emptyView = LayoutInflater.from(requireContext())
                .inflate(android.R.layout.simple_list_item_1, container, false) as android.widget.TextView
            emptyView.text = "暂无可分享的文件"
            container.addView(emptyView)
            return
        }
        entries.forEach { (file, label) ->
            val itemBinding = ItemFileBinding.inflate(LayoutInflater.from(requireContext()), container, false)
            itemBinding.txtFileName.text = label
            itemBinding.txtFileSize.text = file.absolutePath
            itemBinding.btnShare.setOnClickListener {
                if (file.isDirectory) {
                    val zip = zipDirectory(file, "${runId ?: "patch"}_${file.name}")
                    if (zip == null) {
                        Toast.makeText(requireContext(), "打包目录失败", Toast.LENGTH_SHORT).show()
                    } else {
                        shareFiles(listOf(zip))
                    }
                } else {
                    shareFiles(listOf(file))
                }
            }
            container.addView(itemBinding.root)
        }
    }

    private fun zipDirectory(dir: File, prefix: String): File? {
        return runCatching {
            val zipName = "${prefix.ifBlank { "patchmatch" }}.zip"
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

    private fun shareFiles(files: List<File>) {
        if (files.isEmpty()) return
        val ctx = requireContext()
        val uris = files.filter { it.exists() }.map {
            FileProvider.getUriForFile(ctx, ctx.packageName + ".fileprovider", it)
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
        startActivity(Intent.createChooser(intent, "分享 PatchMatch 输出"))
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
