package com.example.rebuild_edge.ui.colmap

import android.app.ActivityManager
import android.content.Intent
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.provider.DocumentsContract
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.documentfile.provider.DocumentFile
import androidx.fragment.app.Fragment
import com.example.rebuild_edge.SfmNative
import com.example.rebuild_edge.data.TaskRecord
import com.example.rebuild_edge.data.TaskStore
import com.example.rebuild_edge.databinding.FragmentColmapBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

class ColmapFragment : Fragment() {

    private var _binding: FragmentColmapBinding? = null
    private val binding get() = _binding!!

    private val ioScope = CoroutineScope(Dispatchers.IO + Job())
    private var datasetTree: Uri? = null
    private var startTimeMs: Long = 0L
    private var timerJob: Job? = null
    private var tailJob: Job? = null
    private var cpuJob: Job? = null
    private var ramJob: Job? = null
    private var kotlinLog: String = ""

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentColmapBinding.inflate(inflater, container, false)
        val root = binding.root

        binding.btnSelectDataset.setOnClickListener {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
                putExtra(DocumentsContract.EXTRA_INITIAL_URI, datasetTree)
            }
            startActivityForResult(intent, REQ_OPEN_TREE)
        }

        val defaultThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        binding.editThreads.setText(defaultThreads.toString())

        binding.btnRunColmap.setOnClickListener {
            runColmapPipeline()
        }

        return root
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_OPEN_TREE) {
            val uri = data?.data ?: return
            requireContext().contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
            datasetTree = uri
            binding.txtDataset.text = "已选择: $uri"
            binding.btnRunColmap.isEnabled = true
        }
    }

    private fun runColmapPipeline() {
        val uri = datasetTree ?: return
        binding.btnRunColmap.isEnabled = false
        startTimeMs = System.currentTimeMillis()
        kotlinLog = "准备数据集...\n"
        binding.txtLog.text = kotlinLog
        binding.txtElapsed.text = "Elapsed: 00:00"
        binding.txtCpu.text = "CPU: --"
        binding.txtRam.text = "RAM: --"
        startTimer()
        startCpuMonitor()
        startRamMonitor()

        ioScope.launch {
            val doc = DocumentFile.fromTreeUri(requireContext(), uri)
            if (doc == null || !doc.isDirectory) {
                appendKotlinLog("选择的URI不是有效的文件夹\n")
                onRunFinished("Error: 无法读取文件夹", null)
                return@launch
            }

            val datasetDir = File(requireContext().filesDir, "colmap_dataset/$startTimeMs")
            datasetDir.deleteRecursively()
            datasetDir.mkdirs()

            val entries = mutableListOf<Pair<DocumentFile, String>>()
            collectImages(doc, "", entries)
            if (entries.isEmpty()) {
                appendKotlinLog("未找到任何JPG/JPEG文件\n")
                onRunFinished("Error: 没有图片", null)
                return@launch
            }

            val sortedEntries = entries.sortedBy { it.second.lowercase(Locale.getDefault()) }
            appendKotlinLog("发现${sortedEntries.size}张JPG，开始复制到应用目录...\n")
            val resolver = requireContext().contentResolver
            sortedEntries.forEachIndexed { index, pair ->
                val (docFile, relativePath) = pair
                val destFile = File(datasetDir, relativePath)
                destFile.parentFile?.mkdirs()
                runCatching {
                    resolver.openInputStream(docFile.uri)?.use { ins ->
                        FileOutputStream(destFile).use { outs -> ins.copyTo(outs) }
                    }
                }.onFailure { err ->
                    Log.e(TAG, "Copy failed for ${docFile.name}", err)
                    appendKotlinLog("复制 ${docFile.name} 失败: ${err.message}\n")
                }
                if ((index + 1) % 20 == 0 || index + 1 == sortedEntries.size) {
                    appendKotlinLog("已复制 ${index + 1} / ${sortedEntries.size}\n")
                }
            }

            val runDir = File(requireContext().getExternalFilesDir(null), "colmap_runs/run_$startTimeMs").apply { mkdirs() }
            val imageListFile = File(runDir, "image_list.txt")
            imageListFile.writeText(
                sortedEntries.joinToString("\n") { it.second.replace(File.separatorChar, '/') }
            )
            val logFile = File(runDir, "colmap_run.log")
            logFile.writeText("")
            withContext(Dispatchers.Main) {
                startTailingLog(logFile)
            }

            val cameraModel = binding.spinnerCameraModel.selectedItem.toString()
            val singleCamera = binding.chkSingleCamera.isChecked
            val sequentialOverlap = binding.editSequentialOverlap.text.toString().trim().toIntOrNull()?.coerceAtLeast(1) ?: 8
            val gpsDist = binding.editGpsDistance.text.toString().trim().toIntOrNull()?.coerceAtLeast(10) ?: 120
            val maxImageSize = binding.editMaxImageSize.text.toString().trim().toIntOrNull()?.coerceAtLeast(512) ?: 4000
            val threads = binding.editThreads.text.toString().trim().toIntOrNull()?.coerceAtLeast(1)
                ?: Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
            val alignGps = binding.chkAlignGps.isChecked
            val alignmentType = binding.spinnerAlignmentType.selectedItem.toString()
            val alignmentError = binding.editAlignmentError.text.toString().trim().toDoubleOrNull()?.coerceAtLeast(1.0)
                ?: 20.0

            appendKotlinLog(
                "Run dir: ${runDir.absolutePath}\n" +
                    "图像目录: ${datasetDir.absolutePath}\n" +
                    "参数: camera=$cameraModel, single=$singleCamera, overlap=$sequentialOverlap, " +
                    "gpsDist=$gpsDist, maxEdge=$maxImageSize, threads=$threads, align=$alignGps\n"
            )

            val result = try {
                SfmNative.runColmapSfm(
                    datasetDir.absolutePath,
                    runDir.absolutePath,
                    imageListFile.absolutePath,
                    logFile.absolutePath,
                    cameraModel,
                    singleCamera,
                    sequentialOverlap,
                    gpsDist,
                    maxImageSize,
                    threads,
                    alignGps,
                    alignmentType,
                    alignmentError
                )
            } catch (t: Throwable) {
                Log.e(TAG, "Native COLMAP crashed", t)
                "Error: ${t.message}"
            }

            val sparseRoot = File(runDir, "sparse")
            val alignedRoot = File(runDir, "sparse_aligned")
            val alignedOk = File(alignedRoot, "images.bin").exists() && File(alignedRoot, "points3D.bin").exists()
            val sparseModel = if (alignedOk) alignedRoot else pickSparseModelDir(sparseRoot)
            val success = result.startsWith("OK") && sparseModel?.let { File(it, "points3D.bin").exists() } == true

            runCatching {
                val ended = System.currentTimeMillis()
                val duration = (ended - startTimeMs).coerceAtLeast(0L)
                val rec = TaskRecord(
                    id = TaskStore.newId(),
                    startedAt = startTimeMs,
                    durationMs = duration,
                    outDir = runDir.absolutePath,
                    logPath = logFile.absolutePath,
                    inputCount = sortedEntries.size,
                    alignUsingGps = alignGps,
                    maxLongEdge = maxImageSize,
                    mode = sequentialOverlap,
                    stride = gpsDist,
                    window = threads,
                    kNeighbors = alignmentError.toInt(),
                    resultSummary = result.take(5000),
                    success = success,
                    taskType = TaskRecord.TYPE_COLMAP,
                    extraJson = JSONObject().apply {
                        put("cameraModel", cameraModel)
                        put("singleCamera", singleCamera)
                        put("maxImageSize", maxImageSize)
                        put("sequentialOverlap", sequentialOverlap)
                        put("maxGpsNeighborDist", gpsDist)
                        put("threads", threads)
                        put("alignmentType", alignmentType)
                        put("alignmentError", alignmentError)
                        put("datasetDir", datasetDir.absolutePath)
                    }.toString()
                )
                TaskStore.add(requireContext(), rec)
            }

            onRunFinished(result, logFile)
        }
    }

    private suspend fun onRunFinished(result: String, logFile: File?) {
        withContext(Dispatchers.Main) {
            stopTailingLog()
            stopTimer()
            stopCpuMonitor()
            stopRamMonitor()
            val nativeLog = logFile?.takeIf { it.exists() }?.let { runCatching { it.readText() }.getOrDefault("") } ?: ""
            binding.txtLog.text = buildString {
                append(kotlinLog)
                if (nativeLog.isNotEmpty()) {
                    append("\n").append(nativeLog)
                }
                append("\n").append(result)
            }
            binding.btnRunColmap.isEnabled = true
        }
    }

    private fun collectImages(node: DocumentFile?, relativePath: String, out: MutableList<Pair<DocumentFile, String>>) {
        if (node == null || !node.exists()) return
        if (node.isDirectory) {
            node.listFiles().forEach { child ->
                val name = child.name ?: return@forEach
                val next = if (relativePath.isEmpty()) name else "$relativePath/$name"
                collectImages(child, next, out)
            }
        } else if (node.isFile) {
            val name = node.name ?: return
            val lower = name.lowercase(Locale.getDefault())
            if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
                val rel = if (relativePath.isEmpty()) name else relativePath
                out.add(node to rel)
            }
        }
    }

    private fun pickSparseModelDir(base: File): File? {
        val preferred = File(base, "0")
        if (preferred.isDirectory) return preferred
        return base.listFiles()?.firstOrNull { it.isDirectory }
    }

    private suspend fun appendKotlinLog(line: String) {
        withContext(Dispatchers.Main) {
            kotlinLog += line
            binding.txtLog.text = kotlinLog
        }
    }

    private fun startTimer() {
        timerJob?.cancel()
        timerJob = CoroutineScope(Dispatchers.Main).launch {
            while (isActive) {
                val elapsed = ((System.currentTimeMillis() - startTimeMs) / 1000).toInt()
                val mm = (elapsed / 60) % 60
                val hh = elapsed / 3600
                val ss = elapsed % 60
                val text = if (hh > 0) {
                    String.format("Elapsed: %02d:%02d:%02d", hh, mm, ss)
                } else {
                    String.format("Elapsed: %02d:%02d", mm, ss)
                }
                binding.txtElapsed.text = text
                delay(1000)
            }
        }
    }

    private fun stopTimer() {
        timerJob?.cancel()
        timerJob = null
    }

    private fun startTailingLog(logFile: File) {
        tailJob?.cancel()
        tailJob = ioScope.launch {
            var lastLen = -1L
            while (isActive) {
                if (logFile.exists()) {
                    val len = logFile.length()
                    if (len != lastLen) {
                        val text = runCatching { logFile.readText() }.getOrDefault("")
                        lastLen = len
                        withContext(Dispatchers.Main) {
                            binding.txtLog.text = kotlinLog + "\n" + text
                        }
                    }
                }
                delay(1000)
            }
        }
    }

    private fun stopTailingLog() {
        tailJob?.cancel()
        tailJob = null
    }

    private fun startCpuMonitor() {
        cpuJob?.cancel()
        cpuJob = ioScope.launch {
            val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
            var lastTotal = -1L
            var lastIdle = -1L
            var lastProcMs = -1L
            var lastWallMs = -1L
            while (isActive) {
                val sysPair = readSystemCpu()
                val total = sysPair?.first ?: -1
                val idleAll = sysPair?.second ?: -1
                val procMs = Process.getElapsedCpuTime()
                val wallMs = SystemClock.elapsedRealtime()

                var sysPct: Int? = null
                var appPct: Int? = null
                if (lastTotal >= 0 && lastIdle >= 0 && total > lastTotal && idleAll >= lastIdle) {
                    val totald = (total - lastTotal).toFloat()
                    val idled = (idleAll - lastIdle).toFloat()
                    val used = (1f - idled / totald) * 100f
                    sysPct = used.coerceIn(0f, 100f).toInt()
                }
                if (lastProcMs >= 0 && lastWallMs >= 0) {
                    val dProc = (procMs - lastProcMs).toFloat()
                    val dWall = (wallMs - lastWallMs).toFloat().coerceAtLeast(1f)
                    val pct = (dProc / (dWall * cores)) * 100f
                    appPct = pct.coerceIn(0f, 100f).toInt()
                }

                lastTotal = if (total >= 0) total else lastTotal
                lastIdle = if (idleAll >= 0) idleAll else lastIdle
                lastProcMs = procMs
                lastWallMs = wallMs

                withContext(Dispatchers.Main) {
                    val sysText = sysPct?.let { "$it%" } ?: "--"
                    val appText = appPct?.let { "$it%" } ?: "--"
                    binding.txtCpu.text = "CPU: sys $sysText | app $appText"
                }
                delay(1000)
            }
        }
    }

    private fun stopCpuMonitor() {
        cpuJob?.cancel()
        cpuJob = null
    }

    private fun startRamMonitor() {
        ramJob?.cancel()
        ramJob = ioScope.launch {
            while (isActive) {
                val sysMem = readSystemRam()
                val appKb = readAppRamKb()

                withContext(Dispatchers.Main) {
                    val sysText = sysMem?.let { (total, used) ->
                        if (total > 0L && used >= 0L) {
                            val usedMb = used / (1024 * 1024)
                            val totalMb = total / (1024 * 1024)
                            val pct = ((used * 100f) / total).toInt().coerceIn(0, 100)
                            "sys ${usedMb}MB/${totalMb}MB (${pct}%)"
                        } else {
                            "--"
                        }
                    } ?: "--"

                    val appText = appKb?.let { kb ->
                        val mb = (kb / 1024).coerceAtLeast(0)
                        "${mb}MB"
                    } ?: "--"

                    binding.txtRam.text = "RAM: $sysText | app $appText"
                }

                delay(1000)
            }
        }
    }

    private fun stopRamMonitor() {
        ramJob?.cancel()
        ramJob = null
    }

    private fun readSystemCpu(): Pair<Long, Long>? {
        return try {
            val line = File("/proc/stat").useLines { seq -> seq.firstOrNull { it.startsWith("cpu ") } }
            if (line == null) return null
            val parts = line.trim().split(Regex("\\s+"))
            if (parts.size < 5) return null
            fun at(i: Int): Long = parts.getOrNull(i)?.toLongOrNull() ?: 0L
            val user = at(1)
            val nice = at(2)
            val system = at(3)
            val idle = at(4)
            val iowait = at(5)
            val irq = at(6)
            val softirq = at(7)
            val steal = at(8)
            val idleAll = idle + iowait
            val nonIdle = user + nice + system + irq + softirq + steal
            val total = idleAll + nonIdle
            total to idleAll
        } catch (_: Throwable) {
            null
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        stopTimer()
        stopTailingLog()
        stopCpuMonitor()
        stopRamMonitor()
        ioScope.cancel()
        _binding = null
    }

    private fun readSystemRam(): Pair<Long, Long>? {
        return try {
            val ctx = context ?: return null
            val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return null
            val info = ActivityManager.MemoryInfo()
            am.getMemoryInfo(info)
            val total = info.totalMem
            val used = total - info.availMem
            total to used
        } catch (_: Throwable) {
            null
        }
    }

    private fun readAppRamKb(): Long? {
        return try {
            val ctx = context ?: return null
            val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return null
            val infos = am.getProcessMemoryInfo(intArrayOf(Process.myPid()))
            if (infos.isNotEmpty()) infos[0].totalPss.toLong() else null
        } catch (_: Throwable) {
            null
        }
    }

    companion object {
        private const val REQ_OPEN_TREE = 2001
        private const val TAG = "ColmapFragment"
    }
}
