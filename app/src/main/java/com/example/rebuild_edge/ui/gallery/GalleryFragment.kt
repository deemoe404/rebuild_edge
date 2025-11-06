package com.example.rebuild_edge.ui.gallery

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.provider.DocumentsContract
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import androidx.fragment.app.Fragment
import com.example.rebuild_edge.SfmNative
import com.example.rebuild_edge.databinding.FragmentGalleryBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class GalleryFragment : Fragment() {

    private var _binding: FragmentGalleryBinding? = null
    private val binding get() = _binding!!

    private var pickedTree: Uri? = null
    private val ioScope = CoroutineScope(Dispatchers.IO + Job())
    private var timerJob: Job? = null
    private var tailJob: Job? = null
    private var cpuJob: Job? = null
    private var startTimeMs: Long = 0L
    private var kotlinLog: String = ""

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentGalleryBinding.inflate(inflater, container, false)
        val root: View = binding.root

        binding.btnSelect.setOnClickListener {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
                putExtra(DocumentsContract.EXTRA_INITIAL_URI, pickedTree)
            }
            startActivityForResult(intent, REQ_OPEN_TREE)
        }
        binding.btnRun.setOnClickListener {
            runSfm()
        }

        return root
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_OPEN_TREE) {
            val uri = data?.data ?: return
            requireContext().contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
            pickedTree = uri
            binding.textGallery.text = "Selected: ${uri}"
            binding.btnRun.isEnabled = true
        }
    }

    private fun runSfm() {
        val uri = pickedTree ?: return
        binding.btnRun.isEnabled = false
        kotlinLog = "Preparing images...\n"
        binding.txtLog.text = kotlinLog
        binding.txtElapsed.text = "Elapsed: 00:00"
        binding.txtCpu.text = "CPU: --"
        startTimer()
        startCpuMonitor()
        ioScope.launch {
            val cacheDir = File(requireContext().cacheDir, "sfm_input").apply {
                mkdirs()
            }
            // Clear previous
            cacheDir.listFiles()?.forEach { it.delete() }

            val doc = DocumentFile.fromTreeUri(requireContext(), uri)
            val imagePaths = mutableListOf<String>()
            val files = doc?.listFiles()?.sortedBy { it.name } ?: emptyList()
            var copied = 0
            val total = files.count { f ->
                val name = (f.name ?: "").lowercase()
                f.isFile && (name.endsWith(".jpg") || name.endsWith(".jpeg"))
            }
            appendKotlinLog("Found $total JPG images. Copying to cache...\n")
            files.forEach { f ->
                val name = (f.name ?: "").lowercase()
                if (f.isFile && (name.endsWith(".jpg") || name.endsWith(".jpeg"))) {
                    val dest = File(cacheDir, f.name!!)
                    requireContext().contentResolver.openInputStream(f.uri)?.use { ins ->
                        FileOutputStream(dest).use { outs -> ins.copyTo(outs) }
                    }
                    imagePaths.add(dest.absolutePath)
                    copied += 1
                    if (copied % 10 == 0 || copied == total) {
                        appendKotlinLog("Copied $copied / $total\n")
                    }
                }
            }
            val outDir = File(requireContext().getExternalFilesDir(null), "sfm_out").apply { mkdirs() }
            val logFile = File(outDir, "sfm_run.log")

            val align = binding.chkAlign.isChecked
            Log.d(TAG, "Selected ${imagePaths.size} images, outDir=${outDir.absolutePath}, alignGps=$align")
            val result = if (imagePaths.size >= 3) {
                try {
                    appendKotlinLog("Starting SFM (alignGps=$align) ...\n")
                    startTailingLog(logFile)
                    val maxLongEdge = runCatching { binding.editMaxLongEdge.text.toString().trim().toInt() }
                        .getOrDefault(2000)
                        .coerceAtLeast(0)
                    appendKotlinLog("Max long edge: $maxLongEdge px\n")
                    SfmNative.runSfm(imagePaths.toTypedArray(), outDir.absolutePath, align, maxLongEdge)
                } catch (t: Throwable) {
                    Log.e(TAG, "Native SFM crashed", t)
                    "Error: ${t.message}"
                }
            } else {
                "Need at least 3 JPG images in the folder"
            }
            withContext(Dispatchers.Main) {
                stopTailingLog()
                stopTimer()
                stopCpuMonitor()
                // If native produced a log, show Kotlin prep log + native log + result summary
                val nativeLog = if (logFile.exists()) runCatching { logFile.readText() }.getOrDefault("") else ""
                if (nativeLog.isNotEmpty()) {
                    binding.txtLog.text = kotlinLog + "\n" + nativeLog + "\n" + result
                } else {
                    binding.txtLog.text = kotlinLog + "\n" + result
                }
                Log.d(TAG, "SFM result:\n$result")
                binding.btnRun.isEnabled = true
            }
        }
    }

    private fun startTimer() {
        startTimeMs = System.currentTimeMillis()
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

    private suspend fun appendKotlinLog(line: String) {
        withContext(Dispatchers.Main) {
            kotlinLog += line
            binding.txtLog.text = kotlinLog
        }
    }

    private fun startCpuMonitor() {
        cpuJob?.cancel()
        cpuJob = ioScope.launch {
            val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
            var lastTotal: Long = -1
            var lastIdle: Long = -1
            var lastProcMs: Long = -1
            var lastWallMs: Long = -1
            while (isActive) {
                // System CPU from /proc/stat
                val sysPair = readSystemCpu() // total, idleAll
                val total = sysPair?.first ?: -1
                val idleAll = sysPair?.second ?: -1

                // Process CPU from Process.getElapsedCpuTime()
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
        } catch (_: Throwable) { null }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        stopTimer()
        stopTailingLog()
        stopCpuMonitor()
        ioScope.cancel()
        _binding = null
    }

    companion object {
        private const val REQ_OPEN_TREE = 1001
        private const val TAG = "GalleryFragment"
    }
}
