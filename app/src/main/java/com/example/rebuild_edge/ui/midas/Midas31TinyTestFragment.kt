package com.example.rebuild_edge.ui.midas

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.provider.OpenableColumns
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.core.content.FileProvider
import com.example.rebuild_edge.databinding.FragmentMidas3TinyBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.FloatBuffer
import kotlin.math.abs

class Midas31TinyTestFragment : Fragment() {
    private var _binding: FragmentMidas3TinyBinding? = null
    private val binding get() = _binding!!
    private val ortEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private var modelSession: OrtSession? = null
    private var selectedImageUri: Uri? = null
    private var lastDepthBitmap: Bitmap? = null
    private val TAG = "Midas31TinyTestFragment"
    private val MODEL_NAME = "dpt_swin2_tiny_256.onnx"
    private var cachedModelFile: File? = null
    private var lastSharedDepthFile: File? = null

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        requireContext().contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        selectedImageUri = uri
        binding.txtSelectedImage.text = "Selected: ${getDisplayName(uri)}"
        binding.imgPreview.setImageURI(uri)
        binding.imgPreview.visibility = View.VISIBLE
    }

    private val pickModelLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        val ctx = context ?: return@registerForActivityResult
        ctx.contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        binding.txtStatus.text = "Copying MiDaS 3.1 Tiny model..."
        viewLifecycleOwner.lifecycleScope.launch {
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    copyModelFile(ctx, uri)
                }
            }
            result.onSuccess {
                binding.txtStatus.text = "Updated model: ${it.name}"
            }.onFailure {
                Log.e(TAG, "model copy failed", it)
                binding.txtStatus.text = "Model copy failed: ${it.localizedMessage ?: it::class.simpleName}"
            }
            refreshModelHint()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMidas3TinyBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.btnSelectImage.setOnClickListener {
            pickImageLauncher.launch(arrayOf("image/*"))
        }
        binding.btnRunInference.setOnClickListener {
            runInference()
        }
        binding.btnShareDepth.setOnClickListener {
            shareDepthBitmap()
        }
        binding.btnSelectModel.setOnClickListener {
            pickModelLauncher.launch(arrayOf("application/octet-stream", "application/onnx", "*/*"))
        }
        refreshModelHint()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        modelSession?.close()
        ortEnvironment.close()
        _binding = null
    }

    private fun runInference() {
        val uri = selectedImageUri
        if (uri == null) {
            binding.txtStatus.text = "Select an image before running inference."
            return
        }
        val width = binding.editWidth.text.toString().toIntOrNull()?.coerceAtLeast(1)
        val height = binding.editHeight.text.toString().toIntOrNull()?.coerceAtLeast(1)
        if (width == null || height == null) {
            binding.txtStatus.text = "Enter positive numbers for width/height."
            return
        }
        if (width != 256 || height != 256) {
            binding.txtStatus.text = "MiDaS 3.1 Tiny requires 256×256 input."
            return
        }
        if (locateModelFile() == null) {
            binding.txtStatus.text = "MiDaS 3.1 Tiny model missing; select a .onnx file first."
            return
        }
        binding.btnShareDepth.isEnabled = false
        binding.txtDepthPreviewLabel.visibility = View.GONE
        binding.imgDepthMap.visibility = View.GONE
        val startCpu = Process.getElapsedCpuTime()
        val startWall = SystemClock.elapsedRealtime()
        val memBefore = usedMemory()

        binding.btnRunInference.isEnabled = false
        binding.txtStatus.text = "Running MiDaS 3.1 Tiny inference..."

        viewLifecycleOwner.lifecycleScope.launch {
            val result = try {
                withContext(Dispatchers.IO) {
                    runModel(uri, width, height)
                }
            } catch (e: Throwable) {
                Log.e(TAG, "inference failed", e)
                binding.txtStatus.text = "Error: ${e.localizedMessage ?: e::class.simpleName}"
                binding.btnRunInference.isEnabled = true
                return@launch
            }

            val elapsed = SystemClock.elapsedRealtime() - startWall
            val cpuElapsed = Process.getElapsedCpuTime() - startCpu
            val memAfter = usedMemory()

            binding.txtElapsed.text = "Elapsed: ${elapsed} ms"
            binding.txtCpu.text = "CPU (ms): $cpuElapsed"
            binding.txtMemory.text = "Memory: ${formatBytes(memAfter)} (${formatDelta(memAfter - memBefore)})"
            binding.txtDepthRange.text = "Depth range: min=${formatDepth(result.minDepth)}, max=${formatDepth(result.maxDepth)}"
            binding.imgPreview.setImageBitmap(result.preview)
            binding.imgPreview.visibility = View.VISIBLE
            lastDepthBitmap = result.depthBitmap
            binding.imgDepthMap.setImageBitmap(result.depthBitmap)
            binding.imgDepthMap.visibility = View.VISIBLE
            binding.txtDepthPreviewLabel.visibility = View.VISIBLE
            binding.btnShareDepth.isEnabled = true
            binding.txtStatus.text = "MiDaS 3.1 Tiny ready"
            binding.btnRunInference.isEnabled = true
        }
    }

    private suspend fun runModel(uri: Uri, width: Int, height: Int): InferenceResult {
        val session = modelSession ?: createSession() ?: throw IllegalStateException("Model file not found. See the hint above.")
        val bitmap = loadBitmap(uri)
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val inputData = bitmapToInputArray(scaled, width, height)

        val shape = longArrayOf(1L, 3L, height.toLong(), width.toLong())
        val inputName = session.inputNames.firstOrNull() ?: throw IllegalStateException("Model input missing")
        OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(inputData), shape).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { outputs ->
                val depthTensor = outputs[0] as? OnnxTensor
                    ?: throw IllegalStateException("MiDaS 3.1 Tiny did not return a tensor")
                depthTensor.use { tensor ->
                    val buffer = tensor.floatBuffer
                    buffer.rewind()
                    val depthValues = FloatArray(buffer.remaining())
                    buffer.get(depthValues)
                    val finite = depthValues.filter { it.isFinite() }
                    val minDepth = finite.minOrNull() ?: Float.NaN
                    val maxDepth = finite.maxOrNull() ?: Float.NaN
                    val depthBitmap = createDepthBitmap(depthValues, width, height, minDepth, maxDepth)
                    return InferenceResult(
                        minDepth = minDepth,
                        maxDepth = maxDepth,
                        preview = scaled,
                        depthBitmap = depthBitmap
                    )
                }
            }
        }
    }

    private fun bitmapToInputArray(bitmap: Bitmap, width: Int, height: Int): FloatArray {
        val pixelData = IntArray(width * height)
        bitmap.getPixels(pixelData, 0, width, 0, 0, width, height)
        val area = width * height
        val result = FloatArray(area * 3)
        for (i in 0 until area) {
            val pixel = pixelData[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            result[i] = r
            result[i + area] = g
            result[i + area * 2] = b
        }
        return result
    }

    private fun createDepthBitmap(depthValues: FloatArray, width: Int, height: Int, minDepth: Float, maxDepth: Float): Bitmap {
        val safeMin = if (minDepth.isFinite()) minDepth else 0f
        val safeMax = if (maxDepth.isFinite()) maxDepth else safeMin + 1f
        val range = (safeMax - safeMin).takeIf { it > 0f } ?: 1f
        val pixels = IntArray(width * height)
        for (i in depthValues.indices) {
            val value = depthValues[i]
            val intensity = if (value.isFinite()) {
                ((value - safeMin) / range).coerceIn(0f, 1f)
            } else {
                0f
            }
            val gray = (intensity * 255).toInt().coerceIn(0, 255)
            pixels[i] = Color.rgb(gray, gray, gray)
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun createSession(): OrtSession? {
        val modelFile = locateModelFile() ?: return null
        return ortEnvironment.createSession(modelFile.absolutePath, OrtSession.SessionOptions()).also {
            modelSession = it
        }
    }

    private fun locateModelFile(): File? {
        cachedModelFile?.takeIf { it.exists() }?.let { return it }
        val ctx = context ?: return null
        val external = ctx.getExternalFilesDir("checkpoints")
        val candidates = listOfNotNull(
            external?.let { File(it, MODEL_NAME) },
            File(ctx.filesDir, MODEL_NAME)
        )
        return candidates.firstOrNull { it.exists() }?.also { cachedModelFile = it }
    }

    private fun refreshModelHint() {
        val ctx = context ?: return
        val externalDir = ctx.getExternalFilesDir("checkpoints") ?: ctx.filesDir
        val expected = File(externalDir, MODEL_NAME)
        val fallback = File(ctx.filesDir, MODEL_NAME)
        val found = cachedModelFile?.takeIf { it.exists() }
            ?: expected.takeIf { it.exists() }
            ?: fallback.takeIf { it.exists() }
        val display = found ?: expected
        val status = if (found != null) "ready" else "missing"
        binding.txtModelPath.text = "Model file: ${display.absolutePath} ($status)"
    }

    private fun loadBitmap(uri: Uri): Bitmap {
        val stream: InputStream = requireContext().contentResolver.openInputStream(uri)
            ?: throw IllegalStateException("Cannot open selected image")
        stream.use {
            return BitmapFactory.decodeStream(it)
                ?: throw IllegalStateException("Unable to decode selected image")
        }
    }

    private fun copyModelFile(ctx: Context, uri: Uri): File {
        val baseDir = ctx.getExternalFilesDir("checkpoints") ?: ctx.filesDir
        baseDir.mkdirs()
        val dest = File(baseDir, MODEL_NAME)
        ctx.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dest).use { output ->
                input.copyTo(output)
            }
        } ?: throw IllegalStateException("Could not read selected model file")
        cachedModelFile = dest
        modelSession?.close()
        modelSession = null
        return dest
    }

    private fun shareDepthBitmap() {
        val bitmap = lastDepthBitmap ?: return
        val ctx = requireContext()
        val shareFile = File(ctx.cacheDir, "midas_depth_${System.currentTimeMillis()}.png")
        val written = runCatching {
            lastSharedDepthFile?.takeIf { it.exists() }?.delete()
            FileOutputStream(shareFile).use { bitmap.compress(Bitmap.CompressFormat.PNG, 90, it) }
            shareFile
        }.getOrNull()
        if (written == null) {
            Toast.makeText(ctx, "无法导出 MiDaS 深度图", Toast.LENGTH_SHORT).show()
            return
        }
        lastSharedDepthFile = written
        val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".fileprovider", written)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Share MiDaS depth map"))
    }

    private fun formatBytes(bytes: Long): String {
        val absBytes = abs(bytes)
        return when {
            absBytes >= 1024 * 1024 -> String.format("%.1f MB", absBytes / (1024f * 1024f))
            absBytes >= 1024 -> String.format("%.1f KB", absBytes / 1024f)
            else -> "$absBytes B"
        }
    }

    private fun formatDepth(value: Float): String {
        return if (value.isFinite()) String.format("%.3f", value) else "--"
    }

    private fun formatDelta(delta: Long): String {
        val sign = if (delta >= 0) "+" else "-"
        return "$sign${formatBytes(delta)}"
    }

    private fun usedMemory(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    private fun getDisplayName(uri: Uri): String {
        val cursor = requireContext().contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null
        )
        cursor?.use {
            if (it.moveToFirst()) {
                val index = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) {
                    return it.getString(index)
                }
            }
        }
        return uri.lastPathSegment ?: uri.toString()
    }

    private data class InferenceResult(
        val minDepth: Float,
        val maxDepth: Float,
        val preview: Bitmap,
        val depthBitmap: Bitmap
    )
}
