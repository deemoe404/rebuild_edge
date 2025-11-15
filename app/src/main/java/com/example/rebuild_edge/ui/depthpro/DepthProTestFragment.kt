package com.example.rebuild_edge.ui.depthpro

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.provider.OpenableColumns
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.rebuild_edge.databinding.FragmentDepthproBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.FloatBuffer
import kotlin.math.abs

class DepthProTestFragment : Fragment() {
    private var _binding: FragmentDepthproBinding? = null
    private val binding get() = _binding!!
    private val ortEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private var modelSession: OrtSession? = null
    private var selectedImageUri: Uri? = null
    private val TAG = "DepthProTestFragment"
    private val MODEL_NAME = "model_uint8.onnx"
    private var cachedModelFile: File? = null

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        requireActivity().contentResolver.takePersistableUriPermission(
            uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
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
            uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        binding.txtStatus.text = "Copying DepthPro model..."
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
        _binding = FragmentDepthproBinding.inflate(inflater, container, false)
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
        if (locateModelFile() == null) {
            binding.txtStatus.text = "DepthPro model missing; select a .onnx file first."
            return
        }
        val startCpu = Process.getElapsedCpuTime()
        val startWall = SystemClock.elapsedRealtime()
        val memBefore = usedMemory()

        binding.btnRunInference.isEnabled = false
        binding.txtStatus.text = "Running DepthPro inference..."

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
            binding.txtStatus.text = "DepthPro ready"
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
                    ?: throw IllegalStateException("DepthPro did not return a tensor")
                depthTensor.use { tensor ->
                    val buffer = tensor.floatBuffer
                    buffer.rewind()
                    val depthValues = FloatArray(buffer.remaining())
                    buffer.get(depthValues)
                    val finite = depthValues.filter { it.isFinite() }
                    return InferenceResult(
                        minDepth = finite.minOrNull() ?: Float.NaN,
                        maxDepth = finite.maxOrNull() ?: Float.NaN,
                        preview = scaled
                    )
                }
            }
        }
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
            result[i] = (r - 0.5f) / 0.5f
            result[i + area] = (g - 0.5f) / 0.5f
            result[i + area * 2] = (b - 0.5f) / 0.5f
        }
        return result
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
        val preview: Bitmap
    )
}
