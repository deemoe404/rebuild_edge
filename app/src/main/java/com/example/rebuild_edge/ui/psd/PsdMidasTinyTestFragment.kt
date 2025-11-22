package com.example.rebuild_edge.ui.psd

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
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.rebuild_edge.databinding.FragmentPsdMidasTinyBinding
import com.example.rebuild_edge.util.NpyReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class PsdMidasTinyTestFragment : Fragment() {

    private var _binding: FragmentPsdMidasTinyBinding? = null
    private val binding get() = _binding!!
    private val ortEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private var modelSession: OrtSession? = null
    private var selectedImageUri: Uri? = null
    private var selectedSparseUri: Uri? = null
    private var cachedModelFile: File? = null

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        requireActivity().contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        selectedImageUri = uri
        binding.txtSelectedImage.text = "Selected RGB: ${getDisplayName(uri)}"
    }

    private val pickSparseLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        requireActivity().contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        selectedSparseUri = uri
        binding.txtSelectedSparse.text = "Selected sparse depth: ${getDisplayName(uri)}"
        binding.txtSparseStats.text = "Sparse depth: pending (run inference to inspect)"
    }

    private val pickModelLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        val ctx = context ?: return@registerForActivityResult
        ctx.contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        binding.txtStatus.text = "Copying PSD Tiny model..."
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
        _binding = FragmentPsdMidasTinyBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.btnSelectImage.setOnClickListener { pickImageLauncher.launch(arrayOf("image/*")) }
        binding.btnSelectSparse.setOnClickListener { pickSparseLauncher.launch(arrayOf("*/*")) }
        binding.btnSelectModel.setOnClickListener {
            pickModelLauncher.launch(arrayOf("application/octet-stream", "application/onnx", "*/*"))
        }
        binding.btnRunInference.setOnClickListener { runInference() }
        refreshModelHint()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        modelSession?.close()
        modelSession = null
        ortEnvironment.close()
        _binding = null
    }

    private fun runInference() {
        val imageUri = selectedImageUri
        val sparseUri = selectedSparseUri
        val ctx = context ?: return
        if (imageUri == null) {
            binding.txtStatus.text = "Select an RGB image before running inference."
            return
        }
        if (sparseUri == null) {
            binding.txtStatus.text = "Select a sparse depth .npy file before running inference."
            return
        }
        val width = binding.editWidth.text.toString().toIntOrNull()?.coerceAtLeast(1)
        val height = binding.editHeight.text.toString().toIntOrNull()?.coerceAtLeast(1)
        if (width == null || height == null) {
            binding.txtStatus.text = "Enter positive numbers for width/height."
            return
        }
        if (width != 256 || height != 256) {
            binding.txtStatus.text =
                "PSD MiDaS Tiny ONNX was exported for 256x256 input. Please set both width and height to 256."
            return
        }
        if (locateModelFile() == null) {
            binding.txtStatus.text = "PSD Tiny model missing; select a .onnx file first."
            return
        }
        val align = binding.switchAlign.isChecked
        val startWall = SystemClock.elapsedRealtime()
        val startCpu = Process.getElapsedCpuTime()
        val memBefore = usedMemory()

        binding.btnRunInference.isEnabled = false
        binding.txtStatus.text = "Running PSD MiDaS Tiny inference..."

        viewLifecycleOwner.lifecycleScope.launch {
            val result = try {
                withContext(Dispatchers.IO) {
                    runModel(ctx.applicationContext, imageUri, sparseUri, width, height, align)
                }
            } catch (e: Throwable) {
                Log.e(TAG, "PSD MiDaS inference failed", e)
                binding.txtStatus.text = "Error: ${e.localizedMessage ?: e::class.simpleName}"
                binding.btnRunInference.isEnabled = true
                return@launch
            }

            val elapsed = SystemClock.elapsedRealtime() - startWall
            val cpuElapsed = Process.getElapsedCpuTime() - startCpu
            val memAfter = usedMemory()

            binding.txtElapsed.text = "Elapsed: ${elapsed} ms"
            binding.txtCpu.text = "CPU (ms): $cpuElapsed"
            binding.txtMemory.text =
                "Memory: ${formatBytes(memAfter)} (${formatDelta(memAfter - memBefore)})"
            binding.txtDepthRange.text = "Depth range (${result.depthWidth}x${result.depthHeight}): min=${formatDepth(result.minDepth)}, max=${formatDepth(result.maxDepth)}"
            val sparse = result.sparseStats
            binding.txtSparseStats.text = "Sparse depth (${sparse.width}x${sparse.height}): points=${sparse.validCount}, range=${formatDepth(sparse.minDepth)}~${formatDepth(sparse.maxDepth)}"
            binding.imgPreview.setImageBitmap(result.preview)
            binding.imgPreview.visibility = View.VISIBLE
            val readySuffix = when {
                result.aligned && result.alignmentNote.isNullOrEmpty() -> "aligned"
                result.aligned -> "aligned (${result.alignmentNote})"
                result.alignmentNote.isNullOrEmpty() -> "raw depth (unaligned)"
                else -> "raw depth (${result.alignmentNote})"
            }
            binding.txtStatus.text = "PSD MiDaS Tiny ready · $readySuffix"
            binding.btnRunInference.isEnabled = true
        }
    }

    private suspend fun runModel(
        appContext: Context,
        imageUri: Uri,
        sparseUri: Uri,
        width: Int,
        height: Int,
        alignWithSparse: Boolean
    ): InferenceResult {
        val session = modelSession ?: createSession() ?: throw IllegalStateException("Model file not found. See the hint above.")
        val rgbBitmap = loadBitmap(appContext, imageUri)
        val scaled = Bitmap.createScaledBitmap(rgbBitmap, width, height, true)
        if (rgbBitmap !== scaled) {
            rgbBitmap.recycle()
        }
        val inputData = bitmapToInputArray(scaled, width, height)
        val shape = longArrayOf(1L, 3L, height.toLong(), width.toLong())
        val inputName = session.inputNames.firstOrNull() ?: throw IllegalStateException("Model input missing")
        val sparse = loadSparseDepth(appContext, sparseUri)
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(inputData), shape)
        val rawDepth: FloatArray
        val outputWidth: Int
        val outputHeight: Int
        inputTensor.use { tensorInput ->
            session.run(mapOf(inputName to tensorInput)).use { outputs ->
                val depthTensor = outputs[0] as? OnnxTensor
                    ?: throw IllegalStateException("PSD MiDaS did not return a tensor")
                depthTensor.use { tensor ->
                    val buffer = tensor.floatBuffer
                    buffer.rewind()
                    val arr = FloatArray(buffer.remaining())
                    buffer.get(arr)
                    val dims = resolveSpatialDims(tensor.info?.shape, width, height, arr.size)
                    outputWidth = dims.first
                    outputHeight = dims.second
                    if (outputWidth * outputHeight != arr.size) {
                        throw IllegalStateException("Unexpected ONNX output shape: ${arr.size} values cannot be reshaped into ${outputWidth}x${outputHeight}.")
                    }
                    rawDepth = arr
                }
            }
        }
        val resized = resizeFloatArray(rawDepth, outputWidth, outputHeight, sparse.width, sparse.height)
        val alignment = if (alignWithSparse) alignInverseDepth(resized, sparse) else AlignmentResult(
            depth = resized,
            aligned = false,
            note = "Alignment disabled"
        )
        val preview = depthToBitmap(alignment.depth, sparse.width, sparse.height)
        val (minDepth, maxDepth) = computeDepthRange(alignment.depth)
        return InferenceResult(
            preview = preview,
            minDepth = minDepth,
            maxDepth = maxDepth,
            depthWidth = sparse.width,
            depthHeight = sparse.height,
            sparseStats = sparse,
            aligned = alignment.aligned,
            alignmentNote = alignment.note
        )
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

    private fun loadBitmap(context: Context, uri: Uri): Bitmap {
        val stream: InputStream = context.contentResolver.openInputStream(uri)
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

    private fun loadSparseDepth(context: Context, uri: Uri): SparseDepth {
        val resolver = context.contentResolver
        val result = resolver.openInputStream(uri)?.use { input ->
            NpyReader.read(input)
        } ?: throw IllegalStateException("Cannot read sparse depth file")
        val stats = computeDepthRange(result.data)
        return SparseDepth(
            width = result.width,
            height = result.height,
            data = result.data,
            validCount = countValidDepth(result.data),
            minDepth = stats.first,
            maxDepth = stats.second
        )
    }

    private fun resizeFloatArray(
        data: FloatArray,
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int
    ): FloatArray {
        if (srcWidth == dstWidth && srcHeight == dstHeight) {
            return data.copyOf()
        }
        if (srcWidth <= 0 || srcHeight <= 0 || dstWidth <= 0 || dstHeight <= 0) {
            return FloatArray(dstWidth * dstHeight)
        }
        val output = FloatArray(dstWidth * dstHeight)
        val scaleX = if (dstWidth == 1) 0f else (srcWidth - 1).toFloat() / (dstWidth - 1).toFloat()
        val scaleY = if (dstHeight == 1) 0f else (srcHeight - 1).toFloat() / (dstHeight - 1).toFloat()
        for (y in 0 until dstHeight) {
            val srcY = min(scaleY * y, (srcHeight - 1).toFloat())
            val y0 = srcY.toInt().coerceIn(0, srcHeight - 1)
            val y1 = min(y0 + 1, srcHeight - 1)
            val yLerp = srcY - y0
            for (x in 0 until dstWidth) {
                val srcX = min(scaleX * x, (srcWidth - 1).toFloat())
                val x0 = srcX.toInt().coerceIn(0, srcWidth - 1)
                val x1 = min(x0 + 1, srcWidth - 1)
                val xLerp = srcX - x0
                val topLeft = data[y0 * srcWidth + x0]
                val topRight = data[y0 * srcWidth + x1]
                val bottomLeft = data[y1 * srcWidth + x0]
                val bottomRight = data[y1 * srcWidth + x1]
                val top = topLeft + (topRight - topLeft) * xLerp
                val bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp
                val value = top + (bottom - top) * yLerp
                output[y * dstWidth + x] = value
            }
        }
        return output
    }

    private fun alignInverseDepth(prediction: FloatArray, sparse: SparseDepth): AlignmentResult {
        if (sparse.validCount < MIN_SPARSE_SAMPLES) {
            return AlignmentResult(
                depth = prediction.copyOf(),
                aligned = false,
                note = "Not enough sparse points (${sparse.validCount})"
            )
        }
        if (!sparse.minDepth.isFinite() || !sparse.maxDepth.isFinite() || sparse.maxDepth <= sparse.minDepth) {
            return AlignmentResult(
                depth = prediction.copyOf(),
                aligned = false,
                note = "Invalid sparse depth range"
            )
        }
        val total = prediction.size
        val mask = BooleanArray(total)
        var active = 0
        for (i in 0 until total) {
            val depth = sparse.data[i]
            if (depth.isFinite() && depth > 0f) {
                mask[i] = true
                active += 1
            }
        }
        if (active < MIN_SPARSE_SAMPLES) {
            return AlignmentResult(
                depth = prediction.copyOf(),
                aligned = false,
                note = "Sparse depth has too few valid pixels ($active)"
            )
        }
        val safeMin = max(sparse.minDepth.toDouble(), 1e-6)
        val safeMax = max(sparse.maxDepth.toDouble(), safeMin + 1e-6)
        var a00 = 0.0
        var a01 = 0.0
        var a11 = 0.0
        var b0 = 0.0
        var b1 = 0.0
        val predictionDouble = DoubleArray(total) { prediction[it].toDouble() }
        for (i in 0 until total) {
            if (!mask[i]) continue
            val depth = sparse.data[i].toDouble()
            val clipped = depth.coerceIn(safeMin, safeMax)
            val inv = 1.0 / max(clipped, 1e-6)
            val pred = predictionDouble[i]
            a00 += pred * pred
            a01 += pred
            a11 += 1.0
            b0 += pred * inv
            b1 += inv
        }
        if (a11 <= 0.0) {
            return AlignmentResult(
                depth = prediction.copyOf(),
                aligned = false,
                note = "Degenerate alignment system"
            )
        }
        val det = (a00 * a11) - (a01 * a01)
        if (det <= 0.0) {
            return AlignmentResult(
                depth = prediction.copyOf(),
                aligned = false,
                note = "Alignment determinant <= 0"
            )
        }
        val scale = ((a11 * b0) - (a01 * b1)) / det
        val shift = ((-a01 * b0) + (a00 * b1)) / det
        val depth = FloatArray(total)
        val minInverse = 1.0 / safeMax
        val maxInverse = 1.0 / safeMin
        for (i in 0 until total) {
            val inv = scale * predictionDouble[i] + shift
            val clippedInv = inv.coerceIn(minInverse, maxInverse)
            val depthVal = (1.0 / max(clippedInv, 1e-6)).toFloat()
            depth[i] = depthVal
        }
        return AlignmentResult(depth = depth, aligned = true, note = null)
    }

    private fun depthToBitmap(depth: FloatArray, width: Int, height: Int): Bitmap {
        val (minDepth, maxDepth) = computeDepthRange(depth)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        val validRange = if (minDepth.isFinite() && maxDepth.isFinite() && maxDepth > minDepth) {
            maxDepth - minDepth
        } else {
            Float.NaN
        }
        val hsv = floatArrayOf(240f, 1f, 1f)
        for (i in depth.indices) {
            val value = depth[i]
            val color = if (value.isFinite() && value > 0f && !validRange.isNaN()) {
                val norm = ((value - minDepth) / validRange).coerceIn(0f, 1f)
                hsv[0] = 240f - (240f * norm)
                Color.HSVToColor(hsv)
            } else {
                Color.argb(255, 32, 32, 32)
            }
            pixels[i] = color
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
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

    private fun usedMemory(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    private fun formatBytes(bytes: Long): String {
        val absBytes = abs(bytes)
        return when {
            absBytes >= 1024L * 1024L -> String.format("%.1f MB", absBytes / (1024f * 1024f))
            absBytes >= 1024L -> String.format("%.1f KB", absBytes / 1024f)
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

    private fun countValidDepth(values: FloatArray): Int {
        var count = 0
        for (value in values) {
            if (value.isFinite() && value > 0f) {
                count += 1
            }
        }
        return count
    }

    private fun computeDepthRange(values: FloatArray): Pair<Float, Float> {
        var minVal = Float.POSITIVE_INFINITY
        var maxVal = Float.NEGATIVE_INFINITY
        for (value in values) {
            if (value.isFinite() && value > 0f) {
                if (value < minVal) minVal = value
                if (value > maxVal) maxVal = value
            }
        }
        if (!minVal.isFinite() || !maxVal.isFinite()) {
            return Float.NaN to Float.NaN
        }
        return minVal to maxVal
    }

    private fun resolveSpatialDims(
        shape: LongArray?,
        fallbackWidth: Int,
        fallbackHeight: Int,
        valueCount: Int
    ): Pair<Int, Int> {
        val dims = shape?.filter { it > 0 && it <= Int.MAX_VALUE } ?: emptyList()
        if (dims.size >= 2) {
            val height = dims[dims.size - 2].toInt()
            val width = dims.last().toInt()
            if (height > 0 && width > 0 && height.toLong() * width.toLong() == valueCount.toLong()) {
                return width to height
            }
        }
        if (fallbackWidth > 0 && fallbackHeight > 0 && fallbackWidth * fallbackHeight == valueCount) {
            return fallbackWidth to fallbackHeight
        }
        return valueCount to 1
    }

    private data class SparseDepth(
        val width: Int,
        val height: Int,
        val data: FloatArray,
        val validCount: Int,
        val minDepth: Float,
        val maxDepth: Float
    )

    private data class AlignmentResult(
        val depth: FloatArray,
        val aligned: Boolean,
        val note: String?
    )

    private data class InferenceResult(
        val preview: Bitmap,
        val minDepth: Float,
        val maxDepth: Float,
        val depthWidth: Int,
        val depthHeight: Int,
        val sparseStats: SparseDepth,
        val aligned: Boolean,
        val alignmentNote: String?
    )

    companion object {
        private const val TAG = "PsdMidasTinyTest"
        private const val MODEL_NAME = "psd_nk_midas_swin2t.onnx"
        private const val MIN_SPARSE_SAMPLES = 10
    }
}
