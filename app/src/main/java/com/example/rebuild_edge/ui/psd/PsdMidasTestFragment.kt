package com.example.rebuild_edge.ui.psd

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
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.rebuild_edge.databinding.FragmentPsdMidasBinding
import com.example.rebuild_edge.ui.psd.PsdDepthCompletionEngine.TensorData
import com.example.rebuild_edge.ui.psd.PsdNative
import com.example.rebuild_edge.util.NpyReader
import java.io.BufferedReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class PsdMidasTestFragment : Fragment() {

    private var _binding: FragmentPsdMidasBinding? = null
    private val binding get() = _binding!!

    private var engine: PsdDepthCompletionEngine? = null
    private var modelBundle: PsdDepthCompletionEngine.ModelBundle? = null
    private var meta: PsdDepthCompletionEngine.ModelMetadata? = null
    private var selectedImageUri: Uri? = null
    private var selectedSparseUri: Uri? = null
    private var latestDepth: FloatArray? = null
    private var latestWidth: Int = 0
    private var latestHeight: Int = 0
    private var latestRawFile: File? = null

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        requireActivity().contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        selectedImageUri = uri
        binding.txtSelectedImage.text = "RGB: ${getDisplayName(uri)}"
    }

    private val pickSparseLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        requireActivity().contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        selectedSparseUri = uri
        binding.txtSelectedSparse.text = "Sparse depth: ${getDisplayName(uri)}"
        binding.txtSparseStats.text = "Sparse depth: pending"
    }

    private val pickModelsLauncher = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNullOrEmpty()) return@registerForActivityResult
        val ctx = context ?: return@registerForActivityResult
        viewLifecycleOwner.lifecycleScope.launch {
            binding.txtStatus.text = "Copying model files..."
            val result = runCatching {
                withContext(Dispatchers.IO) { copyModels(ctx, uris) }
            }
            result.onSuccess { (bundle, metadata, names) ->
                modelBundle = bundle
                meta = metadata
                engine?.close()
                engine = null
                binding.txtModelPath.text = "Models: $names"
                binding.txtMeta.text = "Meta: ${metadata.dataset} rgb=${metadata.rgbWidth}x${metadata.rgbHeight} mde=${metadata.mdeWidth}x${metadata.mdeHeight} bins=${metadata.binCount}"
                binding.txtStatus.text = "Models ready"
                binding.editWidth.setText(metadata.mdeWidth.toString())
                binding.editHeight.setText(metadata.mdeHeight.toString())
            }.onFailure {
                Log.e(TAG, "copy models failed", it)
                binding.txtStatus.text = "Model copy failed: ${it.localizedMessage ?: it::class.simpleName}"
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentPsdMidasBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.btnSelectImage.setOnClickListener { pickImageLauncher.launch(arrayOf("image/*")) }
        binding.btnSelectSparse.setOnClickListener { pickSparseLauncher.launch(arrayOf("*/*")) }
        binding.btnSelectModel.setOnClickListener {
            pickModelsLauncher.launch(arrayOf("application/onnx", "application/octet-stream", "application/json", "*/*"))
        }
        binding.btnRunInference.setOnClickListener { runInference() }
        binding.btnExportDepth.setOnClickListener { shareDepth() }
    }

    override fun onDestroyView() {
        engine?.close()
        engine = null
        _binding = null
        super.onDestroyView()
    }

    private fun runInference() {
        val imageUri = selectedImageUri
        val sparseUri = selectedSparseUri
        val bundle = modelBundle
        val metadata = meta
        val ctx = context ?: return
        if (bundle == null || metadata == null) {
            binding.txtStatus.text = "Select ONNX (mde/res/head) + meta.json first."
            return
        }
        if (imageUri == null) {
            binding.txtStatus.text = "Select an RGB image."
            return
        }
        if (sparseUri == null) {
            binding.txtStatus.text = "Select a sparse depth .npy file."
            return
        }
        binding.btnRunInference.isEnabled = false
        binding.txtStatus.text = "Running PSD..."
        val startWall = SystemClock.elapsedRealtime()
        val startCpu = Process.getElapsedCpuTime()
        val memBefore = usedMemory()

        viewLifecycleOwner.lifecycleScope.launch {
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    runPipeline(ctx, bundle, metadata, imageUri, sparseUri)
                }
            }
            result.onSuccess { (preview, statsText, depthArr, w, h) ->
                latestDepth = depthArr
                latestWidth = w
                latestHeight = h
                latestRawFile = null
                val elapsed = SystemClock.elapsedRealtime() - startWall
                val cpuElapsed = Process.getElapsedCpuTime() - startCpu
                val memAfter = usedMemory()
                binding.imgPreview.setImageBitmap(preview)
                binding.imgPreview.visibility = View.VISIBLE
                binding.txtSparseStats.text = statsText
                binding.txtElapsed.text = "Elapsed: ${elapsed} ms"
                binding.txtCpu.text = "CPU (ms): $cpuElapsed"
                binding.txtMemory.text = "Memory: ${formatBytes(memAfter)} (${formatDelta(memAfter - memBefore)})"
                val depthRange = computeDepthRange(depthArr)
                binding.txtDepthRange.text = "Depth range ($w x $h): ${formatDepth(depthRange.first)} ~ ${formatDepth(depthRange.second)}"
                binding.txtStatus.text = "Done"
            }.onFailure {
                Log.e(TAG, "PSD pipeline failed", it)
                binding.txtStatus.text = "Error: ${it.localizedMessage ?: it::class.simpleName}"
            }
            binding.btnRunInference.isEnabled = true
        }
    }

    private fun shareDepth() {
        val ctx = context ?: return
        val depth = latestDepth ?: run {
            binding.txtStatus.text = "Run inference first."
            return
        }
        val w = latestWidth
        val h = latestHeight
        val file = latestRawFile ?: writeDepthRaw(ctx, depth, w, h).also { latestRawFile = it }
        val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".fileprovider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Share depth"))
    }

    private fun runPipeline(
        context: Context,
        bundle: PsdDepthCompletionEngine.ModelBundle,
        metadata: PsdDepthCompletionEngine.ModelMetadata,
        imageUri: Uri,
        sparseUri: Uri
    ): PipelineResult {
        val eng = engine ?: PsdDepthCompletionEngine().also { engine = it }.apply {
            loadModels(bundle, metadata)
        }

        val sparse = loadSparseDepth(context, sparseUri)
        val rgbBitmap = loadBitmap(context, imageUri)
        val mdeW = metadata.mdeWidth
        val mdeH = metadata.mdeHeight
        val rgbMde = Bitmap.createScaledBitmap(rgbBitmap, mdeW, mdeH, true)
        if (rgbBitmap !== rgbMde) rgbBitmap.recycle()
        val rgbTensor = bitmapToInputArray(rgbMde, mdeW, mdeH)
        val intrinsics = loadIntrinsics(context, sparseUri, sparse.width, sparse.height)
        val mdeOut = eng.runMidas(rgbTensor, intArrayOf(1, 3, mdeH, mdeW), intrinsics)
        val depthMde = mdeOut.depth
        val depthUpsampled = resizeFloatArray(depthMde.data, depthMde.shape[3], depthMde.shape[2], sparse.width, sparse.height)
        val depthInv = FloatArray(depthUpsampled.size) { idx -> 1f / max(depthUpsampled[idx], 1e-6f) }
        val aligned = eng.alignInverseDepthPolyfit(
            TensorData(depthInv, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            minDepth = 0.05f,
            maxDepth = 200f,
            adaptiveMinMax = true
        )
        val ipFilled = PsdNative.fillSparseFast(sparse.data, 1, sparse.height, sparse.width, 4)
        val ipMedian = eng.alignDepthMedian(
            TensorData(ipFilled, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            minDepth = sparse.minDepth.coerceAtLeast(0.05f),
            maxDepth = sparse.maxDepth.coerceAtLeast(sparse.minDepth + 1e-4f),
            adaptiveMinMax = true
        )
        val feat0Up = upsampleFeatureTo(mdeOut.path0, sparse.height, sparse.width)
        val depthDiff = dualDiffusionFull(
            eng,
            feat0Up,
            ipMedian,
            aligned,
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            intrinsics,
            knn = 3,
            scale = 4,
            iteration = 3
        )
        val sparseResidual = FloatArray(sparse.data.size) { i -> (sparse.data[i] - depthDiff.data[i]) }
        val residualOut = eng.runResidualBranch(
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            depthDiff,
            TensorData(sparseResidual, intArrayOf(1, 1, sparse.height, sparse.width)),
            arrayOf(mdeOut.path0, mdeOut.path1, mdeOut.path2, mdeOut.path3)
        )
        val depthResidual = FloatArray(depthDiff.data.size) { i -> depthDiff.data[i] + residualOut.residual.data[i] }
        val bins = eng.computeBins(
            TensorData(residualOut.residual.data, residualOut.residual.shape),
            residualOut.confidence
        )
        val laplace = eng.computeLaplace(
            TensorData(sparseResidual, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            bins
        )
        val headOut = eng.runHead(
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(depthResidual, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(sparseResidual, intArrayOf(1, 1, sparse.height, sparse.width)),
            laplace,
            residualOut.confidence,
            arrayOf(mdeOut.path0, mdeOut.path1, mdeOut.path2, mdeOut.path3),
            bins
        )
        val depthFinal = headOut.depthPix.data
        val preview = depthToBitmap(depthFinal, sparse.width, sparse.height)
        val sparseStats = "Sparse depth (${sparse.width}x${sparse.height}): points=${sparse.validCount}, range=${formatDepth(sparse.minDepth)}~${formatDepth(sparse.maxDepth)}"
        return PipelineResult(preview, sparseStats, depthFinal, sparse.width, sparse.height)
    }

    private fun upsampleFeatureTo(src: TensorData, targetH: Int, targetW: Int): TensorData {
        val c = src.shape[1]
        val h = src.shape[2]
        val w = src.shape[3]
        val out = FloatArray(c * targetH * targetW)
        for (ci in 0 until c) {
            val channel = FloatArray(h * w) { idx -> src.data[ci * h * w + idx] }
            val up = resizeFloatArray(channel, w, h, targetW, targetH)
            val base = ci * targetH * targetW
            System.arraycopy(up, 0, out, base, up.size)
        }
        return TensorData(out, intArrayOf(1, c, targetH, targetW))
    }

    private fun dualDiffusionFull(
        eng: PsdDepthCompletionEngine,
        feat: TensorData,
        ipMedian: TensorData,
        depthPolyfit: TensorData,
        sparse: TensorData,
        intrinsics: FloatArray,
        knn: Int,
        scale: Int,
        iteration: Int
    ): TensorData {
        val h = sparse.shape[2]
        val w = sparse.shape[3]
        val point = PsdNative.depthToPoint(depthPolyfit.data, intrinsics, 1, h, w)
        val sparseDown = eng.sparseDownSample(sparse, scale)
        val ipDown = eng.sparseDownSample(ipMedian, scale)
        val pointDown = eng.depthToPoint(ipDown, intrinsics.copyOf(), scaleDown = scale)
        val featDown = resizeFeature(feat, h / scale, w / scale)
        val depthKnnDown = PsdNative.knnPropagate(
            pointDown.data,
            featDown.data,
            ipDown.data,
            sparseDown.data,
            1,
            featDown.shape[1],
            h / scale,
            w / scale,
            knn,
            sparseDown.data.size
        )
        val depthCspnDown = eng.cspn(
            sparseDown,
            TensorData(depthKnnDown, intArrayOf(1, 1, h / scale, w / scale)),
            featDown,
            kernel = 3,
            iteration = iteration
        )
        val depthUp = resizeFloatArray(depthCspnDown.data, w / scale, h / scale, w, h)
        val featUp = resizeFeature(feat, h, w)
        return eng.cspn(
            sparse,
            TensorData(depthUp, intArrayOf(1, 1, h, w)),
            featUp,
            kernel = 3,
            iteration = iteration
        )
    }

    private fun resizeFeature(src: TensorData, targetH: Int, targetW: Int): TensorData {
        val c = src.shape[1]
        val h = src.shape[2]
        val w = src.shape[3]
        val out = FloatArray(c * targetH * targetW)
        for (ci in 0 until c) {
            val channel = FloatArray(h * w) { idx -> src.data[ci * h * w + idx] }
            val up = resizeFloatArray(channel, w, h, targetW, targetH)
            val base = ci * targetH * targetW
            System.arraycopy(up, 0, out, base, up.size)
        }
        return TensorData(out, intArrayOf(1, c, targetH, targetW))
    }

    private fun loadIntrinsics(context: Context, sparseUri: Uri, targetW: Int, targetH: Int): FloatArray {
        val camFile = findCameraFile(context, sparseUri)
        if (camFile != null && camFile.exists()) {
            runCatching {
                val model = loadCameraModel(camFile)
                val srcW = max((model.cx * 2.0).toInt(), 1)
                val srcH = max((model.cy * 2.0).toInt(), 1)
                val sx = targetW.toFloat() / srcW.toFloat()
                val sy = targetH.toFloat() / srcH.toFloat()
                return floatArrayOf(
                    (model.fx * sx).toFloat(), 0f, (model.cx * sx).toFloat(),
                    0f, (model.fy * sy).toFloat(), (model.cy * sy).toFloat(),
                    0f, 0f, 1f
                )
            }.onFailure {
                Log.w(TAG, "Failed to load camera_poses.json, fallback to default", it)
            }
        }
        return floatArrayOf(
            targetW.toFloat(), 0f, targetW / 2f,
            0f, targetH.toFloat(), targetH / 2f,
            0f, 0f, 1f
        )
    }

    private fun findCameraFile(context: Context, sparseUri: Uri): File? {
        val path = sparseUri.path ?: return null
        val file = if (path.startsWith("/")) File(path) else null
        val bases = mutableListOf<File>()
        file?.parentFile?.let { bases += it }
        file?.parentFile?.parentFile?.let { bases += it }
        val resolver = context.contentResolver
        if (file == null || !file.exists()) {
            // try copy to temp and inspect parent
            return null
        }
        bases.forEach { base ->
            val candidate = File(base, "camera_poses.json")
            if (candidate.exists()) return candidate
        }
        return null
    }

    private data class CameraModel(val fx: Double, val fy: Double, val cx: Double, val cy: Double)

    private fun loadCameraModel(file: File): CameraModel {
        val text = file.readText()
        val obj = JSONObject(text)
        val kArr = obj.optJSONArray("K") ?: throw IllegalStateException("camera_poses.json missing K")
        if (kArr.length() < 6) throw IllegalStateException("K array too short")
        val fx = kArr.optDouble(0)
        val fy = kArr.optDouble(4)
        val cx = kArr.optDouble(2)
        val cy = kArr.optDouble(5)
        return CameraModel(fx, fy, cx, cy)
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
            // cfg_swin2_tiny uses mean/std = 0.5
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
        if (srcWidth == dstWidth && srcHeight == dstHeight) return data.copyOf()
        if (srcWidth <= 0 || srcHeight <= 0 || dstWidth <= 0 || dstHeight <= 0) return FloatArray(dstWidth * dstHeight)
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

    private fun depthToBitmap(depth: FloatArray, width: Int, height: Int): Bitmap {
        val (minDepth, maxDepth) = computeDepthRange(depth)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        val validRange = if (minDepth.isFinite() && maxDepth.isFinite() && maxDepth > minDepth) {
            maxDepth - minDepth
        } else Float.NaN
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

    private fun copyModels(ctx: Context, uris: List<Uri>): Triple<PsdDepthCompletionEngine.ModelBundle, PsdDepthCompletionEngine.ModelMetadata, String> {
        val baseDir = File(ctx.filesDir, "psd_models").apply { mkdirs() }
        var mde: File? = null
        var residual: File? = null
        var head: File? = null
        var metaFile: File? = null
        val names = mutableListOf<String>()
        uris.forEach { uri ->
            ctx.contentResolver.openInputStream(uri)?.use { input ->
                val name = getDisplayName(uri)
                val dest = File(baseDir, name)
                FileOutputStream(dest).use { output -> input.copyTo(output) }
                names.add(name)
                val lower = name.lowercase()
                when {
                    lower.contains("mde") -> mde = dest
                    lower.contains("res") || lower.contains("residual") -> residual = dest
                    lower.contains("head") -> head = dest
                    lower.endsWith(".json") || lower.contains("meta") -> metaFile = dest
                }
            }
        }
        require(mde != null && residual != null && head != null && metaFile != null) { "Need mde, residual, head ONNX and meta.json" }
        val meta = PsdDepthCompletionEngine.ModelMetadata.fromJsonFile(metaFile!!)
        return Triple(PsdDepthCompletionEngine.ModelBundle(mde!!, residual!!, head!!), meta, names.joinToString(", "))
    }

    private fun writeDepthRaw(ctx: Context, depth: FloatArray, width: Int, height: Int): File {
        val file = File(ctx.cacheDir, "psd_depth_${width}x$height.bin")
        FileOutputStream(file).channel.use { channel ->
            val buf = ByteBuffer.allocate(8 + depth.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(width)
            buf.putInt(height)
            depth.forEach { buf.putFloat(it) }
            buf.flip()
            channel.write(buf)
        }
        return file
    }

    private fun getDisplayName(uri: Uri): String {
        val cursor = context?.contentResolver?.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val idx = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) return it.getString(idx)
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

    private fun formatDepth(value: Float): String = if (value.isFinite()) String.format("%.3f", value) else "--"

    private fun formatDelta(delta: Long): String {
        val sign = if (delta >= 0) "+" else "-"
        return "$sign${formatBytes(delta)}"
    }

    private fun countValidDepth(depth: FloatArray): Int = depth.count { it.isFinite() && it > 0f }

    private fun computeDepthRange(depth: FloatArray): Pair<Float, Float> {
        var minD = Float.MAX_VALUE
        var maxD = -Float.MAX_VALUE
        depth.forEach { v ->
            if (v.isFinite() && v > 0f) {
                if (v < minD) minD = v
                if (v > maxD) maxD = v
            }
        }
        if (minD == Float.MAX_VALUE || maxD == -Float.MAX_VALUE) return 0f to 0f
        return minD to maxD
    }

    data class SparseDepth(
        val width: Int,
        val height: Int,
        val data: FloatArray,
        val validCount: Int,
        val minDepth: Float,
        val maxDepth: Float
    )

    data class PipelineResult(
        val preview: Bitmap,
        val sparseStats: String,
        val depth: FloatArray,
        val width: Int,
        val height: Int
    )

    companion object {
        private const val TAG = "PsdMidasTest"
    }
}
