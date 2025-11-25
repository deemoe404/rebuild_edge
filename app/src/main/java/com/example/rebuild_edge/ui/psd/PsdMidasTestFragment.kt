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
import com.example.rebuild_edge.R
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
                applyDiffusionDefaults(metadata.dataset)
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
        val enablePostProcess = binding.switchPostProcess.isChecked
        val postIterations = binding.editPostIterations.text?.toString()?.toIntOrNull()?.coerceAtLeast(0) ?: 0
        val diffusionMode = binding.spinnerDiffusion.selectedItem?.toString() ?: "3D-2D"
        val diffusionKnn = binding.editKnn.text?.toString()?.toIntOrNull()?.coerceAtLeast(1) ?: 1
        val diffusionCandidate = binding.editCandidateLimit.text?.toString()?.toIntOrNull()?.coerceAtLeast(0) ?: 0
        val diffusionScale = binding.editDiffScale.text?.toString()?.toIntOrNull()?.coerceAtLeast(1) ?: 4
        val diffusionIter = binding.editDiffIteration.text?.toString()?.toIntOrNull()?.coerceAtLeast(1) ?: 3
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
        Log.d(TAG, "runInference: Start. enablePostProcess=$enablePostProcess postIterations=$postIterations")
        val startWall = SystemClock.elapsedRealtime()
        val startCpu = Process.getElapsedCpuTime()
        val memBefore = usedMemory()

        viewLifecycleOwner.lifecycleScope.launch {
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    runPipeline(
                        ctx,
                        bundle,
                        metadata,
                        imageUri,
                        sparseUri,
                        enablePostProcess,
                        postIterations,
                        diffusionMode,
                        diffusionKnn,
                        diffusionCandidate,
                        diffusionScale,
                        diffusionIter
                    )
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
                Log.d(TAG, "runInference: Success. Elapsed=${elapsed}ms CPU=${cpuElapsed}ms")
                binding.imgPreview.setImageBitmap(preview)
                binding.imgPreview.visibility = View.VISIBLE
                binding.txtSparseStats.text = statsText
                binding.txtElapsed.text = "Elapsed: ${elapsed} ms"
                binding.txtCpu.text = "CPU (ms): $cpuElapsed"
                binding.txtMemory.text = "Memory: ${formatBytes(memAfter)} (${formatDelta(memAfter - memBefore)})"
                val depthRange = computeDepthRange(depthArr)
                binding.txtDepthRange.text = "Depth range ($w x $h): ${formatDepth(depthRange.first)} ~ ${formatDepth(depthRange.second)}"
                binding.txtStatus.text = if (enablePostProcess && postIterations > 0) {
                    "Done (post-process ${postIterations} iter)"
                } else {
                    "Done"
                }
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
        sparseUri: Uri,
        enablePostProcess: Boolean,
        postIterations: Int,
        diffusionMode: String,
        diffusionKnn: Int,
        diffusionCandidate: Int,
        diffusionScale: Int,
        diffusionIteration: Int
    ): PipelineResult {
        val eng = engine ?: PsdDepthCompletionEngine().also { engine = it }.apply {
            loadModels(bundle, metadata)
        }
        Log.d(TAG, "runPipeline: Engine ready. Loading inputs...")

        val sparseOrig = loadSparseDepth(context, sparseUri)
        val (workW, workH, scaleSparse) = computeWorkingSize(sparseOrig.width, sparseOrig.height)
        val sparse = if (scaleSparse < 0.999f) {
            val data = resizeFloatArray(sparseOrig.data, sparseOrig.width, sparseOrig.height, workW, workH)
            sparseOrig.copy(width = workW, height = workH, data = data)
        } else sparseOrig
        Log.d(TAG, "runPipeline: Sparse loaded. Orig=${sparseOrig.width}x${sparseOrig.height} Work=${sparse.width}x${sparse.height}")
        val rgbBitmap = loadBitmap(context, imageUri)
        val imgW = rgbBitmap.width
        val imgH = rgbBitmap.height
        val mdeW = metadata.mdeWidth
        val mdeH = metadata.mdeHeight
        val rgbMde = Bitmap.createScaledBitmap(rgbBitmap, mdeW, mdeH, true)
        if (rgbBitmap !== rgbMde) rgbBitmap.recycle()
        val rgbTensor = bitmapToInputArray(rgbMde, mdeW, mdeH)
        val intrinsics = loadIntrinsics(
            context,
            imageUri,
            imgW,
            imgH,
            sparseUri,
            sparseOrig.width,
            sparseOrig.height,
            scaleSparse,
            sparse.width,
            sparse.height
        )
        Log.d(TAG, "runPipeline: Inputs ready. Running Midas...")
        val mdeOut = eng.runMidas(rgbTensor, intArrayOf(1, 3, mdeH, mdeW), intrinsics)
        val depthMde = mdeOut.depth
        val depthInverse = resizeFloatArray(depthMde.data, depthMde.shape[3], depthMde.shape[2], sparse.width, sparse.height)
        val minSparse = sparse.minDepth.coerceAtLeast(1e-3f)
        val maxSparse = sparse.maxDepth.coerceAtLeast(minSparse + 1e-3f)
        val aligned = eng.alignInverseDepthPolyfit(
            TensorData(depthInverse, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            minDepth = minSparse,
            maxDepth = maxSparse,
            adaptiveMinMax = true
        )
        val ipFilled = PsdNative.fillSparseFast(
            sparse.data,
            1,
            sparse.height,
            sparse.width,
            1,
            sparse.maxDepth.coerceAtLeast(100f)
        )
        val ipMedian = eng.alignDepthMedian(
            TensorData(ipFilled, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            minDepth = minSparse,
            maxDepth = maxSparse,
            adaptiveMinMax = true
        )
        Log.d(TAG, "runPipeline: Alignment done. Running Dual Diffusion...")
        val depthDiff = dualDiffusionFull(
            eng,
            mdeOut.path0,
            ipMedian,
            aligned,
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            intrinsics,
            mode = diffusionMode,
            knn = diffusionKnn,
            candidateLimit = diffusionCandidate,
            scale = diffusionScale,
            iteration = diffusionIteration
        )
        val sparseResidual = FloatArray(sparse.data.size) { i -> (sparse.data[i] - depthDiff.data[i]) }
        Log.d(TAG, "runPipeline: Dual Diffusion done. Running Residual Branch...")
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
        Log.d(TAG, "runPipeline: Residual done. Running Head...")
        val headOut = eng.runHead(
            TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(depthResidual, intArrayOf(1, 1, sparse.height, sparse.width)),
            TensorData(sparseResidual, intArrayOf(1, 1, sparse.height, sparse.width)),
            laplace,
            residualOut.confidence,
            arrayOf(mdeOut.path0, mdeOut.path1, mdeOut.path2, mdeOut.path3),
            bins
        )
        var depthWork = headOut.depthPix.data
        if (enablePostProcess && postIterations > 0) {
            Log.d(TAG, "runPipeline: Head done. Running Post-Process ($postIterations iter)...")
            val featHead = headOut.feat
            val featFull = resizeFeature3D(featHead, 32, sparse.height, sparse.width)
            val refined = eng.cspn(
                TensorData(sparse.data, intArrayOf(1, 1, sparse.height, sparse.width)),
                TensorData(depthWork, intArrayOf(1, 1, sparse.height, sparse.width)),
                featFull,
                kernel = 3,
                iteration = postIterations
            )
            depthWork = refined.data
        }
        val depthFinal = if (scaleSparse < 0.999f) {
            resizeFloatArray(depthWork, sparse.width, sparse.height, sparseOrig.width, sparseOrig.height)
        } else depthWork
        val preview = depthToBitmap(depthFinal, sparseOrig.width, sparseOrig.height)
        val sparseStats = "Sparse depth (${sparseOrig.width}x${sparseOrig.height} -> ${sparse.width}x${sparse.height}): points=${sparseOrig.validCount}, range=${formatDepth(sparseOrig.minDepth)}~${formatDepth(sparseOrig.maxDepth)}"
        Log.d(TAG, "runPipeline: Finished.")
        return PipelineResult(preview, sparseStats, depthFinal, sparseOrig.width, sparseOrig.height)
    }

    private fun dualDiffusionFull(
        eng: PsdDepthCompletionEngine,
        feat: TensorData,
        ipMedian: TensorData,
        depthPolyfit: TensorData,
        sparse: TensorData,
        intrinsics: FloatArray,
        mode: String,
        knn: Int,
        candidateLimit: Int,
        scale: Int,
        iteration: Int
    ): TensorData {
        val h = sparse.shape[2]
        val w = sparse.shape[3]
        val outH = max(1, h / scale)
        val outW = max(1, w / scale)

        val sparseDown = eng.sparseDownSample(sparse, scale)
        val ipDownData = resizeFloatArray(ipMedian.data, w, h, outW, outH)
        val ipDown = TensorData(ipDownData, intArrayOf(1, 1, outH, outW))
        val depthPolyfitDownData = resizeFloatArray(depthPolyfit.data, w, h, outW, outH)
        val depthPolyfitDown = TensorData(depthPolyfitDownData, intArrayOf(1, 1, outH, outW))
        val pointDown = eng.depthToPoint(depthPolyfitDown, intrinsics.copyOf(), scaleDown = scale)

        val featDown = resizeFeature3D(feat, 9, outH, outW)
        val featFull = resizeFeature3D(feat, 16, h, w)
        val costFeatDown = concatPointAndFeat(pointDown, featDown)

        fun knnStep(depthSeed: TensorData, sparseSrc: TensorData, point: TensorData, featSrc: TensorData, hh: Int, ww: Int): TensorData {
            val depthKnn = PsdNative.knnPropagate(
                point.data,
                featSrc.data,
                depthSeed.data,
                sparseSrc.data,
                1,
                featSrc.shape[1],
                hh,
                ww,
                knn,
                candidateLimit
            )
            return TensorData(depthKnn, intArrayOf(1, 1, hh, ww))
        }

        // 3D-2D path (default)
        val depth3d2d: TensorData = run {
            val depthKnnDown = knnStep(ipDown, sparseDown, pointDown, costFeatDown, outH, outW)
            val depthCspnDown = eng.cspn(
                sparseDown,
                depthKnnDown,
                costFeatDown,
                kernel = 3,
                iteration = iteration
            )
            val depthUpData = resizeFloatArray(depthCspnDown.data, outW, outH, w, h)
            val depthUp = TensorData(depthUpData, intArrayOf(1, 1, h, w))
            eng.cspn(
                sparse,
                depthUp,
                featFull,
                kernel = 3,
                iteration = iteration
            )
        }

        return when (mode.uppercase()) {
            "3D+2D" -> {
                val depthKnnDown = knnStep(ipDown, sparseDown, pointDown, costFeatDown, outH, outW)
                val depthCspn2dDown = eng.cspn(
                    sparseDown,
                    ipDown,
                    costFeatDown,
                    kernel = 3,
                    iteration = iteration
                )
                val depthCspn2dUp = TensorData(
                    resizeFloatArray(depthCspn2dDown.data, outW, outH, w, h),
                    intArrayOf(1, 1, h, w)
                )
                val depthCspnFull = eng.cspn(
                    sparse,
                    depthCspn2dUp,
                    featFull,
                    kernel = 3,
                    iteration = iteration
                )
                val depthKnnUp = TensorData(
                    resizeFloatArray(depthKnnDown.data, outW, outH, w, h),
                    intArrayOf(1, 1, h, w)
                )
                val mixed = FloatArray(depthCspnFull.data.size) { idx ->
                    (depthCspnFull.data[idx] + depthKnnUp.data[idx]) * 0.5f
                }
                TensorData(mixed, depthCspnFull.shape)
            }
            "2D-3D" -> {
                val depthCspn2dDown = eng.cspn(
                    sparseDown,
                    ipDown,
                    costFeatDown,
                    kernel = 3,
                    iteration = iteration
                )
                val depthCspn2dUp = TensorData(
                    resizeFloatArray(depthCspn2dDown.data, outW, outH, w, h),
                    intArrayOf(1, 1, h, w)
                )
                val depthCspnFull = eng.cspn(
                    sparse,
                    depthCspn2dUp,
                    featFull,
                    kernel = 3,
                    iteration = iteration
                )
                val depthCspn2dDownAgain = TensorData(
                    resizeFloatArray(depthCspnFull.data, w, h, outW, outH),
                    intArrayOf(1, 1, outH, outW)
                )
                val depthKnnDown = knnStep(depthCspn2dDownAgain, sparseDown, pointDown, costFeatDown, outH, outW)
                TensorData(
                    resizeFloatArray(depthKnnDown.data, outW, outH, w, h),
                    intArrayOf(1, 1, h, w)
                )
            }
            else -> depth3d2d
        }
    }

    private fun concatPointAndFeat(point: TensorData, feat: TensorData): TensorData {
        val b = point.shape[0]
        val cPoint = point.shape[1]
        val cFeat = feat.shape[1]
        val h = point.shape[2]
        val w = point.shape[3]
        val out = FloatArray(point.data.size + feat.data.size)
        System.arraycopy(point.data, 0, out, 0, point.data.size)
        System.arraycopy(feat.data, 0, out, point.data.size, feat.data.size)
        return TensorData(out, intArrayOf(b, cPoint + cFeat, h, w))
    }

private fun resizeFeature3D(src: TensorData, targetC: Int, targetH: Int, targetW: Int): TensorData {
    val srcC = src.shape[1]
    val srcH = src.shape[2]
    val srcW = src.shape[3]
    val out = FloatArray(targetC * targetH * targetW)

    // Trilinear interpolation
    // We map target coordinates (tc, th, tw) to source coordinates (sc, sh, sw)
    // sc = tc * (srcC / targetC) etc.
    // But for align_corners=False (default in PyTorch interpolate), the formula is:
    // src_idx = target_idx * (src_size / target_size)

    val scaleC = srcC.toFloat() / targetC
    val scaleH = srcH.toFloat() / targetH
    val scaleW = srcW.toFloat() / targetW

    for (tc in 0 until targetC) {
        val sc = (tc + 0.5f) * scaleC - 0.5f
        val c0 = Math.floor(sc.toDouble()).toInt().coerceIn(0, srcC - 1)
        val c1 = (c0 + 1).coerceIn(0, srcC - 1)
        val dc = sc - Math.floor(sc.toDouble()).toFloat()

        for (th in 0 until targetH) {
            val sh = (th + 0.5f) * scaleH - 0.5f
            val h0 = Math.floor(sh.toDouble()).toInt().coerceIn(0, srcH - 1)
            val h1 = (h0 + 1).coerceIn(0, srcH - 1)
            val dh = sh - Math.floor(sh.toDouble()).toFloat()

            for (tw in 0 until targetW) {
                val sw = (tw + 0.5f) * scaleW - 0.5f
                val w0 = Math.floor(sw.toDouble()).toInt().coerceIn(0, srcW - 1)
                val w1 = (w0 + 1).coerceIn(0, srcW - 1)
                val dw = sw - Math.floor(sw.toDouble()).toFloat()

                // 8 corners
                // c0
                val v000 = src.data[c0 * srcH * srcW + h0 * srcW + w0]
                val v001 = src.data[c0 * srcH * srcW + h0 * srcW + w1]
                val v010 = src.data[c0 * srcH * srcW + h1 * srcW + w0]
                val v011 = src.data[c0 * srcH * srcW + h1 * srcW + w1]
                // c1
                val v100 = src.data[c1 * srcH * srcW + h0 * srcW + w0]
                val v101 = src.data[c1 * srcH * srcW + h0 * srcW + w1]
                val v110 = src.data[c1 * srcH * srcW + h1 * srcW + w0]
                val v111 = src.data[c1 * srcH * srcW + h1 * srcW + w1]

                // Interpolate along W
                val c0h0 = v000 * (1 - dw) + v001 * dw
                val c0h1 = v010 * (1 - dw) + v011 * dw
                val c1h0 = v100 * (1 - dw) + v101 * dw
                val c1h1 = v110 * (1 - dw) + v111 * dw

                // Interpolate along H
                val c0_val = c0h0 * (1 - dh) + c0h1 * dh
                val c1_val = c1h0 * (1 - dh) + c1h1 * dh

                // Interpolate along C
                val value = c0_val * (1 - dc) + c1_val * dc

                out[tc * targetH * targetW + th * targetW + tw] = value
            }
        }
    }
    return TensorData(out, intArrayOf(1, targetC, targetH, targetW))
}

    private fun loadIntrinsics(
        context: Context,
        imageUri: Uri,
        imageW: Int,
        imageH: Int,
        sparseUri: Uri,
        srcW: Int,
        srcH: Int,
        scale: Float,
        targetW: Int,
        targetH: Int
    ): FloatArray {
        val sxSparse = if (srcW > 0) targetW.toFloat() / srcW.toFloat() else scale
        val sySparse = if (srcH > 0) targetH.toFloat() / srcH.toFloat() else scale
        val camFile = findCameraFile(context, sparseUri)
        if (camFile != null && camFile.exists()) {
            runCatching {
                val model = loadCameraModel(camFile)
                return floatArrayOf(
                    (model.fx * sxSparse).toFloat(), 0f, (model.cx * sxSparse).toFloat(),
                    0f, (model.fy * sySparse).toFloat(), (model.cy * sySparse).toFloat(),
                    0f, 0f, 1f
                )
            }.onFailure {
                Log.w(TAG, "Failed to load camera_poses.json, will try JPEG metadata", it)
            }
        }
        val kFromJpeg = loadIntrinsicsFromJpegMetadata(context, imageUri, imageW, imageH, targetW, targetH)
        if (kFromJpeg != null) {
            return kFromJpeg
        }
        throw IllegalStateException("Unable to load intrinsics from camera_poses.json or JPEG metadata")
    }

    private fun findCameraFile(context: Context, sparseUri: Uri): File? {
        val path = sparseUri.path ?: return null
        val file = if (path.startsWith("/")) File(path) else null
        val bases = mutableListOf<File>()
        file?.parentFile?.let { bases += it }
        file?.parentFile?.parentFile?.let { bases += it }
        bases.forEach { base ->
            val candidate = File(base, "camera_poses.json")
            if (candidate.exists()) return candidate
        }
        return null
    }

    private data class CameraModel(val fx: Double, val fy: Double, val cx: Double, val cy: Double)

    private data class DjiIntrinsics(val fx: Double, val fy: Double, val cx: Double, val cy: Double)

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

    private fun loadIntrinsicsFromJpegMetadata(
        context: Context,
        imageUri: Uri,
        imageW: Int,
        imageH: Int,
        targetW: Int,
        targetH: Int
    ): FloatArray? {
        val xmp = extractXmpXml(context, imageUri) ?: run {
            Log.w(TAG, "No XMP metadata found in JPEG: $imageUri")
            return null
        }
        val intr = parseDjiIntrinsicsFromXmp(xmp) ?: run {
            Log.w(TAG, "No DJI intrinsics found in XMP")
            return null
        }
        if (imageW <= 0 || imageH <= 0 || targetW <= 0 || targetH <= 0) {
            Log.w(TAG, "Invalid image or target size for intrinsics scaling: image=${imageW}x${imageH}, target=${targetW}x${targetH}")
            return null
        }
        val sx = targetW.toFloat() / imageW.toFloat()
        val sy = targetH.toFloat() / imageH.toFloat()
        return floatArrayOf(
            (intr.fx * sx).toFloat(), 0f, (intr.cx * sx).toFloat(),
            0f, (intr.fy * sy).toFloat(), (intr.cy * sy).toFloat(),
            0f, 0f, 1f
        )
    }

    private fun extractXmpXml(context: Context, imageUri: Uri): String? {
        return runCatching {
            context.contentResolver.openInputStream(imageUri)?.use { input ->
                val bytes = input.readBytes()
                val data = bytes.toString(Charsets.ISO_8859_1)
                val start1 = data.indexOf("<x:xmpmeta")
                val start2 = data.indexOf("<xmpmeta")
                val start = when {
                    start1 >= 0 -> start1
                    start2 >= 0 -> start2
                    else -> return null
                }
                val endTag1 = "</x:xmpmeta>"
                val endTag2 = "</xmpmeta>"
                var end = data.indexOf(endTag1, start)
                if (end >= 0) {
                    end += endTag1.length
                } else {
                    end = data.indexOf(endTag2, start)
                    if (end < 0) return null
                    end += endTag2.length
                }
                data.substring(start, end)
            }
        }.getOrElse {
            Log.w(TAG, "Failed to extract XMP from JPEG", it)
            null
        }
    }

    private fun parseDjiIntrinsicsFromXmp(xmp: String): DjiIntrinsics? {
        fun getAttr(name: String): Double? {
            val regex = Regex("${Regex.escape(name)}=\"([^\"]+)\"")
            val match = regex.find(xmp) ?: return null
            return match.groupValues.getOrNull(1)?.toDoubleOrNull()
        }

        val focalCalib = getAttr("drone-dji:CalibratedFocalLength")
        val cxAttr = getAttr("drone-dji:CalibratedOpticalCenterX")
        val cyAttr = getAttr("drone-dji:CalibratedOpticalCenterY")

        val dewarpRegex = Regex("drone-dji:DewarpData=\"([^\"]+)\"")
        val dewarpMatch = dewarpRegex.find(xmp)
        var fxD = 0.0
        var fyD = 0.0
        var dx = 0.0
        var dy = 0.0
        var hasDewarp = false
        if (dewarpMatch != null) {
            val payload = dewarpMatch.groupValues.getOrNull(1) ?: ""
            val semi = payload.indexOf(';')
            val nums = if (semi >= 0 && semi + 1 < payload.length) {
                payload.substring(semi + 1)
            } else {
                payload
            }
            val values = nums.split(',').mapNotNull { it.trim().toDoubleOrNull() }
            if (values.size >= 4) {
                fxD = values[0]
                fyD = values[1]
                dx = values[2]
                dy = values[3]
                hasDewarp = true
            }
        }

        val hasF = focalCalib != null
        val hasCx = cxAttr != null
        val hasCy = cyAttr != null
        if (!hasF && !hasDewarp) return null

        val fx = if (hasF) focalCalib!! else fxD
        val fy = if (hasF) focalCalib!! else fyD
        val cx = (if (hasCx) cxAttr!! else 0.0) + if (hasDewarp) dx else 0.0
        val cy = (if (hasCy) cyAttr!! else 0.0) + if (hasDewarp) dy else 0.0

        return if (fx > 0.0 && fy > 0.0 && cx > 0.0 && cy > 0.0) {
            DjiIntrinsics(fx, fy, cx, cy)
        } else {
            null
        }
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
        if (srcWidth == dstWidth && srcHeight == dstHeight) return data
        if (srcWidth <= 0 || srcHeight <= 0 || dstWidth <= 0 || dstHeight <= 0) return FloatArray(dstWidth * dstHeight)
        val output = FloatArray(dstWidth * dstHeight)
        val scaleX = srcWidth.toFloat() / dstWidth.toFloat()
        val scaleY = srcHeight.toFloat() / dstHeight.toFloat()
        for (y in 0 until dstHeight) {
            val srcY = (y + 0.5f) * scaleY - 0.5f
            val y0 = Math.floor(srcY.toDouble()).toInt().coerceIn(0, srcHeight - 1)
            val y1 = (y0 + 1).coerceIn(0, srcHeight - 1)
            val yLerp = srcY - Math.floor(srcY.toDouble()).toFloat()
            for (x in 0 until dstWidth) {
                val srcX = (x + 0.5f) * scaleX - 0.5f
                val x0 = Math.floor(srcX.toDouble()).toInt().coerceIn(0, srcWidth - 1)
                val x1 = (x0 + 1).coerceIn(0, srcWidth - 1)
                val xLerp = srcX - Math.floor(srcX.toDouble()).toFloat()
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

    private fun computeWorkingSize(origW: Int, origH: Int): Triple<Int, Int, Float> {
        val maxEdge = 640
        val scale = if (origW > maxEdge || origH > maxEdge) {
            min(maxEdge.toFloat() / origW.toFloat(), maxEdge.toFloat() / origH.toFloat())
        } else 1f
        val workW = max(1, (origW * scale).toInt())
        val workH = max(1, (origH * scale).toInt())
        return Triple(workW, workH, scale)
    }

    private data class DiffusionDefaults(
        val mode: String,
        val knn: Int,
        val candidate: Int,
        val scale: Int = 4,
        val iteration: Int = 3
    )

    private fun applyDiffusionDefaults(dataset: String) {
        val defaults = when (dataset.uppercase()) {
            "VKITTI2" -> DiffusionDefaults("3D+2D", 3, 0)
            "CITYSCAPE" -> DiffusionDefaults("3D-2D", 2, 0)
            "DIMLI" -> DiffusionDefaults("3D-2D", 12, 0)
            "TOFDC" -> DiffusionDefaults("3D-2D", 8, 0)
            "ARGOVERSE" -> DiffusionDefaults("3D-2D", 1, 8000)
            else -> DiffusionDefaults("3D-2D", 1, 0)
        }
        val modes = resources.getStringArray(R.array.psd_diffusion_modes).map { it.uppercase() }
        val idx = modes.indexOf(defaults.mode.uppercase()).coerceAtLeast(0)
        binding.spinnerDiffusion.setSelection(idx)
        binding.editKnn.setText(defaults.knn.toString())
        binding.editCandidateLimit.setText(defaults.candidate.toString())
        binding.editDiffScale.setText(defaults.scale.toString())
        binding.editDiffIteration.setText(defaults.iteration.toString())
    }

    companion object {
        private const val TAG = "PsdMidasTest"
    }
}
