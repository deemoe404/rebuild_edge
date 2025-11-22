package com.example.rebuild_edge.ui.psd

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import org.json.JSONObject

/**
 * Helper that orchestrates the PSD ONNX submodules on Android.
 *
 * The class is intentionally lightweight: it only manages ONNX Runtime sessions and exposes
 * strongly typed helpers for running the MiDaS encoder, residual branch, and image/pixel head. The
 * sparse-depth alignment, dual-space propagation, and other custom math can be implemented in
 * Kotlin/C++ on top of the tensor helpers provided here.
 */
class PsdDepthCompletionEngine(
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
) : Closeable {

    data class ModelBundle(
        val midas: File,
        val residual: File,
        val head: File
    )

    data class ModelMetadata(
        val dataset: String,
        val rgbHeight: Int,
        val rgbWidth: Int,
        val mdeHeight: Int,
        val mdeWidth: Int,
        val binCount: Int,
        val alpha1: Float,
        val beta1: Float,
        val alpha2: Float,
        val beta2: Float
    ) {
        companion object {
            fun fromJsonFile(file: File): ModelMetadata {
                val text = file.readText(Charsets.UTF_8)
                val obj = JSONObject(text)
                return ModelMetadata(
                    dataset = obj.optString("dataset", "unknown"),
                    rgbHeight = obj.getInt("rgb_height"),
                    rgbWidth = obj.getInt("rgb_width"),
                    mdeHeight = obj.getInt("mde_height"),
                    mdeWidth = obj.getInt("mde_width"),
                    binCount = obj.getInt("bin_num"),
                    alpha1 = obj.getDouble("alpha1").toFloat(),
                    beta1 = obj.getDouble("beta1").toFloat(),
                    alpha2 = obj.getDouble("alpha2").toFloat(),
                    beta2 = obj.getDouble("beta2").toFloat()
                )
            }
        }
    }

    data class TensorData(
        val data: FloatArray,
        val shape: IntArray
    ) {
        init {
            require(shape.isNotEmpty()) { "shape cannot be empty" }
            val expected = shape.fold(1L) { acc, dim -> acc * dim }
            require(expected.toInt() == data.size) {
                "data length ${data.size} does not match shape product $expected"
            }
        }

        fun toLongArray(): LongArray = LongArray(shape.size) { shape[it].toLong() }
    }

    data class MdeOutput(
        val depth: TensorData,
        val path0: TensorData,
        val path1: TensorData,
        val path2: TensorData,
        val path3: TensorData
    )

    data class ResidualOutput(
        val residual: TensorData,
        val confidence: TensorData
    )

    data class HeadOutput(
        val similarity: TensorData,
        val confidenceImg: TensorData,
        val feat: TensorData,
        val globalOffset: TensorData,
        val globalConfidence: TensorData,
        val depthImg: TensorData,
        val depthPix: TensorData
    )

    private var metadata: ModelMetadata? = null
    private var mdeSession: OrtSession? = null
    private var residualSession: OrtSession? = null
    private var headSession: OrtSession? = null

    fun loadModels(bundle: ModelBundle, meta: ModelMetadata) {
        close()
        metadata = meta
        mdeSession = environment.createSession(bundle.midas.absolutePath, OrtSession.SessionOptions())
        residualSession = environment.createSession(bundle.residual.absolutePath, OrtSession.SessionOptions())
        headSession = environment.createSession(bundle.head.absolutePath, OrtSession.SessionOptions())
    }

    fun runMidas(rgbInput: FloatArray, inputShape: IntArray, intrinsics: FloatArray): MdeOutput {
        val session = mdeSession ?: throw IllegalStateException("MiDaS ONNX session is not loaded")
        val rgbTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(rgbInput), inputShape.toLongArrayCompat())
        val kTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(intrinsics), longArrayOf(1, 3, 3))
        val names = session.inputNames.toList()
        val inputs: Map<String, OnnxTensor> = when (names.size) {
            1 -> mapOf(names[0] to rgbTensor) // legacy single-input export
            2 -> {
                // Prefer semantic names if present
                val map = mutableMapOf<String, OnnxTensor>()
                val rgbName = names.find { it.contains("rgb", ignoreCase = true) } ?: names[0]
                val kName = names.find { it.contains("intrinsics", ignoreCase = true) || it.contains("k", ignoreCase = true) }
                    ?: names.getOrNull(1)
                    ?: names[0]
                map[rgbName] = rgbTensor
                if (kName != rgbName) {
                    map[kName] = kTensor
                }
                map
            }
            else -> throw IllegalStateException("Unexpected MiDaS input count: ${names.size}")
        }
        try {
            session.run(inputs).use { result ->
                val depth = tensorFrom(result[0] as OnnxTensor)
                val path0 = tensorFrom(result[1] as OnnxTensor)
                val path1 = tensorFrom(result[2] as OnnxTensor)
                val path2 = tensorFrom(result[3] as OnnxTensor)
                val path3 = tensorFrom(result[4] as OnnxTensor)
                return MdeOutput(depth, path0, path1, path2, path3)
            }
        } finally {
            try { rgbTensor.close() } catch (_: Throwable) {}
            try { kTensor.close() } catch (_: Throwable) {}
        }
    }

    fun runResidualBranch(
        sparse: TensorData,
        depthDiff: TensorData,
        sparseResidual: TensorData,
        pathFeats: Array<TensorData>
    ): ResidualOutput {
        val session = residualSession ?: throw IllegalStateException("Residual ONNX session is not loaded")
        require(pathFeats.size == 4) { "Expected 4 path features, found ${pathFeats.size}" }

        val inputs = linkedMapOf(
            "sparse" to createTensor(sparse),
            "depth_diff" to createTensor(depthDiff),
            "sparse_residual" to createTensor(sparseResidual),
            "path0" to createTensor(pathFeats[0]),
            "path1" to createTensor(pathFeats[1]),
            "path2" to createTensor(pathFeats[2]),
            "path3" to createTensor(pathFeats[3])
        )
        try {
            session.run(inputs).use { result ->
                val residual = tensorFrom(result[0] as OnnxTensor)
                val confidence = tensorFrom(result[1] as OnnxTensor)
                return ResidualOutput(residual, confidence)
            }
        } finally {
            inputs.values.forEach { it.closeSilently() }
        }
    }

    fun runHead(
        sparse: TensorData,
        depthResidual: TensorData,
        sparseResidual: TensorData,
        laplace: TensorData,
        confidenceResidual: TensorData,
        pathFeats: Array<TensorData>,
        bins: TensorData
    ): HeadOutput {
        val session = headSession ?: throw IllegalStateException("Head ONNX session is not loaded")
        require(pathFeats.size == 4) { "Expected 4 path features, found ${pathFeats.size}" }
        val inputs = linkedMapOf(
            "sparse" to createTensor(sparse),
            "depth_residual" to createTensor(depthResidual),
            "sparse_residual" to createTensor(sparseResidual),
            "laplace" to createTensor(laplace),
            "confidence_residual" to createTensor(confidenceResidual),
            "path0" to createTensor(pathFeats[0]),
            "path1" to createTensor(pathFeats[1]),
            "path2" to createTensor(pathFeats[2]),
            "path3" to createTensor(pathFeats[3]),
            "bins" to createTensor(bins)
        )
        try {
            session.run(inputs).use { result ->
                val similarity = tensorFrom(result[0] as OnnxTensor)
                val confidenceImg = tensorFrom(result[1] as OnnxTensor)
                val feat = tensorFrom(result[2] as OnnxTensor)
                val globalOffset = tensorFrom(result[3] as OnnxTensor)
                val globalConf = tensorFrom(result[4] as OnnxTensor)
                val depthImg = tensorFrom(result[5] as OnnxTensor)
                val depthPix = tensorFrom(result[6] as OnnxTensor)
                return HeadOutput(
                    similarity,
                    confidenceImg,
                    feat,
                    globalOffset,
                    globalConf,
                    depthImg,
                    depthPix
                )
            }
        } finally {
            inputs.values.forEach { it.closeSilently() }
        }
    }

    fun computeBins(residual: TensorData, confidence: TensorData): TensorData {
        val meta = metadata ?: throw IllegalStateException("Metadata not loaded")
        val batch = residual.shape[0]
        val height = residual.shape[2]
        val width = residual.shape[3]
        require(residual.shape.contentEquals(confidence.shape)) {
            "residual and confidence shapes must match"
        }
        val binCount = meta.binCount
        val bins = FloatArray(batch * binCount * height * width)
        var offset = 0
        var idx = 0
        while (idx < residual.data.size) {
            val resVal = abs(residual.data[idx])
            val confVal = confidence.data[idx]
            val widthVal = resVal * (2f - confVal) + 0.05f
            val maxBin = ((1f + meta.alpha1) * widthVal + meta.beta1).coerceAtLeast(0.05f)
            val minBin = ((-1f + meta.alpha2) * widthVal + meta.beta2).coerceAtMost(-0.05f)
            val step = (maxBin - minBin) / (binCount - 1)
            for (b in 0 until binCount) {
                bins[offset + b] = minBin + step * b
            }
            offset += binCount
            idx += 1
        }
        return TensorData(bins, intArrayOf(batch, binCount, height, width))
    }

    fun computeLaplace(
        sparseResidual: TensorData,
        sparseDepth: TensorData,
        bins: TensorData
    ): TensorData {
        require(sparseResidual.shape.contentEquals(sparseDepth.shape)) {
            "sparse residual shape ${sparseResidual.shape.contentToString()} must match sparse depth shape"
        }
        val batch = sparseResidual.shape[0]
        val height = sparseResidual.shape[2]
        val width = sparseResidual.shape[3]
        val binCount = bins.shape[1]
        val laplace = FloatArray(batch * binCount * height * width)
        var baseResidual = 0
        var baseBins = 0
        var baseSparse = 0
        val logits = FloatArray(binCount)
        for (b in 0 until batch) {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val sr = sparseResidual.data[baseResidual]
                    val sparseVal = sparseDepth.data[baseSparse]
                    val valid = sparseVal > 0f
                    var maxLogit = Float.NEGATIVE_INFINITY
                    for (k in 0 until binCount) {
                        val value = -abs(sr - bins.data[baseBins + k])
                        logits[k] = value
                        maxLogit = max(maxLogit, value)
                    }
                    var sum = 0f
                    for (k in 0 until binCount) {
                        val score = exp((logits[k] - maxLogit).toDouble()).toFloat()
                        logits[k] = score
                        sum += score
                    }
                    for (k in 0 until binCount) {
                        val value = if (valid && sum > 0f) logits[k] / sum else 0f
                        laplace[baseBins + k] = value
                    }
                    baseResidual += 1
                    baseSparse += 1
                    baseBins += binCount
                }
            }
        }
        return TensorData(laplace, bins.shape.copyOf())
    }

    /**
        Align inverse depth to sparse measurements using the 2x2 normal equation
        (polyfit in the original PyTorch). The caller supplies min/max depth to
        clamp the valid mask; set adaptiveMinMax=true to derive min/max from the
        sparse target instead of config constants.
     */
    fun alignInverseDepthPolyfit(
        depthInverse: TensorData,
        targetDepth: TensorData,
        minDepth: Float,
        maxDepth: Float,
        adaptiveMinMax: Boolean = true
    ): TensorData {
        require(depthInverse.shape.contentEquals(targetDepth.shape)) {
            "depthInverse shape ${depthInverse.shape.contentToString()} must match target ${targetDepth.shape.contentToString()}"
        }
        val b = depthInverse.shape[0]
        val h = depthInverse.shape[2]
        val w = depthInverse.shape[3]
        val out = depthInverse.data.copyOf()
        val target = targetDepth.data
        val total = h * w
        val eps = 1e-8f
        for (bi in 0 until b) {
            var localMin = minDepth
            var localMax = maxDepth
            if (adaptiveMinMax) {
                var tMin = Float.MAX_VALUE
                var tMax = -Float.MAX_VALUE
                val base = bi * total
                for (i in 0 until total) {
                    val v = target[base + i]
                    if (v > 0f) {
                        if (v < tMin) tMin = v
                        if (v > tMax) tMax = v
                    }
                }
                if (tMin < tMax && tMin < Float.MAX_VALUE / 2) {
                    localMin = tMin
                    localMax = tMax
                }
            }
            var a00 = 0.0
            var a01 = 0.0
            var a11 = 0.0
            var b0 = 0.0
            var b1 = 0.0
            val base = bi * total
            for (i in 0 until total) {
                val t = target[base + i]
                val valid = t > localMin && t < localMax
                if (valid) {
                    val pred = depthInverse.data[base + i]
                    a00 += (pred * pred).toDouble()
                    a01 += pred.toDouble()
                    a11 += 1.0
                    b0 += (pred * (1f / t)).toDouble()
                    b1 += (1f / t).toDouble()
                }
            }
            val det = a00 * a11 - a01 * a01
            var scale = 1.0
            var shift = 0.0
            if (det > eps) {
                scale = (a11 * b0 - a01 * b1) / det
                shift = (-a01 * b0 + a00 * b1) / det
            }
            for (i in 0 until total) {
                val idx = base + i
                val v = (scale * depthInverse.data[idx] + shift).toFloat()
                val invMin = 1f / localMax
                val invMax = 1f / localMin
                val clamped = when {
                    v < invMin -> invMin
                    v > invMax -> invMax
                    else -> v
                }
                out[idx] = clamped
            }
        }
        // convert back to depth
        for (i in out.indices) {
            out[i] = 1f / (out[i] + eps)
        }
        return TensorData(out, depthInverse.shape.copyOf())
    }

    /**
        Median-based scale alignment on depth (not inverse). Matches the PyTorch
        align_depth variant with adaptive min/max.
     */
    fun alignDepthMedian(
        depth: TensorData,
        targetDepth: TensorData,
        minDepth: Float,
        maxDepth: Float,
        adaptiveMinMax: Boolean = true
    ): TensorData {
        require(depth.shape.contentEquals(targetDepth.shape)) {
            "depth shape ${depth.shape.contentToString()} must match target ${targetDepth.shape.contentToString()}"
        }
        val b = depth.shape[0]
        val h = depth.shape[2]
        val w = depth.shape[3]
        val total = h * w
        val out = depth.data.copyOf()
        for (bi in 0 until b) {
            var localMin = minDepth
            var localMax = maxDepth
            if (adaptiveMinMax) {
                var tMin = Float.MAX_VALUE
                var tMax = -Float.MAX_VALUE
                val base = bi * total
                for (i in 0 until total) {
                    val v = targetDepth.data[base + i]
                    if (v > 0f) {
                        if (v < tMin) tMin = v
                        if (v > tMax) tMax = v
                    }
                }
                if (tMin < tMax && tMin < Float.MAX_VALUE / 2) {
                    localMin = tMin
                    localMax = tMax
                }
            }
            val base = bi * total
            val predList = ArrayList<Float>()
            val targList = ArrayList<Float>()
            for (i in 0 until total) {
                val t = targetDepth.data[base + i]
                if (t > localMin && t < localMax) {
                    predList.add(depth.data[base + i])
                    targList.add(t)
                }
            }
            val scale = if (predList.isNotEmpty()) {
                val predMed = median(predList)
                val targMed = median(targList)
                if (predMed > 0f) targMed / predMed else 1f
            } else {
                1f
            }
            for (i in 0 until total) {
                val idx = base + i
                var v = depth.data[idx] * scale
                if (v < localMin) v = localMin
                if (v > localMax) v = localMax
                out[idx] = v
            }
        }
        return TensorData(out, depth.shape.copyOf())
    }

    /** Simple median for small lists. */
    private fun median(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[mid - 1] + sorted[mid]) / 2f
        } else sorted[mid]
    }

    /**
        Convert depth to point cloud (pinhole camera). K is a 3x3 row-major array.
        Returns TensorData shaped [B,3,H,W].
     */
    fun depthToPoint(depth: TensorData, intrinsics: FloatArray, scaleDown: Int = 1): TensorData {
        val b = depth.shape[0]
        val h = depth.shape[2]
        val w = depth.shape[3]
        require(intrinsics.size == 9) { "intrinsics must be 3x3 row-major" }
        val fx = intrinsics[0] / scaleDown
        val fy = intrinsics[4] / scaleDown
        val cx = intrinsics[2] / scaleDown
        val cy = intrinsics[5] / scaleDown
        val out = FloatArray(b * 3 * h * w)
        var offsetDepth = 0
        var offsetOut = 0
        for (bi in 0 until b) {
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val z = depth.data[offsetDepth]
                    val xn = (x - cx) / fx
                    val yn = (y - cy) / fy
                    out[offsetOut] = xn * z      // X
                    out[offsetOut + 1] = yn * z  // Y
                    out[offsetOut + 2] = z       // Z
                    offsetDepth += 1
                    offsetOut += 3
                }
            }
        }
        return TensorData(out, intArrayOf(b, 3, h, w))
    }

    /**
        Average pooling on sparse depth with validity mask; scale must divide H,W.
     */
    fun sparseDownSample(sparse: TensorData, scale: Int): TensorData {
        val b = sparse.shape[0]
        val h = sparse.shape[2]
        val w = sparse.shape[3]
        val oh = h / scale
        val ow = w / scale
        val out = FloatArray(b * oh * ow)
        var outIdx = 0
        var base = 0
        for (bi in 0 until b) {
            for (yy in 0 until oh) {
                for (xx in 0 until ow) {
                    var sum = 0f
                    var cnt = 0f
                    for (dy in 0 until scale) {
                        for (dx in 0 until scale) {
                            val idx = base + (yy * scale + dy) * w + (xx * scale + dx)
                            val v = sparse.data[idx]
                            if (v > 0f) {
                                sum += v
                                cnt += 1f
                            }
                        }
                    }
                    out[outIdx++] = if (cnt > 0f) sum / cnt else 0f
                }
            }
            base += h * w
        }
        return TensorData(out, intArrayOf(b, 1, oh, ow))
    }

    /**
        Lightweight CSPN implementation (CPU, unoptimized). Uses cosine affinity over feature map.
        feature: [B,C,H,W], sparse/dense: [B,1,H,W].
     */
    fun cspn(
        sparse: TensorData,
        dense: TensorData,
        feature: TensorData,
        kernel: Int = 3,
        iteration: Int = 3
    ): TensorData {
        val b = dense.shape[0]
        val h = dense.shape[2]
        val w = dense.shape[3]
        val c = feature.shape[1]
        require(kernel == 3) { "Only 3x3 kernel supported in CPU CSPN" }
        val out = dense.data.copyOf()
        val mask = BooleanArray(b * h * w)
        for (i in mask.indices) {
            mask[i] = sparse.data[i] > 0f
        }
        // precompute normalized feature vectors
        val featNorm = feature.data.copyOf()
        // L2 norm per pixel
        var idx = 0
        for (bi in 0 until b) {
            for (ci in 0 until c) {
                for (iPos in 0 until h * w) {
                    val v = feature.data[bi * c * h * w + ci * h * w + iPos]
                    featNorm[bi * c * h * w + ci * h * w + iPos] = v
                }
            }
        }
        // iterate
        repeat(iteration) {
            val newOut = out.copyOf()
            for (bi in 0 until b) {
                val baseDense = bi * h * w
                val baseFeat = bi * c * h * w
                for (y in 0 until h) {
                    for (x in 0 until w) {
                        val idxCenter = baseDense + y * w + x
                        if (mask[idxCenter]) {
                            newOut[idxCenter] = sparse.data[idxCenter]
                            continue
                        }
                        var acc = 0f
                        var weightSum = 0f
                        val centerVec = FloatArray(c) { k ->
                            featNorm[baseFeat + k * h * w + y * w + x]
                        }
                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                if (dy == 0 && dx == 0) continue
                                val ny = y + dy
                                val nx = x + dx
                                if (ny !in 0 until h || nx !in 0 until w) continue
                                val nIdx = baseDense + ny * w + nx
                                // cosine similarity
                                var dot = 0f
                                var normN = 0f
                                var normC = 0f
                                for (k in 0 until c) {
                                    val nv = featNorm[baseFeat + k * h * w + ny * w + nx]
                                    val cv = centerVec[k]
                                    dot += nv * cv
                                    normN += nv * nv
                                    normC += cv * cv
                                }
                                val denom = sqrt((normN * normC).toDouble()).coerceAtLeast(1e-6).toFloat()
                                val sim = if (denom > 0f) dot / denom else 0f
                                val wgt = max(sim, 0f)
                                acc += out[nIdx] * wgt
                                weightSum += wgt
                            }
                        }
                        if (weightSum > 1e-6f) {
                            newOut[idxCenter] = acc / weightSum
                        }
                    }
                }
            }
            System.arraycopy(newOut, 0, out, 0, out.size)
        }
        return TensorData(out, dense.shape.copyOf())
    }

    /**
        Simplified dual-space diffusion: only runs CSPN at full resolution. This is
        a CPU fallback; for production consider a native implementation plus KNN
        refinement to match the original Python more closely.
     */
    fun dualDiffusionSimplified(
        feature: TensorData,
        ipMedian: TensorData,
        depthPolyfit: TensorData,
        sparse: TensorData,
        intrinsics: FloatArray,
        iteration: Int = 3
    ): TensorData {
        // Downsample feature to [B,C',H,W] where C' matches feature channels
        val feat = feature
        // Use CSPN only
        val depthCspn = cspn(sparse, ipMedian, feat, kernel = 3, iteration = iteration)
        return depthCspn
    }

    /**
        Basic fill similar to ip_basic.fill_in_fast: iteratively averages valid
        neighbors to fill zeros.
     */
    fun fillSparseDepthFast(sparse: TensorData, iterations: Int = 4): TensorData {
        val b = sparse.shape[0]
        val h = sparse.shape[2]
        val w = sparse.shape[3]
        val out = sparse.data.copyOf()
        repeat(iterations) {
            val tmp = out.copyOf()
            var idx = 0
            for (bi in 0 until b) {
                for (y in 0 until h) {
                    for (x in 0 until w) {
                        val v = out[idx]
                        if (v > 0f) {
                            idx += 1
                            continue
                        }
                        var sum = 0f
                        var cnt = 0
                        for (dy in -1..1) {
                            for (dx in -1..1) {
                                if (dy == 0 && dx == 0) continue
                                val ny = y + dy
                                val nx = x + dx
                                if (ny !in 0 until h || nx !in 0 until w) continue
                                val nIdx = bi * h * w + ny * w + nx
                                val nv = out[nIdx]
                                if (nv > 0f) {
                                    sum += nv
                                    cnt += 1
                                }
                            }
                        }
                        tmp[idx] = if (cnt > 0) sum / cnt else 0f
                        idx += 1
                    }
                }
            }
            System.arraycopy(tmp, 0, out, 0, out.size)
        }
        return TensorData(out, sparse.shape.copyOf())
    }

    private fun tensorFrom(tensor: OnnxTensor): TensorData {
        val buffer = tensor.floatBuffer ?: error("Expected float tensor output")
        val dup = buffer.duplicate()
        dup.rewind()
        val array = FloatArray(dup.remaining())
        dup.get(array)
        val shape = tensor.info.shape.map { it.toInt() }.toIntArray()
        return TensorData(array, shape)
    }

    private fun createTensor(data: TensorData): OnnxTensor {
        return OnnxTensor.createTensor(environment, FloatBuffer.wrap(data.data), data.toLongArray())
    }

    private fun IntArray.toLongArrayCompat(): LongArray = LongArray(size) { this[it].toLong() }

    private fun OnnxTensor.closeSilently() {
        try {
            close()
        } catch (err: Throwable) {
            Log.w(TAG, "Failed closing tensor", err)
        }
    }

    override fun close() {
        mdeSession?.close()
        residualSession?.close()
        headSession?.close()
        mdeSession = null
        residualSession = null
        headSession = null
    }

    companion object {
        private const val TAG = "PsdDepthEngine"
    }
}
