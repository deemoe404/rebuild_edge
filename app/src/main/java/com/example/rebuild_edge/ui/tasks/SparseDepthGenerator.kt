package com.example.rebuild_edge.ui.tasks

import android.graphics.BitmapFactory
import android.util.Log
import com.example.rebuild_edge.data.TaskRecord
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.Channels
import java.util.Locale
import java.util.UUID
import kotlin.math.max
import kotlin.math.roundToInt
import org.json.JSONArray
import org.json.JSONObject

private data class PlyProperty(val type: String, val name: String)

class SparseDepthGenerator(
    private val record: TaskRecord,
    private val extra: JSONObject?
) {

    data class Result(
        val outputDir: File,
        val files: List<File>,
        val pointsWritten: Long,
        val width: Int,
        val height: Int,
        val sourceWidth: Int,
        val sourceHeight: Int,
        val runId: String,
        val metadataFile: File
    )

    fun generate(
        width: Int,
        height: Int,
        onProgress: ((current: Int, total: Int) -> Unit)? = null
    ): Result {
        require(width > 0 && height > 0) { "目标分辨率必须大于0" }
        val outDir = File(record.outDir)
        val cameraFile = File(outDir, CAMERAS_JSON)
        val plyFile = File(outDir, PLY_FILE)
        check(cameraFile.exists()) { "缺少 camera_poses.json" }
        check(plyFile.exists()) { "缺少 reconstruction_points.ply" }
        Log.i(TAG, "start generate sparse depth task=${record.id} size=${width}x$height camera=${cameraFile.absolutePath} ply=${plyFile.absolutePath}")

        val cameraModel = loadCameraModel(cameraFile)
        val points = loadPoints(plyFile)
        val sourceSize = estimateSourceResolution(outDir, extra) ?: estimateFromIntrinsics(cameraModel)
        val srcW = max(sourceSize.first, 1)
        val srcH = max(sourceSize.second, 1)
        Log.i(TAG, "camera model: cams=${cameraModel.cameras.size} fx=${cameraModel.fx} fy=${cameraModel.fy} src=${srcW}x${srcH} pts=${points.size / 3}")
        val scaleX = width.toDouble() / srcW.toDouble()
        val scaleY = height.toDouble() / srcH.toDouble()

        val runId = UUID.randomUUID().toString()
        val depthDir = File(outDir, "sparse_depth_${width}x${height}_$runId")
        if (depthDir.exists()) {
            depthDir.deleteRecursively()
        }
        depthDir.mkdirs()
        Log.i(TAG, "output dir ${depthDir.absolutePath}")

        val fx = cameraModel.fx * scaleX
        val fy = cameraModel.fy * scaleY
        val cx = cameraModel.cx * scaleX
        val cy = cameraModel.cy * scaleY

        val totalCams = cameraModel.cameras.size
        val savedFiles = ArrayList<File>(totalCams)
        var totalActive = 0L
        onProgress?.invoke(0, totalCams)
        cameraModel.cameras.forEachIndexed { idx, pose ->
            val depth = FloatArray(width * height)
            var activeForCam = 0L
            var i = 0
            while (i < points.size) {
                val X = points[i]
                val Y = points[i + 1]
                val Z = points[i + 2]
                val cam = pose.R
                val tx = pose.t[0]
                val ty = pose.t[1]
                val tz = pose.t[2]
                val xc = cam[0] * X + cam[1] * Y + cam[2] * Z + tx
                val yc = cam[3] * X + cam[4] * Y + cam[5] * Z + ty
                val zc = cam[6] * X + cam[7] * Y + cam[8] * Z + tz
                if (zc > 1e-6) {
                    val u = fx * (xc / zc) + cx
                    val v = fy * (yc / zc) + cy
                    val px = u.roundToInt()
                    val py = v.roundToInt()
                    if (px in 0 until width && py in 0 until height) {
                        val depthVal = zc.toFloat()
                        val idx = py * width + px
                        val existing = depth[idx]
                        if (existing == 0f || depthVal < existing) {
                            if (existing == 0f) activeForCam += 1
                            depth[idx] = depthVal
                        }
                    }
                }
                i += 3
            }
            totalActive += activeForCam
            if (idx < 3) {
                Log.d(TAG, "camera[${idx}] ${pose.image} writes=$activeForCam")
            }
            val fname = sanitizeName(pose.image) + ".npy"
            val outFile = File(depthDir, fname)
            writeNpyFloat32(outFile, width, height, depth)
            savedFiles += outFile
            onProgress?.invoke(idx + 1, totalCams)
        }
        Log.i(TAG, "sparse depth done, pixels=$totalActive files=${savedFiles.size}")

        // Save metadata for reference
        val meta = JSONObject().apply {
            put("width", width)
            put("height", height)
            put("sourceWidth", srcW)
            put("sourceHeight", srcH)
            put("points2d", totalActive)
            put("images", savedFiles.size)
            put("generatedAt", System.currentTimeMillis())
            put("runId", runId)
        }
        val metaFile = File(depthDir, "metadata.json")
        metaFile.writeText(meta.toString())
        Log.d(TAG, "metadata=${meta}")

        return Result(depthDir, savedFiles, totalActive, width, height, srcW, srcH, runId, metaFile)
    }

    private fun loadCameraModel(file: File): CameraModel {
        val text = file.readText()
        val obj = JSONObject(text)
        val kArr = obj.optJSONArray("K") ?: throw IllegalStateException("camera_poses.json 缺少 K 数组")
        if (kArr.length() < 6) throw IllegalStateException("K 数组长度不正确")
        val fx = kArr.optDouble(0)
        val fy = kArr.optDouble(4)
        val cx = kArr.optDouble(2)
        val cy = kArr.optDouble(5)
        val list = obj.optJSONArray("cameras") ?: JSONArray()
        val poses = ArrayList<CameraPose>(list.length())
        for (i in 0 until list.length()) {
            val item = list.optJSONObject(i) ?: continue
            val name = item.optString("image")
            val rArr = item.optJSONArray("R") ?: continue
            val tArr = item.optJSONArray("t") ?: continue
            if (rArr.length() < 9 || tArr.length() < 3) continue
            val R = DoubleArray(9) { idx -> rArr.optDouble(idx) }
            val t = DoubleArray(3) { idx -> tArr.optDouble(idx) }
            poses += CameraPose(name, R, t)
        }
        if (poses.isEmpty()) throw IllegalStateException("未在 JSON 中找到任何相机姿态")
        Log.d(TAG, "loadCameraModel ${poses.size} cameras")
        return CameraModel(fx, fy, cx, cy, poses)
    }

    private fun loadPoints(file: File): DoubleArray {
        RandomAccessFile(file, "r").use { raf ->
            var format = "ascii"
            var vertexCount = 0
            val vertexProps = mutableListOf<PlyProperty>()
            var inVertex = false
            while (true) {
                val raw = raf.readLine() ?: break
                val line = raw.trim()
                if (line.startsWith("format")) {
                    format = line.split(Regex("\\s+")).getOrNull(1)?.lowercase(Locale.getDefault()) ?: "ascii"
                } else if (line.startsWith("element")) {
                    val parts = line.split(Regex("\\s+"))
                    inVertex = parts.getOrNull(1) == "vertex"
                    if (inVertex) {
                        vertexCount = parts.getOrNull(2)?.toIntOrNull() ?: 0
                        Log.d(TAG, "PLY header vertex count=$vertexCount")
                        vertexProps.clear()
                    }
                } else if (line.startsWith("property") && inVertex) {
                    val parts = line.split(Regex("\\s+"))
                    if (parts.size >= 3 && parts[1] != "list") {
                        vertexProps.add(PlyProperty(parts[1], parts[2]))
                    }
                } else if (line.equals("end_header", ignoreCase = true)) {
                    break
                }
            }
            if (vertexCount <= 0) {
                throw IllegalStateException("PLY 中未声明顶点数量")
            }
            val points = DoubleArray(vertexCount * 3)
            val actual = when {
                format.contains("binary", ignoreCase = true) -> readBinaryVertices(raf, vertexCount, vertexProps, points)
                else -> readAsciiVertices(raf, vertexCount, points)
            }
            if (actual == 0) {
                throw IllegalStateException("PLY 中没有有效点")
            }
            return if (actual == vertexCount) points else points.copyOf(actual * 3)
        }
    }

    private fun readAsciiVertices(raf: RandomAccessFile, vertexCount: Int, points: DoubleArray): Int {
        var filled = 0
        var lineIdx = 0
        while (filled < vertexCount) {
            val raw = raf.readLine() ?: break
            lineIdx += 1
            val parts = raw.trim().split(Regex("\\s+"))
            if (parts.size < 3) continue
            val x = parts[0].toDoubleOrNull() ?: continue
            val y = parts[1].toDoubleOrNull() ?: continue
            val z = parts[2].toDoubleOrNull() ?: continue
            val idx = filled * 3
            points[idx] = x
            points[idx + 1] = y
            points[idx + 2] = z
            filled += 1
        }
        Log.d(TAG, "PLY ascii parsed points=$filled lines=$lineIdx")
        return filled
    }

    private fun readBinaryVertices(
        raf: RandomAccessFile,
        vertexCount: Int,
        props: List<PlyProperty>,
        points: DoubleArray
    ): Int {
        if (props.size < 3) throw IllegalStateException("PLY 顶点属性不足")
        val propSizes = props.map { plyTypeSize(it.type) }
        val scratch = ByteArray(8)
        val xIndex = props.indexOfFirst { it.name.equals("x", true) }.takeIf { it >= 0 } ?: 0
        val yIndex = props.indexOfFirst { it.name.equals("y", true) }.takeIf { it >= 0 } ?: if (props.size > 1) 1 else 0
        val zIndex = props.indexOfFirst { it.name.equals("z", true) }.takeIf { it >= 0 } ?: if (props.size > 2) 2 else props.lastIndex
        for (i in 0 until vertexCount) {
            var x = 0.0
            var y = 0.0
            var z = 0.0
            props.forEachIndexed { idx, prop ->
                val size = propSizes[idx]
                raf.readFully(scratch, 0, size)
                val value = readLittleEndian(scratch, size, prop.type)
                when (idx) {
                    xIndex -> x = value
                    yIndex -> y = value
                    zIndex -> z = value
                }
            }
            val base = i * 3
            points[base] = x
            points[base + 1] = y
            points[base + 2] = z
        }
        Log.d(TAG, "PLY binary parsed points=$vertexCount props=${props.size}")
        return vertexCount
    }

    private fun plyTypeSize(type: String): Int = when (type.lowercase(Locale.getDefault())) {
        "char", "uchar", "int8", "uint8" -> 1
        "short", "ushort", "int16", "uint16" -> 2
        "int", "uint", "float", "float32", "int32", "uint32" -> 4
        "double", "float64" -> 8
        else -> 4
    }

    private fun readLittleEndian(bytes: ByteArray, size: Int, type: String): Double {
        val order = ByteOrder.LITTLE_ENDIAN
        val buffer = ByteBuffer.wrap(bytes, 0, size).order(order)
        return when (type.lowercase(Locale.getDefault())) {
            "char", "int8" -> buffer.get(0).toDouble()
            "uchar", "uint8" -> (buffer.get(0).toInt() and 0xFF).toDouble()
            "short", "int16" -> buffer.short.toDouble()
            "ushort", "uint16" -> (buffer.short.toInt() and 0xFFFF).toDouble()
            "int", "int32" -> buffer.int.toDouble()
            "uint", "uint32" -> java.lang.Integer.toUnsignedLong(buffer.int).toDouble()
            "float", "float32" -> buffer.float.toDouble()
            "double", "float64" -> buffer.double
            else -> buffer.float.toDouble()
        }
    }

    private fun writeNpyFloat32(file: File, width: Int, height: Int, data: FloatArray) {
        file.outputStream().use { fos ->
            val channel = Channels.newChannel(fos)
            val magic = byteArrayOf(0x93.toByte()) + "NUMPY".toByteArray(Charsets.US_ASCII)
            fos.write(magic)
            fos.write(byteArrayOf(1, 0))
            val dict = "{'descr': '<f4', 'fortran_order': False, 'shape': ($height, $width), }"
            var padding = 16 - ((magic.size + 2 + 2 + dict.length + 1) % 16)
            if (padding == 16) padding = 0
            val header = (dict + " ".repeat(padding) + "\n").toByteArray(Charsets.US_ASCII)
            val headerLen = header.size
            fos.write(byteArrayOf((headerLen and 0xFF).toByte(), ((headerLen ushr 8) and 0xFF).toByte()))
            fos.write(header)
            val buffer = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)
            for (value in data) {
                if (buffer.remaining() < 4) {
                    buffer.flip()
                    while (buffer.hasRemaining()) {
                        channel.write(buffer)
                    }
                    buffer.clear()
                }
                buffer.putFloat(value)
            }
            buffer.flip()
            while (buffer.hasRemaining()) {
                channel.write(buffer)
            }
        }
    }

    private fun sanitizeName(name: String): String {
        val base = name.substringAfterLast('/')
            .substringAfterLast(File.separatorChar)
        return base.replace(Regex("[^A-Za-z0-9._-]"), "_")
    }

    private fun estimateFromIntrinsics(model: CameraModel): Pair<Int, Int> {
        val approxW = (model.cx * 2.0).roundToInt().coerceAtLeast(1)
        val approxH = (model.cy * 2.0).roundToInt().coerceAtLeast(1)
        return approxW to approxH
    }

    data class CameraModel(
        val fx: Double,
        val fy: Double,
        val cx: Double,
        val cy: Double,
        val cameras: List<CameraPose>
    )

    data class CameraPose(
        val image: String,
        val R: DoubleArray,
        val t: DoubleArray
    )

    companion object {
        private const val TAG = "SparseDepthGen"
        private const val CAMERAS_JSON = "camera_poses.json"
        private const val PLY_FILE = "reconstruction_points.ply"

        fun hasArtifacts(record: TaskRecord): Boolean {
            val outDir = File(record.outDir)
            return File(outDir, CAMERAS_JSON).exists() && File(outDir, PLY_FILE).exists()
        }

        fun estimateSourceResolution(outDir: File, extra: JSONObject?): Pair<Int, Int>? {
            val extraWidth = extra?.optInt("inputWidth", 0) ?: 0
            val extraHeight = extra?.optInt("inputHeight", 0) ?: 0
            if (extraWidth > 0 && extraHeight > 0) {
                return extraWidth to extraHeight
            }
            val cameraFile = File(outDir, CAMERAS_JSON)
            if (!cameraFile.exists()) return null
            return try {
                val obj = JSONObject(cameraFile.readText())
                val cams = obj.optJSONArray("cameras") ?: return null
                if (cams.length() == 0) return null
                for (i in 0 until cams.length()) {
                    val cam = cams.optJSONObject(i) ?: continue
                    val image = cam.optString("image")
                    val size = decodeImageSize(image, extra)
                    if (size != null) return size
                }
                val kArr = obj.optJSONArray("K") ?: return null
                if (kArr.length() < 6) return null
                val cx = kArr.optDouble(2)
                val cy = kArr.optDouble(5)
                val w = (cx * 2.0).roundToInt()
                val h = (cy * 2.0).roundToInt()
                if (w > 0 && h > 0) w to h else null
            } catch (e: Exception) {
                Log.e(TAG, "estimateSourceResolution failed", e)
                null
            }
        }

        private fun decodeImageSize(path: String?, extra: JSONObject?): Pair<Int, Int>? {
            if (path.isNullOrBlank()) return null
            val file = File(path)
            val candidates = mutableListOf<File>()
            if (file.isAbsolute && file.exists()) {
                candidates += file
            }
            val datasetDir = extra?.optString("datasetDir")?.takeIf { it.isNotBlank() }?.let { File(it) }
            if (datasetDir != null) {
                val rel = File(datasetDir, path)
                if (rel.exists()) candidates += rel
            }
            for (candidate in candidates) {
                val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(candidate.absolutePath, opts)
                if (opts.outWidth > 0 && opts.outHeight > 0) {
                    return opts.outWidth to opts.outHeight
                }
            }
            return null
        }
    }
}
