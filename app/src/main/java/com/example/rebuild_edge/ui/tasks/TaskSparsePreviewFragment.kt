package com.example.rebuild_edge.ui.tasks

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.documentfile.provider.DocumentFile
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.example.rebuild_edge.databinding.FragmentTaskSparsePreviewBinding
import com.example.rebuild_edge.ui.widgets.ZoomImageView
import com.example.rebuild_edge.util.NpyReader
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class TaskSparsePreviewFragment : Fragment() {

    private var _binding: FragmentTaskSparsePreviewBinding? = null
    private val binding get() = _binding!!

    private var runDir: String? = null
    private var runId: String? = null
    private var files: List<File> = emptyList()
    private var currentIndex = 0
    private var cameraImageMap: Map<String, File> = emptyMap()
    private var rgbFolderUri: Uri? = null
    private var rgbImageMap: Map<String, Uri> = emptyMap()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentTaskSparsePreviewBinding.inflate(inflater, container, false)
        runDir = arguments?.getString("runDir")
        runId = arguments?.getString("runId")
        setupUi()
        return binding.root
    }

    private fun setupUi() {
        val dir = runDir?.let { File(it) }
        if (dir == null || !dir.isDirectory) {
            Toast.makeText(requireContext(), "稀疏深度目录无效", Toast.LENGTH_SHORT).show()
            binding.txtRunInfo.text = "目录无效"
            return
        }
        cameraImageMap = loadCameraImageMap(dir.parentFile)
        files = dir.listFiles { f -> f.isFile && f.name.lowercase(Locale.getDefault()).endsWith(".npy") }?.sortedBy { it.name }
            ?: emptyList()
        if (files.isEmpty()) {
            binding.txtRunInfo.text = "未找到 NPY 文件"
            return
        }
        val suffix = if (cameraImageMap.isEmpty()) " · (缺少相机信息/图像)" else ""
        binding.txtRunInfo.text = "ID=${runId ?: "--"} · ${files.size} 张$suffix"
        binding.btnPrev.setOnClickListener { showFrame(currentIndex - 1) }
        binding.btnNext.setOnClickListener { showFrame(currentIndex + 1) }
        binding.imageDepth.setOnOverlayPointClickListener { point ->
            val depthText = String.format(Locale.getDefault(), "深度: %.3f m", point.depth.toDouble())
            Toast.makeText(requireContext(), depthText, Toast.LENGTH_SHORT).show()
        }
        binding.btnSelectRgbFolder.setOnClickListener { startSelectRgbFolder() }
        binding.txtRgbFolderInfo.visibility = View.GONE
        showFrame(0)
    }

    private fun showFrame(index: Int) {
        if (files.isEmpty()) return
        val newIndex = index.coerceIn(0, files.lastIndex)
        currentIndex = newIndex
        val file = files[newIndex]
        binding.btnPrev.isEnabled = currentIndex > 0
        binding.btnNext.isEnabled = currentIndex < files.lastIndex
        binding.txtFrameInfo.text = "加载 ${file.name} (${newIndex + 1}/${files.size}) ..."
        binding.progressLoading.visibility = View.VISIBLE
        binding.imageDepth.resetZoom()
        viewLifecycleOwner.lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { loadDepthBitmap(file) }
            }
            binding.progressLoading.visibility = View.GONE
            if (result.isSuccess) {
                val frame = result.getOrThrow()
                binding.imageDepth.setImageBitmap(frame.bitmap)
                binding.imageDepth.setOverlayPoints(frame.overlayPoints)
                val stats = frame.stats
                binding.txtFrameInfo.text = buildString {
                    append("${file.name} (${currentIndex + 1}/${files.size})\n")
                    append("分辨率: ${frame.width}x${frame.height} · 有效像素=${stats.validPixels}\n")
                    append("深度范围: %.3f ~ %.3f".format(Locale.getDefault(), stats.minDepth, stats.maxDepth))
                    append("\n采样点: ${frame.overlayPoints.size}")
                }
            } else {
                binding.imageDepth.setImageBitmap(null)
                binding.imageDepth.setOverlayPoints(emptyList())
                val err = result.exceptionOrNull()
                if (err != null) {
                    Log.e("SparsePreview", "Failed to load ${file.absolutePath}", err)
                }
                binding.txtFrameInfo.text = "加载失败 (${file.name}): ${err?.localizedMessage ?: "未知错误"}"
            }
        }
    }

    private fun loadDepthBitmap(file: File): LoadedFrame {
        val arr = NpyReader.read(file)
        val width = arr.width
        val height = arr.height
        val floats = arr.data
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        var valid = 0
        for (value in floats) {
            if (value.isFinite() && value > 0f) {
                valid += 1
                if (value < min) min = value
                if (value > max) max = value
            }
        }
        if (valid == 0) {
            min = 0f
            max = 1f
        }
        val overlayPoints = collectOverlayPoints(floats, width, height, valid)
        val colorBitmap = loadColorBitmapForDepth(file, width, height)
        val displayBitmap = colorBitmap ?: createGrayscaleBitmap(floats, width, height, min, max)
        return LoadedFrame(
            displayBitmap,
            width,
            height,
            Stats(valid, min.toDouble(), max.toDouble()),
            overlayPoints
        )
    }

    private fun collectOverlayPoints(
        data: FloatArray,
        width: Int,
        height: Int,
        validCount: Int
    ): List<ZoomImageView.OverlayPoint> {
        if (validCount == 0) return emptyList()
        val maxStored = 120_000
        val step = (validCount + maxStored - 1) / maxStored
        val points = ArrayList<ZoomImageView.OverlayPoint>((validCount / step).coerceAtLeast(0))
        var seen = 0
        for (y in 0 until height) {
            val rowStart = y * width
            for (x in 0 until width) {
                val value = data[rowStart + x]
                if (value.isFinite() && value > 0f) {
                    if (seen % step == 0) {
                        points.add(ZoomImageView.OverlayPoint(x.toFloat(), y.toFloat(), value))
                    }
                    seen += 1
                }
            }
        }
        return points
    }

    private fun loadColorBitmapForDepth(depthFile: File, targetWidth: Int, targetHeight: Int): Bitmap? {
        val key = normalizedDepthKey(depthFile)
        rgbImageMap[key]?.let { uri ->
            val decoded = decodeImageForPreview(uri, targetWidth, targetHeight)
            Log.d("SparsePreview", "RGB override match for ${depthFile.name}: uri=$uri decoded=${decoded != null}")
            return decoded
        }
        cameraImageMap[key]?.let { imageFile ->
            val decoded = decodeImageForPreview(imageFile, targetWidth, targetHeight)
            Log.d("SparsePreview", "camera_poses match for ${depthFile.name}: file=${imageFile.absolutePath} decoded=${decoded != null}")
            return decoded
        }
        Log.d("SparsePreview", "No RGB match for ${depthFile.name}, key=$key")
        return null
    }

    private fun decodeImageForPreview(imageFile: File, targetWidth: Int, targetHeight: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(imageFile.absolutePath, bounds)
        val srcW = bounds.outWidth
        val srcH = bounds.outHeight
        if (srcW <= 0 || srcH <= 0) return null
        var sample = 1
        while (srcW.toDouble() / sample > targetWidth * 1.3 && srcH.toDouble() / sample > targetHeight * 1.3) {
            sample *= 2
        }
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = BitmapFactory.decodeFile(imageFile.absolutePath, opts) ?: return null
        return if (decoded.width == targetWidth && decoded.height == targetHeight) {
            decoded
        } else {
            Bitmap.createScaledBitmap(decoded, targetWidth, targetHeight, true).also {
                if (it != decoded) decoded.recycle()
            }
        }
    }

    private fun decodeImageForPreview(uri: Uri, targetWidth: Int, targetHeight: Int): Bitmap? {
        val resolver = requireContext().contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        val srcW = bounds.outWidth
        val srcH = bounds.outHeight
        if (srcW <= 0 || srcH <= 0) return null
        var sample = 1
        while (srcW.toDouble() / sample > targetWidth * 1.3 && srcH.toDouble() / sample > targetHeight * 1.3) {
            sample *= 2
        }
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) } ?: return null
        return if (decoded.width == targetWidth && decoded.height == targetHeight) {
            decoded
        } else {
            Bitmap.createScaledBitmap(decoded, targetWidth, targetHeight, true).also {
                if (it != decoded) decoded.recycle()
            }
        }
    }

    private fun createGrayscaleBitmap(
        data: FloatArray,
        width: Int,
        height: Int,
        min: Float,
        max: Float
    ): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val range = if (max > min) max - min else 1f
        val pixels = IntArray(width * height)
        for (i in data.indices) {
            val v = data[i]
            val norm = if (v.isFinite() && v > 0f) ((v - min) / range).coerceIn(0f, 1f) else 0f
            val gray = (norm * 255f).toInt().coerceIn(0, 255)
            pixels[i] = Color.rgb(gray, gray, gray)
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun loadCameraImageMap(parent: File?): Map<String, File> {
        val result = mutableMapOf<String, File>()
        val jsonFile = parent?.let { File(it, "camera_poses.json") } ?: return result
        if (!jsonFile.exists()) return result
        return runCatching {
            val cameras = JSONObject(jsonFile.readText()).optJSONArray("cameras") ?: return@runCatching result
            for (i in 0 until cameras.length()) {
                val item = cameras.optJSONObject(i) ?: continue
                val path = item.optString("image").takeIf { it.isNotBlank() } ?: continue
                val key = normalizedName(File(path).nameWithoutExtension)
                result[key] = File(path)
            }
            result
        }.getOrElse {
            Log.w("SparsePreview", "Failed to load camera_poses.json", it)
            result
        }
    }

    private fun startSelectRgbFolder() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            putExtra(DocumentsContract.EXTRA_INITIAL_URI, rgbFolderUri)
        }
        startActivityForResult(intent, REQ_SELECT_RGB_FOLDER)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQ_SELECT_RGB_FOLDER || resultCode != Activity.RESULT_OK) return
        val uri = data?.data ?: return
        val takeFlags = data.flags and
            (Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        if (takeFlags != 0) {
            requireContext().contentResolver.takePersistableUriPermission(uri, takeFlags)
        }
        rgbFolderUri = uri
        rgbImageMap = loadImagesFromFolder(uri)
        binding.txtRgbFolderInfo.apply {
            text = if (rgbImageMap.isEmpty()) {
                "RGB: ${describeUri(uri)}（未发现可用图片）"
            } else {
                "RGB: ${describeUri(uri)}"
            }
            visibility = View.VISIBLE
        }
        Log.d("SparsePreview", "RGB folder selected: ${describeUri(uri)}, matches=${rgbImageMap.size}")
        if (rgbImageMap.isNotEmpty()) {
            Log.d("SparsePreview", "RGB files: ${rgbImageMap.keys.take(5)}")
        }
        showFrame(currentIndex)
    }

    private fun loadImagesFromFolder(treeUri: Uri): Map<String, Uri> {
        val root = DocumentFile.fromTreeUri(requireContext(), treeUri) ?: return emptyMap()
        val result = mutableMapOf<String, Uri>()
        collectDocumentImages(root, result)
        Log.d("SparsePreview", "Loaded ${result.size} RGB images from ${describeUri(treeUri)}")
        return result
    }

    private fun collectDocumentImages(dir: DocumentFile, map: MutableMap<String, Uri>) {
        dir.listFiles().forEach { child ->
            if (child.isDirectory) {
                collectDocumentImages(child, map)
                return@forEach
            }
            if (!child.isFile) return@forEach
            val name = child.name ?: return@forEach
            if (!isImageFile(name)) return@forEach
            val baseName = name.substringBeforeLast('.')
            if (baseName.isBlank()) return@forEach
            val key = normalizedName(baseName)
            map[key] = child.uri
        }
    }

    private fun isImageFile(name: String): Boolean {
        val lower = name.lowercase(Locale.getDefault())
        return lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png")
    }

    private fun describeUri(uri: Uri): String {
        val doc = DocumentFile.fromTreeUri(requireContext(), uri)
        return doc?.name?.takeUnless { it.isBlank() }
            ?: uri.lastPathSegment
            ?: uri.toString()
    }

    private fun normalizedDepthKey(file: File): String {
        var base = file.nameWithoutExtension
        while (base.contains('.')) {
            base = base.substringBeforeLast('.')
        }
        return normalizedName(base)
    }

    private fun normalizedName(name: String): String {
        return sanitizeName(name).lowercase(Locale.ROOT)
    }

    private fun sanitizeName(name: String): String {
        val base = name.substringAfterLast('/').substringAfterLast(File.separatorChar)
        return base.replace(Regex("[^A-Za-z0-9._-]"), "_")
    }

    data class LoadedFrame(
        val bitmap: Bitmap,
        val width: Int,
        val height: Int,
        val stats: Stats,
        val overlayPoints: List<ZoomImageView.OverlayPoint>
    )
    data class Stats(val validPixels: Int, val minDepth: Double, val maxDepth: Double)

    companion object {
        private const val REQ_SELECT_RGB_FOLDER = 0xAC3E
    }
    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
