package com.example.rebuild_edge.ui.tasks

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.rebuild_edge.data.TaskRecord
import com.example.rebuild_edge.data.TaskStore
import com.example.rebuild_edge.databinding.FragmentTaskDetailBinding
import com.example.rebuild_edge.databinding.ItemFileBinding
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TaskDetailFragment : Fragment() {
    private var _binding: FragmentTaskDetailBinding? = null
    private val binding get() = _binding!!
    private var record: TaskRecord? = null
    private var files: List<File> = emptyList()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentTaskDetailBinding.inflate(inflater, container, false)
        val root = binding.root

        val id = arguments?.getString("taskId")
        record = id?.let { TaskStore.getById(requireContext(), it) }

        binding.recyclerFiles.layoutManager = LinearLayoutManager(requireContext())

        bindUi()
        return root
    }

    private fun bindUi() {
        val rec = record ?: return
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        val ts = fmt.format(Date(rec.startedAt))
        val mins = (rec.durationMs / 1000) / 60
        val secs = (rec.durationMs / 1000) % 60
        val typeLabel = when (rec.taskType) {
            TaskRecord.TYPE_COLMAP -> "COLMAP"
            else -> "OpenCV"
        }
        val extra = rec.extraJson.takeIf { it.isNotBlank() }?.let { runCatching { JSONObject(it) }.getOrNull() }
        binding.txtSummary.text = buildString {
            append(if (rec.success) "✅ 成功" else "⚠️ 可能失败")
            append(" · 类型 ").append(typeLabel)
            append(" · 开始时间 ").append(ts)
            append(" · 用时 ").append(mins).append("分").append(secs).append("秒")
            if (rec.taskType == TaskRecord.TYPE_COLMAP) {
                append("\nCOLMAP 参数: camera=")
                    .append(extra?.optString("cameraModel") ?: "--")
                append(", overlap=")
                    .append(extra?.optInt("sequentialOverlap", rec.mode) ?: rec.mode)
                append(", gps=")
                    .append(extra?.optInt("maxGpsNeighborDist", rec.stride) ?: rec.stride)
                append(", threads=")
                    .append(extra?.optInt("threads", rec.window) ?: rec.window)
                append(", alignType=")
                    .append(extra?.optString("alignmentType") ?: "ecef")
                append(", 对齐=").append(rec.alignUsingGps)
            } else {
                append("\n参数: 对齐=").append(rec.alignUsingGps)
                append(", maxEdge=").append(rec.maxLongEdge)
                append(", 模式=").append(rec.mode)
                append(", stride=").append(rec.stride)
                append(", window=").append(rec.window)
                append(", k=").append(rec.kNeighbors)
            }
        }

        // Load log (best-effort)
        val logText = runCatching { File(rec.logPath).takeIf { it.exists() }?.readText() ?: "" }.getOrDefault("")
        binding.txtLog.text = if (logText.isNotEmpty()) logText else "(没有找到日志文件)"

        // List output files
        val outDir = File(rec.outDir)
        // Only list regular files; skip directories like "sparse"/"sparse_aligned"
        files = outDir.listFiles()
            ?.filter { it.isFile }
            ?.sortedBy { it.name.lowercase(Locale.getDefault()) }
            ?: emptyList()
        binding.recyclerFiles.adapter = FilesAdapter(files, ::shareSingle)

        binding.btnShareLog.setOnClickListener {
            shareUris(listOf(File(rec.logPath)))
        }
        binding.btnShareAll.setOnClickListener {
            shareUris(files)
        }
    }

    private fun shareSingle(file: File) {
        shareUris(listOf(file))
    }

    private fun shareUris(files: List<File>) {
        if (files.isEmpty()) return
        val ctx = requireContext()
        val auth = ctx.packageName + ".fileprovider"
        val uris = files.filter { it.exists() }.map { f ->
            FileProvider.getUriForFile(ctx, auth, f)
        }
        if (uris.isEmpty()) return
        val intent = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).apply {
                type = "*/*"
                putExtra(Intent.EXTRA_STREAM, uris.first())
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                type = "*/*"
                putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList<Uri>(uris))
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }
        startActivity(Intent.createChooser(intent, "分享输出文件"))
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

class FilesAdapter(
    private val files: List<File>,
    private val onShare: (File) -> Unit
) : androidx.recyclerview.widget.RecyclerView.Adapter<FilesAdapter.VH>() {
    class VH(val binding: ItemFileBinding) : androidx.recyclerview.widget.RecyclerView.ViewHolder(binding.root)
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val binding = ItemFileBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return VH(binding)
    }
    override fun getItemCount(): Int = files.size
    override fun onBindViewHolder(holder: VH, position: Int) {
        val f = files[position]
        holder.binding.txtFileName.text = f.name
        holder.binding.btnShare.setOnClickListener { onShare(f) }
        holder.binding.root.setOnClickListener { onShare(f) }
    }
}
