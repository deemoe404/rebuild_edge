package com.example.rebuild_edge.ui.tasks

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.rebuild_edge.data.TaskRecord
import com.example.rebuild_edge.databinding.ItemTaskBinding
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TasksAdapter(
    private val items: List<TaskRecord>,
    private val onClick: (TaskRecord) -> Unit
) : RecyclerView.Adapter<TasksAdapter.VH>() {

    class VH(val binding: ItemTaskBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val binding = ItemTaskBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return VH(binding)
    }

    override fun getItemCount(): Int = items.size

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = items[position]
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        val ts = fmt.format(Date(item.startedAt))
        val mins = (item.durationMs / 1000) / 60
        val secs = (item.durationMs / 1000) % 60
        val typeLabel = when (item.taskType) {
            TaskRecord.TYPE_COLMAP -> "COLMAP"
            else -> "OpenCV"
        }
        holder.binding.txtTitle.text = (if (item.success) "✅ " else "⚠️ ") + "$typeLabel @ $ts"
        val extra = item.extraJson.takeIf { it.isNotBlank() }?.let { runCatching { JSONObject(it) }.getOrNull() }
        val badges = buildBadges(item, extra)
        if (badges.isEmpty()) {
            holder.binding.txtBadges.visibility = View.GONE
        } else {
            holder.binding.txtBadges.visibility = View.VISIBLE
            holder.binding.txtBadges.text = badges.joinToString(" · ")
        }
        val detail = if (item.taskType == TaskRecord.TYPE_COLMAP) {
            val camera = extra?.optString("cameraModel") ?: "OPENCV"
            val overlap = extra?.optInt("sequentialOverlap", item.mode) ?: item.mode
            val gps = extra?.optInt("maxGpsNeighborDist", item.stride) ?: item.stride
            "COLMAP · 相机=$camera · overlap=$overlap · gps=$gps · 输出=${item.outDir}"
        } else {
            "模式${item.mode} · 输出目录：${item.outDir}"
        }
        holder.binding.txtSubtitle.text =
            "用时 ${mins}分${secs}秒 · 输入${item.inputCount}张 · $detail"
        holder.binding.root.setOnClickListener { onClick(item) }
    }

    private fun buildBadges(task: TaskRecord, extra: JSONObject?): List<String> {
        val badges = mutableListOf<String>()
        val tagsArray = extra?.optJSONArray("tags")
        if (tagsArray != null) {
            for (i in 0 until tagsArray.length()) {
                when (val tag = tagsArray.optString(i)) {
                    "sfm" -> badges.addOnce("SFM")
                    "sparse_depth" -> badges.addOnce("稀疏深度")
                    else -> if (tag.isNotBlank()) badges.addOnce(tag)
                }
            }
        }
        if (extra?.has("sparseDepth") == true) {
            badges.addOnce("稀疏深度")
        }
        if (badges.isEmpty()) {
            if (task.taskType == TaskRecord.TYPE_COLMAP) badges.add("COLMAP") else badges.add("SFM")
        }
        return badges
    }

    private fun MutableList<String>.addOnce(label: String) {
        if (!this.any { it == label }) {
            add(label)
        }
    }
}
