package com.example.rebuild_edge.ui.tasks

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import com.example.rebuild_edge.data.TaskRecord
import com.example.rebuild_edge.data.TaskStore
import com.example.rebuild_edge.databinding.FragmentTaskLogBinding
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TaskLogFragment : Fragment() {
    private var _binding: FragmentTaskLogBinding? = null
    private val binding get() = _binding!!
    private var record: TaskRecord? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentTaskLogBinding.inflate(inflater, container, false)
        val id = arguments?.getString("taskId")
        record = id?.let { TaskStore.getById(requireContext(), it) }
        bindUi()
        return binding.root
    }

    private fun bindUi() {
        val rec = record
        if (rec == null) {
            binding.txtLogSummary.text = "找不到任务记录"
            binding.txtLogContent.text = ""
            binding.btnShareLog.isEnabled = false
            return
        }
        val logFile = File(rec.logPath)
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        binding.txtLogSummary.text = buildString {
            append("任务: ").append(rec.id)
            append("\n开始时间: ").append(fmt.format(Date(rec.startedAt)))
            append("\n日志路径: ").append(logFile.absolutePath)
        }
        val logText = runCatching { logFile.takeIf { it.exists() }?.readText() ?: "" }.getOrDefault("")
        binding.txtLogContent.text = if (logText.isNotEmpty()) logText else "(没有找到日志文件)"
        binding.btnShareLog.isEnabled = logFile.exists()
        binding.btnShareLog.setOnClickListener {
            if (!logFile.exists()) {
                Toast.makeText(requireContext(), "找不到日志文件", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            shareLog(logFile)
        }
    }

    private fun shareLog(file: File) {
        val ctx = requireContext()
        val auth = ctx.packageName + ".fileprovider"
        val uri = FileProvider.getUriForFile(ctx, auth, file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "分享日志"))
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
