package com.example.rebuild_edge.ui.tasks

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.rebuild_edge.R
import com.example.rebuild_edge.data.TaskRecord
import com.example.rebuild_edge.data.TaskStore
import com.example.rebuild_edge.databinding.FragmentTasksBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TasksFragment : Fragment() {
    private var _binding: FragmentTasksBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentTasksBinding.inflate(inflater, container, false)
        val root = binding.root

        binding.recyclerTasks.layoutManager = LinearLayoutManager(requireContext())
        refresh()
        return root
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    private fun refresh() {
        val items = TaskStore.loadAll(requireContext()).sortedBy { it.startedAt }
        binding.emptyView.visibility = if (items.isEmpty()) View.VISIBLE else View.GONE
        if (items.isEmpty()) {
            binding.recyclerTasks.adapter = null
        } else {
            binding.recyclerTasks.adapter = TasksAdapter(items.reversed(), ::onTaskClick)
        }
    }

    private fun onTaskClick(task: TaskRecord) {
        val args = Bundle().apply { putString("taskId", task.id) }
        findNavController().navigate(R.id.nav_task_detail, args)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

