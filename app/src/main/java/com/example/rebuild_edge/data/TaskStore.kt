package com.example.rebuild_edge.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

object TaskStore {
    private const val FILE_NAME = "tasks.json"

    private fun file(context: Context): File = File(context.filesDir, FILE_NAME)

    fun loadAll(context: Context): List<TaskRecord> {
        val f = file(context)
        if (!f.exists()) return emptyList()
        val text = runCatching { f.readText() }.getOrDefault("[]")
        val arr = runCatching { JSONArray(text) }.getOrNull() ?: JSONArray()
        val list = ArrayList<TaskRecord>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            list.add(fromJson(o))
        }
        return list
    }

    fun getById(context: Context, id: String): TaskRecord? {
        return loadAll(context).firstOrNull { it.id == id }
    }

    fun add(context: Context, record: TaskRecord) {
        val f = file(context)
        val arr = JSONArray().apply {
            // Append existing first (oldest first kept), then new
            loadAll(context).forEach { put(toJson(it)) }
            put(toJson(record))
        }
        runCatching { f.writeText(arr.toString()) }
    }

    fun newId(): String = UUID.randomUUID().toString()

    private fun toJson(t: TaskRecord): JSONObject {
        return JSONObject().apply {
            put("id", t.id)
            put("startedAt", t.startedAt)
            put("durationMs", t.durationMs)
            put("outDir", t.outDir)
            put("logPath", t.logPath)
            put("inputCount", t.inputCount)
            put("alignUsingGps", t.alignUsingGps)
            put("maxLongEdge", t.maxLongEdge)
            put("mode", t.mode)
            put("stride", t.stride)
            put("window", t.window)
            put("kNeighbors", t.kNeighbors)
            put("resultSummary", t.resultSummary)
            put("success", t.success)
            put("taskType", t.taskType)
            put("extraJson", t.extraJson)
        }
    }

    private fun fromJson(o: JSONObject): TaskRecord {
        return TaskRecord(
            id = o.optString("id"),
            startedAt = o.optLong("startedAt"),
            durationMs = o.optLong("durationMs"),
            outDir = o.optString("outDir"),
            logPath = o.optString("logPath"),
            inputCount = o.optInt("inputCount"),
            alignUsingGps = o.optBoolean("alignUsingGps"),
            maxLongEdge = o.optInt("maxLongEdge"),
            mode = o.optInt("mode"),
            stride = o.optInt("stride"),
            window = o.optInt("window"),
            kNeighbors = o.optInt("kNeighbors"),
            resultSummary = o.optString("resultSummary"),
            success = o.optBoolean("success", false),
            taskType = o.optString("taskType", TaskRecord.TYPE_OPENCV),
            extraJson = o.optString("extraJson", "")
        )
    }
}
