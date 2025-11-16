package com.example.rebuild_edge.util

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.text.Charsets

object NpyReader {
    data class Result(val width: Int, val height: Int, val data: FloatArray)

    fun read(file: File): Result {
        file.inputStream().use { input ->
            return read(input)
        }
    }

    fun read(inputStream: InputStream): Result {
        val bytes = inputStream.readBytes()
        require(bytes.size >= 10) { "NPY 文件过短" }
        val buffer = ByteBuffer.wrap(bytes)
        val magic = ByteArray(6)
        buffer.get(magic)
        val numpyMagic = byteArrayOf(
            0x93.toByte(),
            'N'.code.toByte(),
            'U'.code.toByte(),
            'M'.code.toByte(),
            'P'.code.toByte(),
            'Y'.code.toByte()
        )
        require(magic.contentEquals(numpyMagic)) { "不是有效的 NPY 文件" }

        val major = buffer.get().toInt() and 0xFF
        buffer.get() // minor, unused
        val headerLen = when (major) {
            1 -> buffer.short.toInt() and 0xFFFF
            2 -> buffer.int
            else -> buffer.short.toInt() and 0xFFFF
        }
        require(headerLen in 1..bytes.size) { "NPY header 长度无效: $headerLen" }
        val headerBytes = ByteArray(headerLen)
        buffer.get(headerBytes)
        val header = String(headerBytes, Charsets.US_ASCII)
        val shapePart = Regex("'shape'\\s*:\\s*\\(([^)]*)\\)").find(header)
            ?: error("无法在 NPY header 中解析 shape: $header")
        val dims = Regex("\\d+")
            .findAll(shapePart.groupValues[1])
            .map { it.value.toInt() }
            .toList()
        require(dims.size >= 2) { "NPY shape 不是二维: ${shapePart.groupValues[1]}" }
        val height = dims[0]
        val width = dims[1]
        val total = dims.fold(1L) { acc, dim -> acc * dim }
        require(total > 0) { "NPY 数据为空: dims=${dims.joinToString()}" }

        val descr = Regex("'descr'\\s*:\\s*'([^']+)'")
            .find(header)?.groupValues?.getOrNull(1) ?: "<f4"
        val (byteOrder, typeChar, typeSize) = parseDescr(descr)
        require(typeChar == 'f') { "仅支持浮点类型 NPY，当前 descr=$descr" }
        val fortranOrder = Regex("'fortran_order'\\s*:\\s*(True|False)")
            .find(header)?.groupValues?.getOrNull(1)?.equals("True", ignoreCase = true) == true

        val payloadSize = (total * typeSize).toInt()
        require(payloadSize <= buffer.remaining()) { "NPY 数据长度不足: 需要 $payloadSize, 剩余 ${buffer.remaining()}" }
        val payload = ByteArray(payloadSize)
        buffer.get(payload)
        val dataBuffer = ByteBuffer.wrap(payload).order(byteOrder)
        val flattened = FloatArray(total.toInt())
        when (typeSize) {
            4 -> {
                dataBuffer.asFloatBuffer().get(flattened)
            }
            8 -> {
                var idx = 0
                while (idx < flattened.size) {
                    if (dataBuffer.remaining() < 8) break
                    flattened[idx] = dataBuffer.double.toFloat()
                    idx += 1
                }
                require(idx == flattened.size) { "NPY double 数据长度不足: 写入 $idx / ${flattened.size}" }
            }
            else -> error("不支持的 dtype 大小: $typeSize bytes")
        }

        val data = FloatArray(width * height)
        copyWithOrder(flattened, data, width, height, fortranOrder)
        return Result(width, height, data)
    }

    private fun parseDescr(descr: String): Triple<ByteOrder, Char, Int> {
        if (descr.length < 3) return Triple(ByteOrder.LITTLE_ENDIAN, 'f', 4)
        val orderChar = descr[0]
        val typeChar = descr[1]
        val size = descr.substring(2).toIntOrNull() ?: 4
        val byteOrder = when (orderChar) {
            '<' -> ByteOrder.LITTLE_ENDIAN
            '>' -> ByteOrder.BIG_ENDIAN
            else -> ByteOrder.nativeOrder()
        }
        return Triple(byteOrder, typeChar, size)
    }

    private fun copyWithOrder(src: FloatArray, dst: FloatArray, width: Int, height: Int, fortran: Boolean) {
        if (!fortran) {
            val length = min(src.size, dst.size)
            System.arraycopy(src, 0, dst, 0, length)
            return
        }
        var idx = 0
        for (x in 0 until width) {
            for (y in 0 until height) {
                val dstIndex = y * width + x
                if (dstIndex >= dst.size || idx >= src.size) break
                dst[dstIndex] = src[idx]
                idx += 1
            }
        }
    }
}
