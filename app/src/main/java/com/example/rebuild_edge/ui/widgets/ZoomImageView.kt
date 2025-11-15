package com.example.rebuild_edge.ui.widgets

import android.content.Context
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.appcompat.widget.AppCompatImageView

class ZoomImageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : AppCompatImageView(context, attrs, defStyleAttr) {

    data class OverlayPoint(val x: Float, val y: Float, val depth: Float)

    private val matrixValues = FloatArray(9)
    private val drawMatrix = Matrix()
    private val scaleDetector = ScaleGestureDetector(context, ScaleListener())
    private val gestureDetector = GestureDetector(context, GestureListener())
    private val lastPoint = PointF()
    private var isDragging = false

    private var minScale = 1f
    private var maxScale = 5f

    init {
        scaleType = ScaleType.MATRIX
    }

    private val overlayRadius = resources.displayMetrics.density * 5f
    private val overlayTapRadius = overlayRadius * 2f
    private val overlayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF42A5F5.toInt()
        style = Paint.Style.FILL
    }
    private val overlayStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        style = Paint.Style.STROKE
        strokeWidth = overlayRadius * 0.45f
    }
    private val overlayHighlight = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFC107.toInt()
        style = Paint.Style.STROKE
        strokeWidth = overlayRadius * 0.65f
    }

    private var overlayPoints: List<OverlayPoint> = emptyList()
    private var drawnOverlays: List<DrawnOverlay> = emptyList()
    private var highlightedPoint: OverlayPoint? = null
    private var overlayClickListener: ((OverlayPoint) -> Unit)? = null

    private data class DrawnOverlay(val viewX: Float, val viewY: Float, val point: OverlayPoint)


    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        when (event.action and MotionEvent.ACTION_MASK) {
            MotionEvent.ACTION_DOWN -> {
                lastPoint.set(event.x, event.y)
                isDragging = true
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDragging && !scaleDetector.isInProgress) {
                    val dx = event.x - lastPoint.x
                    val dy = event.y - lastPoint.y
                    drawMatrix.postTranslate(dx, dy)
                    imageMatrix = drawMatrix
                    lastPoint.set(event.x, event.y)
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isDragging = false
            }
        }
        return true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (overlayPoints.isEmpty()) {
            drawnOverlays = emptyList()
            return
        }
        val selected = selectOverlayPoints()
        if (selected.isEmpty()) {
            drawnOverlays = emptyList()
            return
        }
        val matrix = imageMatrix
        val mapped = FloatArray(2)
        val updated = ArrayList<DrawnOverlay>(selected.size)
        for (point in selected) {
            mapped[0] = point.x
            mapped[1] = point.y
            matrix.mapPoints(mapped)
            canvas.drawCircle(mapped[0], mapped[1], overlayRadius, overlayPaint)
            canvas.drawCircle(mapped[0], mapped[1], overlayRadius, overlayStroke)
            if (point == highlightedPoint) {
                canvas.drawCircle(mapped[0], mapped[1], overlayRadius * 1.6f, overlayHighlight)
            }
            updated += DrawnOverlay(mapped[0], mapped[1], point)
        }
        drawnOverlays = updated
    }

    fun resetZoom() {
        drawMatrix.reset()
        imageMatrix = drawMatrix
    }

    private fun currentScale(): Float {
        drawMatrix.getValues(matrixValues)
        return matrixValues[Matrix.MSCALE_X]
    }

    fun setOverlayPoints(points: List<OverlayPoint>) {
        overlayPoints = points
        highlightedPoint = null
        invalidate()
    }

    fun setOnOverlayPointClickListener(listener: ((OverlayPoint) -> Unit)?) {
        overlayClickListener = listener
    }

    private fun selectOverlayPoints(): List<OverlayPoint> {
        if (overlayPoints.isEmpty()) return emptyList()
        val limit = maxPointsForScale(currentScale())
        val rect = computeVisibleRect()
        val currentDrawable = drawable
        val source = if (rect != null) {
            val padded = RectF(rect)
            val padding = (40f / currentScale().coerceAtLeast(0.01f)).coerceAtMost(400f)
            padded.inset(-padding, -padding)
            padded.left = padded.left.coerceAtLeast(0f)
            padded.top = padded.top.coerceAtLeast(0f)
            padded.right = padded.right.coerceAtMost(currentDrawable?.intrinsicWidth?.toFloat() ?: padded.right)
            padded.bottom = padded.bottom.coerceAtMost(currentDrawable?.intrinsicHeight?.toFloat() ?: padded.bottom)
            val filtered = overlayPoints.filter { padded.contains(it.x, it.y) }
            if (filtered.isEmpty()) overlayPoints else filtered
        } else {
            overlayPoints
        }
        return downsampleList(source, limit)
    }

    private fun downsampleList(list: List<OverlayPoint>, limit: Int): List<OverlayPoint> {
        if (list.size <= limit) return list
        val step = (list.size + limit - 1) / limit
        return list.filterIndexed { idx, _ -> idx % step == 0 }.take(limit)
    }

    private fun maxPointsForScale(scale: Float): Int {
        val base = 120
        val safe = scale.coerceAtLeast(0.01f)
        val scaled = (safe * safe * base).toInt()
        return scaled.coerceIn(base, 1200)
    }

    private fun computeVisibleRect(): RectF? {
        val drawable = drawable ?: return null
        val drawableW = drawable.intrinsicWidth.toFloat()
        val drawableH = drawable.intrinsicHeight.toFloat()
        if (drawableW <= 0f || drawableH <= 0f) return null
        val inv = Matrix()
        if (!imageMatrix.invert(inv)) return null
        val corners = floatArrayOf(
            0f, 0f,
            width.toFloat(), 0f,
            width.toFloat(), height.toFloat(),
            0f, height.toFloat()
        )
        inv.mapPoints(corners)
        val xs = listOf(corners[0], corners[2], corners[4], corners[6])
        val ys = listOf(corners[1], corners[3], corners[5], corners[7])
        val rect = RectF(
            xs.minOrNull() ?: 0f,
            ys.minOrNull() ?: 0f,
            xs.maxOrNull() ?: drawableW,
            ys.maxOrNull() ?: drawableH
        )
        rect.left = rect.left.coerceAtLeast(0f)
        rect.top = rect.top.coerceAtLeast(0f)
        rect.right = rect.right.coerceAtMost(drawableW)
        rect.bottom = rect.bottom.coerceAtMost(drawableH)
        return if (rect.width() <= 0f || rect.height() <= 0f) null else rect
    }

    private fun findNearbyOverlay(x: Float, y: Float): DrawnOverlay? {
        val radiusSq = overlayTapRadius * overlayTapRadius
        var best: DrawnOverlay? = null
        var bestDistance = Float.MAX_VALUE
        for (entry in drawnOverlays) {
            val dx = entry.viewX - x
            val dy = entry.viewY - y
            val dist = dx * dx + dy * dy
            if (dist <= radiusSq && dist < bestDistance) {
                bestDistance = dist
                best = entry
            }
        }
        return best
    }

    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val scaleFactor = detector.scaleFactor
            val current = currentScale()
            val newScale = (current * scaleFactor).coerceIn(minScale, maxScale)
            val delta = newScale / current
            drawMatrix.postScale(delta, delta, detector.focusX, detector.focusY)
            imageMatrix = drawMatrix
            return true
        }

        override fun onScaleBegin(detector: ScaleGestureDetector): Boolean {
            isDragging = false
            return true
        }
    }

    private inner class GestureListener : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(e: MotionEvent): Boolean {
            if (currentScale() > minScale + 0.1f) {
                drawMatrix.reset()
            } else {
                val targetScale = maxScale.coerceAtMost(minScale * 2f)
                drawMatrix.postScale(targetScale, targetScale, e.x, e.y)
            }
            imageMatrix = drawMatrix
            return true
        }

        override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
            val match = findNearbyOverlay(e.x, e.y)
            return if (match != null) {
                highlightedPoint = match.point
                overlayClickListener?.invoke(match.point)
                invalidate()
                true
            } else {
                super.onSingleTapConfirmed(e)
            }
        }
    }
}
