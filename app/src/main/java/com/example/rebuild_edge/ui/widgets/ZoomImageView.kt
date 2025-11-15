package com.example.rebuild_edge.ui.widgets

import android.content.Context
import android.graphics.Matrix
import android.graphics.PointF
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

    fun resetZoom() {
        drawMatrix.reset()
        imageMatrix = drawMatrix
    }

    private fun currentScale(): Float {
        drawMatrix.getValues(matrixValues)
        return matrixValues[Matrix.MSCALE_X]
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
    }
}
