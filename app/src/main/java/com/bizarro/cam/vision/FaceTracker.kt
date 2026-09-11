package com.bizarro.cam.vision

import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions

/**
 * Face contour extraction fed by CameraX MLKitAnalyzer in COORDINATE_SYSTEM_VIEW_REFERENCED
 * mode: points arrive already rotated and mapped into the preview surface pixel space,
 * so normalization is a plain divide by result.size. Residual rotation/mirror can be
 * calibrated at runtime (long-press WIRE).
 */
class FaceTracker {

    val detector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setMinFaceSize(0.12f)
            .enableTracking()
            .build()
    )

    @Volatile var frameW = 720
    @Volatile var frameH = 1280
    @Volatile var overlayRotation = 0
    @Volatile var overlayMirror = false
    @Volatile var points = FloatArray(0)
    @Volatile var wire = FloatArray(0)
    @Volatile var boxes = FloatArray(0)
    @Volatile var count = 0
    @Volatile var lastFaces = 0

    private var prevPoints: FloatArray? = null

    private fun finalizeX(nx: Float, ny: Float): Float {
        val rx = when (overlayRotation) {
            1 -> ny
            2 -> 1f - nx
            3 -> 1f - ny
            else -> nx
        }
        return if (overlayMirror) 1f - rx else rx
    }

    private fun finalizeY(nx: Float, ny: Float): Float = when (overlayRotation) {
        1 -> 1f - nx
        2 -> 1f - ny
        3 -> nx
        else -> ny
    }

    fun onFaces(faces: List<Face>) {
        lastFaces = faces.size
        if (faces.isEmpty()) {
            points = FloatArray(0)
            wire = FloatArray(0)
            boxes = FloatArray(0)
            count = 0
            prevPoints = null
            return
        }
        val w = frameW.toFloat().coerceAtLeast(1f)
        val h = frameH.toFloat().coerceAtLeast(1f)
        val pts = ArrayList<Float>(512)
        val wr = ArrayList<Float>(2048)
        val bx = ArrayList<Float>(faces.size * 4)
        val sorted = faces.sortedByDescending { it.boundingBox.width() * it.boundingBox.height() }
        for (face in sorted) {
            val r = face.boundingBox
            bx.add(finalizeX(r.left / w, r.top / h))
            bx.add(finalizeY(r.left / w, r.top / h))
            bx.add(finalizeX(r.right / w, r.bottom / h))
            bx.add(finalizeY(r.right / w, r.bottom / h))
            for (contour in face.allContours) {
                val cp = contour.points
                var px = 0f
                var py = 0f
                var first = true
                for (p in cp) {
                    val nx = p.x / w
                    val ny = p.y / h
                    val fx = finalizeX(nx, ny)
                    val fy = finalizeY(nx, ny)
                    pts.add(fx)
                    pts.add(fy)
                    if (!first) {
                        wr.add(px); wr.add(py); wr.add(fx); wr.add(fy)
                    }
                    px = fx; py = fy; first = false
                }
            }
        }
        val arr0 = pts.toFloatArray()
        var arr = arr0
        val prev = prevPoints
        if (prev != null && prev.size == arr.size) {
            arr = FloatArray(arr0.size) { i -> prev[i] * 0.65f + arr0[i] * 0.35f }
        }
        prevPoints = arr
        points = arr
        count = arr.size / 2
        wire = wr.toFloatArray()
        boxes = bx.toFloatArray()
    }
}
