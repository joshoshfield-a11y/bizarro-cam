package com.bizarro.cam.vision

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

/**
 * ML Kit face contour extraction. Publishes normalized stage-space UV points,
 * wireframe line segments and bounding boxes for the GL overlay and the mesh
 * displacement field. Temporal smoothing happens on the analysis thread.
 */
class FaceTracker {

    @Volatile var frontFacing = false
    @Volatile var overlayRotation = 0
    @Volatile var overlayMirror = false
    @Volatile var points = FloatArray(0)
    @Volatile var wire = FloatArray(0)
    @Volatile var boxes = FloatArray(0)
    @Volatile var count = 0
    @Volatile var lastFaces = 0

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setMinFaceSize(0.12f)
            .enableTracking()
            .build()
    )

    private var prevPoints: FloatArray? = null

    @OptIn(ExperimentalGetImage::class)
    fun analyze(imageProxy: ImageProxy) {
        val media = imageProxy.image
        if (media == null) {
            imageProxy.close()
            return
        }
        val rotation = imageProxy.imageInfo.rotationDegrees
        val w = imageProxy.width
        val h = imageProxy.height
        val input = try {
            InputImage.fromMediaImage(media, rotation)
        } catch (e: Exception) {
            imageProxy.close()
            return
        }
        detector.process(input)
            .addOnSuccessListener { faces -> publish(faces, w, h, rotation) }
            .addOnFailureListener { }
            .addOnCompleteListener { imageProxy.close() }
    }

    private fun mapX(x: Float, y: Float, w: Int, h: Int, rot: Int): Float {
        val nx = when (rot) {
            90 -> y / h
            180 -> 1f - x / w
            270 -> 1f - y / h
            else -> x / w
        }
        return finalizeX(nx, mapY0(x, y, w, h, rot))
    }

    private fun mapY0(x: Float, y: Float, w: Int, h: Int, rot: Int): Float = when (rot) {
        90 -> 1f - x / w
        180 -> 1f - y / h
        270 -> x / w
        else -> y / h
    }

    private fun mapY(x: Float, y: Float, w: Int, h: Int, rot: Int): Float =
        finalizeY(when (rot) {
            90 -> y / h
            180 -> 1f - x / w
            270 -> 1f - y / h
            else -> x / w
        }, mapY0(x, y, w, h, rot))

    // user-calibratable residual transform (long-press WIRE cycles it)
    private fun finalizeX(nx: Float, ny: Float): Float {
        val x = if (frontFacing) 1f - nx else nx
        val rx = when (overlayRotation) {
            1 -> ny
            2 -> 1f - x
            3 -> 1f - ny
            else -> x
        }
        return if (overlayMirror) 1f - rx else rx
    }

    private fun finalizeY(nx: Float, ny: Float): Float {
        val x = if (frontFacing) 1f - nx else nx
        return when (overlayRotation) {
            1 -> 1f - x
            2 -> 1f - ny
            3 -> x
            else -> ny
        }
    }

    private fun publish(faces: List<Face>, w: Int, h: Int, rot: Int) {
        lastFaces = faces.size
        if (faces.isEmpty()) {
            points = FloatArray(0)
            wire = FloatArray(0)
            boxes = FloatArray(0)
            count = 0
            prevPoints = null
            return
        }
        val pts = ArrayList<Float>(512)
        val wr = ArrayList<Float>(2048)
        val bx = ArrayList<Float>(faces.size * 4)
        val sorted = faces.sortedByDescending { it.boundingBox.width() * it.boundingBox.height() }
        for (face in sorted) {
            val r = face.boundingBox
            bx.add(mapX(r.left.toFloat(), r.top.toFloat(), w, h, rot))
            bx.add(mapY(r.left.toFloat(), r.top.toFloat(), w, h, rot))
            bx.add(mapX(r.right.toFloat(), r.bottom.toFloat(), w, h, rot))
            bx.add(mapY(r.right.toFloat(), r.bottom.toFloat(), w, h, rot))
            for (contour in face.allContours) {
                val cp = contour.points
                var px = 0f
                var py = 0f
                var first = true
                for (p in cp) {
                    val nx = mapX(p.x, p.y, w, h, rot)
                    val ny = mapY(p.x, p.y, w, h, rot)
                    pts.add(nx)
                    pts.add(ny)
                    if (!first) {
                        wr.add(px); wr.add(py); wr.add(nx); wr.add(ny)
                    }
                    px = nx; py = ny; first = false
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
