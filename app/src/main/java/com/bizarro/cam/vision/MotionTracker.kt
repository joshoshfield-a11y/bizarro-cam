package com.bizarro.cam.vision

import java.nio.ByteBuffer
import kotlin.math.hypot

/**
 * Moving-object tracker. Consumes the decimated motion-energy readback from the
 * displacement FBO, extracts connected-component blobs, matches them against
 * previous tracks (persistent IDs, EMA smoothing) and publishes stage-space
 * track boxes for overlay rendering and mesh displacement.
 */
class MotionTracker {

    data class Track(val id: Int, val cx: Float, val cy: Float, val w: Float, val h: Float)

    @Volatile var tracks: List<Track> = emptyList()

    private var prev: List<Track> = emptyList()
    private var nextId = 1
    private val gw = 64
    private val gh = 36

    fun update(buf: ByteBuffer, imgW: Int, imgH: Int) {
        val mask = BooleanArray(gw * gh)
        val sx = imgW / gw
        val sy = imgH / gh
        for (gy in 0 until gh) {
            for (gx in 0 until gw) {
                val px = gx * sx + sx / 2
                val py = gy * sy + sy / 2
                val idx = (py * imgW + px) * 4 + 3
                if ((buf.get(idx).toInt() and 0xFF) > 80) mask[gy * gw + gx] = true
            }
        }
        val labels = IntArray(gw * gh) { -1 }
        val blobs = ArrayList<FloatArray>()
        val stack = IntArray(gw * gh)
        var label = 0
        for (i in 0 until gw * gh) {
            if (!mask[i] || labels[i] >= 0) continue
            var sp = 0
            stack[sp++] = i
            labels[i] = label
            var minx = i % gw; var maxx = minx
            var miny = i / gw; var maxy = miny
            var cnt = 0
            while (sp > 0) {
                val cc = stack[--sp]
                cnt++
                val cx = cc % gw
                val cy = cc / gw
                if (cx < minx) minx = cx
                if (cx > maxx) maxx = cx
                if (cy < miny) miny = cy
                if (cy > maxy) maxy = cy
                if (cx > 0 && mask[cc - 1] && labels[cc - 1] < 0) { labels[cc - 1] = label; stack[sp++] = cc - 1 }
                if (cx < gw - 1 && mask[cc + 1] && labels[cc + 1] < 0) { labels[cc + 1] = label; stack[sp++] = cc + 1 }
                if (cy > 0 && mask[cc - gw] && labels[cc - gw] < 0) { labels[cc - gw] = label; stack[sp++] = cc - gw }
                if (cy < gh - 1 && mask[cc + gw] && labels[cc + gw] < 0) { labels[cc + gw] = label; stack[sp++] = cc + gw }
            }
            if (cnt >= 4) blobs.add(floatArrayOf(minx.toFloat(), miny.toFloat(), maxx.toFloat(), maxy.toFloat()))
            label++
        }
        val candidates = ArrayList<Track>()
        for (b in blobs) {
            val x0 = b[0] / gw
            val x1 = (b[2] + 1) / gw
            val y1 = 1f - b[1] / gh
            val y0 = 1f - (b[3] + 1) / gh
            candidates.add(Track(0, (x0 + x1) / 2f, (y0 + y1) / 2f, x1 - x0, y1 - y0))
        }
        val used = BooleanArray(prev.size)
        val out = ArrayList<Track>()
        for (cand in candidates) {
            var best = -1
            var bestD = 0.12f
            for (i in prev.indices) {
                if (used[i]) continue
                val d = hypot(cand.cx - prev[i].cx, cand.cy - prev[i].cy)
                if (d < bestD) { bestD = d; best = i }
            }
            if (best >= 0) {
                used[best] = true
                val p = prev[best]
                out.add(Track(
                    p.id,
                    p.cx * 0.6f + cand.cx * 0.4f,
                    p.cy * 0.6f + cand.cy * 0.4f,
                    p.w * 0.6f + cand.w * 0.4f,
                    p.h * 0.6f + cand.h * 0.4f
                ))
            } else {
                out.add(Track(nextId++, cand.cx, cand.cy, cand.w, cand.h))
            }
        }
        prev = out
        tracks = out
    }

    fun clear() {
        prev = emptyList()
        tracks = emptyList()
    }
}
