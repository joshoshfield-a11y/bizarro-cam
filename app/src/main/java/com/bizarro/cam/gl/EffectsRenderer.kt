package com.bizarro.cam.gl

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLContext
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import androidx.camera.core.SurfaceRequest
import androidx.core.content.ContextCompat
import com.bizarro.cam.record.EglCore
import com.bizarro.cam.record.VideoEncoder
import com.bizarro.cam.vision.FaceTracker
import com.bizarro.cam.vision.MotionTracker
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import javax.microedition.khronos.opengles.GL10

/**
 * Full GPU pipeline:
 *  pass1: Sobel edge + motion-energy displacement map (512x512 FBO)
 *  pass2: displaced 48x86 grid composite with all FX (720x1280 stage FBO, ping-pong)
 *  pass3: center-crop blit to screen + vector overlays
 *  pass4: identical scene re-rendered to the encoder EGL surface when recording
 */
class EffectsRenderer(
    private val context: Context,
    private val faceTracker: FaceTracker,
    private val motionTracker: MotionTracker,
) : GLSurfaceView.Renderer {

    companion object {
        private const val TAG = "EffectsRenderer"
        private const val STAGE_W = 720
        private const val STAGE_H = 1280
        private const val DISP = 512
        private const val GRID_COLS = 48
        private const val GRID_ROWS = 86
        private const val MAXP = 128
    }

    var onReady: (() -> Unit)? = null
    var onRecordingStarted: (() -> Unit)? = null
    var onRecordingStopped: ((String) -> Unit)? = null
    var onEncoderError: ((String) -> Unit)? = null
    var onSnapshotSaved: ((String) -> Unit)? = null
    var onContextRecreated: (() -> Unit)? = null

    @Volatile var wireframe = true
    @Volatile var glReady = false
    @Volatile var takeSnapshot = false
    @Volatile var fps = 0f
    @Volatile var shaderStatus = "?"

    private fun onoff(v: Int): String = if (v > 0) "1" else "0"

    @Volatile var frontCamera = false
        set(v) {
            field = v
            faceTracker.frontFacing = v
        }

    private val mainExecutor = ContextCompat.getMainExecutor(context)
    private val ioExecutor = Executors.newSingleThreadExecutor()
    private val params = ConcurrentHashMap<String, Float>()

    fun param(k: String): Float = params[k] ?: 0f
    fun setParam(k: String, v: Float) {
        params[k] = v
    }

    init {
        params["edge"] = 0.55f
        params["displace"] = 0.30f
        params["morph"] = 0.25f
        params["glitch"] = 0.10f
        params["echo"] = 0.18f
        params["poster"] = 0.0f
        params["chroma"] = 0.20f
        params["hue"] = 0.0f
        params["slit"] = 0.0f
        params["kaleido"] = 0.0f
        params["noise"] = 0.06f
        params["scan"] = 0.10f
        params["invert"] = 0.0f
    }

    private var oesTex = 0
    private var surfaceTexture: SurfaceTexture? = null
    private var cameraSurface: Surface? = null
    private var pendingRequest: SurfaceRequest? = null
    @Volatile private var frameAvailable = false
    private val texMatrix = FloatArray(16)

    private val stageTex = IntArray(2)
    private val stageFbo = IntArray(2)
    private var curIdx = 0
    private var dispTex = 0
    private var dispFbo = 0

    private var progDisp = 0
    private var progComposite = 0
    private var progBlit = 0
    private var progLine = 0
    private var progRaw = 0

    private var quadVbo = 0
    private var gridVbo = 0
    private var gridIbo = 0
    private var gridIndexCount = 0
    private var lineVbo = 0

    private var viewW = 1
    private var viewH = 1
    private val startMs = SystemClock.elapsedRealtime()
    private var fpsCount = 0
    private var fpsTime = 0L
    private var readbackCount = 0
    private var firstFrameLogged = false

    private var eglCtx14: EGLContext? = null
    private var eglCoreEnc: EglCore? = null
    private var encEglSurface: EGLSurface? = null
    @Volatile private var encoder: VideoEncoder? = null

    private var view: GLSurfaceView? = null
    fun setView(v: GLSurfaceView) {
        view = v
    }

    // ---------------------------------------------------------------- camera

    fun attachSurfaceRequest(request: SurfaceRequest) {
        val v = view ?: return
        v.queueEvent { provideSurface(request) }
    }

    private fun provideSurface(request: SurfaceRequest) {
        val st = surfaceTexture
        if (st == null) {
            pendingRequest = request
            return
        }
        st.setDefaultBufferSize(request.resolution.width, request.resolution.height)
        val s = Surface(st)
        cameraSurface = s
        request.provideSurface(s, mainExecutor) {
            cameraSurface = null
            s.release()
        }
    }

    // ---------------------------------------------------------------- record

    fun startRecording(path: String) {
        if (encoder != null) return
        val ctx = eglCtx14
        if (ctx == null) {
            onEncoderError?.let { mainExecutor.execute { it("no egl context") } }
            return
        }
        try {
            val enc = VideoEncoder(STAGE_W, STAGE_H, 10_000_000, path, ctx)
            enc.start()
            encoder = enc
            eglCoreEnc = EglCore(ctx)
            encEglSurface = eglCoreEnc!!.createWindowSurface(enc.inputSurface)
            onRecordingStarted?.let { mainExecutor.execute(it) }
        } catch (e: Exception) {
            Log.e(TAG, "startRecording", e)
            onEncoderError?.let { mainExecutor.execute { it(e.message ?: "start failed") } }
        }
    }

    fun stopRecording() {
        val enc = encoder ?: return
        encoder = null
        val egl = eglCoreEnc
        val es = encEglSurface
        eglCoreEnc = null
        encEglSurface = null
        ioExecutor.execute {
            var err: String? = null
            try {
                enc.stop()
                err = enc.error
            } catch (e: Exception) {
                Log.e(TAG, "stop", e)
                err = e.message
            }
            val path = enc.outputPath
            view?.queueEvent {
                egl?.releaseSurface(es)
                egl?.release()
            }
            if (err != null) onEncoderError?.let { mainExecutor.execute { it(err) } }
            onRecordingStopped?.let { mainExecutor.execute { it(path) } }
        }
    }

    fun forceStopEncoder() {
        val enc = encoder ?: return
        encoder = null
        Thread { try { enc.stop() } catch (_: Exception) {} }.start()
    }

    // ---------------------------------------------------------------- GL

    override fun onSurfaceCreated(gl: GL10?, config: javax.microedition.khronos.egl.EGLConfig?) {
        eglCtx14 = EGL14.eglGetCurrentContext()
        if (surfaceTexture != null) {
            // GL context was recreated while the camera still feeds the old SurfaceTexture:
            // tear it down and force a camera rebind onto a fresh one.
            Log.w(TAG, "GL context recreated - rebinding camera onto fresh SurfaceTexture")
            try { surfaceTexture?.release() } catch (e: Exception) { Log.w(TAG, "st release", e) }
            surfaceTexture = null
            cameraSurface = null
            pendingRequest = null
            frameAvailable = false
            onContextRecreated?.let { mainExecutor.execute(it) }
        }

        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        oesTex = ids[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR.toFloat())
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR.toFloat())
        val st = SurfaceTexture(oesTex)
        st.setOnFrameAvailableListener { frameAvailable = true }
        surfaceTexture = st

        progDisp = buildProgram(ShaderStore.QUAD_VERT, ShaderStore.DISP_FRAG)
        progComposite = buildProgram(ShaderStore.GRID_VERT, ShaderStore.COMPOSITE_FRAG)
        progBlit = buildProgram(ShaderStore.QUAD_VERT, ShaderStore.BLIT_FRAG)
        progLine = buildProgram(ShaderStore.LINE_VERT, ShaderStore.LINE_FRAG)
        progRaw = buildProgram(ShaderStore.QUAD_VERT, ShaderStore.RAW_FRAG)
        shaderStatus = "disp=" + onoff(progDisp) + " comp=" + onoff(progComposite) +
            " blit=" + onoff(progBlit) + " line=" + onoff(progLine) + " raw=" + onoff(progRaw)
        Log.i(TAG, "shader status: " + shaderStatus)

        val quad = floatArrayOf(
            -1f, -1f, 0f, 0f,
             1f, -1f, 1f, 0f,
            -1f,  1f, 0f, 1f,
             1f,  1f, 1f, 1f
        )
        GLES20.glGenBuffers(1, ids, 0)
        quadVbo = ids[0]
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, quad.size * 4, floatBuffer(quad), GLES20.GL_STATIC_DRAW)

        val cols = GRID_COLS + 1
        val rows = GRID_ROWS + 1
        val gv = FloatArray(cols * rows * 4)
        var vi = 0
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val u = c / GRID_COLS.toFloat()
                val vv = r / GRID_ROWS.toFloat()
                gv[vi++] = u * 2f - 1f
                gv[vi++] = vv * 2f - 1f
                gv[vi++] = u
                gv[vi++] = vv
            }
        }
        gridIndexCount = GRID_COLS * GRID_ROWS * 6
        val idx = ShortArray(gridIndexCount)
        var ii = 0
        for (r in 0 until GRID_ROWS) {
            for (c in 0 until GRID_COLS) {
                val a = r * cols + c
                val b = a + 1
                val cc = a + cols
                val d = cc + 1
                idx[ii++] = a.toShort()
                idx[ii++] = cc.toShort()
                idx[ii++] = b.toShort()
                idx[ii++] = b.toShort()
                idx[ii++] = cc.toShort()
                idx[ii++] = d.toShort()
            }
        }
        GLES20.glGenBuffers(1, ids, 0)
        gridVbo = ids[0]
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, gridVbo)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, gv.size * 4, floatBuffer(gv), GLES20.GL_STATIC_DRAW)
        GLES20.glGenBuffers(1, ids, 0)
        gridIbo = ids[0]
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gridIbo)
        GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, idx.size * 2, shortBuffer(idx), GLES20.GL_STATIC_DRAW)

        GLES20.glGenBuffers(1, ids, 0)
        lineVbo = ids[0]

        GLES20.glGenTextures(2, stageTex, 0)
        for (i in 0..1) {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, stageTex[i])
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, STAGE_W, STAGE_H, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        }
        GLES20.glGenFramebuffers(2, stageFbo, 0)
        for (i in 0..1) {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, stageFbo[i])
            GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, stageTex[i], 0)
        }

        GLES20.glGenTextures(1, ids, 0)
        dispTex = ids[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, dispTex)
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, DISP, DISP, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glGenFramebuffers(1, ids, 0)
        dispFbo = ids[0]
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, dispFbo)
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, dispTex, 0)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        glReady = true
        pendingRequest?.let {
            pendingRequest = null
            provideSurface(it)
        }
        onReady?.let { mainExecutor.execute(it) }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewW = width
        viewH = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        val st = surfaceTexture ?: return
        if (frameAvailable) {
            try {
                st.updateTexImage()
                if (!firstFrameLogged) {
                    firstFrameLogged = true
                    Log.i(TAG, "first camera frame latched ts=" + st.timestamp + " texmat=" + texMatrix.contentToString())
                }
            } catch (e: Exception) {
                Log.e(TAG, "updateTexImage failed", e)
            }
            frameAvailable = false
        }
        st.getTransformMatrix(texMatrix)
        val t = (SystemClock.elapsedRealtime() - startMs) / 1000f
        val prevIdx = curIdx xor 1

        if (progDisp > 0) {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, dispFbo)
            GLES20.glViewport(0, 0, DISP, DISP)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            GLES20.glUseProgram(progDisp)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
            GLES20.glUniform1i(glLoc(progDisp, "uTex"), 0)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, stageTex[prevIdx])
            GLES20.glUniform1i(glLoc(progDisp, "uPrev"), 1)
            GLES20.glUniformMatrix4fv(glLoc(progDisp, "uTexMatrix"), 1, false, texMatrix, 0)
            GLES20.glUniform4f(glLoc(progDisp, "uCrop"), 0f, 0f, 1f, 1f)
            GLES20.glUniform2f(glLoc(progDisp, "uResolution"), STAGE_W.toFloat(), STAGE_H.toFloat())
            GLES20.glUniform1f(glLoc(progDisp, "uTime"), t)
            drawQuad(progDisp)
        }

        readbackCount++
        if (readbackCount % 3 == 0) {
            val buf = ByteBuffer.allocateDirect(DISP * DISP * 4).order(ByteOrder.nativeOrder())
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, dispFbo)
            GLES20.glReadPixels(0, 0, DISP, DISP, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
            buf.rewind()
            motionTracker.update(buf, DISP, DISP)
        }

        buildPointArray()
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)

        if (progComposite <= 0 && progRaw > 0) {
            // heavy pipeline failed to compile on this driver: raw camera fallback
            GLES20.glViewport(0, 0, viewW, viewH)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            GLES20.glUseProgram(progRaw)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
            GLES20.glUniform1i(glLoc(progRaw, "uTex"), 0)
            GLES20.glUniformMatrix4fv(glLoc(progRaw, "uTexMatrix"), 1, false, texMatrix, 0)
            GLES20.glUniform4f(glLoc(progRaw, "uCrop"), 0f, 0f, 1f, 1f)
            drawQuad(progRaw)
            val crop2 = computeCrop(viewW, viewH)
            if (wireframe && progLine > 0) drawLines(crop2)
            curIdx = prevIdx
            return
        }

        if (progComposite > 0) {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, stageFbo[curIdx])
            GLES20.glViewport(0, 0, STAGE_W, STAGE_H)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            GLES20.glUseProgram(progComposite)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
            GLES20.glUniform1i(glLoc(progComposite, "uTex"), 0)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, stageTex[prevIdx])
            GLES20.glUniform1i(glLoc(progComposite, "uPrev"), 1)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, dispTex)
            GLES20.glUniform1i(glLoc(progComposite, "uDisp"), 2)
            GLES20.glUniformMatrix4fv(glLoc(progComposite, "uTexMatrix"), 1, false, texMatrix, 0)
            GLES20.glUniform2f(glLoc(progComposite, "uResolution"), STAGE_W.toFloat(), STAGE_H.toFloat())
            GLES20.glUniform1f(glLoc(progComposite, "uTime"), t)
            GLES20.glUniform1f(glLoc(progComposite, "uEdge"), param("edge"))
            GLES20.glUniform1f(glLoc(progComposite, "uDisplace"), param("displace"))
            GLES20.glUniform1f(glLoc(progComposite, "uGlitch"), param("glitch"))
            GLES20.glUniform1f(glLoc(progComposite, "uPoster"), param("poster"))
            GLES20.glUniform1f(glLoc(progComposite, "uChroma"), param("chroma"))
            GLES20.glUniform1f(glLoc(progComposite, "uHue"), param("hue"))
            GLES20.glUniform1f(glLoc(progComposite, "uEcho"), param("echo"))
            GLES20.glUniform1f(glLoc(progComposite, "uNoise"), param("noise"))
            GLES20.glUniform1f(glLoc(progComposite, "uSlit"), param("slit"))
            GLES20.glUniform1f(glLoc(progComposite, "uKaleido"), param("kaleido"))
            GLES20.glUniform1f(glLoc(progComposite, "uInvert"), param("invert"))
            GLES20.glUniform1f(glLoc(progComposite, "uScan"), param("scan"))
            GLES20.glUniform1f(glLoc(progComposite, "uMorph"), param("morph"))
            GLES20.glUniform1f(glLoc(progComposite, "uMeshAmp"), param("displace"))
            GLES20.glUniform1f(glLoc(progComposite, "uPointCount"), pointCount.toFloat())
            if (pointCount > 0) {
                GLES20.glUniform2fv(glLoc(progComposite, "uPoints"), MAXP, pointArr, 0)
            }
            drawGrid()
        }

        if (takeSnapshot) {
            takeSnapshot = false
            saveSnapshot()
        }

        val crop = computeCrop(viewW, viewH)
        drawScene(0, viewW, viewH, stageTex[curIdx], crop)

        val enc = encoder
        if (enc != null) {
            val es = encEglSurface
            val ecore = eglCoreEnc
            if (es != null && ecore != null) {
                ecore.makeCurrent(es)
                drawScene(-1, STAGE_W, STAGE_H, stageTex[curIdx], identityCrop)
                ecore.setPresentationTime(es, st.timestamp)
                ecore.swap(es)
            }
        }

        curIdx = prevIdx

        fpsCount++
        val now = SystemClock.elapsedRealtime()
        if (now - fpsTime > 1000) {
            fps = fpsCount * 1000f / (now - fpsTime).coerceAtLeast(1)
            fpsCount = 0
            fpsTime = now
        }
    }

    // ---------------------------------------------------------------- passes

    private val identityCrop = floatArrayOf(0f, 0f, 1f, 1f)

    private fun drawScene(fbo: Int, w: Int, h: Int, tex: Int, crop: FloatArray) {
        if (fbo >= 0) {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        }
        GLES20.glViewport(0, 0, w, h)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        if (progBlit > 0) {
            GLES20.glUseProgram(progBlit)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex)
            GLES20.glUniform1i(glLoc(progBlit, "uTex"), 0)
            GLES20.glUniform4f(glLoc(progBlit, "uCrop"), crop[0], crop[1], crop[2], crop[3])
            drawQuad(progBlit)
        }
        if (wireframe && progLine > 0) {
            drawLines(crop)
        }
    }

    private var lineGroupCounts = ArrayList<Pair<FloatArray, Int>>()

    private fun drawLines(crop: FloatArray) {
        val verts = buildLineVerts(crop)
        if (verts.isEmpty()) return
        GLES20.glUseProgram(progLine)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, lineVbo)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, verts.size * 4, floatBuffer(verts), GLES20.GL_STREAM_DRAW)
        val aPos = GLES20.glGetAttribLocation(progLine, "aPos")
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 8, 0)
        var offset = 0
        for ((color, count) in lineGroupCounts) {
            GLES20.glUniform4f(glLoc(progLine, "uColor"), color[0], color[1], color[2], color[3])
            GLES20.glDrawArrays(GLES20.GL_LINES, offset, count)
            offset += count
        }
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun buildLineVerts(crop: FloatArray): FloatArray {
        val all = ArrayList<Float>()
        lineGroupCounts = ArrayList()
        fun cropX(x: Float) = crop[0] + x * (crop[2] - crop[0])
        fun cropY(y: Float) = crop[1] + y * (crop[3] - crop[1])
        fun addSeg(list: ArrayList<Float>, x0: Float, y0: Float, x1: Float, y1: Float) {
            list.add(cropX(x0) * 2 - 1)
            list.add(1 - cropY(y0) * 2)
            list.add(cropX(x1) * 2 - 1)
            list.add(1 - cropY(y1) * 2)
        }

        val wr = ArrayList<Float>()
        val w = faceTracker.wire
        var i = 0
        while (i + 3 < w.size) {
            addSeg(wr, w[i], w[i + 1], w[i + 2], w[i + 3])
            i += 4
        }
        if (wr.isNotEmpty()) {
            lineGroupCounts.add(floatArrayOf(0.2f, 1f, 0.9f, 0.85f) to wr.size / 2)
            all.addAll(wr)
        }

        val br = ArrayList<Float>()
        val b = faceTracker.boxes
        i = 0
        while (i + 3 < b.size) {
            val x0 = b[i]; val y0 = b[i + 1]; val x1 = b[i + 2]; val y1 = b[i + 3]
            addSeg(br, x0, y0, x1, y0)
            addSeg(br, x1, y0, x1, y1)
            addSeg(br, x1, y1, x0, y1)
            addSeg(br, x0, y1, x0, y0)
            i += 4
        }
        if (br.isNotEmpty()) {
            lineGroupCounts.add(floatArrayOf(1f, 0.2f, 0.95f, 0.7f) to br.size / 2)
            all.addAll(br)
        }

        val tr = motionTracker.tracks
        if (tr.isNotEmpty()) {
            val buckets = Array(6) { ArrayList<Float>() }
            for (tk in tr) {
                val bk = buckets[((tk.id % 6) + 6) % 6]
                val hw = kotlin.math.max(tk.w * 0.6f, 0.03f)
                val hh = kotlin.math.max(tk.h * 0.6f, 0.03f)
                addSeg(bk, tk.cx - hw, tk.cy - hh, tk.cx + hw, tk.cy - hh)
                addSeg(bk, tk.cx + hw, tk.cy - hh, tk.cx + hw, tk.cy + hh)
                addSeg(bk, tk.cx + hw, tk.cy + hh, tk.cx - hw, tk.cy + hh)
                addSeg(bk, tk.cx - hw, tk.cy + hh, tk.cx - hw, tk.cy - hh)
            }
            val palette = arrayOf(
                floatArrayOf(1f, 0.9f, 0.1f, 0.8f),
                floatArrayOf(0.4f, 1f, 0.2f, 0.8f),
                floatArrayOf(1f, 0.4f, 0.1f, 0.8f),
                floatArrayOf(0.5f, 0.3f, 1f, 0.8f),
                floatArrayOf(1f, 1f, 1f, 0.8f),
                floatArrayOf(0.1f, 0.6f, 1f, 0.8f)
            )
            for (k in 0 until 6) {
                if (buckets[k].isNotEmpty()) {
                    lineGroupCounts.add(palette[k] to buckets[k].size / 2)
                    all.addAll(buckets[k])
                }
            }
        }
        return all.toFloatArray()
    }

    // ---------------------------------------------------------------- utils

    private val pointArr = FloatArray(MAXP * 2)
    private var pointCount = 0

    private fun buildPointArray() {
        var n = 0
        val fp = faceTracker.points
        val fn = kotlin.math.min(faceTracker.count, MAXP - 30)
        for (i in 0 until fn) {
            pointArr[n * 2] = fp[i * 2]
            pointArr[n * 2 + 1] = fp[i * 2 + 1]
            n++
        }
        for (tk in motionTracker.tracks) {
            if (n >= MAXP) break
            pointArr[n * 2] = tk.cx
            pointArr[n * 2 + 1] = tk.cy
            n++
        }
        pointCount = n
    }

    private fun computeCrop(vw: Int, vh: Int): FloatArray {
        val frameA = STAGE_W.toFloat() / STAGE_H
        val viewA = vw.toFloat() / vh
        return if (viewA > frameA) {
            val fh = frameA / viewA
            floatArrayOf(0f, (1f - fh) / 2f, 1f, (1f + fh) / 2f)
        } else {
            val fw = viewA / frameA
            floatArrayOf((1f - fw) / 2f, 0f, (1f + fw) / 2f, 1f)
        }
    }

    private fun saveSnapshot() {
        try {
            val buf = ByteBuffer.allocateDirect(STAGE_W * STAGE_H * 4).order(ByteOrder.nativeOrder())
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, stageFbo[curIdx])
            GLES20.glReadPixels(0, 0, STAGE_W, STAGE_H, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buf)
            buf.rewind()
            ioExecutor.execute {
                try {
                    val bmp = Bitmap.createBitmap(STAGE_W, STAGE_H, Bitmap.Config.ARGB_8888)
                    bmp.copyPixelsFromBuffer(buf)
                    val m = Matrix().apply { postScale(1f, -1f) }
                    val flipped = Bitmap.createBitmap(bmp, 0, 0, STAGE_W, STAGE_H, m, false)
                    val dir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: context.filesDir
                    val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                    val f = File(dir, "BizarroCam_$ts.png")
                    FileOutputStream(f).use { flipped.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    onSnapshotSaved?.let { mainExecutor.execute { it(f.absolutePath) } }
                } catch (e: Exception) {
                    Log.e(TAG, "save", e)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "snapshot", e)
        }
    }

    private fun drawQuad(prog: Int) {
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        val aPos = GLES20.glGetAttribLocation(prog, "aPos")
        val aUV = GLES20.glGetAttribLocation(prog, "aUV")
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 16, 0)
        GLES20.glEnableVertexAttribArray(aUV)
        GLES20.glVertexAttribPointer(aUV, 2, GLES20.GL_FLOAT, false, 16, 8)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aUV)
    }

    private fun drawGrid() {
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, gridVbo)
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gridIbo)
        val aPos = GLES20.glGetAttribLocation(progComposite, "aPos")
        val aUV = GLES20.glGetAttribLocation(progComposite, "aUV")
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 16, 0)
        GLES20.glEnableVertexAttribArray(aUV)
        GLES20.glVertexAttribPointer(aUV, 2, GLES20.GL_FLOAT, false, 16, 8)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, gridIndexCount, GLES20.GL_UNSIGNED_SHORT, 0)
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aUV)
    }

    private fun glLoc(prog: Int, name: String): Int = GLES20.glGetUniformLocation(prog, name)

    private fun floatBuffer(arr: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(arr.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            .apply { put(arr); position(0) }

    private fun shortBuffer(arr: ShortArray): ShortBuffer =
        ByteBuffer.allocateDirect(arr.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
            .apply { put(arr); position(0) }

    private fun buildProgram(vs: String, fs: String): Int {
        val v = compile(GLES20.GL_VERTEX_SHADER, vs)
        val f = compile(GLES20.GL_FRAGMENT_SHADER, fs)
        if (v == 0 || f == 0) return 0
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, v)
        GLES20.glAttachShader(p, f)
        GLES20.glLinkProgram(p)
        val ok = IntArray(1)
        GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, ok, 0)
        if (ok[0] == 0) {
            Log.e(TAG, "link fail: " + GLES20.glGetProgramInfoLog(p))
            return 0
        }
        return p
    }

    private fun compile(type: Int, src: String): Int {
        val s = GLES20.glCreateShader(type)
        GLES20.glShaderSource(s, src)
        GLES20.glCompileShader(s)
        val ok = IntArray(1)
        GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            Log.e(TAG, "shader fail: " + GLES20.glGetShaderInfoLog(s))
            return 0
        }
        return s
    }
}
