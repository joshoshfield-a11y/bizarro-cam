package com.bizarro.cam

import android.Manifest
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.bizarro.cam.gl.EffectsRenderer
import com.bizarro.cam.vision.FaceTracker
import com.bizarro.cam.vision.MotionTracker
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val REQ_PERMS = 1001
        private const val REQ_AUDIO = 1002
        private val SLIDERS = listOf(
            "edge" to "EDGE", "displace" to "DISP", "morph" to "MORPH",
            "glitch" to "GLITCH", "echo" to "ECHO", "poster" to "POSTER",
            "chroma" to "CHROMA", "hue" to "HUE", "slit" to "SLIT",
            "kaleido" to "KALEIDO", "noise" to "NOISE", "scan" to "SCAN"
        )
    }

    private lateinit var glView: GLSurfaceView
    private lateinit var renderer: EffectsRenderer
    private lateinit var faceTracker: FaceTracker
    private lateinit var motionTracker: MotionTracker
    private var camera: Camera? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var recording = false
    private var recStartMs = 0L
    private lateinit var btnRec: Button
    private lateinit var txtStats: TextView
    private val uiHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        supportActionBar?.hide()
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        faceTracker = FaceTracker()
        motionTracker = MotionTracker()
        glView = findViewById(R.id.glView)
        glView.setEGLContextClientVersion(2)
        glView.preserveEGLContextOnPause = true
        renderer = EffectsRenderer(this, faceTracker, motionTracker)
        renderer.setView(glView)
        renderer.onReady = { runOnUiThread { tryBindCamera() } }
        renderer.onRecordingStarted = {
            runOnUiThread {
                recording = true
                recStartMs = SystemClock.elapsedRealtime()
                btnRec.text = "STOP"
            }
        }
        renderer.onRecordingStopped = { path ->
            runOnUiThread {
                recording = false
                btnRec.text = "REC"
                Toast.makeText(this, "Saved:\n$path", Toast.LENGTH_LONG).show()
            }
        }
        renderer.onEncoderError = { msg ->
            runOnUiThread {
                recording = false
                btnRec.text = "REC"
                Toast.makeText(this, "Encoder: $msg", Toast.LENGTH_LONG).show()
            }
        }
        renderer.onContextRecreated = { runOnUiThread { bindCamera() } }
        renderer.onSnapshotSaved = { path ->
            runOnUiThread { Toast.makeText(this, "PNG:\n$path", Toast.LENGTH_SHORT).show() }
        }
        glView.setRenderer(renderer)
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        btnRec = findViewById(R.id.btnRec)
        txtStats = findViewById(R.id.txtStats)
        buildSliders()
        findViewById<Button>(R.id.btnFlip).setOnClickListener { flipCamera() }
        val btnWire = findViewById<Button>(R.id.btnWire)
        btnWire.setOnClickListener {
            renderer.wireframe = !renderer.wireframe
        }
        btnWire.setOnLongClickListener {
            faceTracker.overlayRotation = (faceTracker.overlayRotation + 1) % 4
            if (faceTracker.overlayRotation == 0) faceTracker.overlayMirror = !faceTracker.overlayMirror
            Toast.makeText(
                this,
                "overlay rot=${faceTracker.overlayRotation * 90} mirror=${faceTracker.overlayMirror}",
                Toast.LENGTH_LONG
            ).show()
            true
        }
        findViewById<Button>(R.id.btnFx).setOnClickListener { randomizeFx() }
        findViewById<Button>(R.id.btnReset).setOnClickListener { resetFx() }
        btnRec.setOnClickListener { toggleRecording() }
        findViewById<Button>(R.id.btnShot).setOnClickListener { renderer.takeSnapshot = true }
        checkPermissions()
    }

    private fun buildSliders() {
        val row = findViewById<LinearLayout>(R.id.sliderRow)
        for ((key, label) in SLIDERS) {
            val col = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(170, LinearLayout.LayoutParams.MATCH_PARENT)
            }
            val tv = TextView(this).apply {
                text = label
                textSize = 9f
                setTextColor(0xFF00E5FF.toInt())
                gravity = Gravity.CENTER
            }
            val sb = SeekBar(this).apply {
                rotation = 270f
                layoutParams = LinearLayout.LayoutParams(200, 60)
                max = 1000
                progress = (renderer.param(key) * 1000f).toInt()
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                        renderer.setParam(key, p / 1000f)
                    }
                    override fun onStartTrackingTouch(s: SeekBar?) {}
                    override fun onStopTrackingTouch(s: SeekBar?) {}
                })
            }
            col.addView(tv)
            col.addView(sb)
            row.addView(col)
        }
    }

    private fun randomizeFx() {
        val rnd = java.util.Random()
        val row = findViewById<LinearLayout>(R.id.sliderRow)
        var i = 0
        for ((key, _) in SLIDERS) {
            val v = rnd.nextInt(1001)
            renderer.setParam(key, v / 1000f)
            val col = row.getChildAt(i) as? LinearLayout
            (col?.getChildAt(1) as? SeekBar)?.progress = v
            i++
        }
        renderer.setParam("invert", if (rnd.nextBoolean()) 1f else 0f)
    }

    private fun resetFx() {
        val row = findViewById<LinearLayout>(R.id.sliderRow)
        var i = 0
        for ((key, _) in SLIDERS) {
            renderer.setParam(key, 0f)
            val col = row.getChildAt(i) as? LinearLayout
            (col?.getChildAt(1) as? SeekBar)?.progress = 0
            i++
        }
        renderer.setParam("invert", 0f)
    }

    private fun flipCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        renderer.frontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT
        bindCamera()
    }

    private fun checkPermissions() {
        val need = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.CAMERA)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            need.add(Manifest.permission.RECORD_AUDIO)
        }
        if (need.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, need.toTypedArray(), REQ_PERMS)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_PERMS) tryBindCamera()
        if (requestCode == REQ_AUDIO) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startRecording()
            } else {
                Toast.makeText(this, "Mic denied - cannot record audio", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun tryBindCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        if (!renderer.glReady) return
        bindCamera()
    }

    private fun bindCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            provider.unbindAll()
            val preview = Preview.Builder()
                .setTargetResolution(Size(720, 1280))
                .setTargetRotation(Surface.ROTATION_0)
                .build()
            preview.setSurfaceProvider(Preview.SurfaceProvider { request ->
                glView.queueEvent { renderer.attachSurfaceRequest(request) }
            })
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(480, 640))
                .setTargetRotation(Surface.ROTATION_0)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                faceTracker.analyze(imageProxy)
            }
            val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            try {
                camera = provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (e: SecurityException) {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Camera failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun toggleRecording() {
        if (recording) {
            glView.queueEvent { renderer.stopRecording() }
            return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQ_AUDIO)
            return
        }
        startRecording()
    }

    private fun startRecording() {
        val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val out = File(dir, "BizarroCam_$ts.mp4")
        glView.queueEvent { renderer.startRecording(out.absolutePath) }
    }

    private val statsRunnable = object : Runnable {
        override fun run() {
            val recSec = if (recording) (SystemClock.elapsedRealtime() - recStartMs) / 1000 else 0
            txtStats.text = "%.0f fps | faces %d | tracks %d%s".format(
                renderer.fps, faceTracker.lastFaces, motionTracker.tracks.size,
                if (recording) " | REC ${recSec}s" else ""
            )
            uiHandler.postDelayed(this, 500)
        }
    }

    override fun onResume() {
        super.onResume()
        glView.onResume()
        uiHandler.post(statsRunnable)
    }

    override fun onPause() {
        uiHandler.removeCallbacks(statsRunnable)
        glView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        cameraExecutor.shutdown()
        renderer.forceStopEncoder()
        super.onDestroy()
    }
}
