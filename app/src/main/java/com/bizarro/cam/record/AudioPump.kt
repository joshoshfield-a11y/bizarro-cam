package com.bizarro.cam.record

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log

/** Live mic RMS level meter for audio-reactive effects. */
class AudioPump {
    @Volatile var level = 0f
        private set
    @Volatile var running = false
        private set
    private var thread: Thread? = null

    fun start() {
        if (running) return
        running = true
        thread = Thread({ loop() }, "audio-pump").apply { start() }
    }

    fun stop() {
        running = false
        thread?.join(500)
        thread = null
        level = 0f
    }

    private fun loop() {
        var rec: AudioRecord? = null
        try {
            val minBuf = AudioRecord.getMinBufferSize(
                44100, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            rec = AudioRecord(
                MediaRecorder.AudioSource.MIC, 44100,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuf, 4096)
            )
            rec.startRecording()
            val buf = ShortArray(512)
            var smooth = 0f
            while (running) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) {
                    var sum = 0.0
                    for (i in 0 until n) sum += buf[i].toDouble() * buf[i].toDouble()
                    val rms = (kotlin.math.sqrt(sum / n) / 32768.0).toFloat()
                    smooth = smooth * 0.6f + (rms * 4f).coerceIn(0f, 1f) * 0.4f
                    level = smooth
                }
            }
        } catch (e: Exception) {
            Log.e("AudioPump", "meter failed", e)
        } finally {
            try { rec?.stop() } catch (_: Exception) {}
            try { rec?.release() } catch (_: Exception) {}
        }
    }
}
