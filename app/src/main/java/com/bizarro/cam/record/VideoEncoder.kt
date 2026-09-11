package com.bizarro.cam.record

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.MediaRecorder
import android.opengl.EGLContext
import android.util.Log
import android.view.Surface
import java.nio.ByteOrder
import java.util.concurrent.ArrayBlockingQueue

/**
 * H.264 + AAC MP4 encoder fed from a GL surface (shared EGL context) and an
 * AudioRecord capture loop. A single pump thread drains both codecs and muxes.
 */
class VideoEncoder(
    val width: Int,
    val height: Int,
    bitrate: Int,
    val outputPath: String,
    sharedContext: EGLContext,
) {
    companion object {
        private const val TAG = "VideoEncoder"
        private const val SAMPLE_RATE = 44100
    }

    private val videoCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    private val audioCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
    private val muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    val inputSurface: Surface

    private var videoTrack = -1
    private var audioTrack = -1
    @Volatile private var muxerStarted = false
    @Volatile var isRunning = false
        private set
    @Volatile var finished = false
        private set
    @Volatile var error: String? = null

    private val eglCore = EglCore(sharedContext)
    private var pumpThread: Thread? = null
    private var audioThread: Thread? = null
    private val audioQueue = ArrayBlockingQueue<ShortArray>(128)
    @Volatile private var audioCaptureEnded = false
    private var audioPtsUs = 0L
    private var audioEosQueued = false

    init {
        val vf = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
        vf.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        vf.setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
        vf.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
        vf.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        videoCodec.configure(vf, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        inputSurface = videoCodec.createInputSurface()

        val af = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, SAMPLE_RATE, 1)
        af.setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
        af.setInteger(MediaFormat.KEY_BIT_RATE, 128000)
        audioCodec.configure(af, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    }

    fun start() {
        videoCodec.start()
        audioCodec.start()
        isRunning = true
        pumpThread = Thread({ pumpLoop() }, "enc-pump").apply { start() }
        audioThread = Thread({ audioLoop() }, "enc-audio").apply { start() }
    }

    fun stop() {
        if (!isRunning) return
        isRunning = false
        try { videoCodec.signalEndOfInputStream() } catch (e: Exception) { Log.w(TAG, "signalEOS", e) }
        try { audioThread?.join(2000) } catch (_: InterruptedException) {}
        try { pumpThread?.join(8000) } catch (_: InterruptedException) {}
        try { inputSurface.release() } catch (_: Exception) {}
        finished = true
    }

    private fun audioLoop() {
        var rec: AudioRecord? = null
        try {
            val minBuf = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            rec = AudioRecord(
                MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf * 2
            )
            rec.startRecording()
            val buf = ShortArray(2048)
            while (isRunning) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) {
                    val copy = ShortArray(n)
                    System.arraycopy(buf, 0, copy, 0, n)
                    audioQueue.put(copy)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "audio failed", e)
            error = "audio: ${e.message}"
        } finally {
            try { rec?.stop() } catch (_: Exception) {}
            try { rec?.release() } catch (_: Exception) {}
            audioCaptureEnded = true
        }
    }

    private fun pumpLoop() {
        val vinfo = MediaCodec.BufferInfo()
        val ainfo = MediaCodec.BufferInfo()
        var vEos = false
        var aEos = false
        audioPtsUs = 0L
        try {
            while (!(vEos && aEos)) {
                if (!audioEosQueued) {
                    if (!isRunning && audioQueue.isEmpty()) {
                        val idx = audioCodec.dequeueInputBuffer(10000)
                        if (idx >= 0) {
                            audioCodec.queueInputBuffer(
                                idx, 0, 0, audioPtsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            audioEosQueued = true
                        }
                    } else {
                        val idx = audioCodec.dequeueInputBuffer(0)
                        if (idx >= 0) {
                            val data = audioQueue.poll()
                            val ib = audioCodec.getInputBuffer(idx)
                            if (ib != null) {
                                ib.clear()
                                if (data != null) {
                                    ib.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(data)
                                    audioCodec.queueInputBuffer(idx, 0, data.size * 2, audioPtsUs, 0)
                                    audioPtsUs += data.size * 1_000_000L / SAMPLE_RATE
                                } else {
                                    audioCodec.queueInputBuffer(idx, 0, 0, 0, 0)
                                }
                            }
                        }
                    }
                }

                if (!vEos) {
                    while (true) {
                        val out = videoCodec.dequeueOutputBuffer(vinfo, 0)
                        if (out == MediaCodec.INFO_TRY_AGAIN_LATER) break
                        if (out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                            videoTrack = muxer.addTrack(videoCodec.outputFormat)
                            tryStartMuxer()
                            continue
                        }
                        if (out >= 0) {
                            val ob = videoCodec.getOutputBuffer(out)
                            if (ob != null && vinfo.size > 0 &&
                                (vinfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0 && muxerStarted
                            ) {
                                muxer.writeSampleData(videoTrack, ob, vinfo)
                            }
                            videoCodec.releaseOutputBuffer(out, false)
                            if ((vinfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) vEos = true
                        }
                    }
                }

                if (audioEosQueued && !aEos) {
                    while (true) {
                        val out = audioCodec.dequeueOutputBuffer(ainfo, 0)
                        if (out == MediaCodec.INFO_TRY_AGAIN_LATER) break
                        if (out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                            audioTrack = muxer.addTrack(audioCodec.outputFormat)
                            tryStartMuxer()
                            continue
                        }
                        if (out >= 0) {
                            val ob = audioCodec.getOutputBuffer(out)
                            if (ob != null && ainfo.size > 0 &&
                                (ainfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0 && muxerStarted
                            ) {
                                muxer.writeSampleData(audioTrack, ob, ainfo)
                            }
                            audioCodec.releaseOutputBuffer(out, false)
                            if ((ainfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) aEos = true
                        }
                    }
                }

                if (!(vEos && aEos)) Thread.sleep(3)
            }
        } catch (e: Exception) {
            Log.e(TAG, "pump failed", e)
            error = e.message
        } finally {
            try { videoCodec.stop() } catch (_: Exception) {}
            try { audioCodec.stop() } catch (_: Exception) {}
            try { if (muxerStarted) muxer.stop() } catch (_: Exception) {}
            try { muxer.release() } catch (_: Exception) {}
            try { videoCodec.release() } catch (_: Exception) {}
            try { audioCodec.release() } catch (_: Exception) {}
        }
    }

    private fun tryStartMuxer() {
        if (!muxerStarted && videoTrack >= 0 && audioTrack >= 0) {
            muxer.start()
            muxerStarted = true
        }
    }
}
