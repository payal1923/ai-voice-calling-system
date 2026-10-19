package com.aivoice.agent

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.IOException

class VoiceAgentClient(
    private val context: Context,
    private val listener: VoiceEventListener
) {

    interface VoiceEventListener {
        fun onSessionStarted(callId: Int, customerName: String)
        fun onThinking()
        fun onAssistantResponse(transcript: String, response: String, audioUrl: String?)
        fun onInterrupted()
        fun onError(error: String)
        fun onClosed()
    }

    private val okHttpClient = OkHttpClient.Builder().build()
    private var webSocket: WebSocket? = null
    private var mediaPlayer: MediaPlayer? = null
    private var isPlaying = false

    fun connect(wsUrl: String) {
        val request = Request.Builder()
            .url(wsUrl)
            .addHeader("ngrok-skip-browser-warning", "true")
            .build()

        webSocket = okHttpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                // Connection open; awaiting session_started handshake
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleTextMessage(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                listener.onError("Connection failure: ${t.message}")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                stopAudioPlayback()
                listener.onClosed()
            }
        })
    }

    private fun handleTextMessage(jsonString: String) {
        try {
            val json = JSONObject(jsonString)
            when (json.optString("event")) {
                "session_started" -> {
                    val callId = json.optInt("call_id", 0)
                    val customerName = json.optString("customer_name", "App User")
                    listener.onSessionStarted(callId, customerName)
                }
                "processing" -> {
                    listener.onThinking()
                }
                "assistant_response" -> {
                    val transcript = json.optString("transcript", "")
                    val response = json.optString("response", "")
                    val audioUrl = json.optString("audio_url", "")
                    
                    val fullAudioUrl = if (audioUrl.isNotEmpty()) {
                        val base = Config.getServerUrl(context)
                        if (audioUrl.startsWith("http")) audioUrl else "$base$audioUrl"
                    } else null

                    listener.onAssistantResponse(transcript, response, fullAudioUrl)

                    if (fullAudioUrl != null) {
                        playRemoteAudio(fullAudioUrl)
                    }
                }
                "interrupted" -> {
                    stopAudioPlayback()
                    listener.onInterrupted()
                }
                "error" -> {
                    listener.onError(json.optString("message", "Server error"))
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun sendSpeechTurn(wavBytes: ByteArray) {
        webSocket?.send(wavBytes.toByteString())
    }

    fun interrupt() {
        stopAudioPlayback()
        val interruptPayload = JSONObject().apply {
            put("event", "interrupt")
        }
        webSocket?.send(interruptPayload.toString())
    }

    fun endCall() {
        stopAudioPlayback()
        val endPayload = JSONObject().apply {
            put("event", "end_call")
        }
        webSocket?.send(endPayload.toString())
        webSocket?.close(1000, "User ended call")
        webSocket = null
    }

    private fun playRemoteAudio(url: String) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                stopAudioPlayback()
                mediaPlayer = MediaPlayer().apply {
                    setAudioAttributes(
                        AudioAttributes.Builder()
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                            .build()
                    )
                    setDataSource(url)
                    setOnPreparedListener { mp ->
                        isPlaying = true
                        mp.start()
                    }
                    setOnCompletionListener {
                        isPlaying = false
                        stopAudioPlayback()
                    }
                    setOnErrorListener { _, _, _ ->
                        isPlaying = false
                        true
                    }
                    prepareAsync()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun stopAudioPlayback() {
        try {
            if (mediaPlayer != null) {
                if (mediaPlayer?.isPlaying == true) {
                    mediaPlayer?.stop()
                }
                mediaPlayer?.release()
                mediaPlayer = null
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        isPlaying = false
    }

    fun isAssistantSpeaking(): Boolean = isPlaying
}
