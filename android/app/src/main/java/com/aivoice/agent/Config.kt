package com.aivoice.agent

import android.content.Context
import android.content.SharedPreferences

object Config {
    private const val PREFS_NAME = "ai_voice_agent_prefs"
    private const val KEY_SERVER_URL = "key_server_url"

    // Default HTTPS public tunnel endpoint provided by active backend
    const val DEFAULT_SERVER_URL = "https://drift-dwarf-matron.ngrok-free.dev"

    fun getServerUrl(context: Context): String {
        val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_SERVER_URL, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL
    }

    fun setServerUrl(context: Context, url: String) {
        val cleanUrl = url.trim().removeSuffix("/")
        val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_SERVER_URL, cleanUrl).apply()
    }

    fun getCallPageUrl(context: Context): String {
        return "${getServerUrl(context)}/call"
    }

    fun getWebSocketUrl(context: Context, token: String): String {
        val base = getServerUrl(context)
        val wsScheme = if (base.startsWith("https://", ignoreCase = true)) "wss://" else "ws://"
        val hostAndPort = base.substringAfter("://")
        return "$wsScheme$hostAndPort/api/ws/live-call?token=$token"
    }

    fun getSessionUrl(context: Context): String {
        return "${getServerUrl(context)}/api/auth/caller-session"
    }
}
