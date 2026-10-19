package com.aivoice.agent

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.view.View
import android.webkit.*
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.aivoice.agent.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var pendingPermissionRequest: PermissionRequest? = null

    // Register audio record permission launcher
    private val requestAudioPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            pendingPermissionRequest?.grant(pendingPermissionRequest?.resources)
            pendingPermissionRequest = null
            loadVoiceCallingInterface()
        } else {
            Toast.makeText(this, getString(R.string.mic_permission_required), Toast.LENGTH_LONG).show()
            pendingPermissionRequest?.deny()
            pendingPermissionRequest = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupTopBar()
        setupWebView()

        // Check and request microphone permission
        if (hasMicrophonePermission()) {
            loadVoiceCallingInterface()
        } else {
            requestAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        binding.btnRetry.setOnClickListener {
            binding.layoutError.visibility = View.GONE
            loadVoiceCallingInterface()
        }
    }

    private fun hasMicrophonePermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun setupTopBar() {
        binding.btnSettings.setOnClickListener {
            showServerSettingsDialog()
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        val webView = binding.webView
        val settings = webView.settings

        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.databaseEnabled = true
        settings.mediaPlaybackRequiresUserGesture = false
        settings.allowFileAccess = true
        settings.loadsImagesAutomatically = true
        settings.useWideViewPort = true
        settings.loadWithOverviewMode = true

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            settings.safeBrowsingEnabled = false
        }

        // WebChromeClient to grant HTML5 audio capture permission
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                val hasAudio = request.resources.contains(PermissionRequest.RESOURCE_AUDIO_CAPTURE)
                if (hasAudio) {
                    if (hasMicrophonePermission()) {
                        runOnUiThread {
                            request.grant(request.resources)
                        }
                    } else {
                        pendingPermissionRequest = request
                        requestAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                } else {
                    request.deny()
                }
            }

            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                if (newProgress < 100) {
                    binding.progressBar.visibility = View.VISIBLE
                    binding.progressBar.progress = newProgress
                } else {
                    binding.progressBar.visibility = View.GONE
                }
            }
        }

        // WebViewClient to handle loading, custom headers, and errors
        webView.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                binding.layoutError.visibility = View.GONE
                binding.tvStatusBadge.text = "CONNECTING"
                binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.accent_cyan))
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                binding.progressBar.visibility = View.GONE
                binding.tvStatusBadge.text = "LIVE VOICE ACTIVE"
                binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.accent_emerald))
            }

            override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                if (request?.isForMainFrame == true) {
                    binding.progressBar.visibility = View.GONE
                    binding.layoutError.visibility = View.VISIBLE
                    binding.tvStatusBadge.text = "OFFLINE"
                    binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.accent_rose))
                }
            }
        }
    }

    private fun loadVoiceCallingInterface() {
        val targetUrl = Config.getCallPageUrl(this)
        val headers = mapOf("ngrok-skip-browser-warning" to "true")
        binding.webView.loadUrl(targetUrl, headers)
    }

    private fun showServerSettingsDialog() {
        val currentUrl = Config.getServerUrl(this)
        val input = EditText(this).apply {
            setText(currentUrl)
            hint = "e.g. https://your-domain.ngrok-free.dev"
            setSelection(text.length)
        }

        AlertDialog.Builder(this)
            .setTitle("Server Backend URL")
            .setMessage("Set the public HTTPS backend URL for the AI Voice Calling server:")
            .setView(input)
            .setPositiveButton("Save & Connect") { _, _ ->
                val newUrl = input.text.toString().trim()
                if (newUrl.isNotEmpty()) {
                    Config.setServerUrl(this, newUrl)
                    Toast.makeText(this, "Updated server to: $newUrl", Toast.LENGTH_SHORT).show()
                    loadVoiceCallingInterface()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onBackPressed() {
        if (binding.webView.canGoBack()) {
            binding.webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        binding.webView.destroy()
        super.onDestroy()
    }
}
