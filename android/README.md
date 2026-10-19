# AI Voice Agent — Android Application Companion

This Android application provides direct, real-time internet-based voice calling with your AI Voice Agent backend. It uses your device microphone and plays AI speech responses in real-time, eliminating telephone call costs through Exotel.

---

## Key Features

1. **Internet-Based Voice Calling**: Communicates with the FastAPI WebSocket pipeline (`/api/ws/live-call`) over secure HTTPS/WSS.
2. **Hands-Free Audio Pipeline**:
   - WebChromeClient automatically captures microphone input using HTML5 Web Audio & MediaRecorder.
   - 400ms pre-roll speech preservation ensures opening phonemes are never clipped.
   - Low-latency Faster-Whisper STT, Ollama LLM, and Piper TTS audio output.
3. **Barge-In / Instant Interruption**: Users can speak anytime or tap the interrupt button to halt AI playback and redirect the conversation immediately.
4. **Data Isolation & Security**: Uses scoped `caller-session` JWT tokens, isolating customer records and preventing exposure of administrative CRM data.
5. **Configurable Backend Endpoint**: Easily change backend URL in settings (defaults to the live public ngrok tunnel).

---

## Project Structure

```
android/
├── build.gradle                 # Root project Gradle configuration
├── settings.gradle              # Module includes (:app)
├── app/
│   ├── build.gradle             # App module dependencies & SDK configuration
│   └── src/
│       └── main/
│           ├── AndroidManifest.xml # Permissions (RECORD_AUDIO, INTERNET, etc.)
│           ├── java/com/aivoice/agent/
│           │   ├── MainActivity.kt        # Primary activity with WebView & permission bridge
│           │   ├── Config.kt              # Backend endpoint persistence
│           │   ├── VoiceAgentClient.kt    # Native OkHttp WebSocket voice client
│           │   └── AudioRecorderService.kt# 16kHz PCM/WAV native recorder
│           └── res/
│               ├── layout/activity_main.xml
│               └── values/{strings,colors,themes}.xml
```

---

## Permissions Required

The app requests:
- `android.permission.RECORD_AUDIO`: Required for microphone voice input.
- `android.permission.INTERNET`: Required for WebSocket and API communication.
- `android.permission.MODIFY_AUDIO_SETTINGS`: Required for voice communication audio routing.

---

## How to Build & Run

### Option 1: Using Android Studio
1. Open Android Studio.
2. Select **File > Open** and choose the `android/` directory in this workspace.
3. Let Gradle sync project dependencies.
4. Connect an Android device (or launch an Android Virtual Device / Emulator).
5. Click the green **Run (Shift+F10)** button.

### Option 2: Using Command Line (Gradle)
```bash
cd android
./gradlew assembleDebug
```
The generated APK will be at:
`android/app/build/outputs/apk/debug/app-debug.apk`

To install on a connected Android device:
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## Testing Audio Flow & WebSocket

You can verify Android backend connectivity and audio transmission using the automated test suite:
```bash
python tests/test_android_flow.py
```
This tests session token issuance, rate limiting, audio turn delivery, and barge-in from the Android client perspective.
