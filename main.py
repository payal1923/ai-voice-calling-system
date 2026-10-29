import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services.llm_service import generate_response
from app.api.auth_routes import router as auth_router
from app.api.call_routes import router as call_router
from app.api.automation_routes import router as automation_router
from app.api.escalation_routes import router as escalation_router
from app.api.telephony_routes import router as telephony_router

app = FastAPI(title="AI Voice Calling Platform", version="2.0.0")

# Enable CORS for frontend clients (e.g., Live Server on port 5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.ngrok-free\.(app|dev))$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
os.makedirs("recordings", exist_ok=True)
os.makedirs("static/waiting", exist_ok=True)

# Mount recordings audio directory
app.mount(
    "/audio",
    StaticFiles(directory="recordings", html=False),
    name="audio_files",
)

# Mount static directory for waiting audio and assets
app.mount(
    "/static",
    StaticFiles(directory="static", html=False),
    name="static_files",
)

# Include Routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(call_router, prefix="/api", tags=["Calls & Customers"])
app.include_router(automation_router, prefix="/api/automation", tags=["Automated Calling"])
app.include_router(escalation_router, prefix="/api/escalations", tags=["Human Assistance"])
app.include_router(telephony_router, prefix="/api/telephony", tags=["Telephony Operations"])


@app.on_event("startup")
async def warmup_services():
    """
    Pre-initializes STT, TTS, and Ollama LLM models during FastAPI server startup
    to eliminate cold-start latency overhead on Turn 1 caller requests.
    """
    try:
        print("[STARTUP] Pre-warming STT, TTS, and Ollama LLM models...")
        from app.services.stt_service import STTService
        from app.services.tts_service import TTSService
        from app.services.llm_service import generate_response

        # 1. Warm Faster-Whisper STT model & CTranslate2 graph
        print("[STARTUP] Loading STT model...")
        stt = STTService()
        print("[STARTUP] Warming STT inference...")
        warmup_sample = os.path.join("static", "waiting", "waiting_1.wav")
        if os.path.exists(warmup_sample):
            try:
                stt.transcribe(warmup_sample)
            except Exception as stt_err:
                print(f"[STARTUP WARNING] STT warmup notice: {stt_err}")
        print("[STARTUP] STT warmup complete.")

        # 2. Warm Piper TTS model & ONNX inference session
        print("[STARTUP] Loading and warming Piper TTS model...")
        tts = TTSService()
        tts_warmup_path = os.path.join("recordings", "warmup_tts.wav")
        try:
            tts.synthesize("System ready.", tts_warmup_path, language="en")
            print("[STARTUP] Piper TTS warmup complete.")
        except Exception as tts_err:
            print(f"[STARTUP WARNING] Piper TTS warmup notice: {tts_err}")

        # Pre-render static waiting audio clips if missing
        waiting_clips = {
            "waiting_1.wav": "Please hold on for a moment while I process your request.",
            "waiting_2.wav": "Thank you for your patience, I am still processing your information.",
            "waiting_3.wav": "Just a moment longer, your response is almost ready.",
        }
        os.makedirs(os.path.join("static", "waiting"), exist_ok=True)
        for clip_name, clip_text in waiting_clips.items():
            clip_path = os.path.join("static", "waiting", clip_name)
            if not os.path.exists(clip_path) or os.path.getsize(clip_path) < 1000:
                try:
                    tts.synthesize(clip_text, clip_path, language="en")
                    print(f"[STARTUP] Generated missing waiting audio clip: {clip_path}")
                except Exception as ex:
                    print(f"[STARTUP WARNING] Failed to generate waiting clip {clip_name}: {ex}")

        # 3. Warm Ollama LLM model weights
        await generate_response("hi", num_predict=5)
        print("[STARTUP] Warmup complete. All models hot in RAM.")
    except Exception as e:
        print(f"[STARTUP WARNING] Model warmup warning: {e}")



@app.get("/")
def home():
    return FileResponse(
        "frontend/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/call")
def public_voice_call():
    """Serves the original Voice Operations Hub website directly."""
    return FileResponse(
        "frontend/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi import Response
    return Response(status_code=204)


@app.post("/ai-test")
async def ai_test(prompt: str):
    response = await generate_response(prompt)
    return {
        "prompt": prompt,
        "response": response,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)