import asyncio
import httpx

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

_ollama_lock = asyncio.Lock()


def get_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=180.0)



async def generate_response(prompt: str, num_predict: int = 40, return_details: bool = False):
    import time
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "60m",
        "options": {
            "num_predict": num_predict,
            "temperature": 0.1,
            "num_ctx": 1024,
        },
    }

    t_lock_start = time.perf_counter()
    async with _ollama_lock:
        t_lock_acquired = time.perf_counter()
        client = get_client()
        
        t_ollama_start = time.perf_counter()
        response = await client.post(
            OLLAMA_URL,
            json=payload
        )
        t_ollama_end = time.perf_counter()

        response.raise_for_status()

        data = response.json()
        resp_text = data["response"]

        if return_details:
            return resp_text, {
                "t_lock_start": t_lock_start,
                "t_lock_acquired": t_lock_acquired,
                "t_ollama_start": t_ollama_start,
                "t_ollama_end": t_ollama_end,
                "lock_wait_sec": t_lock_acquired - t_lock_start,
                "ollama_req_sec": t_ollama_end - t_ollama_start,
            }
        return resp_text


async def generate_response_stream(prompt: str, num_predict: int = 40):
    """
    Streams tokens asynchronously from Ollama /api/generate endpoint.
    Yields individual string tokens as they arrive.
    """
    import json
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "60m",
        "options": {
            "num_predict": num_predict,
            "temperature": 0.1,
            "num_ctx": 1024,
        },
    }

    async with _ollama_lock:
        client = get_client()
        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue
