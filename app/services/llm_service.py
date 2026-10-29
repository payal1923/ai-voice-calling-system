import asyncio
import httpx

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

_ollama_lock = asyncio.Lock()


_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=180.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=300.0)
        )
    return _client


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
            "num_ctx": 512,
            "num_thread": 4,
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


async def generate_response_stream(prompt: str, num_predict: int = 40, timings: dict = None):
    """
    Streams tokens asynchronously from Ollama /api/generate endpoint.
    Yields individual string tokens as they arrive.
    Optionally records TTFT (time-to-first-token) and total LLM duration in `timings`.
    """
    import json
    import time

    t_start = time.perf_counter()
    if timings is not None:
        timings["llm_start"] = t_start
    first_token_seen = False

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "60m",
        "options": {
            "num_predict": num_predict,
            "temperature": 0.1,
            "num_ctx": 512,
            "num_thread": 4,
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
                        if not first_token_seen:
                            first_token_seen = True
                            if timings is not None:
                                timings["llm_ttft"] = time.perf_counter() - t_start
                        yield token
                except json.JSONDecodeError:
                    continue

    if timings is not None:
        timings["llm_total"] = time.perf_counter() - t_start
