import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


CLI_PATH = shutil.which("codex")
CLI_VERSION = "unknown"

MODEL_LIST = [
    m.strip()
    for m in os.getenv(
        "CODEX_PROXY_MODELS", "gpt-5.3-codex,gpt-5.1,gpt-4.1-mini"
    ).split(",")
    if m.strip()
]
DEFAULT_MODEL = os.getenv(
    "CODEX_PROXY_DEFAULT_MODEL", MODEL_LIST[0] if MODEL_LIST else "gpt-5.3-codex"
)
REQUEST_TIMEOUT_SECONDS = int(os.getenv("CODEX_PROXY_TIMEOUT", "600"))
CHUNK_SIZE = int(os.getenv("CODEX_PROXY_STREAM_CHUNK_SIZE", "80"))
MODELS_DEV_URL = os.getenv("CODEX_PROXY_MODELS_URL", "https://models.dev/api.json")
MODELS_CACHE_TTL_SECONDS = int(os.getenv("CODEX_PROXY_MODELS_CACHE_TTL", "3600"))
TEST_HTML_PATH = Path(__file__).parent / "index.html"
MODELS_CACHE: dict[str, Any] = {"at": 0.0, "data": []}


def _check_cli() -> None:
    global CLI_VERSION
    if not CLI_PATH:
        raise RuntimeError("codex CLI not found. Install Codex CLI and add it to PATH.")

    try:
        proc = subprocess.run(
            [CLI_PATH, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        CLI_VERSION = (proc.stdout or "").strip() or "unknown"
    except Exception as exc:
        raise RuntimeError(f"failed to check codex version: {exc}") from exc


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
            elif isinstance(part, str):
                chunks.append(part)
        return "".join(chunks)
    return str(content or "")


def _build_prompt(
    messages: list[dict[str, Any]], system_hint: str | None = None
) -> str:
    lines: list[str] = []
    if system_hint:
        lines.append(f"system: {system_hint}")

    for msg in messages:
        role = str(msg.get("role", "user"))
        text = _extract_text_content(msg.get("content", ""))
        lines.append(f"{role}: {text}")

    return "\n".join(lines).strip()


def _parse_json_line(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _extract_agent_text(event: dict[str, Any]) -> str | None:
    if event.get("type") != "item.completed":
        return None
    item = event.get("item", {})
    if not isinstance(item, dict):
        return None
    if item.get("type") != "agent_message":
        return None
    text = item.get("text")
    return text if isinstance(text, str) and text else None


def _extract_usage(event: dict[str, Any]) -> dict[str, int] | None:
    if event.get("type") != "turn.completed":
        return None
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
    }


def _chat_usage_payload(usage: dict[str, int]) -> dict[str, int]:
    prompt_tokens = int(usage.get("input_tokens", 0))
    completion_tokens = int(usage.get("output_tokens", 0))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _response_usage_payload(usage: dict[str, int]) -> dict[str, int]:
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _response_object(
    response_id: str, model: str, text: str, usage: dict[str, int]
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "usage": _response_usage_payload(usage),
    }


def _models_from_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for provider_id, provider in sorted(catalog.items(), key=lambda x: str(x[0])):
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if not isinstance(models, dict):
            continue

        for model_obj in models.values():
            if not isinstance(model_obj, dict):
                continue
            model_id = str(model_obj.get("id", "")).strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            rows.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": str(provider_id),
                }
            )

    rows.sort(key=lambda m: str(m.get("id", "")))
    return rows


def _fetch_models_dev_sync() -> list[dict[str, Any]]:
    req = urllib.request.Request(
        MODELS_DEV_URL,
        headers={"User-Agent": "codex-cli-proxy/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        return []
    return _models_from_catalog(parsed)


async def _get_models_data() -> list[dict[str, Any]]:
    now = time.time()
    cached = MODELS_CACHE.get("data")
    cached_at = float(MODELS_CACHE.get("at", 0.0) or 0.0)
    if cached and now - cached_at < MODELS_CACHE_TTL_SECONDS:
        return cached

    try:
        remote_models = await asyncio.to_thread(_fetch_models_dev_sync)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        remote_models = []

    if remote_models:
        MODELS_CACHE["at"] = now
        MODELS_CACHE["data"] = remote_models
        return remote_models

    fallback = [
        {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "openai",
        }
        for model in MODEL_LIST
    ]

    if DEFAULT_MODEL and DEFAULT_MODEL not in {m["id"] for m in fallback}:
        fallback.insert(
            0,
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            },
        )

    return fallback


def _messages_from_responses_input(
    raw_input: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    system_text = None

    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}], None

    if not isinstance(raw_input, list):
        return [], None

    messages: list[dict[str, Any]] = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content", "")

        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type in {"input_text", "output_text", "text"}:
                        parts.append(
                            {"type": "text", "text": str(block.get("text", ""))}
                        )
                elif isinstance(block, str):
                    parts.append({"type": "text", "text": block})
            content = parts

        text_content = _extract_text_content(content)
        if role == "system":
            system_text = text_content
        else:
            messages.append({"role": role, "content": text_content})

    return messages, system_text


async def _iter_codex_events(
    prompt: str, model: str | None
) -> tuple[asyncio.subprocess.Process, AsyncGenerator[dict[str, Any], None]]:
    args = [CLI_PATH, "exec", "--skip-git-repo-check", "--json"]
    if model:
        args.extend(["-m", model])
    args.append(prompt)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def gen() -> AsyncGenerator[dict[str, Any], None]:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue
            parsed = _parse_json_line(decoded)
            if parsed:
                yield parsed

    return proc, gen()


async def _run_codex(prompt: str, model: str | None) -> tuple[str, dict[str, int], str]:
    proc, events = await _iter_codex_events(prompt, model)
    pieces: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}

    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            async for event in events:
                text = _extract_agent_text(event)
                if text:
                    pieces.append(text)
                parsed_usage = _extract_usage(event)
                if parsed_usage:
                    usage = parsed_usage
            stderr_bytes = await proc.stderr.read()
            await proc.wait()
    except TimeoutError as exc:
        if proc.returncode is None:
            proc.kill()
        raise HTTPException(status_code=504, detail="Codex request timed out") from exc

    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
    final_text = "\n\n".join([p for p in pieces if p]).strip()

    if proc.returncode != 0 and not final_text:
        detail = stderr_text or "Codex CLI execution failed"
        raise HTTPException(status_code=502, detail=detail[:1000])

    return final_text, usage, stderr_text


@asynccontextmanager
async def lifespan(_: FastAPI):
    _check_cli()
    yield


app = FastAPI(title="Codex CLI Proxy", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "cli": "codex",
            "cli_path": CLI_PATH,
            "cli_version": CLI_VERSION,
            "default_model": DEFAULT_MODEL,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if TEST_HTML_PATH.exists():
        return HTMLResponse(TEST_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    data = await _get_models_data()
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))

    system_text = None
    non_system_messages: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            system_text = _extract_text_content(msg.get("content", ""))
            continue
        non_system_messages.append(msg)

    prompt = _build_prompt(non_system_messages, system_text)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream:

        async def event_stream() -> AsyncGenerator[str, None]:
            first_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

            proc, events = await _iter_codex_events(prompt, model)
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                    async for event in events:
                        text = _extract_agent_text(event)
                        if not text:
                            continue
                        for i in range(0, len(text), CHUNK_SIZE):
                            chunk = text[i : i + CHUNK_SIZE]
                            payload = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    stderr_bytes = await proc.stderr.read()
                    await proc.wait()
                    if proc.returncode != 0:
                        err = stderr_bytes.decode("utf-8", errors="replace").strip()[
                            :600
                        ]
                        if err:
                            payload = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": f"\n[proxy warning] {err}"
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                payload = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "\n[proxy error] request timeout"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            finally:
                end_payload = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(end_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, usage, _ = await _run_codex(prompt, model)

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": _chat_usage_payload(usage),
    }


@app.post("/v1/responses")
async def responses_api(request: Request) -> Any:
    body = await request.json()
    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))

    raw_input = body.get("input")
    messages, system_text = _messages_from_responses_input(raw_input)
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="input is required (string or message array)",
        )

    prompt = _build_prompt(messages, system_text)
    response_id = f"resp_{uuid.uuid4().hex[:12]}"

    if stream:

        async def response_stream() -> AsyncGenerator[str, None]:
            created_event = {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "in_progress",
                    "model": model,
                },
            }
            yield f"data: {json.dumps(created_event, ensure_ascii=False)}\n\n"

            proc, events = await _iter_codex_events(prompt, model)
            usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                    async for event in events:
                        maybe_usage = _extract_usage(event)
                        if maybe_usage:
                            usage = maybe_usage

                        text = _extract_agent_text(event)
                        if not text:
                            continue
                        for i in range(0, len(text), CHUNK_SIZE):
                            chunk = text[i : i + CHUNK_SIZE]
                            delta_event = {
                                "type": "response.output_text.delta",
                                "response_id": response_id,
                                "delta": chunk,
                            }
                            yield f"data: {json.dumps(delta_event, ensure_ascii=False)}\n\n"

                    stderr_bytes = await proc.stderr.read()
                    await proc.wait()
                    if proc.returncode != 0:
                        err = stderr_bytes.decode("utf-8", errors="replace").strip()[
                            :600
                        ]
                        if err:
                            error_event = {
                                "type": "response.error",
                                "response_id": response_id,
                                "error": {
                                    "message": err,
                                },
                            }
                            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            except TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                error_event = {
                    "type": "response.error",
                    "response_id": response_id,
                    "error": {
                        "message": "request timeout",
                    },
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            finally:
                done_event = {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "model": model,
                        "usage": _response_usage_payload(usage),
                    },
                }
                yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            response_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text, usage, _ = await _run_codex(prompt, model)
    return _response_object(response_id, model, text, usage)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
