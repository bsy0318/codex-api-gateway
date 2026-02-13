# Codex CLI Proxy API

This project runs Codex CLI behind an OpenAI-compatible local proxy API.
It can also be consumed by external clients with nearly the same usage pattern as the real OpenAI API.

## Endpoints
- `GET /health`
- `GET /` (web console)
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## Setup
```bash
codex --version
codex login
pip install -r requirements.txt
copy .env.example .env
```

## Run
```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

## Port Configuration
- Choose port in command:
```bash
uvicorn server:app --host 127.0.0.1 --port 8999
```
- Or with env var:
```bash
set PORT=8999
python server.py
```
- You can also change target Host/Port in the web console.

## External Usage (like a real API)
- Bind for external access:
```bash
uvicorn server:app --host 0.0.0.0 --port 8999
```
- Connect from another machine:
```python
from openai import OpenAI

client = OpenAI(base_url="http://<SERVER_IP>:8999/v1", api_key="not-needed")
```

## Environment Variables
- `CODEX_PROXY_DEFAULT_MODEL`
- `CODEX_PROXY_MODELS`
- `CODEX_PROXY_TIMEOUT`
- `CODEX_PROXY_STREAM_CHUNK_SIZE`
- `HOST`, `PORT`

`.env` is loaded automatically at startup.

## Legal Disclaimer
- This project is an unofficial community proxy and is not affiliated with, endorsed by, or sponsored by OpenAI, Anthropic, or models.dev.
- You are solely responsible for how you deploy and use this software, including API policy compliance, data protection, access control, and applicable laws/regulations.
- The software is provided "as is" without warranties of any kind; use in production is at your own risk.
