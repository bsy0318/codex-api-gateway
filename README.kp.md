# Codex CLI Proxy API

이 봉사기는 로컬에서 Codex CLI를 OpenAI 호환 API처럼 리용하게 하는 중계봉사기입니다.
외부 측에서도 OpenAI API와 비슷한 방식으로 그대로 련동할수 있습니다.

## 봉사 주소
- `GET /health`
- `GET /` (웹 조작창)
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## 준비
```bash
codex --version
codex login
```

## 실행
```bash
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

## 포구(포트) 설정
- 실행 때 포구 지정:
```bash
uvicorn server:app --host 127.0.0.1 --port 8999
```
- 또는 환경변수:
```bash
set PORT=8999
python server.py
```
- 웹 조작창에서도 Host/Port 값을 바꿀 수 있습니다.

## 외부 련동
- 외부 접근을 허용하여 실행:
```bash
uvicorn server:app --host 0.0.0.0 --port 8999
```
- 다른 장치에서 API처럼 접속:
```python
from openai import OpenAI

client = OpenAI(base_url="http://<SERVER_IP>:8999/v1", api_key="not-needed")
```

## 환경변수
- `CODEX_PROXY_DEFAULT_MODEL`
- `CODEX_PROXY_MODELS`
- `CODEX_PROXY_TIMEOUT`
- `CODEX_PROXY_STREAM_CHUNK_SIZE`
- `HOST`, `PORT`

## 법적 면책
- 이 프로잭트는 비공식 중계봉사기이며 OpenAI, Anthropic, models.dev와 제휴 또는 공식승인이 없습니다.
- 본 소프트웨어의 리용 및 배포에서 API 정책준수, 자료보호, 접근통제, 해당 법규준수 책임은 전적으로 리용자에게 있습니다.
- 본 소프트웨어는 그 어떤 담보도 없이 "있는 그대로" 제공되며 운용환경 리용위험은 리용자 책임입니다.
