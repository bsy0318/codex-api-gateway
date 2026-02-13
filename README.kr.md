# CCodex CLI Proxy API

로컬에서 Codex CLI를 OpenAI 호환 API처럼 사용할 수 있는 프록시 서버입니다.
외부 클라이언트에서도 OpenAI API와 유사한 방식으로 그대로 연동할 수 있습니다.

## 지원 엔드포인트
- `GET /health`
- `GET /` (웹 콘솔)
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## 준비
```bash
codex --version
codex login
copy .env.example .env
```

## 실행
```bash
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

## 포트 설정
- 서버 실행 시 포트 변경:
```bash
uvicorn server:app --host 127.0.0.1 --port 8999
```
- 또는 환경 변수:
```bash
set PORT=8999
python server.py
```
- 웹 콘솔(`index.html`)에서도 Host/Port를 직접 바꿔 API 대상을 변경할 수 있습니다.

## 외부에서 실제 API처럼 쓰기
- 서버를 외부 접근 가능하게 바인딩:
```bash
uvicorn server:app --host 0.0.0.0 --port 8999
```
- 다른 장치/서버에서는 아래처럼 연결:
```python
from openai import OpenAI

client = OpenAI(base_url="http://<SERVER_IP>:8999/v1", api_key="not-needed")
```
- 운영 환경에서는 반드시 방화벽/리버스프록시/접근제어를 함께 설정하세요.

## 사용 예시
```bash
curl http://127.0.0.1:8000/v1/models
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
res = client.chat.completions.create(
    model="gpt-5.3-codex",
    messages=[{"role": "user", "content": "안녕하세요"}],
)
print(res.choices[0].message.content)
```

## 환경 변수
- `CODEX_PROXY_DEFAULT_MODEL`
- `CODEX_PROXY_MODELS`
- `CODEX_PROXY_TIMEOUT`
- `CODEX_PROXY_STREAM_CHUNK_SIZE`
- `HOST`, `PORT`

서버 시작 시 `.env` 파일을 자동으로 로드합니다.

## 참고
- 내부적으로 `codex exec --json --skip-git-repo-check`를 실행합니다.
- 기본 바인딩은 localhost입니다. 외부 공개 시 보안 설정을 꼭 적용하세요.

## 법적 면책사항
- 이 프로젝트는 비공식 커뮤니티 프록시이며 OpenAI, Anthropic, models.dev와 제휴/보증/공식 지원 관계가 없습니다.
- 본 소프트웨어의 배포 및 사용에 따른 API 정책 준수, 개인정보/데이터 보호, 접근통제, 관련 법규 준수 책임은 전적으로 사용자에게 있습니다.
- 본 소프트웨어는 어떠한 보증 없이 "있는 그대로" 제공되며, 운영 환경 사용에 따른 위험은 사용자 책임입니다.
