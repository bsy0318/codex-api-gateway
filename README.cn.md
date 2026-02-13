# Codex CLI Proxy API

这是一个本地代理服务，可把 Codex CLI 作为 OpenAI 兼容 API 使用。
也可以给外部客户端使用，调用方式基本和真实 OpenAI API 一致。

## 支持接口
- `GET /health`
- `GET /`（Web 控制台）
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## 准备
```bash
codex --version
codex login
```

## 启动
```bash
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

## 端口设置
- 启动时指定端口：
```bash
uvicorn server:app --host 127.0.0.1 --port 8999
```
- 或环境变量：
```bash
set PORT=8999
python server.py
```
- 也可在 `index.html` 的 Host/Port 输入框中切换目标地址。

## 外部调用（像真实 API 一样）
- 允许外部访问启动：
```bash
uvicorn server:app --host 0.0.0.0 --port 8999
```
- 其他设备连接：
```python
from openai import OpenAI

client = OpenAI(base_url="http://<SERVER_IP>:8999/v1", api_key="not-needed")
```

## 环境变量
- `CODEX_PROXY_DEFAULT_MODEL`
- `CODEX_PROXY_MODELS`
- `CODEX_PROXY_TIMEOUT`
- `CODEX_PROXY_STREAM_CHUNK_SIZE`
- `HOST`, `PORT`

## 法律免责声明
- 本项目为非官方社区代理，与 OpenAI、Anthropic、models.dev 无隶属、背书或官方支持关系。
- 你需自行承担部署和使用本软件的全部责任，包括 API 政策合规、数据保护、访问控制及适用法律法规。
- 本软件按“现状”提供，不附带任何形式的保证；用于生产环境的风险由使用者自行承担。
