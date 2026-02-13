# ChatGPT Proxy

ローカルで Codex CLI を OpenAI 互換 API として利用するためのプロキシです。
外部クライアントからも、実際の OpenAI API とほぼ同じ呼び方で連携できます。

## エンドポイント
- `GET /health`
- `GET /` (Web コンソール)
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## セットアップ
```bash
codex --version
codex login
```

## 起動
```bash
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

## ポート設定
- 起動時に変更:
```bash
uvicorn server:app --host 127.0.0.1 --port 8999
```
- または環境変数:
```bash
set PORT=8999
python server.py
```
- `index.html` の Host/Port 入力でも接続先を変更できます。

## 外部利用 (実 API のように使用)
- 外部アクセス可能で起動:
```bash
uvicorn server:app --host 0.0.0.0 --port 8999
```
- 別マシンから接続:
```python
from openai import OpenAI

client = OpenAI(base_url="http://<SERVER_IP>:8999/v1", api_key="not-needed")
```

## 環境変数
- `CODEX_PROXY_DEFAULT_MODEL`
- `CODEX_PROXY_MODELS`
- `CODEX_PROXY_TIMEOUT`
- `CODEX_PROXY_STREAM_CHUNK_SIZE`
- `HOST`, `PORT`

## 免責事項（法的）
- 本プロジェクトは非公式のコミュニティ製プロキシであり、OpenAI、Anthropic、models.dev とは提携・承認・公式サポート関係にありません。
- 本ソフトウェアの利用・公開に伴う API ポリシー準拠、データ保護、アクセス制御、関連法令遵守の責任は利用者にあります。
- 本ソフトウェアは現状有姿（"as is"）で提供され、いかなる保証もありません。運用利用は自己責任で行ってください。
