"""FastAPI 入口：提供聊天接口、索引刷新接口和一个极简网页。"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from agent import MiniOpsAgent
from config import load_config
from models import ChatRequest, ChatResponse, CitationPayload
from retrieval import RunbookIndex
from runtime_logs import log_store_from_environment
from session_store import JsonSessionStore
from tools import MiniOpsTools


ROOT = Path(__file__).resolve().parent
# 在创建检索器之前加载向量接口配置，直接运行 uvicorn 也能读取本地 .env。
load_config()
index = RunbookIndex(ROOT / "runbooks", ROOT / "data")
runtime_logs = log_store_from_environment(ROOT)
tools = MiniOpsTools(index, runtime_logs)
session_store = JsonSessionStore(ROOT / "data" / "sessions")
agent = MiniOpsAgent(tools, session_store)

app = FastAPI(title="MiniOps Agent", version="0.1.0")


@app.on_event("startup")
def prepare_index() -> None:
    index.ensure_ready()
    # 正式服务启动后写入真实时间戳，日志工具可以立即查询本次运行记录。
    runtime_logs.record(
        "miniops", "INFO", "MiniOps service started", event="app.startup"
    )


@app.get("/")
def home() -> FileResponse:
    """每次读取最新页面，并禁止浏览器复用旧版 HTML。"""

    return FileResponse(
        ROOT / "web" / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "chunks": index.ensure_ready()}


@app.post("/api/reindex")
def reindex() -> dict[str, int]:
    return {"chunks": index.rebuild()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or uuid.uuid4().hex
    result = agent.chat(session_id, request.message)
    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        action=result.action,
        citations=[
            CitationPayload(
                document=item.document,
                title=item.title,
                score=item.score,
                excerpt=item.text[:280],
            )
            for item in result.citations
        ],
        traces=[asdict(item) for item in result.traces],
    )
