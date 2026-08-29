"""项目中共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass(slots=True)
class SearchHit:
    """一条带分数的知识库召回结果。"""

    chunk_id: str
    document: str
    title: str
    text: str
    score: float


@dataclass(slots=True)
class ToolTrace:
    """记录 Agent 为什么调用工具以及工具返回了什么。"""

    tool: str
    arguments: dict[str, Any]
    summary: str


@dataclass(slots=True)
class AgentAnswer:
    """Agent 的统一输出，既供 API 使用，也供评测统计。"""

    answer: str
    citations: list[SearchHit] = field(default_factory=list)
    traces: list[ToolTrace] = field(default_factory=list)
    action: str = "answer"


class ChatRequest(BaseModel):
    """网页发送给后端的一轮对话请求。"""

    message: str = Field(min_length=1, max_length=3000)
    session_id: str | None = Field(default=None, max_length=80)


class CitationPayload(BaseModel):
    document: str
    title: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    action: str
    citations: list[CitationPayload]
    traces: list[dict[str, Any]]
