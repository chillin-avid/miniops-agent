"""Agent 可以使用的少量受控工具。"""

from __future__ import annotations

from typing import Any

from models import SearchHit
from retrieval import RunbookIndex
from runtime_logs import JsonlLogStore
from service_catalog import SERVICE_CATALOG, catalog_for_prompt, resolve_service


class MiniOpsTools:
    """封装工具参数校验和执行结果，模型不能直接访问文件系统。"""

    def __init__(self, index: RunbookIndex, log_store: JsonlLogStore) -> None:
        self.index = index
        self.log_store = log_store

    @property
    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_runbooks",
                    "description": "检索内部故障手册，返回带来源的相关片段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_logs",
                    "description": (
                        "按服务、级别、关键词和最近分钟数查询只读 JSONL 运行日志。"
                        "必须提供服务或关键词；服务目录：" + catalog_for_prompt()
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {"type": "string"},
                            "level": {"type": "string"},
                            "keyword": {"type": "string"},
                            "minutes": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 1440,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_runbooks":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("检索问题不能为空")
            hits = self.index.search(
                query, min(max(int(arguments.get("limit", 3)), 1), 5)
            )
            self.log_store.record(
                "retrieval",
                "INFO",
                f"search_runbooks returned {len(hits)} hits",
                event="tool.search_runbooks",
            )
            return {"hits": [_hit_payload(hit) for hit in hits]}
        if name == "query_logs":
            # 并把“支付接口”等用户说法统一成日志中的真实服务名。
            normalized = self.normalize_log_arguments(arguments)
            if not normalized.get("service") and not normalized.get("keyword"):
                raise ValueError("查询日志前请提供明确的服务或关键词")
            logs = self._query_logs(normalized)
            self.log_store.record(
                "log-query",
                "INFO",
                f"query_logs returned {len(logs)} records",
                event="tool.query_logs",
            )
            return {"logs": logs}
        raise ValueError(f"不允许调用工具：{name}")

    def normalize_log_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """复制日志参数，并把服务的常见说法转换成目录中的统一名称。"""

        normalized = dict(arguments)
        raw_service = str(arguments.get("service", "")).strip()
        if raw_service:
            normalized["service"] = resolve_service(raw_service)
        return normalized

    def valid_log_evidence(
        self,
        arguments: dict[str, Any],
        logs: list[dict[str, str]],
        _context: str,
    ) -> bool:
        """模型负责语义选服务，程序只验证服务范围和返回记录是否匹配。"""

        if not logs:
            return False
        service = str(arguments.get("service", "")).strip()
        keyword = str(arguments.get("keyword", "")).strip().lower()
        if not service and not keyword:
            return False
        if service and service not in SERVICE_CATALOG:
            return False

        for item in logs:
            if service and str(item.get("service", "")).lower() != service.lower():
                return False
            if keyword and keyword not in str(item.get("message", "")).lower():
                return False
        return True

    def _query_logs(self, arguments: dict[str, Any]) -> list[dict[str, str]]:
        service = str(arguments.get("service", "")).strip()
        level = str(arguments.get("level", "")).strip()
        keyword = str(arguments.get("keyword", "")).strip()
        minutes = min(max(int(arguments.get("minutes", 60)), 1), 1440)
        return self.log_store.query(
            service=service,
            level=level,
            keyword=keyword,
            minutes=minutes,
        )


def _hit_payload(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "document": hit.document,
        "title": hit.title,
        "text": hit.text,
        "score": hit.score,
    }
