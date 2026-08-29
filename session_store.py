"""按会话编号持久化 MiniOps 最近的多轮对话。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


MAX_SESSION_MESSAGES = 12


class JsonSessionStore:
    """每个会话保存一个 JSON，服务重启后可以继续追问。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """读取最近消息；文件不存在或损坏时按新会话处理。"""

        target = self._path(session_id)
        if not target.is_file():
            return []
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        if not isinstance(messages, list):
            return []
        selected: list[dict[str, Any]] = []
        for item in messages[-MAX_SESSION_MESSAGES:]:
            if not isinstance(item, dict) or item.get("role") not in {
                "user",
                "assistant",
            }:
                continue
            selected.append(
                {
                    "role": str(item["role"]),
                    "content": str(item.get("content", ""))[:12_000],
                }
            )
        return selected

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """先写临时文件再替换，避免断电留下半个会话 JSON。"""

        target = self._path(session_id)
        temporary = target.with_suffix(".json.tmp")
        payload = {
            "session_id": session_id,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "messages": messages[-MAX_SESSION_MESSAGES:],
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)

    def _path(self, session_id: str) -> Path:
        """只接受十六进制编号，避免利用会话编号拼接任意路径。"""

        if (
            not session_id
            or len(session_id) > 80
            or any(character not in "0123456789abcdef" for character in session_id)
        ):
            raise ValueError("Invalid session id")
        return self.root / f"{session_id}.json"
