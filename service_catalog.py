"""统一日志中的服务名称和用户常用说法。"""

from __future__ import annotations

import re
from typing import Any


SERVICE_CATALOG: dict[str, dict[str, Any]] = {
    "payment-api": {
        "description": "负责支付请求的接口服务",
        "signals": ("支付", "付款", "payment"),
    },
    "order-worker": {
        "description": "负责后台处理订单的任务服务",
        "signals": ("订单", "order"),
    },
    "gateway": {
        "description": "接收外部请求的网关服务",
        "signals": ("网关", "gateway"),
    },
    "miniops": {
        "description": "MiniOps 故障排查助手自身",
        "signals": ("miniops", "故障助手"),
    },
}


def resolve_service(text: str) -> str:
    """输入用户或模型给出的服务说法，返回统一服务名；无法确认时返回空串。"""

    normalized = _normalize(text)
    if not normalized:
        return ""
    matches: list[tuple[int, str]] = []
    for service, item in SERVICE_CATALOG.items():
        candidates = (service, *item["signals"])
        positions = [
            normalized.rfind(_normalize(candidate)) for candidate in candidates
        ]
        latest = max(positions)
        if latest >= 0:
            matches.append((latest, service))

    return max(matches)[1] if matches else ""


def catalog_for_prompt() -> str:
    """把可查询服务整理成简短中文说明，供模型填写日志工具参数。"""

    lines = []
    for service, item in SERVICE_CATALOG.items():
        signals = "、".join(item["signals"])
        lines.append(f"{service}（{item['description']}；常见说法：{signals}）")
    return "；".join(lines)


def _normalize(text: str) -> str:
    """忽略大小写、空格和常见连接符后再匹配服务信号。"""

    return re.sub(r"[\s_-]+", "", str(text).lower())
