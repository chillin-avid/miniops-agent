"""统一读取 MiniOps 与主项目的本地环境变量文件。"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_ENV = ROOT / ".env"
SHARED_ENV = ROOT.parent / "hello-agents-code-review-agent" / ".env"
ALLOWED_NAMES = {
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL_ID",
    "LLM_TIMEOUT",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_TIMEOUT",
    "RERANK_BASE_URL",
    "RERANK_MODEL_ID",
    "RERANK_CANDIDATES",
    "RERANK_TIMEOUT",
    "MINIOPS_RUNTIME_LOG_PATH",
    "MINIOPS_LOG_DIRECTORIES",
    "MINIOPS_LOG_MAX_SCAN_BYTES",
    "MINIOPS_LOG_MAX_FILE_BYTES",
    "MINIOPS_LOG_BACKUP_COUNT",
}


def load_config() -> None:
    """先读本项目配置，再用主项目配置补齐仍然缺少的项目。"""

    for path in (LOCAL_ENV, SHARED_ENV):
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if (
                name in ALLOWED_NAMES
                and name not in os.environ
                and value
                and value != "replace-me"
                and "your-openai-compatible-endpoint" not in value
            ):
                os.environ[name] = value
