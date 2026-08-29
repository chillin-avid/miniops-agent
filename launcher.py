"""双击启动入口：先启动服务，再延迟打开浏览器。"""

from __future__ import annotations

import os
import threading
import webbrowser

import uvicorn

from config import load_config


# 版本参数用于绕开已经打开的旧标签页和浏览器缓存。
URL = "http://127.0.0.1:8090/?ui=4"


def open_browser() -> None:
    """稍等服务开始监听后，再用系统默认浏览器打开页面。"""

    if os.getenv("MINIOPS_NO_BROWSER") != "1":
        webbrowser.open(URL)


def main() -> None:
    load_config()
    print("Starting MiniOps Agent...")
    print(f"Browser URL: {URL}")
    configured = all(
        os.getenv(name, "").strip()
        for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL_ID")
    )
    print("Agent mode: model" if configured else "Agent mode: offline fallback")
    timer = threading.Timer(1.5, open_browser)
    timer.daemon = True
    timer.start()
    uvicorn.run("app:app", host="127.0.0.1", port=8090, log_level="info")


if __name__ == "__main__":
    main()
