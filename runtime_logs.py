"""记录并查询带真实时间戳的本地 JSONL 运行日志。"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


MAX_LOG_FILES = 20
MAX_RESULTS = 20
DEFAULT_SCAN_BYTES = 2_000_000
DEFAULT_MAX_FILE_BYTES = 5_000_000
DEFAULT_BACKUP_COUNT = 5


class JsonlLogStore:
    """追加自身运行日志，并只读检索获批目录中的结构化日志。"""

    def __init__(
        self,
        runtime_log: Path,
        query_directories: list[Path] | None = None,
        *,
        max_scan_bytes: int = DEFAULT_SCAN_BYTES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
    ) -> None:
        self.runtime_log = runtime_log.resolve()
        directories = [self.runtime_log.parent, *(query_directories or [])]
        self.query_directories = tuple(
            dict.fromkeys(path.resolve() for path in directories)
        )
        self.max_scan_bytes = max(1_024, int(max_scan_bytes))
        self.max_file_bytes = max(1_024, int(max_file_bytes))
        self.backup_count = min(max(int(backup_count), 1), 20)
        self._write_lock = threading.Lock()

    def record(
        self,
        service: str,
        level: str,
        message: str,
        *,
        event: str = "",
    ) -> bool:
        """以当前真实时间追加一条 JSONL；记录失败不能拖垮主流程。"""

        item = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "service": str(service).strip()[:120] or "miniops",
            "level": str(level).strip().upper()[:20] or "INFO",
            "message": str(message).replace("\r", " ").replace("\n", " ")[:2_000],
        }
        if event:
            item["event"] = str(event).strip()[:120]
        try:
            self.runtime_log.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(item, ensure_ascii=False) + "\n"
            with self._write_lock:
                # 写入前按文件大小轮转，限制本地日志长期增长占用的空间。
                self._rotate_if_needed(len(line.encode("utf-8")))
                with self.runtime_log.open("a", encoding="utf-8") as stream:
                    stream.write(line)
            return True
        except OSError:
            return False

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        """当前文件将超过上限时移动历史文件，并删除最旧备份。"""

        if not self.runtime_log.is_file():
            return
        if self.runtime_log.stat().st_size + incoming_bytes <= self.max_file_bytes:
            return
        oldest = self._backup_path(self.backup_count)
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.is_file():
                source.replace(self._backup_path(index + 1))
        self.runtime_log.replace(self._backup_path(1))

    def _backup_path(self, index: int) -> Path:
        """生成 miniops.1.jsonl 形式的轮转文件名。"""

        return self.runtime_log.with_name(
            f"{self.runtime_log.stem}.{index}{self.runtime_log.suffix}"
        )

    def query(
        self,
        *,
        service: str = "",
        level: str = "",
        keyword: str = "",
        minutes: int = 60,
        limit: int = MAX_RESULTS,
    ) -> list[dict[str, str]]:
        """按真实时间窗口过滤日志，结果按时间从新到旧返回。"""

        now = datetime.now().astimezone()
        earliest = now - timedelta(minutes=min(max(int(minutes), 1), 1_440))
        service_filter = service.strip().lower()
        level_filter = level.strip().upper()
        keyword_filter = keyword.strip().lower()
        selected: list[tuple[datetime, dict[str, str]]] = []
        for path in self._log_files():
            for item in self._recent_records(path):
                timestamp = _parse_timestamp(item.get("timestamp"), now)
                if (
                    timestamp is None
                    or timestamp < earliest
                    or timestamp > now + timedelta(seconds=5)
                ):
                    continue
                item_service = str(item.get("service", "")).strip()
                item_level = str(item.get("level", "")).strip().upper()
                message = str(item.get("message", "")).strip()
                if service_filter and item_service.lower() != service_filter:
                    continue
                if level_filter and item_level != level_filter:
                    continue
                if keyword_filter and keyword_filter not in message.lower():
                    continue
                selected.append(
                    (
                        timestamp,
                        {
                            "time": timestamp.isoformat(timespec="seconds"),
                            "service": item_service,
                            "level": item_level,
                            "message": message,
                        },
                    )
                )
        selected.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in selected[: min(max(int(limit), 1), MAX_RESULTS)]]

    def _log_files(self) -> list[Path]:
        """只扫描配置目录中的 JSONL 普通文件，限制数量避免读取失控。"""

        files: dict[Path, float] = {}
        for directory in self.query_directories:
            if not directory.is_dir():
                continue
            for path in directory.glob("*.jsonl"):
                if path.is_symlink():
                    continue
                resolved = path.resolve()
                if resolved.is_file():
                    try:
                        files[resolved] = resolved.stat().st_mtime
                    except OSError:
                        continue
        return [
            path
            for path, _ in sorted(
                files.items(), key=lambda item: item[1], reverse=True
            )[:MAX_LOG_FILES]
        ]

    def _recent_records(self, path: Path) -> list[dict[str, Any]]:
        """只读文件尾部的有限字节，跳过损坏行而保留其他真实记录。"""

        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                start = max(0, size - self.max_scan_bytes)
                stream.seek(start)
                if start:
                    stream.readline()
                raw_lines = stream.read().splitlines()
        except OSError:
            return []
        records: list[dict[str, Any]] = []
        for raw_line in raw_lines:
            try:
                item = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                records.append(item)
        return records


def log_store_from_environment(root: Path) -> JsonlLogStore:
    """相对配置基于项目目录，并允许追加外部只读日志目录。"""

    default_log = root / "data" / "runtime-logs" / "miniops.jsonl"
    runtime_value = os.getenv("MINIOPS_RUNTIME_LOG_PATH", "").strip()
    runtime_log = Path(runtime_value) if runtime_value else default_log
    if not runtime_log.is_absolute():
        runtime_log = root / runtime_log
    directories = []
    for value in os.getenv("MINIOPS_LOG_DIRECTORIES", "").split(";"):
        value = value.strip()
        if not value:
            continue
        path = Path(value)
        directories.append(path if path.is_absolute() else root / path)
    try:
        scan_bytes = int(
            os.getenv("MINIOPS_LOG_MAX_SCAN_BYTES", str(DEFAULT_SCAN_BYTES))
        )
    except ValueError:
        scan_bytes = DEFAULT_SCAN_BYTES
    try:
        max_file_bytes = int(
            os.getenv("MINIOPS_LOG_MAX_FILE_BYTES", str(DEFAULT_MAX_FILE_BYTES))
        )
        backup_count = int(
            os.getenv("MINIOPS_LOG_BACKUP_COUNT", str(DEFAULT_BACKUP_COUNT))
        )
    except ValueError:
        max_file_bytes = DEFAULT_MAX_FILE_BYTES
        backup_count = DEFAULT_BACKUP_COUNT
    return JsonlLogStore(
        runtime_log,
        directories,
        max_scan_bytes=scan_bytes,
        max_file_bytes=max_file_bytes,
        backup_count=backup_count,
    )


def write_relative_fixture(source: Path, target: Path) -> Path:
    """仅供测试和评测把相对分钟样例转换为当前时间的 JSONL。"""

    records = json.loads(source.read_text(encoding="utf-8"))
    now = datetime.now().astimezone()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in records:
        lines.append(
            json.dumps(
                {
                    "timestamp": (
                        now - timedelta(minutes=int(item.get("minutes_ago", 0)))
                    ).isoformat(timespec="seconds"),
                    "service": item["service"],
                    "level": item["level"],
                    "message": item["message"],
                },
                ensure_ascii=False,
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _parse_timestamp(value: Any, now: datetime) -> datetime | None:
    """接受 ISO 时间戳，缺少时区时按本机时区解释。"""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=now.tzinfo)
    return timestamp.astimezone(now.tzinfo)
