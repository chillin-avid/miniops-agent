"""运行小型真实 Agent 测评，输出尽量少而明确的指标。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from agent import MiniOpsAgent
from config import load_config
from retrieval import RunbookIndex
from runtime_logs import JsonlLogStore, write_relative_fixture
from tools import MiniOpsTools


ROOT = Path(__file__).resolve().parent


def run_benchmark(max_cases: int | None = None) -> dict[str, object]:
    # 命令行直接运行测评时，也使用与页面相同的本地配置加载逻辑。
    load_config()
    cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))
    if max_cases is not None:
        cases = cases[:max_cases]
    index = RunbookIndex(ROOT / "runbooks", ROOT / "data" / "benchmark-index")
    index.rebuild()
    details = []
    citation_checks = 0
    citation_passes = 0
    top1_passes = 0
    task_passes = 0
    latencies = []

    try:
        with TemporaryDirectory(prefix="miniops-benchmark-logs-") as temporary:
            temporary_root = Path(temporary)
            fixture = write_relative_fixture(
                ROOT / "eval" / "logs.json",
                temporary_root / "fixtures" / "logs.jsonl",
            )
            log_store = JsonlLogStore(
                temporary_root / "runtime" / "miniops.jsonl",
                [fixture.parent],
            )
            agent = MiniOpsAgent(MiniOpsTools(index, log_store))
            for case in cases:
                started = time.perf_counter()
                expected = case["expected_document"]

                session_id = uuid.uuid4().hex
                result = None
                case_citations = []
                for message in case["messages"]:
                    result = agent.chat(session_id, message, require_model=True)
                    # 多轮任务按整个会话累计证据，不能因最后一轮只查日志就丢掉
                    # 前一轮已经引用的正确手册。
                    known_chunks = {item.chunk_id for item in case_citations}
                    case_citations.extend(
                        item
                        for item in result.citations
                        if item.chunk_id not in known_chunks
                    )
                assert result is not None
                # 检索指标使用 Agent 实际返回的引用顺序，日志和多轮问题也按真实检索词统计。
                rank = next(
                    (
                        i
                        for i, hit in enumerate(case_citations, 1)
                        if hit.document == expected
                    ),
                    None,
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                latencies.append(latency_ms)
                tools_called = {trace.tool for trace in result.traces}
                expected_tool = case["expected_tool"]
                citation_ok = True
                if expected:
                    citation_checks += 1
                    citation_ok = any(
                        item.document == expected and item.text
                        for item in case_citations
                    )
                    citation_passes += citation_ok
                    # 正确手册排第一才计入 Top-1，用来观察模型精排是否真正改善顺序。
                    top1_passes += rank == 1
                task_ok = result.action == case["expected_action"] and (
                    expected_tool is None or expected_tool in tools_called
                )
                task_passes += task_ok
                details.append(
                    {
                        "id": case["id"],
                        "rank": rank,
                        "action": result.action,
                        "tools": sorted(tools_called),
                        "citation_ok": citation_ok,
                        "task_ok": task_ok,
                        "latency_ms": latency_ms,
                    }
                )
    finally:
        index.close()

    report = {
        "case_count": len(cases),
        "evidence_hit_rate": round(citation_passes / citation_checks, 4),
        "top1_hit_rate": round(top1_passes / citation_checks, 4),
        "task_completion_rate": round(task_passes / len(cases), 4),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "details": details,
        "note": "真实模型小型自建评测，只用于验证当前工具选择与证据回答，不代表通用准确率。",
    }
    output = ROOT / "data" / "benchmark-report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=None)
    arguments = parser.parse_args()
    print(json.dumps(run_benchmark(arguments.max_cases), ensure_ascii=False, indent=2))
