"""比较原混合检索与知识图谱增强检索在多跳问题上的证据覆盖。"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval import RunbookIndex


ROOT = Path(__file__).resolve().parent


def _evaluate(
    index: RunbookIndex,
    cases: list[dict[str, object]],
    *,
    include_paths: bool,
    latency_repeats: int = 20,
) -> dict[str, object]:
    covered_documents = 0
    expected_documents = 0
    fully_covered = 0
    path_covered = 0
    latencies: list[float] = []
    details: list[dict[str, object]] = []

    for case in cases:
        query = str(case["query"])
        expected = list(map(str, case["expected_documents"]))
        hits, graph = index.search_with_context(query, limit=3)
        # 毫秒级操作单次测量波动很大，预热后重复计时再取全部用例的中位数。
        for _ in range(latency_repeats):
            started = time.perf_counter()
            index.search_with_context(query, limit=3)
            latencies.append((time.perf_counter() - started) * 1000)
        returned = [hit.document for hit in hits]
        covered = [document for document in expected if document in returned]
        graph_documents = {str(item["document"]) for item in graph.paths}

        covered_documents += len(covered)
        expected_documents += len(expected)
        fully_covered += len(covered) == len(expected)
        if include_paths:
            path_covered += all(document in graph_documents for document in expected)
        details.append(
            {
                "id": case["id"],
                "returned_documents": returned,
                "covered_documents": covered,
                "matched_entities": [item["name"] for item in graph.matched_entities],
            }
        )

    return {
        "document_recall_at_3": round(covered_documents / expected_documents, 4),
        "full_case_coverage_at_3": round(fully_covered / len(cases), 4),
        "relation_path_coverage": (
            round(path_covered / len(cases), 4) if include_paths else 0.0
        ),
        "median_retrieval_latency_ms": round(statistics.median(latencies), 2),
        "details": details,
    }


def run_graph_benchmark() -> dict[str, object]:
    """在相同本地检索配置下做关闭/开启图谱的消融对照。"""

    cases = json.loads(
        (ROOT / "eval" / "graph_cases.json").read_text(encoding="utf-8")
    )
    with TemporaryDirectory(prefix="miniops-graph-benchmark-") as temporary:
        storage = Path(temporary)
        baseline = RunbookIndex(
            ROOT / "runbooks", storage / "baseline", use_graph=False
        )
        enhanced = RunbookIndex(
            ROOT / "runbooks", storage / "enhanced", use_graph=True
        )
        try:
            baseline.rebuild()
            enhanced.rebuild()
            baseline_result = _evaluate(baseline, cases, include_paths=False)
            enhanced_result = _evaluate(enhanced, cases, include_paths=True)
        finally:
            baseline.close()
            enhanced.close()

    report = {
        "case_count": len(cases),
        "baseline": baseline_result,
        "graph_enhanced": enhanced_result,
        "note": (
            "小型自建多跳检索消融实验；固定使用相同的本地向量与关键词配置，"
            "只比较服务依赖图是否改善证据覆盖，不代表通用问答准确率。"
        ),
    }
    output = ROOT / "data" / "graph-benchmark-report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_graph_benchmark(), ensure_ascii=False, indent=2))
