"""只测试最重要的边界，避免小项目被测试代码反客为主。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import MiniOpsAgent, _finalize_model_result  # noqa: E402
from knowledge_graph import KnowledgeGraph  # noqa: E402
from retrieval import RunbookIndex  # noqa: E402
from runtime_logs import JsonlLogStore  # noqa: E402
from session_store import JsonSessionStore  # noqa: E402
from service_catalog import resolve_service  # noqa: E402
from tools import MiniOpsTools  # noqa: E402


@pytest.fixture(autouse=True)
def disable_external_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """单元测试固定走本地逻辑，避免消耗额度或依赖网络。"""

    for name in (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL_ID",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL_ID",
        "EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_retrieval_finds_expected_runbook(tmp_path: Path) -> None:
    index = RunbookIndex(ROOT / "runbooks", tmp_path)
    index.rebuild()
    hits = index.search("RedisTimeoutError 连接池读取超时", 3)
    assert hits[0].document == "redis-timeout.md"
    index.close()


def test_graph_expands_service_dependencies_with_explainable_paths() -> None:
    graph = KnowledgeGraph(ROOT / "knowledge_graph.json")
    expansion = graph.expand("order-worker 处理速度下降，检查它的下游依赖")

    documents = {item["document"] for item in expansion.paths}
    assert "queue-backlog.md" in documents
    assert "mysql-pool.md" in documents
    assert any(
        item["relations"][:1] == ["DEPENDS_ON"]
        and item["document"] == "mysql-pool.md"
        for item in expansion.paths
    )


def test_graph_catalog_exposes_entry_nodes_but_not_document_nodes() -> None:
    graph = KnowledgeGraph(ROOT / "knowledge_graph.json")
    catalog = graph.catalog_for_prompt()
    assert "服务：" in catalog
    assert "组件：" in catalog
    assert "service.payment=payment-api" in catalog
    assert "支付接口" in catalog
    assert "doc.redis" not in catalog


def test_model_selected_graph_entry_resolves_typo_without_query_rewrite() -> None:
    graph = KnowledgeGraph(ROOT / "knowledge_graph.json")
    query = "支付aoi访问缓存一直超时"
    alias_matches = {item["id"] for item in graph.expand(query).matched_entities}
    assert "component.redis" in alias_matches
    assert "service.payment" not in alias_matches

    expansion = graph.expand(query, entity_ids=["service.payment", "component.redis"])
    assert expansion.selection_source == "model"
    assert [item["id"] for item in expansion.matched_entities] == [
        "service.payment",
        "component.redis",
    ]
    assert graph.selection_is_grounded(query, ["service.payment", "component.redis"])
    assert not graph.selection_is_grounded(
        "Kubernetes Pod一直Pending，该怎么处理？",
        ["component.host", "symptom.startup_failure"],
    )
    assert "redis-timeout.md" in {item["document"] for item in expansion.paths}


def test_graph_rejects_invalid_or_document_entry_nodes() -> None:
    graph = KnowledgeGraph(ROOT / "knowledge_graph.json")
    with pytest.raises(ValueError, match="不存在对应"):
        graph.expand("支付接口超时", entity_ids=["service.missing"])
    with pytest.raises(ValueError, match="文档节点不能"):
        graph.expand("支付接口超时", entity_ids=["doc.redis"])


def test_search_tool_keeps_query_separate_from_graph_entries(tmp_path: Path) -> None:
    index = RunbookIndex(ROOT / "runbooks", tmp_path / "index")
    index.rebuild()
    tools = MiniOpsTools(index, JsonlLogStore(tmp_path / "runtime.jsonl"))
    query = "支付aoi访问缓存一直超时"
    result = tools.execute(
        "search_runbooks",
        {
            "query": query,
            "entity_ids": ["service.payment", "component.redis"],
            "limit": 3,
        },
    )

    assert result["entity_selection"] == "model"
    assert {item["id"] for item in result["matched_entities"]} == {
        "service.payment",
        "component.redis",
    }
    assert any(item["document"] == "redis-timeout.md" for item in result["hits"])
    index.close()


def test_graph_enhancement_improves_multi_hop_document_coverage(tmp_path: Path) -> None:
    query = "gateway 调用支付服务失败，应该沿依赖继续查什么？"
    baseline = RunbookIndex(ROOT / "runbooks", tmp_path / "base", use_graph=False)
    enhanced = RunbookIndex(ROOT / "runbooks", tmp_path / "graph", use_graph=True)
    baseline.rebuild()
    enhanced.rebuild()

    baseline_documents = {item.document for item in baseline.search(query, 3)}
    enhanced_documents = {item.document for item in enhanced.search(query, 3)}
    assert "redis-timeout.md" not in baseline_documents
    assert "redis-timeout.md" in enhanced_documents
    baseline.close()
    enhanced.close()


def test_invalid_or_missing_graph_falls_back_to_original_retrieval(
    tmp_path: Path,
) -> None:
    index = RunbookIndex(
        ROOT / "runbooks",
        tmp_path / "index",
        graph_path=tmp_path / "missing.json",
    )
    index.rebuild()
    assert index.search("RedisTimeoutError", 1)[0].document == "redis-timeout.md"
    index.close()


def test_agent_clarifies_vague_question(tmp_path: Path) -> None:
    index = RunbookIndex(ROOT / "runbooks", tmp_path)
    index.rebuild()
    agent = MiniOpsAgent(MiniOpsTools(index, JsonlLogStore(tmp_path / "runtime.jsonl")))
    result = agent.chat("test", "服务有问题")
    assert result.action == "clarify"
    assert not result.citations
    index.close()


def test_log_tool_is_read_only_and_filtered(tmp_path: Path) -> None:
    index = RunbookIndex(ROOT / "runbooks", tmp_path)
    index.rebuild()
    logs = tmp_path / "external" / "service.jsonl"
    logs.parent.mkdir()
    now = datetime.now().astimezone()
    records = [
        {
            "timestamp": now.isoformat(timespec="seconds"),
            "service": "payment-api",
            "level": "ERROR",
            "message": "RedisTimeoutError while reading cache",
        },
        {
            "timestamp": (now - timedelta(minutes=90)).isoformat(timespec="seconds"),
            "service": "payment-api",
            "level": "ERROR",
            "message": "old timeout outside the requested window",
        },
    ]
    logs.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )
    store = JsonlLogStore(tmp_path / "runtime" / "miniops.jsonl", [logs.parent])
    tools = MiniOpsTools(index, store)
    result = tools.execute(
        "query_logs",
        {"service": "payment-api", "level": "ERROR", "keyword": "redis", "minutes": 60},
    )
    assert result["logs"]
    assert all(item["service"] == "payment-api" for item in result["logs"])
    assert all("Redis" in item["message"] for item in result["logs"])
    index.close()


def test_service_catalog_accepts_common_chinese_and_mixed_names() -> None:
    """常见中文说法和中英混写都能映射到真实日志服务名。"""

    assert resolve_service("支付API为什么超时") == "payment-api"
    assert resolve_service("payment 接口最近报错") == "payment-api"
    assert resolve_service("订单后台任务积压") == "order-worker"
    assert resolve_service("之前支付服务超时，现在订单任务又积压") == "order-worker"


def test_log_tool_rejects_broad_query_without_service_or_keyword(
    tmp_path: Path,
) -> None:
    """只有时间范围的宽泛查询不能拿任意日志冒充相关证据。"""

    index = RunbookIndex(ROOT / "runbooks", tmp_path / "index")
    index.rebuild()
    tools = MiniOpsTools(index, JsonlLogStore(tmp_path / "runtime.jsonl"))
    with pytest.raises(ValueError, match="服务或关键词"):
        tools.execute("query_logs", {"minutes": 60})
    index.close()


def test_log_evidence_trusts_agent_semantics_but_checks_records(tmp_path: Path) -> None:
    """允许模型理解自然别称，但服务范围和日志内容仍由程序检查。"""

    index = RunbookIndex(ROOT / "runbooks", tmp_path / "index")
    index.rebuild()
    tools = MiniOpsTools(index, JsonlLogStore(tmp_path / "runtime.jsonl"))
    logs = [
        {
            "time": datetime.now().astimezone().isoformat(),
            "service": "payment-api",
            "level": "ERROR",
            "message": "RedisTimeoutError while reading cache",
        }
    ]
    assert tools.valid_log_evidence(
        {"service": "payment-api"}, logs, "支付 API 最近为什么超时"
    )
    assert tools.valid_log_evidence({"service": "payment-api"}, logs, "收银台一直转圈")
    assert not tools.valid_log_evidence({}, logs, "最近发生了什么")
    assert not tools.valid_log_evidence(
        {"service": "unknown-service"}, logs, "未知系统超时"
    )
    index.close()


def test_model_cannot_answer_without_tools_or_evidence() -> None:
    """模型跳过工具直接下结论时，主程序会覆盖成证据不足。"""

    text, action = _finalize_model_result(
        "根据经验判断就是 Redis 连接池耗尽。",
        has_traces=False,
        has_evidence=False,
    )
    assert action == "insufficient"
    assert "没有取得可核验" in text

    question, action = _finalize_model_result(
        "请补充是哪个服务，以及大约什么时候开始。",
        has_traces=False,
        has_evidence=False,
    )
    assert action == "clarify"
    assert question.startswith("请补充")


def test_runtime_log_uses_real_iso_timestamp(tmp_path: Path) -> None:
    store = JsonlLogStore(tmp_path / "runtime" / "miniops.jsonl")
    assert store.record("miniops", "INFO", "service started")
    item = json.loads(store.runtime_log.read_text(encoding="utf-8"))
    timestamp = datetime.fromisoformat(item["timestamp"])
    assert timestamp.tzinfo is not None
    assert abs((datetime.now().astimezone() - timestamp).total_seconds()) < 5
    assert store.query(service="miniops", minutes=1)[0]["message"] == "service started"


def test_runtime_log_rotates_by_size_and_limits_backups(tmp_path: Path) -> None:
    """文件达到上限后轮转，并只保留配置数量的历史文件。"""

    runtime_log = tmp_path / "runtime" / "miniops.jsonl"
    store = JsonlLogStore(runtime_log, max_file_bytes=1_400, backup_count=2)
    for index in range(4):
        assert store.record("miniops", "INFO", f"event-{index}-" + "x" * 1_000)

    assert runtime_log.is_file()
    assert (runtime_log.parent / "miniops.1.jsonl").is_file()
    assert (runtime_log.parent / "miniops.2.jsonl").is_file()
    assert not (runtime_log.parent / "miniops.3.jsonl").exists()
    messages = [item["message"] for item in store.query(service="miniops", minutes=1)]
    assert any(message.startswith("event-3-") for message in messages)
    assert not any(message.startswith("event-0-") for message in messages)


def test_session_history_recovers_after_agent_restart(tmp_path: Path) -> None:
    """重新创建 Agent 后，仍能按会话编号恢复最近对话。"""

    index = RunbookIndex(ROOT / "runbooks", tmp_path / "index")
    index.rebuild()
    tools = MiniOpsTools(index, JsonlLogStore(tmp_path / "runtime.jsonl"))
    store = JsonSessionStore(tmp_path / "sessions")
    session_id = "a" * 32
    first_agent = MiniOpsAgent(tools, store)
    first_agent.chat(session_id, "服务有问题")

    second_agent = MiniOpsAgent(tools, store)
    second_agent.chat(session_id, "payment-api 从十分钟前开始一直超时")

    assert second_agent.sessions[session_id][0]["content"] == "服务有问题"
    assert len(store.load(session_id)) == 4
    index.close()
