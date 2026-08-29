"""单 Agent 的工具调用循环和离线兜底逻辑。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from models import AgentAnswer, SearchHit, ToolTrace
from session_store import JsonSessionStore, MAX_SESSION_MESSAGES
from tools import MiniOpsTools


SYSTEM_PROMPT = """你是轻量级故障排查助手。你只能使用提供的只读工具。
先判断信息是否足以定位服务和现象；关键信息缺失时应简短追问。
只要用户给出了明确故障现象、错误码或异常行为，即使没有服务名，也应先调用 search_runbooks 给出基于手册的通用排查；只有“服务有问题”这类完全没有现象的问题才追问。
需要事实时调用 search_runbooks，用户明确询问近期日志或问题依赖运行现象时调用 query_logs。
最终用中文给出：判断、排查步骤、依据。没有证据时明确说证据不足，不要编造。
引用手册时使用 [文档名 / 章节]，不要声称执行了重启、删除或修改操作。"""


class MiniOpsAgent:
    """保存少量对话记忆，并运行有最大轮数限制的 Function Calling 循环。"""

    def __init__(
        self,
        tools: MiniOpsTools,
        session_store: JsonSessionStore | None = None,
    ) -> None:
        self.tools = tools
        self.session_store = session_store
        self.sessions: dict[str, list[dict[str, Any]]] = {}

    def chat(
        self,
        session_id: str,
        message: str,
        *,
        require_model: bool = False,
    ) -> AgentAnswer:
        """接收会话和本轮问题，返回回答、引用和可观察的工具轨迹。"""

        if session_id not in self.sessions:
            self.sessions[session_id] = (
                self.session_store.load(session_id) if self.session_store else []
            )
        history = self.sessions[session_id]
        history.append({"role": "user", "content": message})
        context = " ".join(str(item.get("content", "")) for item in history[-5:])
        if _needs_clarification(message.strip(), context):
            # 完全没有服务或异常线索时由主程序稳定追问，避免模型把同一句话
            # 随机判成“证据不足”或盲目检索；明确故障仍由 Agent 自主选择工具。
            answer = AgentAnswer(
                answer="我还缺少具体服务或异常现象。请补充：哪个服务、出现了什么报错，以及大致从什么时候开始。",
                action="clarify",
            )
        elif require_model:
            if not _model_configured():
                raise RuntimeError("真实 Agent 测评需要完整的 LLM 模型配置")
            # 测评模式禁止降级，否则规则兜底会被误算成 Agent 成功。
            answer = self._model_chat(history)
        else:
            try:
                answer = (
                    self._model_chat(history)
                    if _model_configured()
                    else self._offline_chat(history)
                )
            except Exception as exc:
                self.tools.log_store.record(
                    "agent",
                    "ERROR",
                    f"model fallback: {_safe_error(exc)}",
                    event="agent.model_fallback",
                )
                answer = self._offline_chat(history)
                answer.traces.insert(
                    0, ToolTrace("model_fallback", {}, _safe_error(exc))
                )
        history.append({"role": "assistant", "content": answer.answer})
        # 只保留最近六轮，演示短期记忆，同时避免上下文无限增长。
        self.sessions[session_id] = history[-MAX_SESSION_MESSAGES:]
        if self.session_store:
            self.session_store.save(session_id, self.sessions[session_id])
        self.tools.log_store.record(
            "agent",
            "INFO",
            f"chat completed action={answer.action} traces={len(answer.traces)} citations={len(answer.citations)}",
            event="agent.chat_completed",
        )
        return answer

    def _model_chat(self, history: list[dict[str, Any]]) -> AgentAnswer:
        from openai import OpenAI

        current_question = str(history[-1].get("content", ""))
        recent_context = " ".join(str(item.get("content", "")) for item in history[-5:])
        client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "").strip(),
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=1,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-10:],
        ]
        traces: list[ToolTrace] = []
        citations: dict[str, SearchHit] = {}
        has_log_evidence = False
        final_text = ""
        for _turn in range(4):
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL_ID", "").strip(),
                temperature=0,
                messages=messages,
                tools=self.tools.specs,
                tool_choice="auto",
            )
            model_message = response.choices[0].message
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": model_message.content or "",
            }
            calls = model_message.tool_calls or []
            if not calls:
                final_text = model_message.content or "当前没有足够证据完成判断。"
                break
            assistant["tool_calls"] = [call.model_dump() for call in calls]
            messages.append(assistant)
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if call.function.name == "query_logs":
                        # 模型可以写“支付接口”，主程序在执行前统一成 payment-api。
                        arguments = self.tools.normalize_log_arguments(arguments)
                    result = self.tools.execute(call.function.name, arguments)
                except Exception as exc:
                    arguments, result = {}, {"error": _safe_error(exc)}
                traces.append(
                    ToolTrace(
                        call.function.name,
                        arguments,
                        _tool_summary(call.function.name, result),
                    )
                )
                raw_hits = result.get("hits", [])
                if raw_hits:
                    candidate_hits = [SearchHit(**item) for item in raw_hits]
                    query = str(arguments.get("query", ""))
                    # 模型负责决定搜什么，主程序只校验证据是否足够相关。
                    if (
                        candidate_hits[0].score >= 0.15
                        and self.tools.index.has_known_identifier(query)
                        and self.tools.index.has_known_identifier(current_question)
                    ):
                        for hit in candidate_hits:
                            citations[hit.chunk_id] = hit
                raw_logs = result.get("logs", [])
                if raw_logs and self.tools.valid_log_evidence(
                    arguments, raw_logs, recent_context
                ):
                    # 且每条记录都要真正满足模型提交的服务和关键词过滤。
                    has_log_evidence = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False)[:12000],
                    }
                )
        final_text, action = _finalize_model_result(
            final_text,
            has_traces=bool(traces),
            has_evidence=bool(citations) or has_log_evidence,
        )
        return AgentAnswer(
            answer=final_text or "已达到工具调用上限，请缩小问题范围后重试。",
            citations=list(citations.values()),
            traces=traces,
            action=action,
        )

    def _offline_chat(self, history: list[dict[str, Any]]) -> AgentAnswer:
        """离线模式复用历史中的服务名，检索后按证据生成简洁的结构化回答。"""

        current = str(history[-1]["content"]).strip()
        context = " ".join(str(item.get("content", "")) for item in history[-5:])
        if _needs_clarification(current, context):
            return AgentAnswer(
                answer="我还缺少具体服务或异常现象。请补充：哪个服务、出现了什么报错，以及大致从什么时候开始。",
                action="clarify",
            )

        logs: list[dict[str, str]] = []
        service = _detect_service(context)
        if _asks_for_logs(current):
            result = self.tools.execute(
                "query_logs",
                {
                    "service": service,
                    "keyword": _detect_log_keyword(context),
                    "minutes": 1440,
                },
            )
            logs = result["logs"]
        # 日志中的真实错误信息比“最近发生了什么”更适合检索，因此先查日志再组织检索词。
        retrieval_query = context + " " + " ".join(item["message"] for item in logs)
        hits = self.tools.index.search(retrieval_query, 3)
        traces = [
            ToolTrace(
                "search_runbooks",
                {"query": retrieval_query, "limit": 3},
                f"召回 {len(hits)} 个片段",
            )
        ]
        if _asks_for_logs(current):
            traces.insert(
                0,
                ToolTrace(
                    "query_logs", {"service": service}, f"命中 {len(logs)} 条日志"
                ),
            )

        # 分数阈值负责过滤弱相关中文问题；英文技术标识完全不在语料中时直接拒答，
        # 避免被“失败、处理”等泛化词误导到无关手册。
        if (
            not hits
            or hits[0].score < 0.15
            or not self.tools.index.has_known_identifier(current)
        ):
            return AgentAnswer(
                answer="当前知识库没有足够接近的处理手册，我不能据此给出确定结论。请提供更具体的错误码，或交给值班人员继续排查。",
                traces=traces,
                action="insufficient",
            )
        answer = _compose_offline_answer(hits, logs)
        return AgentAnswer(answer=answer, citations=hits, traces=traces)


def _compose_offline_answer(hits: list[SearchHit], logs: list[dict[str, str]]) -> str:
    main = hits[0]
    lines = [
        f"初步判断：最相关的是“{main.title}”，但仍应按下面步骤验证，不要直接把它当成最终根因。",
        "",
        "建议排查：",
    ]
    steps = [line for line in main.text.splitlines() if line.strip()][:4]
    lines.extend(f"{index}. {step}" for index, step in enumerate(steps, 1))
    if logs:
        lines.extend(["", "日志证据："])
        lines.extend(f"- {item['service']}: {item['message']}" for item in logs[:5])
    lines.extend(["", "依据："])
    lines.extend(f"- [{hit.document} / {hit.title}]" for hit in hits)
    return "\n".join(lines)


def _model_configured() -> bool:
    return all(
        os.getenv(name, "").strip()
        for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL_ID")
    )


def _needs_clarification(current: str, context: str) -> bool:
    vague = len(current) < 14 or current in {"报错了", "服务挂了", "怎么回事", "有问题"}
    has_signal = bool(
        re.search(
            r"redis|mysql|502|504|cpu|内存|磁盘|证书|dns|超时|积压|启动|连接池",
            context,
            re.I,
        )
    )
    return vague and not has_signal


def _asks_for_logs(text: str) -> bool:
    return bool(re.search(r"日志|最近|刚才|报错记录|发生了什么", text, re.I))


def _detect_service(text: str) -> str:

    from service_catalog import resolve_service

    return resolve_service(text)


def _detect_log_keyword(text: str) -> str:
    for word in ("timeout", "connection", "502", "oom", "certificate", "disk"):
        if word in text.lower():
            return word
    return ""


def _looks_like_question(text: str) -> bool:
    return bool(re.search(r"请补充|能否提供|哪个服务|什么时候", text))


def _finalize_model_result(
    final_text: str,
    *,
    has_traces: bool,
    has_evidence: bool,
) -> tuple[str, str]:
    """根据工具轨迹和有效证据统一决定回答、追问或拒答。"""

    if not has_traces:
        if _looks_like_question(final_text):
            return final_text, "clarify"

        # 或明确说明缺少可核验的手册和日志证据。
        return (
            "我还没有取得可核验的手册或日志证据，暂时不能给出确定结论。"
            "请补充具体服务、错误码或时间范围。",
            "insufficient",
        )
    if has_evidence:
        return final_text, "answer"
    return (
        "当前工具没有返回足够相关的手册或日志证据，我不能据此给出确定结论。"
        "请补充更具体的服务、错误码或时间范围。",
        "insufficient",
    )


def _tool_summary(name: str, result: dict[str, Any]) -> str:
    if "error" in result:
        return str(result["error"])
    if name == "search_runbooks":
        return f"召回 {len(result.get('hits', []))} 个手册片段"
    return f"命中 {len(result.get('logs', []))} 条日志"


def _safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"

    for name in ("LLM_API_KEY", "EMBEDDING_API_KEY"):
        secret = os.getenv(name, "").strip()
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[-500:]
