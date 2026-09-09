"""加载可审计的运维知识图谱，并为手册检索提供关系扩展。"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEMANTIC_RELATIONS = {
    "DEPENDS_ON",
    "INDICATES",
    "AFFECTS",
    "CHECK_WITH",
}
DOCUMENT_RELATION = "DOCUMENTED_BY"
REVERSE_RELATIONS = {
    "DEPENDS_ON": "IS_DEPENDENCY_OF",
    "INDICATES": "IS_INDICATED_BY",
    "AFFECTS": "IS_AFFECTED_BY",
    "CHECK_WITH": "IS_CHECK_FOR",
}
ENTITY_TYPE_LABELS = (
    ("service", "服务"),
    ("component", "组件"),
    ("symptom", "故障现象"),
    ("cause", "可能原因"),
)
MAX_EXPLICIT_ENTITIES = 5


@dataclass(frozen=True, slots=True)
class GraphExpansion:
    """一次问题匹配得到的实体、文档加权和可解释路径。"""

    matched_entities: tuple[dict[str, str], ...] = ()
    document_boosts: tuple[tuple[str, float], ...] = ()
    paths: tuple[dict[str, Any], ...] = ()
    selection_source: str = ""

    def boost_for(self, document: str) -> float:
        return dict(self.document_boosts).get(document, 0.0)

    def as_dict(self, documents: set[str] | None = None) -> dict[str, Any]:
        paths = (
            self.paths
            if documents is None
            else tuple(item for item in self.paths if item["document"] in documents)
        )
        return {
            "matched_entities": [dict(item) for item in self.matched_entities],
            "graph_paths": [dict(item) for item in paths],
            "entity_selection": self.selection_source,
        }


class KnowledgeGraph:
    """使用一个小型 JSON 图谱完成实体匹配和一至两跳关系扩展。"""

    def __init__(self, path: Path) -> None:
        source = json.loads(path.read_text(encoding="utf-8"))
        nodes = source.get("nodes")
        edges = source.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("知识图谱必须包含 nodes 和 edges 列表")

        self.nodes: dict[str, dict[str, Any]] = {}
        for raw in nodes:
            if not isinstance(raw, dict):
                raise ValueError("知识图谱节点必须是对象")
            node_id = str(raw.get("id", "")).strip()
            name = str(raw.get("name", "")).strip()
            node_type = str(raw.get("type", "")).strip()
            if not node_id or not name or not node_type:
                raise ValueError("知识图谱节点缺少 id、name 或 type")
            if node_id in self.nodes:
                raise ValueError(f"知识图谱节点 ID 重复：{node_id}")
            aliases = tuple(
                item.strip()
                for item in map(str, raw.get("aliases", []))
                if item.strip()
            )
            self.nodes[node_id] = {
                "id": node_id,
                "name": name,
                "type": node_type,
                "aliases": aliases,
                "document": str(raw.get("document", "")).strip(),
            }

        self.semantic_neighbors: dict[str, list[tuple[str, str]]] = {
            node_id: [] for node_id in self.nodes
        }
        self.documents: dict[str, list[tuple[str, str]]] = {
            node_id: [] for node_id in self.nodes
        }
        for raw in edges:
            if not isinstance(raw, dict):
                raise ValueError("知识图谱边必须是对象")
            source_id = str(raw.get("source", "")).strip()
            target_id = str(raw.get("target", "")).strip()
            relation = str(raw.get("relation", "")).strip()
            if source_id not in self.nodes or target_id not in self.nodes:
                raise ValueError(f"知识图谱边引用了不存在的节点：{source_id} -> {target_id}")
            if relation == DOCUMENT_RELATION:
                document = self.nodes[target_id].get("document", "")
                if not document:
                    raise ValueError("DOCUMENTED_BY 的目标节点必须提供 document")
                self.documents[source_id].append((target_id, relation))
            elif relation in SEMANTIC_RELATIONS:
                # 故障排查时需要从任一已知实体反查上下游，因此语义边按双向遍历。
                self.semantic_neighbors[source_id].append((target_id, relation))
                self.semantic_neighbors[target_id].append(
                    (source_id, REVERSE_RELATIONS[relation])
                )
            else:
                raise ValueError(f"不支持的知识图谱关系：{relation}")

    def catalog_for_prompt(self) -> str:
        """按类型生成精简入口目录，让模型只选择图谱中真实存在的节点。"""

        groups: list[str] = []
        for node_type, label in ENTITY_TYPE_LABELS:
            entries: list[str] = []
            for node in self.nodes.values():
                if node["type"] != node_type:
                    continue
                aliases = "、".join(node["aliases"])
                alias_text = f"（别名：{aliases}）" if aliases else ""
                entries.append(f'{node["id"]}={node["name"]}{alias_text}')
            if entries:
                groups.append(f"{label}：" + "；".join(entries))
        return " | ".join(groups)

    def validate_entity_ids(self, entity_ids: list[str] | None) -> list[str]:
        """校验模型选择的入口节点，拒绝不存在的节点和文档终点。"""

        if not entity_ids:
            return []
        if len(entity_ids) > MAX_EXPLICIT_ENTITIES:
            raise ValueError(f"图谱入口最多允许 {MAX_EXPLICIT_ENTITIES} 个节点")

        validated: list[str] = []
        for raw_id in entity_ids:
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError("图谱入口节点 ID 必须是非空字符串")
            node_id = raw_id.strip()
            node = self.nodes.get(node_id)
            if node is None:
                raise ValueError(f"不存在对应的图谱节点：{node_id}")
            if node["type"] == "document":
                raise ValueError(f"文档节点不能作为图谱入口：{node_id}")
            if node_id not in validated:
                validated.append(node_id)
        return validated

    def selection_is_grounded(self, query: str, entity_ids: list[str] | None) -> bool:
        """判断模型选择的入口是否至少有一个得到原问题或对话中的实体信号支持。"""

        selected_ids = set(self.validate_entity_ids(entity_ids))
        if not selected_ids:
            return False
        return bool(selected_ids & set(self._match_entities(query)))

    def expand(
        self,
        query: str,
        max_hops: int = 2,
        *,
        entity_ids: list[str] | None = None,
    ) -> GraphExpansion:
        """使用模型选择或别名匹配的入口，扩展最多两跳并返回关系路径。"""

        selected_ids = self.validate_entity_ids(entity_ids)
        matched_ids = selected_ids or self._match_entities(query)
        if not matched_ids:
            return GraphExpansion()
        selection_source = "model" if selected_ids else "alias"

        best_documents: dict[str, tuple[float, dict[str, Any]]] = {}
        for start_id in matched_ids:
            queue: deque[tuple[str, tuple[tuple[str, str, str], ...]]] = deque(
                [(start_id, ())]
            )
            best_depth = {start_id: 0}
            while queue:
                node_id, semantic_path = queue.popleft()
                depth = len(semantic_path)
                for document_id, relation in self.documents[node_id]:
                    document = str(self.nodes[document_id]["document"])
                    boost = (0.24, 0.20, 0.12)[min(depth, 2)]
                    path = self._path_payload(
                        start_id,
                        (*semantic_path, (node_id, relation, document_id)),
                        document,
                    )
                    current = best_documents.get(document)
                    if current is None or boost > current[0]:
                        best_documents[document] = (boost, path)

                if depth >= max_hops:
                    continue
                for neighbor_id, relation in self.semantic_neighbors[node_id]:
                    next_depth = depth + 1
                    if best_depth.get(neighbor_id, max_hops + 1) <= next_depth:
                        continue
                    best_depth[neighbor_id] = next_depth
                    queue.append(
                        (
                            neighbor_id,
                            (*semantic_path, (node_id, relation, neighbor_id)),
                        )
                    )

        ordered = sorted(
            best_documents.items(), key=lambda item: (-item[1][0], item[0])
        )
        return GraphExpansion(
            matched_entities=tuple(
                {
                    "id": node_id,
                    "name": str(self.nodes[node_id]["name"]),
                    "type": str(self.nodes[node_id]["type"]),
                }
                for node_id in matched_ids
            ),
            document_boosts=tuple((document, value[0]) for document, value in ordered),
            paths=tuple(value[1] for _, value in ordered),
            selection_source=selection_source,
        )

    def _match_entities(self, query: str) -> list[str]:
        normalized = _normalize(query)
        matched: list[tuple[int, str]] = []
        for node_id, node in self.nodes.items():
            if node["type"] == "document":
                continue
            candidates = (str(node["name"]), *node["aliases"])
            lengths = [
                len(_normalize(candidate))
                for candidate in candidates
                if _normalize(candidate) and _normalize(candidate) in normalized
            ]
            if lengths:
                matched.append((max(lengths), node_id))
        # 更具体的长别名优先；同时保留多个独立实体供多线索联合扩展。
        return [node_id for _, node_id in sorted(matched, reverse=True)]

    def _path_payload(
        self,
        start_id: str,
        steps: tuple[tuple[str, str, str], ...],
        document: str,
    ) -> dict[str, Any]:
        labels = [str(self.nodes[start_id]["name"])]
        relations: list[str] = []
        for _, relation, target_id in steps:
            relations.append(relation)
            labels.append(str(self.nodes[target_id]["name"]))
        text = labels[0]
        for relation, label in zip(relations, labels[1:], strict=True):
            text += f" -[{relation}]-> {label}"
        return {
            "nodes": labels,
            "relations": relations,
            "document": document,
            "text": text,
        }


def _normalize(text: str) -> str:
    return re.sub(r"[\s_./\\-]+", "", str(text).lower())
