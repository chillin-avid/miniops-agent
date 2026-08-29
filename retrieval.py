"""把 Markdown 手册切成片段，向量化后存入本地 Qdrant。"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import HashingVectorizer

from models import SearchHit


COLLECTION = "miniops_runbooks"
VECTOR_SIZE = 768


class RunbookIndex:
    """管理本地手册索引，并提供向量与关键词混合检索。"""

    def __init__(self, runbooks_dir: Path, storage_dir: Path) -> None:
        self.runbooks_dir = runbooks_dir.resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str((storage_dir / "qdrant").resolve()))
        self.vector_size = int(os.getenv("EMBEDDING_DIMENSIONS", str(VECTOR_SIZE)))
        if self.vector_size < 1:
            raise ValueError("EMBEDDING_DIMENSIONS 必须大于 0")
        self.embedding_model = os.getenv("EMBEDDING_MODEL_ID", "").strip()
        embedding_configured = all(
            os.getenv(name, "").strip()
            for name in (
                "EMBEDDING_BASE_URL",
                "EMBEDDING_API_KEY",
                "EMBEDDING_MODEL_ID",
            )
        )
        self.embedding_client: Any | None = None
        if embedding_configured:
            from openai import OpenAI

            # 只给检索器向量接口权限；对话模型与向量模型使用不同配置。
            self.embedding_client = OpenAI(
                base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
                api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
                timeout=float(os.getenv("EMBEDDING_TIMEOUT", "45")),
                max_retries=1,
            )
        self.rerank_model = os.getenv("RERANK_MODEL_ID", "").strip()
        self.rerank_candidate_count = max(int(os.getenv("RERANK_CANDIDATES", "8")), 1)
        rerank_configured = all(
            os.getenv(name, "").strip()
            for name in ("RERANK_BASE_URL", "RERANK_MODEL_ID", "EMBEDDING_API_KEY")
        )
        self.rerank_client: Any | None = None
        if rerank_configured:
            from openai import OpenAI

            # 重排与向量模型复用百炼密钥，但使用独立的兼容接口地址。
            self.rerank_client = OpenAI(
                base_url=os.getenv("RERANK_BASE_URL", "").strip(),
                api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
                timeout=float(os.getenv("RERANK_TIMEOUT", "45")),
                max_retries=1,
            )
        self.index_signature = (
            f"api:{self.embedding_model}:{self.vector_size}"
            if self.embedding_client
            else f"hashing-char-ngram-v1:{self.vector_size}"
        )
        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=self.vector_size,
            alternate_sign=False,
            norm="l2",
        )

    def rebuild(self) -> int:
        """重建小型索引；数据量很小时全量重建更简单也更可靠。"""

        chunks = self._load_chunks()
        # 先取得全部向量再替换旧集合，接口失败时不会提前删掉可用索引。
        vectors = self._vectors([item["search_text"] for item in chunks])
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        points = []
        for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            points.append(
                models.PointStruct(
                    id=position,
                    vector=vector.tolist(),
                    payload={**chunk, "_index_signature": self.index_signature},
                )
            )
        if points:
            self.client.upsert(collection_name=COLLECTION, points=points, wait=True)
        return len(points)

    def ensure_ready(self) -> int:
        """首次运行自动建索引；向量来源变化时自动丢弃旧索引并重建。"""

        if not self.client.collection_exists(COLLECTION):
            return self.rebuild()
        # 哈希向量与模型向量即使同为 768 维也不能混用，用来源标记判断。
        points, _ = self.client.scroll(
            collection_name=COLLECTION,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        signature = (
            str((points[0].payload or {}).get("_index_signature", "")) if points else ""
        )
        if signature != self.index_signature:
            return self.rebuild()
        return int(self.client.count(COLLECTION, exact=True).count)

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        """接收自然语言问题，融合向量相似度和关键词覆盖率后返回前几条。"""

        self.ensure_ready()
        vector_response = self.client.query_points(
            collection_name=COLLECTION,
            query=self._vectors([query])[0].tolist(),
            limit=max(limit * 3, 8),
            with_payload=True,
        )
        query_terms = _terms(query)
        # 关键词候选从全库选，避免错误码只因向量排名靠后就永远没有融合机会。
        all_points, _ = self.client.scroll(
            collection_name=COLLECTION,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
        vector_scores = {
            str(point.id): float(point.score) for point in vector_response.points
        }
        hits: list[SearchHit] = []
        for point in all_points:
            payload = dict(point.payload or {})
            text = str(payload.get("search_text", ""))
            terms = _terms(text)
            keyword_score = len(query_terms & terms) / max(len(query_terms), 1)
            vector_score = vector_scores.get(str(point.id), 0.0)
            score = round(vector_score * 0.75 + keyword_score * 0.25, 4)
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document=str(payload.get("document", "")),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    score=score,
                )
            )
        # 混合分数先负责扩大候选覆盖，再由模型联合阅读问题和原文进行精排。
        candidates = sorted(hits, key=lambda item: item.score, reverse=True)[
            : max(self.rerank_candidate_count, limit)
        ]
        return self._rerank(query, candidates, limit)

    def has_known_identifier(self, query: str) -> bool:
        """判断问题中的英文技术标识是否至少有一个真实出现在知识库。"""

        ignored = {
            "how",
            "what",
            "why",
            "the",
            "and",
            "api",
            "worker",
            "service",
            "log",
            "logs",
            "recent",
            "latest",
            "error",
            "failed",
            "failure",
        }
        identifiers = {
            item
            for item in re.findall(r"[a-z][a-z0-9_.-]{2,}", query.lower())
            if item not in ignored and "-" not in item
        }
        if not identifiers:
            return True
        points, _ = self.client.scroll(
            collection_name=COLLECTION,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
        corpus = " ".join(
            str(point.payload.get("search_text", "")).lower()
            for point in points
            if point.payload
        )
        return any(identifier in corpus for identifier in identifiers)

    def close(self) -> None:
        self.client.close()

    def _vectors(self, texts: list[str]) -> np.ndarray:
        """优先批量调用真实向量接口，未配置时执行原来的本地哈希。"""

        if self.embedding_client is None:
            return self.vectorizer.transform(texts).astype(np.float32).toarray()
        vectors: list[list[float]] = []
        # qwen3.7-text-embedding 单次最多接收 20 条文本，因此分批但不逐条请求。
        for start in range(0, len(texts), 20):
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts[start : start + 20],
                dimensions=self.vector_size,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        if len(vectors) != len(texts) or any(
            len(item) != self.vector_size for item in vectors
        ):
            raise RuntimeError("向量接口返回的数量或维度与请求不一致")
        return np.asarray(vectors, dtype=np.float32)

    def _rerank(
        self,
        query: str,
        candidates: list[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        """用专用模型精排候选；接口失败时保留混合检索的原顺序。"""

        if self.rerank_client is None or len(candidates) <= 1:
            return candidates[:limit]
        documents = [
            f"文档：{item.document}\n章节：{item.title}\n{item.text}"
            for item in candidates
        ]
        try:
            response = self.rerank_client.post(
                "/reranks",
                body={
                    "model": self.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": min(limit, len(documents)),
                },
                cast_to=object,
            )
            raw_results = (
                response.get("results", [])
                if isinstance(response, dict)
                else getattr(response, "results", [])
            )
            ranked: list[SearchHit] = []
            for result in sorted(
                raw_results,
                key=lambda item: float(item["relevance_score"]),
                reverse=True,
            ):
                index = int(result["index"])
                if not 0 <= index < len(candidates):
                    raise ValueError("重排接口返回了越界的候选编号")
                candidate = candidates[index]
                ranked.append(
                    SearchHit(
                        chunk_id=candidate.chunk_id,
                        document=candidate.document,
                        title=candidate.title,
                        text=candidate.text,
                        score=round(float(result["relevance_score"]), 4),
                    )
                )
            if len(ranked) != min(limit, len(candidates)):
                raise ValueError("重排接口返回的候选数量不完整")
            return ranked
        except Exception:
            # 精排只是提升顺序，失败不能让已经成功的混合检索一起失效。
            return candidates[:limit]

    def _load_chunks(self) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        for path in sorted(self.runbooks_dir.glob("*.md")):
            document_title, sections = _split_markdown(path.read_text(encoding="utf-8"))
            for title, text in sections:
                raw_id = f"{path.name}:{title}:{text}"
                chunks.append(
                    {
                        "chunk_id": hashlib.sha1(raw_id.encode()).hexdigest()[:16],
                        "document": path.name,
                        "title": title,
                        "text": text,
                        "search_text": f"{path.stem} {document_title} {title} {text}",
                    }
                )
        if not chunks:
            raise RuntimeError(f"没有在 {self.runbooks_dir} 找到 Markdown 手册")
        return chunks


def _split_markdown(source: str) -> tuple[str, list[tuple[str, str]]]:
    """按二级标题切分，保留章节名作为检索和引用元数据。"""

    document_title = "未命名手册"
    current_title = "概览"
    lines: list[str] = []
    sections: list[tuple[str, str]] = []
    for line in source.splitlines():
        if line.startswith("# "):
            document_title = line[2:].strip()
        elif line.startswith("## "):
            if lines:
                sections.append((current_title, "\n".join(lines).strip()))
            current_title, lines = line[3:].strip(), []
        elif line.strip():
            lines.append(line.strip().lstrip("- "))
    if lines:
        sections.append((current_title, "\n".join(lines).strip()))
    return document_title, [(title, text) for title, text in sections if text]


def _terms(text: str) -> set[str]:
    """提取英文标识、数字和中文二元片段，用于补足精确错误码匹配。"""

    lowered = text.lower()
    words = set(re.findall(r"[a-z][a-z0-9_.-]+|\d{3,}", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    words.update(chinese[index : index + 2] for index in range(len(chinese) - 1))
    return words
