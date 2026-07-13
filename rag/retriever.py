"""FAISS 기반 조항 검색기 (FR-02).

search("원금 보장 확정 수익률", k=5) -> [{score, law, article, text, ...}, ...]
"""

from __future__ import annotations

import json

import faiss
import numpy as np

from rag.config import (
    EMBEDDING_MODEL,
    INDEX_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    META_PATH,
)
from rag.ingest import embed_texts

_index: faiss.Index | None = None
_meta: dict | None = None


def _load() -> tuple[faiss.Index, dict]:
    """인덱스와 메타데이터를 최초 1회만 로드해 재사용."""
    global _index, _meta
    if _index is None or _meta is None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"인덱스가 없습니다: {INDEX_PATH}\n"
                "먼저 실행하세요: uv run python -m rag.ingest"
            )
        _index = faiss.read_index(str(INDEX_PATH))
        _meta = json.loads(META_PATH.read_text())
    return _index, _meta


def search(
    query: str, k: int = 5, principle_filter: str | None = None
) -> list[dict]:
    """질의와 코사인 유사도가 높은 조항 청크 Top-k.

    principle_filter: "광고 규제" 등 6대 원칙명. 지정 시 해당 태그 청크만 대상.
    반환 dict의 low_confidence: 최고 유사도가 임계값 미만이면 True — 근거 부족 신호로
    W08 Guardrail(판정 보류)에서 사용한다.
    """
    index, meta = _load()
    chunks = meta["chunks"]
    # 색인 구축에 쓴 모델과 다른 모델로 질의를 임베딩하면 유사도가 무의미해진다.
    model = meta.get("embedding_model", EMBEDDING_MODEL)

    vector = np.array(embed_texts([query], model=model), dtype="float32")
    faiss.normalize_L2(vector)

    # 태그 필터가 있으면 전체를 훑은 뒤 걸러낸다(청크 수가 수백 규모라 비용이 무시할 만함).
    depth = index.ntotal if principle_filter else min(k, index.ntotal)
    scores, ids = index.search(vector, depth)

    results: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if principle_filter and chunk["principle"] != principle_filter:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) == k:
            break

    top_score = results[0]["score"] if results else 0.0
    low_confidence = top_score < LOW_CONFIDENCE_THRESHOLD
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
        item["low_confidence"] = low_confidence
    return results
