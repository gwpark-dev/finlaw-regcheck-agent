"""W09 PoC — 로컬 임베딩(bge-m3) 교체 실측 (ADR-003 절차 재사용).

Tier 2(완전 on-premise)의 임베딩 대가를 재는 일회성 실험이다. 기존 인덱스/캐시
(data/index/, data/cache/)는 건드리지 않고 data/index_poc/에 별도로 재인덱싱한다.

  uv run --extra poc python scripts/w09_poc_embed.py

변수 통제: 청킹(chunk_document)·임베딩 입력 텍스트(embed_input)·recall 판정 로직은
production과 동일하게 두고 **임베더만** text-embedding-3-large → BAAI/bge-m3로 바꾼다.
3-large 기준선은 recall@5 = 1.00 (W06, 10건).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import EVAL_DIR, LAWS_DIR  # noqa: E402
from rag.ingest import chunk_document, embed_input  # noqa: E402

POC_MODEL = "BAAI/bge-m3"
POC_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index_poc"
QUERY_FILE = EVAL_DIR / "w06_queries.txt"
K = 5


def build_chunks() -> list[dict]:
    files = sorted(
        p for p in LAWS_DIR.glob("*")
        if p.suffix.lower() in {".pdf", ".txt"} and not p.name.startswith(".")
    )
    chunks: list[dict] = []
    for path in files:
        doc_chunks, _ = chunk_document(path)
        chunks.extend(doc_chunks)
    return chunks


def load_queries() -> list[tuple[str, list[str]]]:
    out = []
    for line in QUERY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        query, _, expected = line.partition("|||")
        wanted = [e.strip() for e in expected.split(",") if e.strip()]
        out.append((query.strip(), wanted))
    return out


def top_articles(scores, ids, chunks, k):
    """search_demo와 동일 — 청크 결과를 조(條) 단위로 접어 Top-k."""
    seen, out = set(), []
    for score, idx in zip(scores, ids):
        if idx < 0:
            continue
        c = chunks[idx]
        key = (c["source_file"], c["article"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) == k:
            break
    return out


def main() -> int:
    from sentence_transformers import SentenceTransformer

    print(f"[1] 청킹 (production과 동일: chunk_document)")
    chunks = build_chunks()
    print(f"    총 청크 {len(chunks)}개")

    print(f"[2] 임베딩 모델 로드: {POC_MODEL} (CPU)")
    t0 = time.perf_counter()
    model = SentenceTransformer(POC_MODEL, device="cpu")
    print(f"    로드 {time.perf_counter() - t0:.1f}초")

    print(f"[3] 코퍼스 임베딩 (embed_input 텍스트, production과 동일)")
    t0 = time.perf_counter()
    passages = [embed_input(c) for c in chunks]
    corpus = model.encode(
        passages, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    ).astype("float32")
    print(f"    {len(chunks)}청크 임베딩 {time.perf_counter() - t0:.1f}초, dim={corpus.shape[1]}")

    # production과 동일한 코사인=정규화 내적 (IndexFlatIP)
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)
    POC_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(POC_INDEX_DIR / "faiss_poc.index"))
    (POC_INDEX_DIR / "meta_poc.json").write_text(
        json.dumps(
            {"embedding_model": POC_MODEL, "embedding_dim": int(corpus.shape[1]),
             "n_chunks": len(chunks)},
            ensure_ascii=False, indent=2,
        )
    )

    print(f"\n[4] recall@{K} 측정 — W06 질의 {'':>0}")
    queries = load_queries()
    q_vecs = model.encode(
        [q for q, _ in queries], normalize_embeddings=True
    ).astype("float32")

    hits = 0
    for (query, expected), qv in zip(queries, q_vecs):
        scores, ids = index.search(qv.reshape(1, -1), K * 8)
        arts = top_articles(scores[0], ids[0], chunks, K)
        found = {f"{Path(c['source_file']).stem}:{c['article']}" for c in arts}
        ok = bool(found & set(expected))
        hits += ok
        mark = "✓ HIT" if ok else "✗ MISS"
        print(f"  {mark}  {query[:42]:<42}  기대 {expected[0] if expected else '-'}")
        if not ok:
            top = ", ".join(sorted(found))[:90]
            print(f"        Top-{K}: {top}")

    recall = hits / len(queries)
    print("\n" + "=" * 70)
    print(f"bge-m3 recall@{K} = {hits}/{len(queries)} = {recall:.2f}")
    print(f"3-large recall@{K} = 1.00 (기준선, W06)")
    print("=" * 70)

    (POC_INDEX_DIR / "recall_poc.json").write_text(
        json.dumps(
            {"model": POC_MODEL, "recall_at_5": round(recall, 2),
             "hits": hits, "n_queries": len(queries),
             "baseline_3large_recall_at_5": 1.00},
            ensure_ascii=False, indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
