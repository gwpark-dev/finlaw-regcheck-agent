"""법령 원문 → 조 단위 청킹 → 임베딩 → FAISS 인덱스 구축 (FR-01).

실행: uv run python -m rag.ingest [--dry-run]
  --dry-run: 임베딩/인덱싱 없이 청킹 결과만 확인 (API 비용 0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

from rag.config import (
    CACHE_DIR,
    CHUNK_PARAMS,
    EMBED_CACHE_PATH,
    EMBEDDING_BATCH,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_PATH,
    LAW_NAMES,
    LAWS_DIR,
    META_PATH,
    PRINCIPLE_SOURCE,
    PRINCIPLES,
)

# --- 원문 파싱용 정규식 -------------------------------------------------------

# 국가법령정보센터 PDF의 페이지 꼬리말
FOOTER_RE = re.compile(r"^법제처\s*\d+\s*국가법령정보센터\s*$")
# 부칙 시작 지점. 부칙은 조항 번호가 제1조부터 다시 시작해 본문 조항과 충돌하므로 잘라낸다.
SUPPLEMENTARY_RE = re.compile(r"^부\s?칙\s*[<(]", re.M)
# 장/절 제목 줄. 조 사이에 끼어 있어 그대로 두면 앞 조항 본문에 붙는다.
HEADING_RE = re.compile(r"^제\d+(장|절|관)\s")
# 조 시작. 제목 괄호를 필수로 요구해 본문 중 "제6조제3항" 같은 인용과 구분한다.
ARTICLE_RE = re.compile(r"^제(\d+)조(?:의(\d+))?\s*\(([^)\n]*)\)", re.M)
# 항 기호 ①~⑳
PARAGRAPH_MARKS = "".join(chr(c) for c in range(0x2460, 0x2474))
PARAGRAPH_SPLIT_RE = re.compile(rf"(?=[{PARAGRAPH_MARKS}])")
# 구조가 시작되는 줄(= 앞줄의 이어진 내용이 아닌 줄)
STRUCT_LINE_RE = re.compile(
    rf"^(제\d+조|제\d+(장|절|관)\s|[{PARAGRAPH_MARKS}]|\d+\.\s|[가나다라마바사아자차카타파하]\.\s|<|\[)"
)
# [시행 2026. 1. 2.]
EFFECTIVE_DATE_RE = re.compile(r"\[시행\s*([\d.\s]+?)\]")


def extract_pages(path: Path) -> list[str]:
    """PDF/TXT에서 페이지별 텍스트를 뽑는다. txt는 단일 페이지로 취급."""
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]
    return [path.read_text(encoding="utf-8")]


def strip_page_noise(pages: list[str]) -> str:
    """페이지마다 반복되는 머리말(법령명)과 꼬리말(법제처 N 국가법령정보센터)을 제거."""
    first_lines = [p.split("\n", 1)[0].strip() for p in pages if p.strip()]
    running_header, count = (
        Counter(first_lines).most_common(1)[0] if first_lines else ("", 0)
    )
    # 절반 이상의 페이지에 같은 첫 줄이 있으면 머리말로 간주
    is_header = count >= max(2, len(first_lines) // 2)

    out: list[str] = []
    for page in pages:
        lines = page.split("\n")
        if is_header and lines and lines[0].strip() == running_header:
            lines = lines[1:]
        out.extend(ln for ln in lines if not FOOTER_RE.match(ln.strip()))
    return "\n".join(out)


def join_wrapped_lines(text: str) -> str:
    """PDF 줄바꿈 복원.

    한글 법령 PDF는 단어 중간에서 줄이 끊기므로("금융\\n소비자"), 구조 기호로
    시작하지 않는 줄은 앞줄에 공백 없이 이어 붙여야 원문이 복원된다.
    """
    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if not line:
            continue
        if out and not STRUCT_LINE_RE.match(line):
            out[-1] += line
        else:
            out.append(line)
    return "\n".join(out)


def load_document(path: Path) -> tuple[str, str | None]:
    """원문 텍스트와 시행일을 반환."""
    pages = extract_pages(path)
    head = pages[0] if pages else ""
    m = EFFECTIVE_DATE_RE.search(head)
    effective_date = m.group(1).strip() if m else None

    text = strip_page_noise(pages)
    if CHUNK_PARAMS["drop_supplementary"]:
        cut = SUPPLEMENTARY_RE.search(text)
        if cut:
            text = text[: cut.start()]
    text = join_wrapped_lines(text)
    text = "\n".join(ln for ln in text.split("\n") if not HEADING_RE.match(ln))
    return text, effective_date


# --- 청킹 --------------------------------------------------------------------


def split_by_chars(text: str) -> list[str]:
    """조 구조가 없는 문서(가이드라인 등)용 폴백. 문자 윈도우 + 오버랩."""
    size = CHUNK_PARAMS["fallback_chunk_chars"]
    overlap = CHUNK_PARAMS["fallback_overlap_chars"]
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def split_article_body(body: str) -> list[tuple[str | None, str]]:
    """긴 조는 항(①②③) 단위로 쪼갠다. 반환: [(항 기호|None, 텍스트)]"""
    if len(body) <= CHUNK_PARAMS["max_chunk_chars"]:
        return [(None, body)]

    parts = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(body) if p.strip()]
    # 항 기호가 없는 긴 조(제1항이 생략된 단항 조문 등)는 문자 윈도우로 폴백
    if len(parts) <= 1:
        return [(None, c) for c in split_by_chars(body)]

    marks = [p[0] if p[0] in PARAGRAPH_MARKS else None for p in parts]

    # ① 앞의 머리글(= "제17조(적합성원칙)" 조 제목 줄)은 그 자체로 검색될 내용이 없다.
    # 독립 청크로 두면 빈 껍데기가 되므로 첫 항에 붙인다.
    if marks[0] is None:
        head = parts.pop(0)
        marks.pop(0)
        parts[0] = f"{head}\n{parts[0]}"

    return list(zip(marks, parts))


def chunk_document(path: Path) -> tuple[list[dict], str | None]:
    """법령 파일 하나 → 청크 리스트."""
    stem = path.stem
    law = LAW_NAMES.get(stem, stem)
    text, effective_date = load_document(path)

    matches = list(ARTICLE_RE.finditer(text))
    chunks: list[dict] = []

    if not matches:
        # 조 구조가 없는 문서 → 문자 윈도우 청킹, 조항 메타는 비움
        for i, body in enumerate(split_by_chars(text)):
            chunks.append(
                {
                    "chunk_id": f"{stem}#{i}",
                    "law": law,
                    "source_file": path.name,
                    "article": None,
                    "article_title": None,
                    "paragraph": None,
                    "principle": None,
                    "effective_date": effective_date,
                    "text": body,
                }
            )
        return chunks, effective_date

    for i, m in enumerate(matches):
        num, sub, title = int(m.group(1)), m.group(2), m.group(3).strip()
        article = f"제{num}조의{sub}" if sub else f"제{num}조"
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : end].strip()

        # 6대 원칙 태그는 금소법 본법 제17~22조 '본조'에만 (제17조의2 등 가지조항 제외)
        principle = (
            PRINCIPLES.get(num) if (stem == PRINCIPLE_SOURCE and not sub) else None
        )

        for j, (mark, part) in enumerate(split_article_body(body)):
            chunks.append(
                {
                    "chunk_id": f"{stem}#{article}#{j}",
                    "law": law,
                    "source_file": path.name,
                    "article": article,
                    "article_title": title,
                    "paragraph": mark,
                    "principle": principle,
                    "effective_date": effective_date,
                    "text": part,
                }
            )
    return chunks, effective_date


def embed_input(chunk: dict) -> str:
    """임베딩에 넣을 텍스트. 항 단위로 쪼개면 조 제목이 사라지므로 헤더를 붙여 문맥을 유지한다."""
    header = chunk["law"]
    if chunk["article"]:
        header += f" {chunk['article']}({chunk['article_title']})"
    return f"{header}\n{chunk['text']}"


# --- 임베딩 ------------------------------------------------------------------


def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """OpenAI 임베딩 + 로컬 캐시. 같은 텍스트는 재실행해도 API를 다시 호출하지 않는다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[float]] = {}
    if EMBED_CACHE_PATH.exists():
        cache = json.loads(EMBED_CACHE_PATH.read_text())

    keys = [_cache_key(t, model) for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in cache]

    if todo:
        load_dotenv()  # API 키는 .env에서만 읽는다 (NFR-06)
        client = OpenAI()
        for i in range(0, len(todo), EMBEDDING_BATCH):
            batch = todo[i : i + EMBEDDING_BATCH]
            print(f"  임베딩 {i + len(batch)}/{len(todo)} ...", flush=True)
            resp = client.embeddings.create(
                model=model, input=[t for _, t in batch]
            )
            for (key, _), item in zip(batch, resp.data):
                cache[key] = item.embedding
            # 중간에 끊겨도 이미 쓴 비용은 보존
            EMBED_CACHE_PATH.write_text(json.dumps(cache))

    return [cache[k] for k in keys]


# --- 파이프라인 --------------------------------------------------------------

GUIDE = f"""
data/laws/ 에 법령 원문 파일이 없습니다.

아래 공개 자료를 PDF(권장) 또는 TXT로 받아 data/laws/ 에 넣어주세요.
파일명(확장자 제외)이 법령명 매핑 키가 됩니다 — rag/config.py: LAW_NAMES 참고.

  금소법.pdf              금융소비자 보호에 관한 법률        (국가법령정보센터)
  금소법_시행령.pdf        금융소비자 보호에 관한 법률 시행령  (국가법령정보센터)
  금소법_감독규정.pdf      금융소비자 보호에 관한 감독규정     (국가법령정보센터)
  AI기본법.pdf            인공지능 발전과 신뢰 기반 조성 등에 관한 기본법 (보조)
  금융AI가이드라인.pdf     금융분야 AI 가이드라인 (금융위)     (보조)

넣은 뒤 다시 실행: uv run python -m rag.ingest
경로: {LAWS_DIR}
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="법령 원문 → FAISS 인덱스 구축")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="임베딩/인덱싱 없이 청킹 결과만 출력 (API 호출 없음)",
    )
    args = parser.parse_args()

    files = sorted(
        p
        for p in LAWS_DIR.glob("*")
        if p.suffix.lower() in {".pdf", ".txt"} and not p.name.startswith(".")
    )
    if not files:
        print(GUIDE)
        return 0

    chunks: list[dict] = []
    sources: list[dict] = []
    for path in files:
        doc_chunks, effective_date = chunk_document(path)
        chunks.extend(doc_chunks)
        articles = {c["article"] for c in doc_chunks if c["article"]}
        tagged = sum(1 for c in doc_chunks if c["principle"])
        sources.append(
            {
                "file": path.name,
                "law": LAW_NAMES.get(path.stem, path.stem),
                "effective_date": effective_date,
                "articles": len(articles),
                "chunks": len(doc_chunks),
            }
        )
        print(
            f"[{path.name}] 조 {len(articles)}개 → 청크 {len(doc_chunks)}개"
            + (f" (원칙 태그 {tagged}개)" if tagged else "")
        )

    if not chunks:
        print("청크가 하나도 만들어지지 않았습니다. 원문 추출을 확인하세요.")
        return 1

    if args.dry_run:
        print(f"\n[dry-run] 총 청크 {len(chunks)}개. 임베딩·인덱싱은 건너뜁니다.")
        sample = next((c for c in chunks if c["principle"]), chunks[0])
        print(f"\n샘플 청크 ({sample['chunk_id']}, 원칙={sample['principle']}):")
        print(embed_input(sample)[:400] + " ...")
        return 0

    print(f"\n총 청크 {len(chunks)}개 임베딩 ({EMBEDDING_MODEL})")
    vectors = embed_texts([embed_input(c) for c in chunks])

    # 코사인 유사도 = L2 정규화 후 내적(IndexFlatIP)
    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,  # NFR-07
        "embedding_dim": int(matrix.shape[1]),
        "chunk_params": CHUNK_PARAMS,
        "similarity": "cosine (L2-normalized inner product)",
        "sources": sources,
        "chunks": chunks,  # FAISS 행 순서와 1:1 대응
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n완료: {INDEX_PATH} ({index.ntotal} vectors)")
    print(f"      {META_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
