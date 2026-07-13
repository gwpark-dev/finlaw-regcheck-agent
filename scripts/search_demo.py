"""검색기 검증용 CLI (W06 DoD).

  uv run python scripts/search_demo.py "원금 보장되고 수익률 확실한 상품입니다"
  uv run python scripts/search_demo.py                  # 평가 질의 10건 일괄 실행 + recall@5
  uv run python scripts/search_demo.py "..." --k 5 --principle "광고 규제"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 넣어야 스크립트 직접 실행 시 rag 패키지를 찾는다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import EVAL_DIR, LOW_CONFIDENCE_THRESHOLD  # noqa: E402
from rag.retriever import search  # noqa: E402

QUERY_FILE = EVAL_DIR / "w06_queries.txt"
LINE_WIDTH = 78


def load_queries(path: Path) -> list[tuple[str, list[str]]]:
    """평가 파일 파싱: '질의문 ||| 파일stem:조항,...' → (질의, 기대조항 리스트)"""
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        query, _, expected = line.partition("|||")
        wanted = [e.strip() for e in expected.split(",") if e.strip()]
        queries.append((query.strip(), wanted))
    return queries


def format_hit(hit: dict) -> str:
    source = Path(hit["source_file"]).stem
    where = hit["article"] or "본문"
    if hit["article_title"]:
        where += f"({hit['article_title']})"
    if hit["paragraph"]:
        where += f" {hit['paragraph']}항"
    tag = f" [{hit['principle']}]" if hit["principle"] else ""

    body = " ".join(hit["text"].split())
    if len(body) > 160:
        body = body[:160] + "…"
    return (
        f"  {hit['rank']}. [{hit['score']:.3f}] {hit['law']} {where}{tag}\n"
        f"     └ {body}\n"
        f"     ({source} / 시행 {hit['effective_date'] or '-'})"
    )


def top_articles(query: str, k: int, principle: str | None) -> list[dict]:
    """조(條) 단위 Top-k.

    인용의 단위는 청크가 아니라 조항이다. 한 조가 여러 항으로 쪼개진 경우(예: 제22조는
    7개 항) 그 형제 청크들이 Top-k 자리를 다 차지해 버리므로, 조마다 최고 점수 청크만
    대표로 남긴다.
    """
    hits = search(query, k=k * 8, principle_filter=principle)

    seen: set[tuple[str, str | None]] = set()
    out: list[dict] = []
    for hit in hits:
        key = (hit["source_file"], hit["article"])
        if key in seen:
            continue
        seen.add(key)
        hit["rank"] = len(out) + 1
        out.append(hit)
        if len(out) == k:
            break
    return out


def run_one(query: str, k: int, principle: str | None, expected: list[str]) -> bool:
    """질의 1건 실행 → Top-k 조항 출력. 기대조항이 있으면 hit 여부를 반환."""
    hits = top_articles(query, k, principle)

    print("─" * LINE_WIDTH)
    print(f"Q: {query}")
    if expected:
        print(f"   기대 조항: {', '.join(expected)}")
    print()

    if not hits:
        print("  (검색 결과 없음)")
        return False

    for hit in hits:
        print(format_hit(hit))

    if hits[0]["low_confidence"]:
        print(
            f"\n  ⚠ low_confidence: 최고 유사도 {hits[0]['score']:.3f} "
            f"< {LOW_CONFIDENCE_THRESHOLD} → 근거 부족 (W08에서 판정 보류 처리)"
        )

    if not expected:
        return True

    found = {f"{Path(h['source_file']).stem}:{h['article']}" for h in hits}
    ok = bool(found & set(expected))
    print(f"\n  {'✓ HIT' if ok else '✗ MISS'} (기대 조항이 Top-{k}에 {'있음' if ok else '없음'})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="RegCheck 조항 검색 데모")
    parser.add_argument("query", nargs="?", help="광고/상담 문구 (생략 시 평가셋 일괄 실행)")
    parser.add_argument("-k", type=int, default=5, help="반환 개수 (기본 5)")
    parser.add_argument(
        "--principle", default=None, help="원칙 태그 필터 (예: '광고 규제')"
    )
    args = parser.parse_args()

    if args.query:
        run_one(args.query, args.k, args.principle, expected=[])
        return 0

    if not QUERY_FILE.exists():
        print(f"평가 질의 파일이 없습니다: {QUERY_FILE}")
        return 1

    queries = load_queries(QUERY_FILE)
    print(f"평가셋 {len(queries)}건 일괄 실행 (k={args.k})\n")
    results = [run_one(q, args.k, args.principle, exp) for q, exp in queries]


    hits = sum(results)
    print("═" * LINE_WIDTH)
    print(f"recall@{args.k} = {hits}/{len(results)} = {hits / len(results):.2f}  (목표 ≥ 0.80)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as e:  # 인덱스 미구축 — traceback 대신 안내만 보여준다
        print(e)
        sys.exit(1)
