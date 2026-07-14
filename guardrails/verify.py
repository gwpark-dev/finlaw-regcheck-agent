"""인용 검증 Guardrail — Guardrail 출력단 (FR-08, NFR-02/03).

이 프로젝트가 존재하는 이유가 "LLM이 조문을 지어낸다"는 실패를 막는 것이다. 여기가
그 강제 지점이다.

판정이 인용한 조항이 **그 원칙에 실제로 제시된 컨텍스트**(본법 고정 주입 + 하위규정
보강 검색, ADR-005)에 존재하는지 대조한다. 존재하지 않는 인용은 버리고, 남은 근거가
없으면 verdict를 NEEDS_REVIEW로 강등한다.

지식베이스 전체가 아니라 '그 원칙의 컨텍스트'를 기준으로 삼는 이유: 색인에는 있지만
이번 판정에 제시되지 않은 조항을 모델이 기억으로 끌어다 쓰는 것도 환각이다. 기준을
좁게 잡는 편이 안전하다.
"""

from __future__ import annotations

import re

# "제22조(금융상품등에 관한 광고 관련 준수사항)" → "제22조"
# 모델이 조 제목이나 항 표시를 섞어 써도 조(條) 단위로 정규화해 대조한다.
_ARTICLE_RE = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?")
_SPACE_RE = re.compile(r"\s+")


def _norm_article(article: str) -> str | None:
    m = _ARTICLE_RE.search(article or "")
    return _SPACE_RE.sub("", m.group()) if m else None


def _norm_law(law: str) -> str:
    return _SPACE_RE.sub("", law or "")


def allowed_citations(chunks: list[dict]) -> set[tuple[str, str]]:
    """컨텍스트 청크 → 인용이 허용되는 (법령, 조항) 집합."""
    out: set[tuple[str, str]] = set()
    for c in chunks:
        article = _norm_article(c.get("article") or "")
        if article:
            out.add((_norm_law(c["law"]), article))
    return out


def verify_verdict(verdict: dict, chunks: list[dict]) -> tuple[dict, list[dict]]:
    """판정 1건의 인용을 검증. 반환: (검증된 판정, 차단된 인용 목록)"""
    allowed = allowed_citations(chunks)

    kept, dropped = [], []
    for ev in verdict["evidence"]:
        article = _norm_article(ev.get("article", ""))
        key = (_norm_law(ev.get("law", "")), article)
        if article and key in allowed:
            kept.append(ev)
        else:
            dropped.append(
                {
                    "principle": verdict["principle"],
                    "law": ev.get("law", ""),
                    "article": ev.get("article", ""),
                    "reason": "컨텍스트에 없는 조항" if article else "조항 번호를 알아볼 수 없음",
                }
            )

    verified = {**verdict, "evidence": kept}

    if dropped and kept:
        verified["reason"] = (
            f"{verdict['reason']} "
            f"[Guardrail: 컨텍스트에 없는 인용 {len(dropped)}건 제거]"
        )

    # 근거가 하나도 남지 않으면 판정을 유지할 수 없다 (명세서 §7.3, NFR-02).
    if not kept:
        verified["verdict"] = "NEEDS_REVIEW"
        verified["suggestion"] = ""
        verified["reason"] = (
            f"[Guardrail: 유효한 근거 조항 없음 → 판정 보류] "
            f"(원 판정: {verdict['verdict']} / {verdict['reason']})"
        )

    return verified, dropped


def verify_report(report: dict, chunks_by_principle: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    """리포트 전체의 인용을 검증. 반환: (검증된 리포트, 차단된 인용 목록)"""
    verdicts, dropped_all = [], []
    for verdict in report["verdicts"]:
        chunks = chunks_by_principle.get(verdict["principle"], [])
        verified, dropped = verify_verdict(verdict, chunks)
        verdicts.append(verified)
        dropped_all.extend(dropped)

    return {**report, "verdicts": verdicts}, dropped_all
