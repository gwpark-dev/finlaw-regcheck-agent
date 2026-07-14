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


# --- 인용 중복 게이트 (FR-09, ADR-007) ---------------------------------------
#
# 오탐의 정체는 "한 행위를 이웃 원칙에 돌려쓰기"다. 모델이 문구에서 문제 행위를 하나
# 찾으면 그 행위를 근거로 여러 원칙을 동시에 위반이라 판정한다. 라벨링 규칙 2
# ("하나의 행위는 그 행위를 열거한 조문 하나에만 걸린다")를 코드로 옮긴 게이트다.
#
# 자기 고유의 근거(다른 원칙이 쓰지 않은 인용)가 하나도 없는 VIOLATION은, 남의 행위를
# 빌려 온 것으로 보고 NEEDS_REVIEW로 강등한다. OK로 내리지 않는다 — 사람 검토로
# 올려보내는 것이지 위반 가능성을 지우는 것이 아니다.

# 제19조제3항은 "거짓으로 또는 왜곡(= 불확실한 사항에 단정적 판단을 제공하거나 확실하다고
# 오인하게 하는 행위)하여 설명"을 금지하고, 제21조는 제1호(단정적 판단)·제2호(사실과
# 다르게 알림)를 금지한다. 두 조문이 같은 행위 집합을 함께 열거하므로, 이 사이의 인용
# 공유는 돌려쓰기가 아니라 법이 예정한 중복이다 (labeling_criteria 규칙 2-1).
LEGAL_OVERLAP = {
    "설명의무": {"C3", "C4"},
    "부당권유행위 금지": {"E1", "E2"},
}

_QUOTE_NOISE = re.compile(r"[\s.,·…\"'”“]+")


def _norm_quote(q: str) -> str:
    return _QUOTE_NOISE.sub("", q or "")


def _same_act(a: str, b: str) -> bool:
    """같은 행위를 가리키는 인용인가. 모델이 같은 문장을 길고 짧게 인용하므로 포함 관계로 본다."""
    a, b = _norm_quote(a), _norm_quote(b)
    return bool(a) and bool(b) and (a in b or b in a)


def _is_legal_overlap(p1: str, e1: str, p2: str, e2: str) -> bool:
    return (
        {p1, p2} == set(LEGAL_OVERLAP)
        and e1 in LEGAL_OVERLAP.get(p1, set())
        and e2 in LEGAL_OVERLAP.get(p2, set())
    )


def duplicate_citation_gate(
    report: dict, elements_by_principle: dict[str, list[dict]]
) -> tuple[dict, list[dict]]:
    """인용 중복 게이트. 반환: (게이트 적용 리포트, 강등 내역)"""
    violating = [v["principle"] for v in report["verdicts"] if v["verdict"] == "VIOLATION"]

    def has_own_evidence(principle: str) -> bool:
        mine = elements_by_principle.get(principle) or []
        for el in mine:
            shared = any(
                _same_act(el["quote"], other["quote"])
                and not _is_legal_overlap(principle, el["id"], p2, other["id"])
                for p2 in violating
                if p2 != principle
                for other in elements_by_principle.get(p2, [])
            )
            if not shared:
                return True  # 이 원칙만의 근거가 있다
        return False

    verdicts, demoted = [], []
    for verdict in report["verdicts"]:
        principle = verdict["principle"]
        elements = elements_by_principle.get(principle) or []
        # 인용을 남긴 VIOLATION만 대상 (누락형 위반은 인용이 비어 있어 중복 판단 불가)
        if verdict["verdict"] != "VIOLATION" or not elements or has_own_evidence(principle):
            verdicts.append(verdict)
            continue

        borrowed = ", ".join(f'"{e["quote"][:30]}…"' for e in elements)
        verdicts.append(
            {
                **verdict,
                "verdict": "NEEDS_REVIEW",
                "suggestion": "",
                "reason": (
                    f"[Guardrail: 다른 원칙과 같은 행위를 근거로 삼음 → 판정 보류] "
                    f"이 원칙만의 독립적 근거가 없다(인용: {borrowed}). "
                    f"(원 판정: VIOLATION / {verdict['reason']})"
                ),
            }
        )
        demoted.append(
            {
                "principle": principle,
                "reason": "인용 중복 — 독립 근거 없음",
                "elements": [e["id"] for e in elements],
            }
        )

    return {**report, "verdicts": verdicts}, demoted
