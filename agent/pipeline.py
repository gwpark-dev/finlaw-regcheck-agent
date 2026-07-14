"""점검 파이프라인 — 명세서 §6의 [1]~[8]을 잇는다.

  마스킹 → 유형 확정 → 판정(컨텍스트 구성 + LLM) → 인용 검증 → 인용 중복 게이트
  → 리포트 + 감사 로그

  uv run python -m agent.pipeline 상담 "이 상품은 원금이 보장되고 수익률도 확실합니다"

호출부는 check()만 쓴다. judge()를 직접 부르면 마스킹·인용 검증·게이트를 건너뛰게 되므로,
UI(W10)를 포함한 모든 경로는 check()를 통과해야 한다.
"""

from __future__ import annotations

import json
import sys

from agent import memory
from agent.classifier import classify
from agent.judge import JUDGE_MODEL, PROMPT_PATH, input_hash, judge, prompt_version
from guardrails.masking import has_pii, mask
from guardrails.verify import duplicate_citation_gate, verify_report
from logging_ import audit

INPUT_TYPES = ("광고", "상담")


class PipelineError(RuntimeError):
    pass


def suggest_input_type(text: str) -> dict:
    """분류기의 유형 제안 + 상품군. UI 기본값과 불일치 경고에 쓴다 (ADR-008)."""
    return classify(text)


def check(
    text: str,
    input_type: str,
    session=None,
    force_recheck: bool = False,
) -> dict:
    """문구 1건 점검. 반환: 명세서 §7.3 리포트.

    input_type: "광고" | "상담" — **필수**. 판정 엔진의 국면 게이트(제22조는 광고
      국면만)가 이 값에 의존하는데, 규칙 기반 분류기의 광고/상담 정확도가 0.535로
      "무조건 상담"(0.837)보다 낮고 오분류가 전부 게이트를 여는 방향이라 신뢰할 수
      없다. 사내 검수 도구이므로 담당자가 유형을 지정한다 (ADR-008).
      분류기는 상품군 분류와 UI 기본값 제안으로 역할을 한정한다.

    force_recheck: 저장된 판정이 있어도 다시 판정한다. 기본은 캐시 재사용 —
      같은 입력에 같은 출력을 보장하기 위함이다 (ADR-007, NFR-07).
    """
    if input_type not in INPUT_TYPES:
        raise PipelineError(
            f"input_type은 {INPUT_TYPES} 중 하나여야 합니다 (ADR-008: 사용자 필수 입력). 받은 값: {input_type!r}"
        )

    # [1] Guardrail-In — 어떤 API 호출보다 먼저 (FR-07, NFR-04)
    masked, pii_findings = mask(text)
    if has_pii(masked):  # 방어적 확인 — 마스킹이 새면 API로 원문이 나간다
        raise PipelineError("마스킹 후에도 개인정보 패턴이 남아 있습니다. 호출을 중단합니다.")

    session_id = getattr(session, "session_id", "-")
    version, model = prompt_version(PROMPT_PATH), JUDGE_MODEL

    # 판정 캐시 — 같은 (문구, 프롬프트 버전, 모델)이면 저장된 판정을 그대로 (ADR-007)
    if not force_recheck:
        cached = memory.find_cached(input_hash(masked), version, model)
        if cached is not None:
            audit.log_check(
                cached,
                masked_input=masked,
                chunks_by_principle={},
                pii_findings=pii_findings,
                dropped_citations=[],
                demoted=[],
                session_id=session_id,
                cache={"hit": True, "forced_recheck": False},
            )
            if session is not None:
                session.turns.append({"text": masked, "report": cached})
            return cached

    # [2] 유형 확정 — 유형은 사용자가 준다. 분류기는 상품군만 (ADR-008)
    cls = {**classify(masked), "type": input_type}

    # [3][4][5] 컨텍스트 구성 + 원칙별 판정
    chunks_by_principle: dict[str, list[dict]] = {}
    elements_by_principle: dict[str, list[dict]] = {}
    report = judge(
        masked,
        cls_override=cls,
        chunks_out=chunks_by_principle,
        elements_out=elements_by_principle,
    )

    # [6] Guardrail-Out — 인용 검증 (FR-08, NFR-02/03) → 인용 중복 게이트 (FR-09)
    report, dropped = verify_report(report, chunks_by_principle)
    report, demoted = duplicate_citation_gate(report, elements_by_principle)

    # [7][8] 리포트 + 이력 + 감사 로그 (FR-06, FR-10, NFR-05)
    audit.log_check(
        report,
        masked_input=masked,
        chunks_by_principle=chunks_by_principle,
        pii_findings=pii_findings,
        dropped_citations=dropped,
        demoted=demoted,
        session_id=session_id,
        cache={"hit": False, "forced_recheck": force_recheck},
    )
    if session is not None:
        session.record(masked, report)
    else:
        memory.append_history(report, session_id=session_id)

    return report


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in INPUT_TYPES:
        print(__doc__)
        print(f"유형은 {INPUT_TYPES} 중 하나 (ADR-008: 사용자 필수 입력)")
        return 1

    from agent.memory import Session, summarize

    input_type, text = sys.argv[1], sys.argv[2]
    hint = suggest_input_type(text)
    if hint["type"] != input_type:
        print(
            f"⚠ 분류기 제안({hint['type']})과 지정하신 유형({input_type})이 다릅니다. "
            f"지정하신 값으로 판정합니다.",
            file=sys.stderr,
        )

    session = Session()
    report = check(text, input_type=input_type, session=session)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n요약: {summarize(report)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
