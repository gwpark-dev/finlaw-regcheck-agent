"""점검 파이프라인 — 명세서 §6의 [1]~[8]을 잇는다.

  마스킹 → 유형 분류 → 판정(컨텍스트 구성 + LLM) → 인용 검증 → 리포트 + 감사 로그

  uv run python -m agent.pipeline "이 상품은 원금이 보장되고 수익률도 확실합니다"

호출부는 이 함수만 쓴다. judge()를 직접 부르면 마스킹과 인용 검증을 건너뛰게 되므로,
UI(W10)를 포함한 모든 경로는 check()를 통과해야 한다.
"""

from __future__ import annotations

import json
import sys

from agent.classifier import classify
from agent.judge import judge
from guardrails.masking import has_pii, mask
from guardrails.verify import verify_report
from logging_ import audit


class PipelineError(RuntimeError):
    pass


def check(text: str, input_type: str | None = None, session=None) -> dict:
    """문구 1건 점검. 반환: 명세서 §7.3 리포트.

    input_type: "광고" | "상담". 지정하면 분류기 대신 이 값을 쓴다. v2.1의 국면
      게이트가 유형에 의존하므로(제22조는 광고 국면만), 검수자가 유형을 알고 있다면
      넣어주는 편이 안전하다 — 분류기 오분류 시 게이트가 반대로 작동한다.
    """
    # [1] Guardrail-In — 어떤 API 호출보다 먼저 (FR-07, NFR-04)
    masked, pii_findings = mask(text)
    if has_pii(masked):  # 방어적 확인 — 마스킹이 새면 API로 원문이 나간다
        raise PipelineError("마스킹 후에도 개인정보 패턴이 남아 있습니다. 호출을 중단합니다.")

    # [2] 유형 분류
    cls = classify(masked)
    if input_type:
        cls = {**cls, "type": input_type}

    # [3][4][5] 컨텍스트 구성 + 원칙별 판정
    chunks_by_principle: dict[str, list[dict]] = {}
    report = judge(masked, cls_override=cls, chunks_out=chunks_by_principle)

    # [6] Guardrail-Out — 인용 검증 (FR-08, NFR-02/03)
    report, dropped = verify_report(report, chunks_by_principle)

    # [7][8] 리포트 + 감사 로그 (FR-10, NFR-05)
    audit.log_check(
        report,
        masked_input=masked,
        chunks_by_principle=chunks_by_principle,
        pii_findings=pii_findings,
        dropped_citations=dropped,
        session_id=getattr(session, "session_id", "-"),
    )
    if session is not None:
        session.record(masked, report)

    return report


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    from agent.memory import Session, summarize

    session = Session()
    report = check(sys.argv[1], session=session)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n요약: {summarize(report)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
