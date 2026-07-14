"""W08 DoD 검증 (명세서 §10).

  1. 개인정보 샘플 마스킹 100%
  2. 지식베이스 외 조항 인용 0건 (인용 검증 Guardrail이 실제로 차단하는지 포함)
  3. 전 요청 로그 기록

  uv run python scripts/w08_dod.py            # 1·2 정적 검증 + 3 (홀드아웃 12건 실행)
  uv run python scripts/w08_dod.py --no-llm   # LLM 호출 없이 1·2의 정적 부분만
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import check  # noqa: E402
from guardrails.masking import has_pii, mask  # noqa: E402
from guardrails.verify import verify_verdict  # noqa: E402
from logging_ import audit  # noqa: E402
from rag.config import EVAL_DIR  # noqa: E402

W = 84

# 개인정보 샘플 — 한국어 조사가 붙는 형태를 일부러 섞었다(\b가 깨지는 지점).
PII_SAMPLES = [
    "고객 홍길동님 주민번호 900101-1234567로 확인했습니다.",
    "연락처는 010-1234-5678이고 자택은 02-345-6789입니다.",
    "입금 계좌 110-234-567890으로 보내주세요.",
    "카드번호 5327-1234-5678-9012입니다.",
    "메일은 hong.gildong@example.co.kr로 보내드렸습니다.",
    "사업자등록번호 123-45-67890 확인 부탁드립니다.",
    "01098765432 로도 연락 가능합니다.",
    "주민 8801012345678 아니고 880101-2345678 입니다.",
    "계좌 1002-345-678901, 연락처 010.9876.5432",
    "문의: support@regcheck.kr / 대표전화 031-123-4567",
]
# 마스킹되면 안 되는 것들 (오탐 확인)
CLEAN_SAMPLES = [
    "연 3.2% 금리, 2026.7.1. 기준입니다.",
    "이 펀드는 원금 손실이 발생할 수 있습니다.",
    "금소법 제22조제3항제3호다목에 따릅니다.",
    "대출 실행 2년 차에 중도상환수수료 1.2%가 부과됩니다.",
]


def check_masking() -> bool:
    print("═" * W)
    print("DoD 1 — 개인정보 샘플 마스킹 100%")
    print("═" * W)

    masked_ok = 0
    for s in PII_SAMPLES:
        m, findings = mask(s)
        ok = not has_pii(m) and findings
        masked_ok += bool(ok)
        if not ok:
            print(f"  ✗ 미마스킹 잔존: {m}")
    print(f"  개인정보 샘플 {masked_ok}/{len(PII_SAMPLES)} 마스킹 완료")

    fp = [s for s in CLEAN_SAMPLES if mask(s)[1]]
    for s in fp:
        print(f"  ✗ 오탐(마스킹되면 안 됨): {s}")
    print(f"  정상 문구 오탐 {len(fp)}/{len(CLEAN_SAMPLES)}건")

    # findings에 값이 새지 않는지 (NFR-04)
    leaked = [s for s in PII_SAMPLES if any(
        any(ch.isdigit() or "@" in str(v) for ch in str(v))
        for f in mask(s)[1] for k, v in f.items() if k != "count"
    )]
    print(f"  탐지 요약에 원본 값 유출 {len(leaked)}건")

    ok = masked_ok == len(PII_SAMPLES) and not fp and not leaked
    print(f"  → {'통과' if ok else '실패'}\n")
    return ok


def check_citation_guard() -> bool:
    """Guardrail이 컨텍스트 밖 인용을 실제로 차단하는지 (음성 테스트)."""
    print("═" * W)
    print("DoD 2-a — 인용 검증 Guardrail 음성 테스트 (날조 인용 차단)")
    print("═" * W)

    chunks = [
        {"law": "금융소비자 보호에 관한 법률", "article": "제21조", "source_file": "금소법.pdf"},
        {"law": "금융소비자 보호에 관한 감독규정", "article": "제15조", "source_file": "금소법_감독규정.pdf"},
    ]
    cases = [
        ("존재하지 않는 조항 날조", "금융소비자 보호에 관한 법률", "제999조", "NEEDS_REVIEW"),
        ("색인엔 있으나 이 원칙 컨텍스트엔 없는 조항", "금융소비자 보호에 관한 법률", "제20조", "NEEDS_REVIEW"),
        ("컨텍스트에 있는 조항 (정상)", "금융소비자 보호에 관한 법률", "제21조", "VIOLATION"),
        ("조 제목이 붙은 표기 (정상 처리돼야)", "금융소비자 보호에 관한 법률", "제21조(부당권유행위 금지)", "VIOLATION"),
    ]

    passed = 0
    for label, law, article, expected in cases:
        verdict = {
            "principle": "부당권유행위 금지",
            "verdict": "VIOLATION",
            "confidence": 0.9,
            "evidence": [{"law": law, "article": article, "quote": "..."}],
            "reason": "테스트",
            "suggestion": "수정안",
        }
        verified, dropped = verify_verdict(verdict, chunks)
        ok = verified["verdict"] == expected
        passed += ok
        print(f"  {'✓' if ok else '✗'} {label:<38} → {verified['verdict']} (차단 {len(dropped)}건)")

    ok = passed == len(cases)
    print(f"  → {'통과' if ok else '실패'}\n")
    return ok


def check_pipeline(cases: list[dict]) -> bool:
    print("═" * W)
    print(f"DoD 2-b/3 — 실입력 {len(cases)}건: 컨텍스트 외 인용 0건 + 전 요청 로그 기록")
    print("═" * W)

    before = len(audit.load_log())
    cited = dropped = 0

    for i, case in enumerate(cases, start=1):
        # 캐시가 있으면 저장된 판정을 돌려주므로, DoD는 재판정을 강제해 실제 호출 경로를 본다.
        report = check(case["text"], input_type=case["input_type"], force_recheck=True)
        rec = audit.load_log()[-1]
        d = len(rec["guardrails"]["dropped_citations"])
        c = sum(len(v["cited"]) for v in rec["verdicts"])
        cited += c
        dropped += d
        print(f"  [{i:>2}/{len(cases)}] {case['id']}  인용 {c}건 / 차단 {d}건")

    after = len(audit.load_log())
    logged = after - before

    print(f"\n  총 인용 {cited}건 중 컨텍스트 외 인용 차단 {dropped}건")
    print(f"  감사 로그 기록 {logged}/{len(cases)}건")

    # 로그에 개인정보가 남지 않았는지 — 원문이 흘러들 수 있는 필드만 본다.
    # 레코드 전체를 훑으면 timestamp("2026-07-14")가 계좌번호 패턴에 걸려 오탐한다.
    leaky = [
        r["input_hash"][:8]
        for r in audit.load_log()
        if has_pii(r["masked_input"])
    ]
    pii_in_log = bool(leaky)
    print(f"  감사 로그 masked_input에 개인정보 잔존: "
          f"{f'{len(leaky)}건(실패)' if pii_in_log else '없음'}")

    ok = dropped == 0 and logged == len(cases) and not pii_in_log
    print(f"  → {'통과' if ok else '실패'}")
    print("  (차단 0건 = 모델이 컨텍스트 밖 조항을 인용하지 않았다는 뜻. "
          "인용했다면 Guardrail이 NEEDS_REVIEW로 강등했을 것이다.)\n")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="W08 DoD 검증")
    parser.add_argument("--no-llm", action="store_true", help="LLM 호출 없이 정적 검증만")
    args = parser.parse_args()

    results = [check_masking(), check_citation_guard()]

    if not args.no_llm:
        cases = [
            json.loads(line)
            for line in (EVAL_DIR / "w08_holdout.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        results.append(check_pipeline(cases))

    print("═" * W)
    print(f"W08 DoD: {'전 항목 통과' if all(results) else '미통과 항목 있음'}")
    print("═" * W)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
