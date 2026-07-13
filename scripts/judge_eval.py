"""판정 정확도 평가 (W07 DoD, FR-13).

  uv run python scripts/judge_eval.py              # 평가셋 전체
  uv run python scripts/judge_eval.py --limit 5    # 앞 5건만 (스모크)

측정 항목
  - 스키마 준수율: LLM 출력이 §7.3 스키마로 파싱·검증된 비율 (목표 100%)
  - 판정 정확도: 원칙별 verdict 일치율 (목표 ≥ 0.75)
  - 오탐률: 정상(OK) 라벨을 VIOLATION으로 판정한 비율 (목표 ≤ 0.15)
  - 재현율: 위반(VIOLATION) 라벨을 VIOLATION으로 잡아낸 비율

라벨이 VIOLATION/OK 두 가지뿐이므로 모델의 NEEDS_REVIEW는 항상 불일치로 집계된다.
보류가 오답인 것은 아니므로 별도 열로 함께 보고한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.judge import JudgeError, PRINCIPLE_ORDER, judge  # noqa: E402
from rag.config import EVAL_DIR  # noqa: E402

EVAL_PATH = EVAL_DIR / "w07_eval.jsonl"
RESULT_PATH = EVAL_DIR / "w07_results.json"
WIDTH = 92


def load_cases(limit: int | None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases[:limit] if limit else cases


def main() -> int:
    parser = argparse.ArgumentParser(description="RegCheck 판정 정확도 평가")
    parser.add_argument("--limit", type=int, default=None, help="앞 N건만 실행")
    args = parser.parse_args()

    if not EVAL_PATH.exists():
        print(f"평가셋이 없습니다: {EVAL_PATH}")
        return 1

    cases = load_cases(args.limit)
    print(f"평가셋 {len(cases)}건 × 6원칙 = {len(cases) * 6}개 판정\n")

    # 원칙별 혼동행렬: (정답, 예측) → 건수
    matrix: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    confusions: list[dict] = []
    records: list[dict] = []
    schema_ok = schema_total = 0
    started = time.perf_counter()

    for i, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        try:
            report = judge(case["text"])
            # 스키마 검증은 judge 내부(_validate)에서 이미 통과한 상태다.
            schema_ok += 6
            failed = None
        except JudgeError as e:
            failed = str(e)
            report = None
        schema_total += 6
        elapsed = time.perf_counter() - t0

        if report is None:
            print(f"[{i:>2}/{len(cases)}] {case['id']}  스키마 실패: {failed}")
            records.append({"id": case["id"], "schema_error": failed})
            continue

        by_principle = {v["principle"]: v for v in report["verdicts"]}
        wrong = []
        for principle in PRINCIPLE_ORDER:
            gold = case["labels"][principle]
            pred = by_principle[principle]["verdict"]
            matrix[principle][(gold, pred)] += 1
            if gold != pred:
                wrong.append(principle)
                confusions.append(
                    {
                        "id": case["id"],
                        "principle": principle,
                        "gold": gold,
                        "pred": pred,
                        "confidence": by_principle[principle]["confidence"],
                        "reason": by_principle[principle]["reason"],
                    }
                )

        mark = "✓" if not wrong else f"✗ {', '.join(wrong)}"
        print(f"[{i:>2}/{len(cases)}] {case['id']}  {elapsed:5.1f}초  {mark}")
        records.append(
            {
                "id": case["id"],
                "target": case["target"],
                "labels": case["labels"],
                "predicted": {p: v["verdict"] for p, v in by_principle.items()},
                "report": report,
            }
        )

    total_time = time.perf_counter() - started
    summary = report_metrics(matrix, schema_ok, schema_total, confusions, len(cases), total_time)

    RESULT_PATH.write_text(
        json.dumps(
            {"summary": summary, "confusions": confusions, "records": records},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n상세 결과 저장: {RESULT_PATH}")
    return 0


def report_metrics(
    matrix: dict,
    schema_ok: int,
    schema_total: int,
    confusions: list[dict],
    n_cases: int,
    total_time: float,
) -> dict:
    print("\n" + "═" * WIDTH)
    print("판정 정확도 (원칙별)")
    print("═" * WIDTH)
    print(f"{'원칙':<20}{'정확도':>10}{'위반 재현율':>14}{'오탐률':>10}{'보류':>8}{'셀':>6}")
    print("─" * WIDTH)

    totals = defaultdict(int)
    per_principle = {}

    for principle in PRINCIPLE_ORDER:
        cells = matrix[principle]
        n = sum(cells.values())
        if not n:
            continue
        correct = sum(c for (g, p), c in cells.items() if g == p)
        gold_v = sum(c for (g, _), c in cells.items() if g == "VIOLATION")
        gold_ok = sum(c for (g, _), c in cells.items() if g == "OK")
        hit_v = cells.get(("VIOLATION", "VIOLATION"), 0)
        false_pos = cells.get(("OK", "VIOLATION"), 0)
        review = sum(c for (_, p), c in cells.items() if p == "NEEDS_REVIEW")

        acc = correct / n
        recall = hit_v / gold_v if gold_v else float("nan")
        fpr = false_pos / gold_ok if gold_ok else float("nan")

        totals["n"] += n
        totals["correct"] += correct
        totals["gold_v"] += gold_v
        totals["gold_ok"] += gold_ok
        totals["hit_v"] += hit_v
        totals["false_pos"] += false_pos
        totals["review"] += review

        per_principle[principle] = {
            "accuracy": round(acc, 3),
            "violation_recall": None if gold_v == 0 else round(recall, 3),
            "false_positive_rate": None if gold_ok == 0 else round(fpr, 3),
            "needs_review": review,
            "cells": n,
        }
        print(
            f"{principle:<20}{acc:>9.2f} {hit_v:>6}/{gold_v:<7}"
            f"{false_pos:>4}/{gold_ok:<5}{review:>8}{n:>6}"
        )

    print("─" * WIDTH)
    acc = totals["correct"] / totals["n"]
    recall = totals["hit_v"] / totals["gold_v"]
    fpr = totals["false_pos"] / totals["gold_ok"]
    print(
        f"{'전체':<20}{acc:>9.2f} {totals['hit_v']:>6}/{totals['gold_v']:<7}"
        f"{totals['false_pos']:>4}/{totals['gold_ok']:<5}{totals['review']:>8}{totals['n']:>6}"
    )

    schema_rate = schema_ok / schema_total if schema_total else 0.0
    print("\n" + "═" * WIDTH)
    print("DoD 지표")
    print("═" * WIDTH)
    verdict = lambda ok: "달성" if ok else "미달"  # noqa: E731
    print(f"  스키마 준수율   {schema_rate:>6.1%}  (목표 100%)   {verdict(schema_rate == 1.0)}")
    print(f"  판정 정확도     {acc:>6.1%}  (목표 ≥75%)   {verdict(acc >= 0.75)}")
    print(f"  오탐률          {fpr:>6.1%}  (목표 ≤15%)   {verdict(fpr <= 0.15)}")
    print(f"  위반 재현율     {recall:>6.1%}  (참고)")
    print(f"\n  평균 소요       {total_time / n_cases:>6.1f}초/건  (NFR-01: 30초)")

    if confusions:
        print("\n" + "═" * WIDTH)
        print(f"혼동 사례 {len(confusions)}건")
        print("═" * WIDTH)
        for c in confusions:
            reason = " ".join(c["reason"].split())[:70]
            print(f"  {c['id']} [{c['principle']}] 정답 {c['gold']} → 판정 {c['pred']} (conf {c['confidence']:.2f})")
            print(f"    └ {reason}…")

    return {
        "schema_compliance": round(schema_rate, 4),
        "accuracy": round(acc, 4),
        "violation_recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "needs_review_cells": totals["review"],
        "cells": totals["n"],
        "seconds_per_case": round(total_time / n_cases, 1),
        "per_principle": per_principle,
    }


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
