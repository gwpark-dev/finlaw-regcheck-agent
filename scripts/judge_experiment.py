"""오탐 원인 규명 실험 (W08 선행).

  uv run python scripts/judge_experiment.py B C          # 조건 골라 실행
  uv run python scripts/judge_experiment.py C --repeat 2 # 재현성(뒤집힘 비율) 측정
  uv run python scripts/judge_experiment.py --report     # 저장된 결과만 표로

평가셋·라벨은 동결. 바꾸는 것은 (모델 / judge 프롬프트 버전 / input_type 출처) 셋뿐이다.
결과는 data/eval/experiments/<조건>.json 에 저장하고, 다시 실행하면 덮어쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.judge import PRINCIPLE_ORDER, JudgeError, judge  # noqa: E402
from rag.config import EVAL_DIR, ROOT  # noqa: E402

PROMPTS = ROOT / "agent" / "prompts"
OUT_DIR = EVAL_DIR / "experiments"
EVAL_PATH = EVAL_DIR / "w07_eval.jsonl"

# $/1M 토큰 (input, output)
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
}

CONDITIONS = {
    "A": {"model": "gpt-4o-mini", "prompt": "judge_v1.0.md", "oracle": False},
    "B": {"model": "gpt-4o-mini", "prompt": "judge_v1.0.md", "oracle": True},
    "C": {"model": "gpt-4o-mini", "prompt": "judge_v2.0.md", "oracle": True},
    "F": {"model": "gpt-4o-mini", "prompt": "judge_v2.1.md", "oracle": True},
    "G": {"model": "gpt-4.1-mini", "prompt": "judge_v2.1.md", "oracle": True},
    "H": {"model": "gpt-4.1", "prompt": "judge_v2.1.md", "oracle": True},
}

# 구성요건 오지정 — v2에서 남은 오탐의 핵심 유형. 조문 항목을 잘못 고르고 무관한 문장을
# 인용해 위반을 만든다. 설명의무·부당권유에서 집중 발생(C 조건 23건).
MISATTRIB_PRINCIPLES = ("설명의무", "부당권유행위 금지")

# 대표 오답 4건 — 조건별로 고쳐지는지 추적한다.
WATCH = {
    ("w07-029", "광고 규제"): "명시 문장을 누락 판정",
    ("w07-030", "광고 규제"): "명시 문장을 누락 판정",
    ("w07-015", "설명의무"): "위험 고지를 위반 근거화",
    ("w07-031", "부당권유행위 금지"): "비투자성 재권유",
}


def load_cases(path: Path = EVAL_PATH) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_condition(key: str, cases: list[dict], repeat: int) -> dict:
    cfg = CONDITIONS[key]
    prompt_path = PROMPTS / cfg["prompt"]
    runs: list[dict] = []

    for r in range(repeat):
        preds: dict[str, dict] = {}
        reasons: dict[str, dict] = {}
        tin = tout = 0
        fingerprints: set[str] = set()
        schema_ok = schema_total = 0
        started = time.perf_counter()

        for i, case in enumerate(cases, start=1):
            override = (
                {"type": case["input_type"], "product": case["product"]}
                if cfg["oracle"]
                else None
            )
            try:
                report = judge(
                    case["text"],
                    model=cfg["model"],
                    prompt_path=prompt_path,
                    cls_override=override,
                )
                schema_ok += 6
            except JudgeError as e:
                print(f"  [{key}#{r + 1}] {case['id']} 스키마 실패: {e}")
                schema_total += 6
                continue
            schema_total += 6

            by = {v["principle"]: v for v in report["verdicts"]}
            preds[case["id"]] = {p: by[p]["verdict"] for p in PRINCIPLE_ORDER}
            reasons[case["id"]] = {p: by[p]["reason"] for p in PRINCIPLE_ORDER}
            tin += report["meta"]["prompt_tokens"]
            tout += report["meta"]["completion_tokens"]
            fingerprints.update(report["meta"]["system_fingerprint"])

            wrong = [p for p in PRINCIPLE_ORDER if case["labels"][p] != preds[case["id"]][p]]
            mark = "✓" if not wrong else f"✗ {', '.join(wrong)}"
            print(f"  [{key}#{r + 1}] {i:>2}/{len(cases)} {case['id']} {mark}")

        pin, pout = PRICING[cfg["model"]]
        runs.append(
            {
                "predicted": preds,
                "reasons": reasons,
                "schema_compliance": schema_ok / schema_total if schema_total else 0.0,
                "seconds_per_case": (time.perf_counter() - started) / len(cases),
                "cost_usd": tin / 1e6 * pin + tout / 1e6 * pout,
                "prompt_tokens": tin,
                "completion_tokens": tout,
                "system_fingerprint": sorted(fingerprints),
            }
        )

    return {"key": key, **cfg, "runs": runs}


def score(preds: dict, cases: list[dict]) -> dict:
    per = defaultdict(lambda: defaultdict(int))
    n = c = gv = hv = gok = fp = nr = 0
    for case in cases:
        if case["id"] not in preds:
            continue
        for p in PRINCIPLE_ORDER:
            gold, pred = case["labels"][p], preds[case["id"]][p]
            n += 1
            c += gold == pred
            per[p]["n"] += 1
            per[p]["c"] += gold == pred
            if gold == "VIOLATION":
                gv += 1
                hv += pred == "VIOLATION"
            else:
                gok += 1
                fp += pred == "VIOLATION"
            nr += pred == "NEEDS_REVIEW"
    # 구성요건 오지정: 설명의무·부당권유에서 정상(OK)을 위반으로 찍은 셀 수
    misattrib = sum(
        1
        for case in cases
        if case["id"] in preds
        for p in MISATTRIB_PRINCIPLES
        if case["labels"][p] == "OK" and preds[case["id"]][p] == "VIOLATION"
    )
    return {
        "accuracy": c / n if n else 0.0,
        "recall": hv / gv if gv else 0.0,
        "fpr": fp / gok if gok else 0.0,
        "needs_review": nr,
        "cells": n,
        "false_positives": fp,
        "misattrib": misattrib,
        "per_principle": {p: v["c"] / v["n"] for p, v in per.items()},
    }


def flip_rate(runs: list[dict]) -> float | None:
    """동일 조건 2회 실행 간 판정이 뒤집힌 셀 비율."""
    if len(runs) < 2:
        return None
    a, b = runs[0]["predicted"], runs[1]["predicted"]
    shared = set(a) & set(b)
    total = len(shared) * len(PRINCIPLE_ORDER)
    flips = sum(1 for i in shared for p in PRINCIPLE_ORDER if a[i][p] != b[i][p])
    return flips / total if total else None


def report(cases: list[dict], suffix: str = "") -> None:
    results = []
    for key in CONDITIONS:
        path = OUT_DIR / f"{key}{suffix}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        print("저장된 실험 결과가 없습니다.")
        return

    W = 108
    print("\n" + "═" * W)
    print("실험 결과 (평가셋 31건 / 186셀 동결, 확정 라벨)")
    print("═" * W)
    print(f"{'조건':<4}{'모델':<14}{'judge':<7}{'유형':<10}"
          f"{'정확도':>8}{'오탐률':>8}{'재현율':>8}{'오지정':>7}{'초/건':>7}{'비용':>8}{'뒤집힘':>8}")
    print("─" * W)
    for res in results:
        s = score(res["runs"][0]["predicted"], cases)
        run = res["runs"][0]
        fr = flip_rate(res["runs"])
        print(
            f"{res['key']:<4}{res['model']:<14}"
            f"{res['prompt'].replace('judge_', '').replace('.md', ''):<7}"
            f"{'oracle' if res['oracle'] else 'classifier':<10}"
            f"{s['accuracy']:>8.3f}{s['fpr']:>8.3f}{s['recall']:>8.3f}{s['misattrib']:>7}"
            f"{run['seconds_per_case']:>7.1f}"
            f"{'$' + format(run['cost_usd'], '.2f'):>8}"
            f"{(format(fr, '.1%') if fr is not None else '-'):>8}"
        )
    print("(오지정 = 설명의무·부당권유에서 정상 문구를 위반으로 찍은 셀 수. C 기준 23건)")

    print("\n" + "═" * W)
    print("원칙별 정확도")
    print("═" * W)
    print(f"{'조건':<6}" + "".join(f"{p[:6]:>10}" for p in PRINCIPLE_ORDER))
    print("─" * W)
    for res in results:
        s = score(res["runs"][0]["predicted"], cases)
        print(f"{res['key']:<6}" + "".join(f"{s['per_principle'][p]:>10.2f}" for p in PRINCIPLE_ORDER))

    print("\n" + "═" * W)
    print("대표 오답 추적 (정답 대비)")
    print("═" * W)
    by_case = {c["id"]: c for c in cases}
    print(f"{'사례':<32}" + "".join(f"{r['key']:>8}" for r in results))
    print("─" * W)
    for (cid, principle), label in WATCH.items():
        if cid not in by_case:  # 홀드아웃에는 w07 대표 오답 문항이 없다
            continue
        gold = by_case[cid]["labels"][principle]
        row = f"{cid} {label:<20}"[:32].ljust(32)
        for res in results:
            pred = res["runs"][0]["predicted"].get(cid, {}).get(principle, "-")
            row += f"{('✓' if pred == gold else '✗') + ' ' + pred[:4]:>8}"
        print(row)
    print(f"\n(정답: w07-029/030 광고=OK, w07-015 설명의무=OK, w07-031 부당권유=OK)")


def main() -> int:
    parser = argparse.ArgumentParser(description="오탐 원인 규명 실험")
    parser.add_argument("conditions", nargs="*", help=f"실행할 조건 {list(CONDITIONS)}")
    parser.add_argument("--repeat", type=int, default=1, help="동일 조건 반복 실행 횟수")
    parser.add_argument(
        "--append",
        action="store_true",
        help="기존 결과에 실행을 1회 덧붙인다(재현성 측정용 — 조건을 통째로 다시 돌리지 않는다)",
    )
    parser.add_argument("--report", action="store_true", help="저장된 결과만 표로 출력")
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="평가셋 대신 홀드아웃(w08_holdout.jsonl)에 실행. 결과는 <조건>_holdout.json",
    )
    args = parser.parse_args()

    eval_path = (EVAL_DIR / "w08_holdout.jsonl") if args.holdout else EVAL_PATH
    suffix = "_holdout" if args.holdout else ""
    cases = load_cases(eval_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"평가 대상: {eval_path.name} ({len(cases)}건 / {len(cases) * 6}셀)")

    for key in args.conditions:
        if key not in CONDITIONS:
            print(f"알 수 없는 조건: {key}")
            return 1
        print(f"\n▶ 조건 {key}: {CONDITIONS[key]}")
        res = run_condition(key, cases, args.repeat)
        path = OUT_DIR / f"{key}{suffix}.json"
        if args.append and path.exists():
            prev = json.loads(path.read_text(encoding="utf-8"))
            res["runs"] = prev["runs"] + res["runs"]
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.conditions or args.report:
        report(cases, suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
