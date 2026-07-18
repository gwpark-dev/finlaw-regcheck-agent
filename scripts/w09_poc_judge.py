"""W09 PoC — 로컬 LLM(Ollama) 판정 실측. §7 게이트 지표로 F/H와 비교.

Tier 2의 판정 LLM 대가를 잰다. judge.py는 수정하지 않는다 — build_context()(검색·임베딩)와
judge_principle_v2()(LLM 호출)가 이미 분리돼 있어, 여기서 조립만 한다.

  변수 통제: 검색은 production 그대로(text-embedding-3-large, data/index/) 고정 →
  F/H(둘 다 3-large 검색)와 동일 조건에서 **판정 LLM만** gpt-4.1 → 로컬 모델로 바꾼다.
  프롬프트는 judge_v2.1 동결. 국면 게이트(phase_gate) on = H와 동일.

  검색 컨텍스트 구성에만 OpenAI 임베딩 질의가 쓰인다(홀드아웃 6건 × 6원칙, < $0.01).
  판정 LLM 호출은 전부 로컬(Ollama) = 판정 API 비용 0.

  uv run python scripts/w09_poc_judge.py <ollama_model> [--limit-v 3 --limit-ok 3]

스키마 파싱 실패는 고치지 않는다 — 실패율 자체가 지표다(로컬 모델은 JSON이 흔들린다).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.judge import (  # noqa: E402
    PRINCIPLE_ORDER, PROMPT_PATH, SEED, JudgeError,
    build_context, input_hash, judge_principle_v2, load_prompt, prompt_version,
    _new_usage,
)
from rag.config import EVAL_DIR  # noqa: E402

HOLDOUT = EVAL_DIR / "w08_holdout.jsonl"
OLLAMA_URL = "http://localhost:11434/v1"


def pick_cases(limit_v: int, limit_ok: int) -> list[dict]:
    """홀드아웃에서 위반 limit_v건 + 정상 limit_ok건. target!=null이면 위반 보유."""
    cases = [json.loads(l) for l in HOLDOUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    viol = [c for c in cases if c.get("target")][:limit_v]
    clean = [c for c in cases if not c.get("target")][:limit_ok]
    return viol + clean


def judge_local(client: OpenAI, model: str, text: str, cls: dict, prompts: dict,
                version: str) -> tuple[dict, dict]:
    """홀드아웃 1건 → {원칙: verdict} + {원칙: schema_ok}. judge()의 조립을 복제하되
    검색은 OpenAI(build_context), 판정은 Ollama(judge_principle_v2)로 나눈다."""
    phase_gate = version >= "v2.1"
    usage = _new_usage()
    verdicts, schema = {}, {}
    for p in PRINCIPLE_ORDER:
        context, _ = build_context(text, p)  # OpenAI 3-large 검색 (production 고정)
        try:
            v = judge_principle_v2(
                client, text, cls, p, prompts, context,
                model=model, seed=SEED, usage=usage,
                phase_gate=phase_gate, elements_out={},
            )
            verdicts[p] = v["verdict"]
            schema[p] = True
        except JudgeError as e:
            verdicts[p] = None  # 스키마 실패 — verdict 없음 (로컬 모델 JSON 흔들림)
            schema[p] = False
            print(f"      [{p}] 스키마 실패: {str(e)[:80]}")
        except Exception as e:  # noqa: BLE001 — 호출 오류(타임아웃 등)도 셀 단위로 격리
            verdicts[p] = None
            schema[p] = False
            print(f"      [{p}] 호출 오류({type(e).__name__}): {str(e)[:80]}")
    return verdicts, schema


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Ollama 모델명 (예: exaone3.5:7.8b)")
    ap.add_argument("--limit-v", type=int, default=3)
    ap.add_argument("--limit-ok", type=int, default=3)
    args = ap.parse_args()

    cases = pick_cases(args.limit_v, args.limit_ok)
    prompts = load_prompt(PROMPT_PATH)
    version = prompt_version(PROMPT_PATH)
    # 판정 LLM만 Ollama로. 임베딩은 build_context 내부에서 별도 OpenAI() 클라이언트가
    # 처리하므로 env를 건드리지 않는다 → 검색은 3-large 유지.
    # 로컬 7B는 비한정 checks 배열을 못 닫고 폭주(runaway)한다 — 컨텍스트 한계까지 무한
    # 생성. 프롬프트·스키마는 동결이라(지시), 생성 시간 상한으로 폭주를 실패로 확정한다.
    # 240초 = 정상 완료(프롬프트 처리 ~50s + 유효 출력 ~600토큰@6t/s ~100s)는 통과,
    # 폭주는 컷. 타임아웃은 호출 오류로 기록돼 스키마 실패율에 잡힌다.
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=240, max_retries=0)

    print(f"모델: {args.model} (Ollama, CPU) / 프롬프트: {version} / 홀드아웃 {len(cases)}건")
    print(f"  위반보유 {args.limit_v} + 정상 {args.limit_ok}\n")

    matrix: dict[tuple[str, str], int] = defaultdict(int)
    schema_ok = schema_total = 0
    per_case = []
    started = time.perf_counter()

    for i, case in enumerate(cases, start=1):
        cls = {"type": case["input_type"], "product": case["product"]}  # oracle (H와 동일)
        t0 = time.perf_counter()
        verdicts, schema = judge_local(client, args.model, case["text"], cls, prompts, version)
        elapsed = time.perf_counter() - t0

        wrong = []
        for p in PRINCIPLE_ORDER:
            gold = case["labels"][p]
            pred = verdicts[p]
            schema_total += 1
            if schema[p]:
                schema_ok += 1
            if pred is not None:
                matrix[(gold, pred)] += 1
                if gold != pred:
                    wrong.append(f"{p}:{gold}→{pred}")
        print(f"[{i}/{len(cases)}] {case['id']} ({case['product']}/{case['input_type']}) "
              f"{elapsed:5.1f}초  {'✓' if not wrong else '✗ ' + '; '.join(wrong)}")
        per_case.append({"id": case["id"], "seconds": round(elapsed, 1),
                         "gold": case["labels"], "pred": verdicts, "schema": schema})

    total_time = time.perf_counter() - started
    summarize(args.model, matrix, schema_ok, schema_total, per_case, total_time, len(cases))
    return 0


def summarize(model, matrix, schema_ok, schema_total, per_case, total_time, n):
    gold_v = sum(c for (g, _), c in matrix.items() if g == "VIOLATION")
    gold_ok = sum(c for (g, _), c in matrix.items() if g == "OK")
    correct = sum(c for (g, p), c in matrix.items() if g == p)
    parsed = sum(matrix.values())
    hit_v = matrix.get(("VIOLATION", "VIOLATION"), 0)
    miss_v = matrix.get(("VIOLATION", "OK"), 0)  # 위반 누락 — 안전 지표
    false_pos = matrix.get(("OK", "VIOLATION"), 0)
    review = sum(c for (_, p), c in matrix.items() if p == "NEEDS_REVIEW")

    acc = correct / parsed if parsed else 0.0
    fpr = false_pos / gold_ok if gold_ok else float("nan")
    miss_rate = miss_v / gold_v if gold_v else float("nan")
    recall = hit_v / gold_v if gold_v else float("nan")
    schema_rate = schema_ok / schema_total if schema_total else 0.0

    print("\n" + "=" * 78)
    print(f"§7 게이트 지표 — {model}  (홀드아웃 {n}건 / {schema_total}셀)")
    print("=" * 78)
    print(f"  스키마 준수율   {schema_rate:>6.1%}  ({schema_ok}/{schema_total})  목표 100%")
    print(f"  판정 정확도     {acc:>6.1%}  (파싱된 {parsed}셀 기준)  목표 ≥75%")
    print(f"  오탐률          {fpr:>6.1%}  ({false_pos}/{gold_ok})  목표 ≤15%")
    print(f"  위반 누락률     {miss_rate:>6.1%}  ({miss_v}/{gold_v})  목표 0%  ← 안전 지표")
    print(f"  위반 재현율     {recall:>6.1%}  ({hit_v}/{gold_v})  참고")
    print(f"  NEEDS_REVIEW    {review}셀")
    print(f"  처리 속도       {total_time / n:>6.1f}초/건 (NFR-01: 30초, F/H는 병렬 12~13초)")

    out = {
        "model": model, "n_cases": n, "cells": schema_total,
        "schema_compliance": round(schema_rate, 4), "accuracy_on_parsed": round(acc, 4),
        "parsed_cells": parsed, "false_positive_rate": round(fpr, 4) if gold_ok else None,
        "violation_miss_rate": round(miss_rate, 4) if gold_v else None,
        "violation_recall": round(recall, 4) if gold_v else None,
        "needs_review_cells": review, "seconds_per_case": round(total_time / n, 1),
        "per_case": per_case,
    }
    path = EVAL_DIR / f"w09_poc_judge_{model.replace(':', '_').replace('/', '_')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n저장: {path}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
