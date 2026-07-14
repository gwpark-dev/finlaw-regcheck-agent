"""감사 로그 — append-only JSONL (FR-10, NFR-05/07).

전 요청을 한 줄씩 남긴다. 지우거나 고치지 않는다 — 감사 대응 근거(UC-4)다.

기록 내용은 "이 판정이 왜 그렇게 나왔는지"를 사후에 재구성할 수 있는 최소 집합이다.
- 무엇을 봤는가: 마스킹된 입력, 입력 해시, 유형·상품군
- 무엇을 근거로 삼았는가: 원칙별로 컨텍스트에 제시된 조항 ID, 판정이 인용한 조항
- 어떻게 판정했는가: 원칙별 verdict·confidence
- 무엇으로 판정했는가: model·prompt_version·seed·system_fingerprint (NFR-07)
- Guardrail이 무엇을 했는가: 마스킹 유형·건수, 차단된 인용

입력은 **마스킹된 것만** 남긴다. 원문 개인정보는 로그에 들어가지 않는다 (NFR-04).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rag.config import ROOT

AUDIT_DIR = ROOT / "data" / "audit"
AUDIT_PATH = AUDIT_DIR / "audit.jsonl"


def _article_ids(chunks: list[dict]) -> list[str]:
    """컨텍스트 청크 → 조 단위 ID 목록 (중복 제거, 순서 유지)."""
    seen, out = set(), []
    for c in chunks:
        source = Path(c["source_file"]).stem
        cid = f"{source}:{c['article']}" if c["article"] else f"{source}:본문"
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def log_check(
    report: dict,
    masked_input: str,
    chunks_by_principle: dict[str, list[dict]],
    pii_findings: list[dict],
    dropped_citations: list[dict],
    session_id: str = "-",
    path: Path = AUDIT_PATH,
) -> dict:
    """점검 1건을 감사 로그에 append. 반환: 기록된 레코드."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "input_hash": report["input_hash"],
        "masked_input": masked_input,  # 원문 아님 (NFR-04)
        "input_type": report["input_type"],
        "product_category": report["product_category"],
        "retrieved": {
            principle: _article_ids(chunks)
            for principle, chunks in chunks_by_principle.items()
        },
        "verdicts": [
            {
                "principle": v["principle"],
                "verdict": v["verdict"],
                "confidence": v["confidence"],
                "cited": [f"{e['law']} {e['article']}" for e in v["evidence"]],
            }
            for v in report["verdicts"]
        ],
        "guardrails": {
            "pii_masked": pii_findings,  # 유형·건수만, 값은 없음
            "dropped_citations": dropped_citations,
        },
        "meta": report["meta"],  # model / prompt_version / seed / system_fingerprint
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_log(path: Path = AUDIT_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
