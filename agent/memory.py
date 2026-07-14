"""점검 이력 Memory (FR-06).

두 층으로 나뉜다.

- **세션 메모리**: 한 번의 대화 동안 오간 점검 결과를 들고 있다. 프로세스 안에만
  존재하며, 최근 점검 결과를 다시 참조할 수 있게 한다.
- **점검 이력**: data/history/history.jsonl에 append-only로 쌓는다 (NFR-05).
  타임스탬프와 입력 해시를 포함하고, 지우거나 고치지 않는다 — 감사 대응 근거(UC-4).

원문 문구는 파일에 저장하지 않는다. 개인정보 마스킹(W08, FR-07)이 아직 없어
"마스킹되지 않은 입력은 로그에 남기지 않는다"는 불변 조건(NFR-04)을 지키려면
지금은 해시만 남기는 것이 맞다. 원문은 세션 메모리(프로세스 내)에만 둔다.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag.config import ROOT

HISTORY_DIR = ROOT / "data" / "history"
HISTORY_PATH = HISTORY_DIR / "history.jsonl"


# --- 점검 이력 (파일) ---------------------------------------------------------


def append_history(report: dict, session_id: str = "-") -> dict:
    """리포트 1건을 이력에 append. 반환: 기록된 레코드."""
    record = {
        "session_id": session_id,
        "timestamp": report["meta"]["timestamp"],
        "input_hash": report["input_hash"],
        "report": report,
    }
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def recent(n: int = 5) -> list[dict]:
    """최근 점검 N건 (최신순)."""
    return load_history()[-n:][::-1]


def find_by_hash(input_hash: str) -> list[dict]:
    """입력 해시로 과거 점검 이력 조회. 같은 문구를 다시 점검했으면 여러 건이 나온다."""
    return [r for r in load_history() if r["input_hash"] == input_hash]


# --- 판정 캐시 (FR-09, ADR-007) ----------------------------------------------
#
# LLM은 temperature=0 + seed 고정에도 같은 입력을 다르게 판정한다(실측 1.4~3.2% 뒤집힘).
# 모델의 비결정성은 없앨 수 없으므로, 도구의 출력을 결정적으로 만든다 — 같은 (문구,
# 프롬프트 버전, 모델)이면 저장된 판정을 그대로 돌려준다. NFR-07의 충족 지점을
# "모델"에서 "도구"로 옮긴 것이다. 재판정은 명시적 요청으로만 하고 이력에 남는다.


def find_cached(input_hash: str, prompt_version: str, model: str) -> dict | None:
    """같은 입력·프롬프트·모델의 가장 최근 판정. 없으면 None.

    프롬프트나 모델이 바뀌면 키가 달라져 캐시가 자동으로 무효화된다.
    """
    for record in reversed(load_history()):
        if record["input_hash"] != input_hash:
            continue
        meta = record["report"]["meta"]
        if meta.get("prompt_version") == prompt_version and meta.get("model") == model:
            return record["report"]
    return None


# --- 세션 메모리 --------------------------------------------------------------


def summarize(report: dict) -> str:
    """리포트 1건 → 한 줄 요약. 세션 맥락으로 다시 읽히는 형태다."""
    flagged = [v["principle"] for v in report["verdicts"] if v["verdict"] == "VIOLATION"]
    held = [v["principle"] for v in report["verdicts"] if v["verdict"] == "NEEDS_REVIEW"]
    parts = []
    if flagged:
        parts.append(f"위반 소지: {', '.join(flagged)}")
    if held:
        parts.append(f"보류: {', '.join(held)}")
    return " / ".join(parts) if parts else "위반 소지 없음"


class Session:
    """세션 내 대화 맥락. 점검할 때마다 결과를 쌓고, 이력 파일에도 함께 남긴다."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        # 원문(text)은 여기(프로세스 메모리)에만 둔다 — 파일에는 해시만 남는다.
        self.turns: list[dict] = []

    def record(self, text: str, report: dict) -> dict:
        """점검 결과를 세션에 쌓고 이력 파일에 append."""
        self.turns.append({"text": text, "report": report})
        return append_history(report, session_id=self.session_id)

    def recent(self, n: int = 3) -> list[dict]:
        """세션 내 최근 점검 N건 (최신순)."""
        return self.turns[-n:][::-1]

    def context(self, n: int = 3) -> str:
        """최근 점검 결과를 텍스트 맥락으로. "방금 그 문구" 같은 참조에 쓴다."""
        if not self.turns:
            return "(이 세션에서 점검한 문구가 아직 없습니다)"
        lines = []
        for i, turn in enumerate(self.recent(n), start=1):
            snippet = " ".join(turn["text"].split())[:60]
            lines.append(f"{i}. \"{snippet}…\" → {summarize(turn['report'])}")
        return "\n".join(lines)
