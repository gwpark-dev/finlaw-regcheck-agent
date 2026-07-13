"""6대 원칙 판정 엔진 (FR-04, FR-05).

ADR-005 구조 — 원칙마다 [본법 조항 고정 주입 + 하위규정 유사도 보강] 컨텍스트를
만들고, 원칙당 1회씩 개별 LLM 호출로 판정한다. 임베딩 검색이 약한 부분(본법)은
결정적 주입으로 대체하고, 검색은 잘하는 일(하위규정 발굴)만 시킨다.

  uv run python -m agent.judge "이 상품은 원금이 보장되고 수익률도 확실합니다"
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent.classifier import classify
from rag.config import EMBEDDING_MODEL, PRINCIPLES, ROOT
from rag.retriever import chunks_by_principle, search

JUDGE_MODEL = "gpt-4o-mini"
PROMPT_PATH = ROOT / "agent" / "prompts" / "judge_v1.0.md"

# 하위규정 보강 검색: 청크 SEARCH_DEPTH개를 뽑아 조(條) 단위로 접은 뒤 상위 SUPPLEMENT_K개.
# 한 조가 여러 항으로 쪼개져 있어(제22조는 7개 항) 청크를 그대로 쓰면 형제 항들이
# 자리를 다 차지한다.
SUPPLEMENT_K = 5
SEARCH_DEPTH = 40

# 원칙별 호출은 서로 독립이라 병렬로 돌린다. 순차 실행은 실측 56초로 NFR-01(30초)을
# 넘겼고, 병렬은 23초였다. ADR-005가 예고한 "지연 누적 시 병렬 전환" 지점.
MAX_WORKERS = 6

PRINCIPLE_ORDER = [PRINCIPLES[num] for num in sorted(PRINCIPLES)]
ARTICLE_OF = {name: f"제{num}조" for num, name in PRINCIPLES.items()}

VERDICTS = ("VIOLATION", "OK", "NEEDS_REVIEW")

# 필드 순서가 곧 모델의 사고 순서다(JSON을 앞에서부터 생성하므로).
# verdict를 맨 앞에 두면 근거를 만들기도 전에 판정을 확정해 버려, 문구가 나빠 보이면
# 6개 원칙을 전부 VIOLATION으로 찍는 오탐이 난다(실측). 그래서
# 적용 대상 판단 → 근거 선별 → 이유 → 판정 순으로 강제한다.
# principle은 호출부가 아는 값이라 모델에게 묻지 않는다.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "applicability": {
            "type": "string",
            "description": "이 문구가 해당 원칙의 적용 대상 상황인지 먼저 판단한 근거. 적용 대상이 아니면 verdict는 OK.",
        },
        "matched_conduct": {
            "type": "string",
            "description": (
                "이 원칙의 조문에 열거된 행위 유형 중 문구가 실제로 해당하는 것을 적고, "
                "그 근거가 되는 문구를 그대로 인용한다. 해당하는 행위가 없으면 빈 문자열 — "
                "그 경우 verdict는 OK."
            ),
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "law": {"type": "string"},
                    "article": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["law", "article", "quote"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "confidence": {"type": "number"},
        "suggestion": {"type": "string"},
    },
    "required": [
        "applicability",
        "matched_conduct",
        "evidence",
        "reason",
        "verdict",
        "confidence",
        "suggestion",
    ],
    "additionalProperties": False,
}


class JudgeError(RuntimeError):
    """LLM 응답이 스키마를 지키지 않음. 평가 스크립트가 스키마 준수율로 집계한다."""


# --- 프롬프트 ----------------------------------------------------------------

_SECTION_RE = re.compile(r"^# (.+)$", re.M)


def load_prompt(path: Path = PROMPT_PATH) -> dict[str, str]:
    """버전 파일을 "# 섹션명" 단위로 파싱. SYSTEM / USER / CHECKLIST:<원칙명>"""
    parts = _SECTION_RE.split(path.read_text(encoding="utf-8"))
    # parts[0]은 첫 헤더 이전(주석 블록), 이후 [이름, 본문, 이름, 본문, ...]
    sections = {
        parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)
    }
    missing = [p for p in PRINCIPLE_ORDER if f"CHECKLIST:{p}" not in sections]
    if missing or "SYSTEM" not in sections or "USER" not in sections:
        raise JudgeError(f"프롬프트 파일에 섹션이 없습니다: {path.name} ({missing})")
    return sections


def prompt_version(path: Path = PROMPT_PATH) -> str:
    """judge_v1.0.md → "v1.0" (NFR-07: 리포트 meta에 기록)"""
    return path.stem.split("_")[-1]


def render(template: str, **values: str) -> str:
    # str.format을 쓰면 프롬프트 본문의 중괄호가 깨진다.
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


# --- 컨텍스트 구성 (ADR-005) --------------------------------------------------


def _format_chunk(chunk: dict) -> str:
    """법령명·조항번호를 라벨로 분리해 제시한다.

    모델이 evidence.law/article에 이 값을 그대로 복사하게 만들기 위함이다. 인용 문자열이
    정규화돼 있어야 W08 인용 검증 Guardrail이 색인과 대조할 수 있다.
    """
    article = chunk["article"] or "본문"
    title = f" ({chunk['article_title']})" if chunk["article_title"] else ""
    return (
        f"법령: {chunk['law']}\n"
        f"조항: {article}{title}\n"
        f"원문: {chunk['text']}"
    )


def _top_articles(hits: list[dict], k: int) -> list[dict]:
    """청크 검색 결과를 조(條) 단위로 접는다 — 조마다 최고 점수 청크만 대표로."""
    seen: set[tuple[str, str | None]] = set()
    out: list[dict] = []
    for hit in hits:  # search()가 이미 유사도 내림차순으로 준다
        key = (hit["source_file"], hit["article"])
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) == k:
            break
    return out


def build_context(text: str, principle: str) -> tuple[str, list[dict]]:
    """원칙 하나의 판정 컨텍스트. 반환: (프롬프트에 넣을 문자열, 사용된 청크 목록)

    본법은 검색 없이 태그로 전량 주입하고(누락 0%), 유사도 검색은 하위규정 발굴에만
    쓴다. 주입한 청크는 검색 결과에서 제외해 중복을 막는다.

    보강 검색 질의에 원칙명을 붙이는 이유: 입력 문구만으로 검색하면 6개 원칙이 모두
    같은 하위규정을 받는다. 꺾기 문구로 '광고 규제'를 판정할 때 불공정영업 조항이
    컨텍스트에 들어오면 모델이 그쪽에 끌려가 오탐이 난다(실측). 원칙명을 붙이면 해당
    원칙의 규제 체인(ADR-004)이 올라온다 — 광고 규제 → 시행령 제19·20조, 감독규정 제17·19조.
    """
    base = chunks_by_principle(principle)
    if not base:
        raise JudgeError(f"원칙 태그 청크가 색인에 없습니다: {principle}")

    injected = {c["chunk_id"] for c in base}
    query = f"{principle}\n{text}"
    hits = search(query, k=SEARCH_DEPTH, exclude_chunk_ids=injected)
    supplement = _top_articles(hits, SUPPLEMENT_K)

    lines = ["[본법 — 이 원칙의 정의 조항, 항상 제공]"]
    lines += [_format_chunk(c) for c in base]
    lines.append("")
    lines.append("[하위규정 — 유사도 검색 결과, 무관한 조항이 섞여 있을 수 있음]")
    for i, hit in enumerate(supplement, start=1):
        lines.append(f"({i}) [유사도 {hit['score']:.3f}]\n{_format_chunk(hit)}")

    return "\n\n".join(lines), base + supplement


# --- 판정 --------------------------------------------------------------------


def _validate(data: dict, principle: str) -> None:
    """모델이 강제 스키마를 벗어나지 않았는지 확인. 스키마 준수율의 측정 근거다."""
    for key in VERDICT_SCHEMA["required"]:
        if key not in data:
            raise JudgeError(f"[{principle}] 필드 누락: {key}")
    if data["verdict"] not in VERDICTS:
        raise JudgeError(f"[{principle}] 알 수 없는 verdict: {data['verdict']}")
    if not isinstance(data["confidence"], (int, float)):
        raise JudgeError(f"[{principle}] confidence가 숫자가 아님")
    for item in data["evidence"]:
        if not all(k in item for k in ("law", "article", "quote")):
            raise JudgeError(f"[{principle}] evidence 항목 필드 누락")


def judge_principle(
    client: OpenAI,
    text: str,
    cls: dict,
    principle: str,
    prompts: dict[str, str],
    context: str,
) -> dict:
    """원칙 1개 판정 → 명세서 §7.3 verdicts[] 항목 1개."""
    user = render(
        prompts["USER"],
        principle=principle,
        article=ARTICLE_OF[principle],
        checklist=prompts[f"CHECKLIST:{principle}"],
        input_type=cls["type"],
        product=cls["product"],
        context=context,
        text=text,
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,  # 판정은 재현 가능해야 한다 (NFR-07)
        messages=[
            {"role": "system", "content": prompts["SYSTEM"]},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "principle_verdict",
                "strict": True,
                "schema": VERDICT_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError) as e:
        raise JudgeError(f"[{principle}] JSON 파싱 실패: {e}") from e
    _validate(data, principle)

    # 명세서 §7.3: evidence가 비면 verdict는 자동으로 NEEDS_REVIEW (NFR-02/03 강제).
    # 근거 없는 확신은 이 프로젝트가 막으려는 실패 그 자체다.
    if not data["evidence"]:
        data["verdict"] = "NEEDS_REVIEW"
        data["reason"] = f"근거 조항 인용 없음 → 판정 보류. (모델 사유: {data['reason']})"

    # applicability·matched_conduct는 모델을 먼저 생각하게 만드는 발판일 뿐,
    # 리포트 스키마(§7.3)에는 없다.
    return {
        "principle": principle,
        "verdict": data["verdict"],
        "confidence": data["confidence"],
        "evidence": data["evidence"],
        "reason": data["reason"],
        "suggestion": data["suggestion"],
    }


def judge(text: str, parallel: bool = True) -> dict:
    """문구 1건 → 6대 원칙 판정 리포트 (명세서 §7.3 스키마).

    parallel=False면 원칙별 호출을 순차 실행한다(디버깅·지연 측정용).
    """
    load_dotenv(ROOT / ".env")  # NFR-06: 키는 .env에서만
    client = OpenAI()
    prompts = load_prompt()
    cls = classify(text)

    # 컨텍스트 구성(= 검색)은 순차로 돌린다. 검색은 질의를 임베딩하며 임베딩 캐시
    # 파일을 갱신하는데, 이를 병렬로 부르면 스레드들이 같은 파일을 동시에 덮어써
    # 캐시가 깨진다(실제로 겪음). 병렬화가 필요한 건 느린 쪽 — LLM 호출뿐이다.
    contexts = {p: build_context(text, p)[0] for p in PRINCIPLE_ORDER}

    def run(principle: str) -> dict:
        return judge_principle(
            client, text, cls, principle, prompts, contexts[principle]
        )

    if parallel:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            verdicts = list(pool.map(run, PRINCIPLE_ORDER))
    else:
        verdicts = [run(p) for p in PRINCIPLE_ORDER]

    return {
        "input_hash": input_hash(text),
        "input_type": cls["type"],
        "product_category": cls["product"],
        "verdicts": verdicts,
        "meta": {  # NFR-07 재현성
            "model": JUDGE_MODEL,
            "embedding": EMBEDDING_MODEL,
            "prompt_version": prompt_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- CLI ---------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    from agent.memory import Session, summarize

    text = sys.argv[1]
    report = judge(text)
    session = Session()
    session.record(text, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n요약: {summarize(report)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
