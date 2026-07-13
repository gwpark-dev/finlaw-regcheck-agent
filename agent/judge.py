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

# temperature=0으로도 판정이 재현되지 않았다(동일 30문항 재실행 시 180셀 중 10셀 뒤집힘).
# seed를 고정하고 응답의 system_fingerprint를 meta에 남겨, 결과가 달라졌을 때
# "모델 쪽이 바뀐 것"인지 구분할 수 있게 한다 (NFR-07).
SEED = 20260713

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


# --- v2.0: 2단 구성요건 판정 --------------------------------------------------
#
# 1단(LLM): 조문이 열거한 구성요건을 항목별로 해당/미해당/판단불가 판정 + 근거 문구 인용.
# 2단(코드): 1단 결과'만'으로 verdict를 결정한다. LLM에게 verdict를 묻지 않는다.
#
# v1.0의 실패는 모델이 "문구가 나쁘다 → 이 원칙도 위반" 식으로 하나의 행위를 6개 원칙에
# 돌려쓴 것이었다. verdict를 코드가 계산하면 그 경로가 구조적으로 막힌다 — 모델이 특정
# 구성요건을 문구 인용과 함께 지목하지 못하면 VIOLATION이 나올 수 없다.

CHECK_STATUS = ("해당", "미해당", "판단불가")

# 조문이 상품군을 명시해 한정한 구성요건. 해당 상품군이 아니면 성립할 수 없으므로
# 2단(코드)에서 강제로 미해당 처리한다 — 모델이 "보장성 광고 요건 누락"을 펀드 광고에
# 갖다 붙이는 오답이 실측됐다. 조문의 문언을 코드로 옮긴 것이지 휴리스틱이 아니다.
#   투자성=펀드 / 보장성=보험 / 대출성=대출 / 예금성=예금
ELEMENT_PRODUCT_SCOPE = {
    "D1": {"대출"},   # 꺾기 — 대출성 상품 계약체결과 관련
    "D4": {"대출"},   # 대출 상환방식 강요
    "D5": {"대출"},   # 중도상환수수료
    "D6": {"대출"},   # 개인 대출 제3자 연대보증
    "E5": {"보험"},   # 제21조5호 — 보장성 상품의 경우
    "E6": {"펀드"},   # 제21조6호가 — 투자성 상품의 경우 (불초청 권유)
    "E7": {"펀드"},   # 제21조6호나 — 투자성 상품의 경우 (거부 후 재권유)
    "F3": {"펀드"},   # 제22조3항3호나 — 투자성
    "F4": {"보험"},   # 제22조3항3호가 — 보장성
    "F5": {"대출"},   # 제22조3항3호라 — 대출성
    "F6": {"예금"},   # 제22조3항3호다 — 예금성
    "F8": {"펀드"},   # 제22조4항2호 — 투자성
    "F9": {"보험"},   # 제22조4항1호 — 보장성
    "F10": {"대출"},  # 제22조4항4호 — 대출성
    "F11": {"예금"},  # 제22조4항3호 — 예금성
}

V2_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "string",
            "description": "이 문구가 해당 원칙의 적용 국면인지에 대한 판단 근거.",
        },
        "applicable": {
            "type": "boolean",
            "description": "적용 국면이면 true. false면 verdict는 OK로 확정된다.",
        },
        "checks": {
            "type": "array",
            "description": "제시된 구성요건 목록 전부에 대해 각각 하나씩. 빠뜨리지 말 것.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "finding": {"type": "string"},
                    "quote": {
                        "type": "string",
                        "description": "근거가 되는 문구를 원문 그대로. 없으면 빈 문자열.",
                    },
                    "status": {"type": "string", "enum": list(CHECK_STATUS)},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "finding", "quote", "status", "confidence"],
                "additionalProperties": False,
            },
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
        "suggestion": {"type": "string"},
    },
    "required": ["scope", "applicable", "checks", "evidence", "suggestion"],
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


def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema: dict,
    name: str,
    seed: int | None,
    principle: str,
    usage: dict,
) -> dict:
    """LLM 1회 호출 → 검증된 JSON. 토큰·fingerprint를 usage에 누적한다."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # 판정은 재현 가능해야 한다 (NFR-07)
        seed=seed,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    )
    u = response.usage
    usage["prompt_tokens"] += u.prompt_tokens
    usage["completion_tokens"] += u.completion_tokens
    if response.system_fingerprint:
        usage["fingerprints"].add(response.system_fingerprint)

    try:
        return json.loads(response.choices[0].message.content)
    except (TypeError, json.JSONDecodeError) as e:
        raise JudgeError(f"[{principle}] JSON 파싱 실패: {e}") from e


def judge_principle(
    client: OpenAI,
    text: str,
    cls: dict,
    principle: str,
    prompts: dict[str, str],
    context: str,
    model: str = JUDGE_MODEL,
    seed: int | None = SEED,
    usage: dict | None = None,
) -> dict:
    """v1.0 — 원칙 1개 판정 → 명세서 §7.3 verdicts[] 항목 1개."""
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
    data = _call(
        client, model, prompts["SYSTEM"], user, VERDICT_SCHEMA,
        "principle_verdict", seed, principle, usage if usage is not None else _new_usage(),
    )
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


def judge_principle_v2(
    client: OpenAI,
    text: str,
    cls: dict,
    principle: str,
    prompts: dict[str, str],
    context: str,
    model: str = JUDGE_MODEL,
    seed: int | None = SEED,
    usage: dict | None = None,
) -> dict:
    """v2.0 — 1단(LLM 구성요건 판정) → 2단(코드 verdict 결정)."""
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
    data = _call(
        client, model, prompts["SYSTEM"], user, V2_SCHEMA,
        "principle_elements", seed, principle, usage if usage is not None else _new_usage(),
    )
    _validate_v2(data, principle)
    return _decide(data, principle, cls["product"])


def _validate_v2(data: dict, principle: str) -> None:
    for key in V2_SCHEMA["required"]:
        if key not in data:
            raise JudgeError(f"[{principle}] 필드 누락: {key}")
    if not isinstance(data["applicable"], bool):
        raise JudgeError(f"[{principle}] applicable이 bool이 아님")
    for c in data["checks"]:
        if c["status"] not in CHECK_STATUS:
            raise JudgeError(f"[{principle}] 알 수 없는 status: {c['status']}")


def _out_of_scope(check: dict, product: str) -> bool:
    """조문이 상품군을 한정한 구성요건인데 상품군이 다르면, 성립할 수 없다."""
    scope = ELEMENT_PRODUCT_SCOPE.get(check["id"])
    # 상품군을 모르면(기타) 억제하지 않는다 — 판단을 모델에게 남긴다.
    return bool(scope) and product != "기타" and product not in scope


def _decide(data: dict, principle: str, product: str) -> dict:
    """2단 — 1단의 구성요건 판정 결과'만' 보고 verdict를 계산한다.

    LLM은 여기에 관여하지 않는다. 원문을 다시 읽지 않으므로 "문구가 나쁘니 이 원칙도
    위반"이라는 비약이 구조적으로 불가능하다.
    """
    checks = data["checks"]
    suppressed = [c for c in checks if _out_of_scope(c, product)]
    checks = [c for c in checks if not _out_of_scope(c, product)]
    matched = [c for c in checks if c["status"] == "해당"]
    unclear = [c for c in checks if c["status"] == "판단불가"]

    def conf(items: list[dict], default: float) -> float:
        return max((c["confidence"] for c in items), default=default)

    if not data["applicable"]:
        verdict = "OK"
        confidence = 1.0
        reason = f"이 원칙의 적용 국면이 아님 — {data['scope']}"
    elif matched:
        verdict = "VIOLATION"
        confidence = conf(matched, 0.5)
        parts = [f"[{c['id']}] {c['finding']} (인용: \"{c['quote']}\")" for c in matched]
        reason = "구성요건 충족: " + " / ".join(parts)
    elif unclear:
        verdict = "NEEDS_REVIEW"
        confidence = conf(unclear, 0.5)
        parts = [f"[{c['id']}] {c['finding']}" for c in unclear]
        reason = "판단불가 구성요건 있음 → 사람 검토 필요: " + " / ".join(parts)
    else:
        verdict = "OK"
        confidence = min((c["confidence"] for c in checks), default=1.0)
        reason = "적용 국면이나 조문이 열거한 구성요건 중 해당하는 것이 없음"

    dropped = [c["id"] for c in suppressed if c["status"] == "해당"]
    if dropped:
        reason += f" (상품군 불일치로 제외된 구성요건: {', '.join(dropped)} — {product} 상품에 적용되지 않음)"

    # 명세서 §7.3: evidence가 비면 판정 보류 (NFR-02/03)
    evidence = data["evidence"]
    if not evidence:
        verdict = "NEEDS_REVIEW"
        reason = f"근거 조항 인용 없음 → 판정 보류. (사유: {reason})"

    return {
        "principle": principle,
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "reason": reason,
        "suggestion": data["suggestion"] if verdict == "VIOLATION" else "",
    }


def _new_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "fingerprints": set()}


def judge(
    text: str,
    parallel: bool = True,
    model: str = JUDGE_MODEL,
    prompt_path: Path = PROMPT_PATH,
    cls_override: dict | None = None,
    seed: int | None = SEED,
) -> dict:
    """문구 1건 → 6대 원칙 판정 리포트 (명세서 §7.3 스키마).

    parallel=False면 원칙별 호출을 순차 실행한다(디버깅·지연 측정용).
    prompt_path의 버전이 v2로 시작하면 2단 구성요건 판정 구조를 쓴다.
    cls_override: 유형 분류를 주입한다. 분류기 오류가 판정에 미치는 영향을 분리 측정할 때 쓴다.
    """
    load_dotenv(ROOT / ".env")  # NFR-06: 키는 .env에서만
    client = OpenAI()
    prompts = load_prompt(prompt_path)
    version = prompt_version(prompt_path)
    cls = cls_override or classify(text)
    judge_one = judge_principle_v2 if version.startswith("v2") else judge_principle

    # 컨텍스트 구성(= 검색)은 순차로 돌린다. 검색은 질의를 임베딩하며 임베딩 캐시
    # 파일을 갱신하는데, 이를 병렬로 부르면 스레드들이 같은 파일을 동시에 덮어써
    # 캐시가 깨진다(실제로 겪음). 병렬화가 필요한 건 느린 쪽 — LLM 호출뿐이다.
    contexts = {p: build_context(text, p)[0] for p in PRINCIPLE_ORDER}
    usage = _new_usage()

    def run(principle: str) -> dict:
        return judge_one(
            client, text, cls, principle, prompts, contexts[principle],
            model=model, seed=seed, usage=usage,
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
            "model": model,
            "embedding": EMBEDDING_MODEL,
            "prompt_version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "system_fingerprint": sorted(usage["fingerprints"]),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
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
