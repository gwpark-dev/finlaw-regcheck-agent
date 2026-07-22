"""RegCheck Streamlit 데모 (W10, FR-12).

문구 입력 → 6대 원칙 위반 리포트. UI는 파이프라인의 **소비자**다 — 마스킹·판정·인용
검증·게이트·로깅은 전부 agent.pipeline.check()가 하고, 여기서는 렌더링만 한다.

  uv run streamlit run app.py --server.address localhost

내부망 localhost 실행 전용 (W09 §5). 외부 공개 터널 금지.
"""

from __future__ import annotations

import html
import json
import os
import re
import time

import streamlit as st

from agent import memory
from agent.judge import JUDGE_MODEL, PROMPT_PATH, input_hash, prompt_version
from agent.memory import Session, summarize
from agent.pipeline import INPUT_TYPES, PipelineError, check, suggest_input_type
from guardrails.masking import mask
from rag.config import META_PATH, PRINCIPLES

VERDICT_STYLE = {
    "VIOLATION": ("🔴", "#c0392b", "위반 소지"),
    "OK": ("🟢", "#27ae60", "문제 없음"),
    "NEEDS_REVIEW": ("🟡", "#e67e22", "판정 보류"),
}

# judge가 만든 사유 문자열(judge.py _decide, 동결)을 표시용으로 분해한다. judge 출력은
# 건드리지 않고 UI에서 파싱만 한다 — "구성요건 충족: [F1] 설명… (인용: "…") / [F3] …"
# 또는 "판단불가 … 필요: [E6] … / [E1] …" 형태.
_PREFIXES = ("구성요건 충족: ", "판단불가 구성요건 있음 → 사람 검토 필요: ")
_DROPPED_RE = re.compile(r"\s*\(상품군 불일치로 제외된 구성요건:\s*(.+?)\)\s*$")
_QUOTE_RE = re.compile(r'\s*\(인용:\s*"(.*)"\)\s*$', re.S)


def parse_elements(reason: str) -> tuple[list[dict] | None, str | None]:
    """사유 문자열 → (구성요건 항목 리스트|None, 상품군 제외 주석|None).

    항목: {"finding": 사람이 읽는 문장, "quote": 인용문 | ""(누락형) | None(판단불가)}.
    코드([F1] 등)는 감사용이라 여기서 떼어 반환하지 않는다(호출부가 원문에서 별도 표시).
    포맷이 아니면 (None, ...) — 호출부가 원문을 그대로 보여준다.
    """
    dropped = None
    m = _DROPPED_RE.search(reason)
    if m:
        dropped, reason = m.group(1), reason[: m.start()]

    prefix = next((p for p in _PREFIXES if reason.startswith(p)), None)
    if prefix is None:
        return None, dropped

    body = reason[len(prefix):]
    items = []
    for part in re.split(r"\s*/\s*(?=\[\w+\])", body):  # " / [코드]" 경계에서만 분리
        part = re.sub(r"^\[\w+\]\s*", "", part.strip())  # 선두 [코드] 제거(본문에선 숨김)
        qm = _QUOTE_RE.search(part)
        if qm:
            items.append({"finding": part[: qm.start()].strip(), "quote": qm.group(1)})
        else:
            items.append({"finding": part, "quote": None})
    return (items or None), dropped

st.set_page_config(page_title="RegCheck — 금소법 컴플라이언스 점검", page_icon="⚖️", layout="wide")


def _secret(key: str) -> str | None:
    """st.secrets에서 값을 읽되, secrets.toml이 없는 로컬에선 조용히 None.
    (st.secrets 접근은 파일 부재 시 예외를 던지므로 감싼다.)"""
    try:
        return st.secrets.get(key)
    except Exception:
        return None


# API 키 로드 순서: 기존 .env 방식이 우선(로컬), 없을 때만 st.secrets로 폴백(클라우드).
# judge/ingest가 호출 시점에 load_dotenv(override=False)로 .env를 읽으므로, env가 비었을
# 때만 secrets 값을 승격해 두면 로컬 .env 동작에는 영향이 없다.
if not os.environ.get("OPENAI_API_KEY"):
    _key = _secret("OPENAI_API_KEY")
    if _key:
        os.environ["OPENAI_API_KEY"] = _key


def _gate() -> None:
    """DEMO_PASSWORD가 secrets에 설정돼 있으면 비밀번호를 요구. 미설정(로컬)이면 통과."""
    expected = _secret("DEMO_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return
    st.title("⚖️ RegCheck")
    pw = st.text_input("접속 비밀번호", type="password")
    if pw == expected:
        st.session_state.authed = True
        st.rerun()
    if pw:
        st.error("비밀번호가 올바르지 않습니다.")
    st.stop()


_gate()

# 세션 1개를 Streamlit 세션 동안 유지 — 점검 이력이 같은 session_id로 쌓인다.
if "session" not in st.session_state:
    st.session_state.session = Session()
if "report" not in st.session_state:
    st.session_state.report = None
    st.session_state.meta_ui = {}


def will_hit_cache(text: str, force: bool) -> bool:
    """check()가 캐시를 반환할지 미리 판단. check() 내부와 동일한 키(마스킹된 입력
    해시, 프롬프트 버전, 모델)로 조회한다. UI는 파이프라인을 수정하지 않으므로, 캐시
    히트 여부를 이렇게 읽어서 배지로만 표시한다."""
    if force:
        return False
    masked, _ = mask(text)
    return memory.find_cached(input_hash(masked), prompt_version(PROMPT_PATH), JUDGE_MODEL) is not None


def run_check(text: str, input_type: str, force: bool) -> None:
    cache_hit = will_hit_cache(text, force)
    t0 = time.perf_counter()
    report = check(text, input_type=input_type, session=st.session_state.session, force_recheck=force)
    elapsed = time.perf_counter() - t0
    st.session_state.report = report
    st.session_state.meta_ui = {"elapsed": elapsed, "cache_hit": cache_hit, "forced": force}


# --- 근거 조항: 지식베이스 원문 직접 조회 + 위반한 호 강조 -----------------------
# 화면의 조문은 항상 지식베이스 원본이어야 한다(NFR-02 취지). LLM이 재인용한 quote는
# 쓰지 않는다. 충족된 구성요건은 조문의 특정 호에 대응하므로(judge 체크리스트 매핑) 그
# 호를 강조하고 나머지는 접는다. 파이프라인·judge는 무수정 — 조회·렌더링만 한다.

PRINCIPLE_ARTICLE = {name: f"제{num}조" for num, name in PRINCIPLES.items()}

# 구성요건 코드 → (항 기호, 호 번호). 호 열거형 조문만. 단일 항 조는 항="".
# 목(가/나) 단위 코드는 그 호를 강조한다(호의 원문 전체 = 목 포함).
ELEMENT_CLAUSE = {
    "불공정영업행위 금지": {  # 제20조 (단일 항)
        "D1": ("", "1"), "D2": ("", "2"), "D3": ("", "3"),
        "D4": ("", "4"), "D5": ("", "4"), "D6": ("", "4"), "D7": ("", "5"),
    },
    "부당권유행위 금지": {  # 제21조 (단일 항)
        "E1": ("", "1"), "E2": ("", "2"), "E3": ("", "3"), "E4": ("", "4"),
        "E5": ("", "5"), "E6": ("", "6"), "E7": ("", "6"), "E8": ("", "7"),
    },
    "광고 규제": {  # 제22조 (항별)
        "F1": ("③", "1"), "F2": ("③", "2"), "F3": ("③", "3"), "F4": ("③", "3"),
        "F5": ("③", "3"), "F6": ("③", "3"), "F7": ("②", ""),
        "F8": ("④", "2"), "F9": ("④", "1"), "F10": ("④", "4"), "F11": ("④", "3"),
    },
}


@st.cache_data
def _kb_chunks() -> list[dict]:
    """지식베이스 인덱스 메타데이터의 조문 청크 (읽기 전용, API 호출 없음)."""
    return json.loads(META_PATH.read_text(encoding="utf-8"))["chunks"]


def _norm(s: str) -> str:
    return re.sub(r"\s", "", s or "")


def _kb_article(law: str, article: str) -> list[dict]:
    """(법령명, 조) → 그 조의 KB 청크들(항 순서). 공백 무시 대조."""
    nl, na = _norm(law), _norm(article)
    return [c for c in _kb_chunks() if _norm(c["law"]) == nl and _norm(c["article"]) == na]


def _split_ho(text: str) -> tuple[str, list[tuple[str, str]]]:
    """조문/항 텍스트 → (두문, [(호번호, 호원문)]). 호는 줄머리 'N. '로 구분."""
    parts = re.split(r"\n(?=\d+\.\s)", text.strip())
    hos = []
    for p in parts[1:]:
        m = re.match(r"(\d+)\.\s", p)
        if m:
            hos.append((m.group(1), p.strip()))
    return parts[0].strip(), hos


def _strip_title(head: str) -> str:
    """두문 앞의 '제N조(제목)'을 제거(헤더에 별도 표시하므로)."""
    return re.sub(r"^제\d+조(?:의\d+)?\([^)]*\)\s*", "", head).strip()


def matched_codes(v: dict) -> set[str]:
    """VIOLATION 사유에서 충족된 구성요건 코드. 상품군 제외분·비위반은 뺀다."""
    if v["verdict"] != "VIOLATION":
        return set()
    reason = _DROPPED_RE.sub("", v["reason"])  # "(상품군 불일치로 제외…)" 제거
    if not reason.startswith("구성요건 충족: "):
        return set()
    return set(re.findall(r"\[(\w+)\]", reason))


# 조문 표시용 줄바꿈 포매터(렌더링 전용 — 인덱스 원문은 무수정).
# 목 마커: 앞이 한글/숫자가 아닌 단독 '가.'~'하.'. 문장 끝 '…다.'는 앞이 한글이라 제외.
_MOK_RE = re.compile(r"\s*(?<![가-힣0-9])([가나다라마바사아자차카타파하]\.)\s+")
# 세목 마커: '(' 뒤가 아니고 **뒤에 공백이 오는** 'N)'. 참조 '1)부터'는 공백이 없어 제외된다.
_SEMOK_RE = re.compile(r"(?<!\()(\d+\))(?=\s)")


def format_clause(text: str) -> str:
    """목(가/나/다) -> 줄바꿈+1단 들여쓰기, 세목(1) 2) 3)) -> 줄바꿈+2단 들여쓰기.
    마커는 보수적으로만 인식하고 애매하면 원문을 유지한다(오검출 방지)."""
    t = html.escape(re.sub(r"\s+", " ", text.strip()))
    t = _SEMOK_RE.sub(r"@@SUB@@\1", t)   # 세목(2단) 자리표시
    t = _MOK_RE.sub(r"@@MOK@@\1 ", t)    # 목(1단) 자리표시
    return (t.replace("@@SUB@@", "<br>&emsp;&emsp;")
             .replace("@@MOK@@", "<br>&emsp;").strip())


def _clause(text: str, highlight: bool = False) -> None:
    body = format_clause(text)
    style = "border-left:3px solid #c0392b;padding-left:8px;font-weight:600;" if highlight else ""
    st.markdown(f"<div style='margin:.3em 0;{style}'>{body}</div>", unsafe_allow_html=True)


def render_article(law: str, article: str, highlight: set) -> None:
    """조문을 KB 원문으로 렌더. highlight={(항,호)}면 그 호만 강조·나머지는 접는다."""
    chunks = _kb_article(law, article)
    st.markdown(f"**{law} {article}**"
                + (f" ({chunks[0]['article_title']})" if chunks and chunks[0].get("article_title") else ""))
    if not chunks:
        st.caption("(지식베이스에서 원문을 찾지 못함)")
        return
    for c in chunks:
        para = c.get("paragraph") or ""
        head, hos = _split_ho(c["text"])
        hi = [(n, t) for n, t in hos if (para, n) in highlight]
        para_hi = (para, "") in highlight  # 호 없는 항 두문 강조(예: F7 제2항)
        if highlight and not hi and not para_hi:
            continue  # 이 항엔 위반한 호가 없음 → 생략
        head_disp = _strip_title(head)
        if head_disp:
            _clause(head_disp, highlight=para_hi)
        for _, t in hi:
            _clause(t, highlight=True)  # 위반한 호 강조
        rest = [(n, t) for n, t in hos if (n, t) not in hi]
        if highlight and rest:
            with st.expander(f"이 {'항' if para else '조'}의 다른 호 {len(rest)}개"):
                for _, t in rest:
                    _clause(t)
        elif not highlight:  # 강조 대상 없음(정상·보류 등) → 전체 호 표시
            for _, t in hos:
                _clause(t)


def render_evidence(v: dict) -> None:
    """근거 조항을 조 단위로 묶어 KB 원문으로 표시. VIOLATION은 위반한 호 강조."""
    if not v["evidence"]:
        st.caption("근거 조항 인용 없음")
        return
    articles = {}  # 같은 조가 여러 evidence로 오면 하나로 묶는다
    for e in v["evidence"]:
        articles.setdefault((_norm(e["law"]), _norm(e["article"])), e)
    codes = matched_codes(v)
    clause = ELEMENT_CLAUSE.get(v["principle"], {})
    main = _norm(PRINCIPLE_ARTICLE.get(v["principle"], ""))
    # inline 렌더 — expander로 감싸지 않는다(render_article의 '다른 호' expander와 중첩 방지).
    st.markdown(f"**근거 조항** {len(articles)}건")
    for (nl, na), e in articles.items():
        hl = {clause[c] for c in codes if c in clause} if na == main else set()
        render_article(e["law"], e["article"], hl)


def render_reason(reason: str) -> None:
    """사유를 읽기 쉬운 불릿으로 렌더. 구성요건 코드([F1] 등)는 화면 어디에도 남기지
    않는다 — 감사용 원본은 감사 로그(JSONL)에 전부 있고, UI는 사람이 읽는 표현만 맡는다."""
    items, _ = parse_elements(reason)
    if not items:
        st.write(reason)  # 구조화 포맷이 아닌 사유(게이트 강등 등)는 원문 그대로
        return
    for it in items:
        st.markdown(f"- {it['finding']}")
        if it["quote"]:
            st.markdown(f"> {it['quote']}")  # 인용문은 인용 블록으로 구분
        elif it["quote"] == "":
            st.caption("↳ 문구에 해당 내용 없음")  # 누락형 위반


def render_verdict_card(v: dict) -> None:
    emoji, color, label = VERDICT_STYLE.get(v["verdict"], ("⚪", "#7f8c8d", v["verdict"]))
    with st.container(border=True):
        st.markdown(
            f"<div style='border-left:5px solid {color};padding-left:10px'>"
            f"<b>{emoji} {v['principle']}</b> — "
            f"<span style='color:{color};font-weight:600'>{v['verdict']} ({label})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        render_reason(v["reason"])

        render_evidence(v)

        if v["verdict"] == "NEEDS_REVIEW":
            st.info("👤 **사람 검토 필요** — 도구가 확정하지 못한 건입니다. 준법감시 담당자가 최종 판단합니다.", icon="ℹ️")
        if v["verdict"] == "VIOLATION" and v["suggestion"]:
            st.markdown(f"**수정안:** {v['suggestion']}")


def render_report() -> None:
    report = st.session_state.report
    ui = st.session_state.meta_ui
    # 심각도 순으로 묶는다 — 검수자는 문제 있는 것부터 봐야 한다.
    groups = {k: [v for v in report["verdicts"] if v["verdict"] == k]
              for k in ("VIOLATION", "NEEDS_REVIEW", "OK")}

    # 요약 지표 + 판정 상태 배지 (이모지는 icon= 한 곳에서만 — 문자열에는 넣지 않는다)
    cols = st.columns([1, 1, 1, 3])
    cols[0].metric("위반 소지", len(groups["VIOLATION"]))
    cols[1].metric("검토 필요", len(groups["NEEDS_REVIEW"]))
    cols[2].metric("문제 없음", len(groups["OK"]))
    with cols[3]:
        if ui.get("cache_hit"):
            st.success(f"캐시된 판정 (동일 문구 재검사) · {ui['elapsed']*1000:.0f}ms — 동일 문구엔 동일 판정", icon="⚡")
        elif ui.get("forced"):
            st.warning(f"재판정 (force_recheck) · {ui['elapsed']:.1f}초 — 감사 로그에 기록됨", icon="♻️")
        else:
            st.info(f"신규 판정 · {ui['elapsed']:.1f}초", icon="🆕")

    st.caption(f"유형: **{report['input_type']}** · 상품군: **{report['product_category']}**")

    # 기술 정보는 검수 화면에서 접어둔다 — 사람이 읽는 요약만 (NFR-07)
    with st.expander("판정 정보"):
        st.markdown(f"**판정 모델:** `{report['meta']['model']}` — 위반 여부를 판단한 AI 모델")
        st.markdown(f"**판정 기준 버전:** `{report['meta']['prompt_version']}` — 같은 버전에서는 같은 문구에 같은 결과 보장")
        st.markdown(f"**문구 식별번호:** `{report['input_hash'][:12]}…` — 감사 기록에서 이 점검을 찾을 때 쓰는 번호")
        st.caption("감사 기록 원본은 로그 파일 참조")

    st.divider()

    # 위반·검토 섹션은 항상 펼침, 문제 없음은 접어서 개수만 — 카드 없는 섹션은 숨김
    if groups["VIOLATION"]:
        st.markdown("#### 🔴 위반 소지")
        for v in groups["VIOLATION"]:
            render_verdict_card(v)
    if groups["NEEDS_REVIEW"]:
        st.markdown("#### 🟡 검토 필요")
        for v in groups["NEEDS_REVIEW"]:
            render_verdict_card(v)
    if groups["OK"]:
        with st.expander(f"🟢 문제 없음 {len(groups['OK'])}건"):
            for v in groups["OK"]:
                render_verdict_card(v)

    st.divider()
    if st.button("♻️ 재판정 (force_recheck)", help="캐시를 무시하고 다시 판정합니다. 이 사실은 감사 로그에 남습니다."):
        with st.spinner("재판정 중… (원칙별 판정 6회)"):
            run_check(st.session_state.last_text, st.session_state.last_type, force=True)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
st.title("⚖️ RegCheck")
st.caption("금소법 6대 판매규제 컴플라이언스 1차 점검 — AI 스크리닝, 최종 판단은 사람 (Human-in-the-loop)")

tab_check, tab_history = st.tabs(["🔍 점검", "📜 점검 이력"])

with tab_check:
    text = st.text_area(
        "점검할 광고/상담 문구",
        height=140,
        placeholder="예) 이 상품은 원금이 보장되고 수익률도 확실합니다. 지금 가입하세요!",
        key="input_text",
    )

    # 유형: 분류기 제안을 기본값으로, 사용자 선택이 우선 (ADR-008)
    suggested = suggest_input_type(text) if text.strip() else {"type": "광고", "product": "-"}
    default_idx = INPUT_TYPES.index(suggested["type"]) if suggested["type"] in INPUT_TYPES else 0
    c1, c2 = st.columns([1, 2])
    with c1:
        input_type = st.radio("유형 (필수)", INPUT_TYPES, index=default_idx, horizontal=True)
    with c2:
        st.write("")
        if text.strip():
            st.caption(f"분류기 제안: 유형 **{suggested['type']}** · 상품군 **{suggested['product']}** — 제안은 참고용이며 유형 선택은 검수자가 확정합니다")
            if input_type != suggested["type"]:
                st.warning(f"⚠️ 분류기 제안({suggested['type']})과 선택({input_type})이 다릅니다. 선택하신 값으로 판정합니다.", icon="⚠️")

    if st.button("🔍 검사", type="primary", disabled=not text.strip()):
        st.session_state.last_text, st.session_state.last_type = text, input_type
        try:
            with st.spinner("점검 중…"):
                run_check(text, input_type, force=False)
        except PipelineError as e:
            st.error(f"파이프라인 오류: {e}")

    if st.session_state.report is not None:
        st.divider()
        render_report()

with tab_history:
    st.subheader("최근 점검 이력")
    n = st.slider("표시 건수", 3, 20, 5)
    history = memory.recent(n)
    if not history:
        st.info("아직 점검 이력이 없습니다.")
    for rec in history:
        rep = rec["report"]
        with st.container(border=True):
            st.markdown(f"**{rec['timestamp'][:19].replace('T', ' ')}** · `{rec['input_hash'][:12]}…` · {rep['input_type']}/{rep['product_category']}")
            st.caption(summarize(rep))
