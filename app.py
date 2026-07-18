"""RegCheck Streamlit 데모 (W10, FR-12).

문구 입력 → 6대 원칙 위반 리포트. UI는 파이프라인의 **소비자**다 — 마스킹·판정·인용
검증·게이트·로깅은 전부 agent.pipeline.check()가 하고, 여기서는 렌더링만 한다.

  uv run streamlit run app.py --server.address localhost

내부망 localhost 실행 전용 (W09 §5). 외부 공개 터널 금지.
"""

from __future__ import annotations

import re
import time

import streamlit as st

from agent import memory
from agent.judge import JUDGE_MODEL, PROMPT_PATH, input_hash, prompt_version
from agent.memory import Session, summarize
from agent.pipeline import INPUT_TYPES, PipelineError, check, suggest_input_type
from guardrails.masking import mask

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

        if v["evidence"]:
            with st.expander(f"근거 조항 {len(v['evidence'])}건", expanded=v["verdict"] == "VIOLATION"):
                for e in v["evidence"]:
                    st.markdown(f"**{e['law']} {e['article']}**")
                    st.caption(f"“{e['quote']}”")
        else:
            st.caption("근거 조항 인용 없음")

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
        flagged = groups["VIOLATION"] + groups["NEEDS_REVIEW"]
        if flagged:
            st.markdown("---")
            st.markdown("**원칙별 판정 상세**")
            for v in flagged:
                st.markdown(f"**{VERDICT_STYLE[v['verdict']][0]} {v['principle']}**")
                render_reason(v["reason"])
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
            with st.spinner("점검 중… (마스킹 → 원칙별 판정 6회 → 인용 검증 → 중복 게이트)"):
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
