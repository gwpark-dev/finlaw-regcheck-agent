# CLAUDE.md — RegCheck

Project-specific context for Claude Code. Merges with the global guidelines at `~/.claude/CLAUDE.md` (those define general engineering principles; this file defines project-specific information only).

---

## 1. Project Identity

RegCheck — 금소법(금융소비자보호법) compliance-checking RAG agent. Takes financial-product ad copy or consultation scripts and flags potential violations of the 6대 판매규제, citing the exact legal articles as evidence and suggesting revisions. Internal (B2B) tool concept; AI does first-pass screening, humans make the final call.

University course project (AI 핀테크 Agent 분석과 설계), 5-week build: W06–W10.

**Primary spec**: `docs/RegCheck_프로젝트명세서.md` — the top-priority reference. Modify only with explicit reasoning. No silent drift from spec.

**The one non-negotiable** (spec NFR-02/03): every violation verdict MUST cite retrieved article chunks. No citation → verdict is `NEEDS_REVIEW`, never a confident judgment. A cited article must exist in the knowledge base — fabricated citations are the failure mode this project exists to prevent.

**Weekly roadmap** (DoD details in spec §10):

| Week | Scope | Status |
|------|-------|--------|
| W06 | Law PDF chunking/embedding → FAISS RAG | **Done** — recall@5 = 1.00 (10 queries, target 0.80) |
| W07 | 6-principle verdict engine (Tool) + inspection-history Memory | **Done** — judge v2.1 + gpt-4.1 (ADR-006). 평가셋 31건/186셀: 정확도 **0.903**, 오탐률 **0.096**, 재현율 1.00, 스키마 100%, 8~13초/건. 홀드아웃 12건에서도 목표 유지(0.917 / 0.076) — 과적합 없음 |
| W08 ★ | Guardrails (PII masking, no-evidence hold, false-positive control) + audit logging | **Done** — DoD 3항목 통과(마스킹 100%, 지식베이스 외 인용 0건, 전 요청 로그). 인용 중복 게이트로 오탐률 0.096→**0.018**(홀드아웃 0.076→**0.000**). 판정 캐시로 NFR-07 충족 |
| W09 | On-premise architecture design doc | **Done** — 설계 문서(3티어·경계·로그 정책) + 전환 PoC 실측(ADR-009). 임베딩 로컬 대체 가능(bge-m3 recall@5 **0.90**), 판정 LLM은 **CPU 서빙 불가**(qwen2.5:7b가 judge_v2.1 출력 폭주 → OOM) → **Tier 2는 GPU 전제** |
| W10 | Streamlit demo | **Done** — `app.py`(check() 소비자, 파이프라인 무수정). 입력→6원칙 리포트 E2E, 유형 필수(ADR-008)·캐시 배지·force_recheck·이력 탭. 시연 3종 검증(위반=광고 규제 VIOLATION / 정상=전원칙 OK / 보류=게이트 4원칙 NEEDS_REVIEW). 대본 `docs/W10_데모_시나리오.md`, 문구 `data/demo/` |

**W07 → W08 인계 항목 (처리 결과)**

1. ~~재현성 (NFR-07 미충족)~~ → **해소 (ADR-007)**. LLM의 비결정성(1.4~3.2% 뒤집힘)은
   없앨 수 없으므로 **판정 캐시**로 도구의 출력을 결정적으로 만들었다 — 같은 (입력 해시,
   프롬프트 버전, 모델)이면 저장된 판정을 반환. NFR-07의 충족 지점을 "모델"에서 "도구"로
   옮긴 것이다. 재판정은 `force_recheck`로만, 그 사실도 감사 로그에 남는다.
2. ~~classifier–게이트 결합 위험~~ → **해소 (ADR-008)**. classifier의 광고/상담 정확도가
   0.535로 "무조건 상담"(0.837)보다 낮고 오분류가 전부 게이트를 여는 방향이었다.
   **`input_type`을 사용자 필수 입력으로** 바꾸고 classifier는 상품군 분류·기본값 제안으로
   역할을 축소했다. (W05 하이브리드 분류기 이식 계획은 폐기)
3. **부당권유 원칙의 구성요건 오지정** → **알려진 한계로 확정 (아래)**.

---

## 알려진 한계 (Known Limitations)

의도적으로 남긴 것들이다. 고치려 들지 말고, 새로 발견한 것처럼 보고하지도 말 것.

### 1. 부당권유 원칙의 구성요건 오지정 — 수정하지 않는다

6대 원칙 중 부당권유(제21조)가 가장 약하다(정확도 0.74~0.75). 모델이 E3(가치에 중대한
영향을 미치는 사항 미고지)·E8(적합성 회피)을 포괄 조항처럼 끌어다 쓴다.

**W09/W10에서 다루지 않기로 결정했다.** 인용 중복 게이트(FR-09)가 실무 영향을 이미
흡수했고(오탐 16건 → 3건, 홀드아웃 5건 → 0건), 추가 프롬프트 튜닝은 수확 체감 구간인
데다 평가셋 과적합 위험이 있다 — W07에서 3건짜리 스모크셋에 프롬프트를 맞추다 whack-a-mole에
빠진 전례가 있다. 남은 오탐은 **이미 위반인 문항에서 이웃 원칙까지 지적하는 형태**이고
순수 정상 문구 오탐은 0건이므로, 검수자가 어차피 들여다볼 건에만 발생한다.

### 2. 인용 중복 게이트는 진짜 위반도 NEEDS_REVIEW로 강등한다 — 설계된 동작이다

게이트는 진짜 위반 7건(평가셋 5 + 홀드아웃 2)을 NEEDS_REVIEW로 강등한다. 위반의 근거가
"여러 원칙이 함께 지목한 그 행위" 하나뿐일 때 도구가 귀속을 결정하지 못하기 때문이다.

**실패가 아니라 UC-5(판정 보류 에스컬레이션)다.** 위반이 `OK`로 분류된 경우는 **0건**이고,
강등된 건은 전부 사람 검토로 올라간다. 그래서 안전 지표는 재현율이 아니라 **위반 누락률
(VIOLATION→OK) = 0%** 이다(명세서 §11). 재현율(0.75/0.67)이 낮다고 게이트를 되돌리지 말 것.

### 3. 마스킹은 정규식 기반 — 이름·주소는 잡지 못한다

명세서 §6.1이 적은 NER 병용은 새 라이브러리가 필요해 W08 범위에서 제외했다. 이 프로젝트는
공개/합성 데이터만 다루므로(§8) 실사용 전 보완 지점.

---

측정 상세: `data/eval/w08_experiment_results.md`. 평가셋·라벨은 **동결**.

---

## 2. Architecture

```
regcheck/
├── app.py                  # Streamlit UI (W10)
├── agent/
│   ├── classifier.py       # input type/product classification (W05 코드 이식)
│   ├── judge.py            # 6-principle verdict engine (W07)
│   ├── memory.py           # session + inspection history (W07)
│   └── prompts/            # versioned prompts (v1.0.md, ...)
├── rag/
│   ├── ingest.py           # collect → chunk → embed → index
│   └── retriever.py        # FAISS top-k search
├── guardrails/
│   ├── masking.py          # PII regex masking (W08)
│   └── verify.py           # citation verification, confidence gate (W08)
├── logging_/audit.py       # append-only JSONL audit log (W08)
├── scripts/search_demo.py  # W06 DoD verification CLI
├── data/
│   ├── laws/               # law PDFs (public domain — committed for reproducibility; re-download from law.go.kr on amendment)
│   ├── cache/              # embedding cache (gitignored)
│   ├── index/              # FAISS index + metadata JSON (gitignored)
│   └── eval/               # labeled eval sets, test queries
└── docs/
    ├── RegCheck_프로젝트명세서.md
    └── decisions/          # ADRs
```

**Boundaries**: `rag/` knows nothing about verdicts; `agent/judge.py` consumes retriever output but never calls OpenAI embeddings directly. Guardrails wrap the pipeline (input masking before any API call, output verification after). Verdict JSON schema is fixed in spec §7.3 — do not extend it casually.

---

## 3. Tech Stack and Key Decisions

| Layer | Choice |
|-------|--------|
| Python | 3.11+ (managed by uv) |
| Package manager | uv (single project, pyproject.toml — no requirements.txt) |
| LLM | OpenAI `gpt-4.1` (verdicts — ADR-006; gpt-4o-mini는 오탐률 목표 미달) |
| Judge prompt | `agent/prompts/judge_v2.1.md` (2단 구성요건 판정; v1.0/v2.0은 실험 기록용 동결) |
| Embeddings | OpenAI `text-embedding-3-large`, locally cached (ADR-003 — 3-small measured recall@5 0.20) |
| Vector store | FAISS (faiss-cpu) + metadata JSON |
| PDF parsing | pdfplumber |
| UI | Streamlit (W10 only) |

Keep dependencies minimal. Adding a new library requires asking the user first.

**Architectural decisions** (full rationale in `docs/decisions/`):

- ADR-001: Law PDF parsing strategy — strip headers/footers, exclude 부칙, restore line breaks
- ADR-002: 6-principle tags limited to 금소법 본법 제17~22조 (시행령/감독규정 untagged; W07 must revisit — "본법 태그 필터 + 시행령 보강 검색" 설계)
- ADR-003: Embeddings switched to `text-embedding-3-large` (spec §7.1 said 3-small; measured recall@5 0.20 → 0.70). Finer chunking and lexical hybrid measured worse — do not retry without new evidence. HyDE deferred.
- ADR-004: Retrieval eval labels follow the regulatory chain (본법·시행령·감독규정 all count as correct). recall@5 = 1.00 under this rule, 0.70 if 본법-only — the 본법 gap is real and W07 must close it with `principle_filter`.
- ADR-005: 판정 컨텍스트는 [본법 조항 고정 주입(태그 기반, 검색 없음) + 하위규정 유사도 보강], 원칙당 1회씩 개별 LLM 호출. ADR-002/004의 미결 사항을 정산. 보강 검색 질의에는 원칙명을 붙인다 — 문구만으로 검색하면 6개 원칙이 같은 하위규정을 받아 오탐이 난다.
- ADR-007: FR-09 재설계 — confidence 임계값 폐기(정답 위반과 오탐의 confidence가 최소값부터 겹쳐 변별력 0), **인용 중복 탐지**로 교체. 다른 원칙과 같은 행위를 근거로 삼고 독립 근거가 없는 VIOLATION은 NEEDS_REVIEW로 강등(법정 중복쌍 {C3,C4}↔{E1,E2}은 예외). 더해 **판정 캐시**로 NFR-07 충족.
- ADR-008: `input_type`은 **사용자 필수 입력**, classifier는 상품군 분류·기본값 제안으로 강등. 국면 게이트가 유형에 의존하는데 classifier 이진 정확도가 0.535라 게이트가 역작동했다.
- ADR-009: 로컬 전환 PoC — 임베딩(bge-m3)은 로컬 대체 가능(recall@5 0.90), 판정 LLM의 CPU 서빙은 불가(judge_v2.1 출력 폭주 → KV 메모리 잠식 → OOM). Tier 2의 실질 전제는 GPU. 부정적 결과도 실측으로 기록한다. 측정: `data/eval/w09_poc_results.md`.
- ADR-006: judge v2.1(2단 구성요건 판정) + 판정 모델 gpt-4.1. 오탐의 지배적 원인은 모델이 아니라 **구조**였다(구조 +0.232 / 모델 +0.080 / classifier +0.021). 단 오탐률 목표(0.15)는 gpt-4.1이 있어야 통과한다. **LLM에게 verdict를 묻지 않는다** — 1단은 구성요건별 충족/불충족 + 문구 인용까지만, 2단(코드)이 verdict를 계산한다. 조문이 결정적으로 한정한 요건(제22조=광고 국면, 제21조제6호=투자성 상품)은 코드 게이트로 강제.

These decisions are settled. Do not change them silently. When a decision worth recording comes up, flag it as "ADR 필요" to the user — but do NOT write ADR bodies yourself. ADRs are drafted in the user's chat session and saved by the user; you may create `docs/decisions/` files only when handed finalized content.

---

## 4. Dev Environment and Commands

**Host**: WSL2 Ubuntu 24.04 on Windows 11.
**Working directory**: `/mnt/c/Users/박건우/Desktop/연세대학교/5학기/AI핀테크Agent분석과 설계/agent 개발 과제/regcheck` (Windows mount — slow I/O).
**venv**: lives at `~/venvs/regcheck` to avoid /mnt/c I/O. Requires `export UV_PROJECT_ENVIRONMENT=~/venvs/regcheck` (in ~/.bashrc).
**Shell**: bash (not PowerShell).

```bash
uv sync                              # sync dependencies
uv add <dep>                         # add dependency (ask user first for new libs)
uv run python -m rag.ingest          # build knowledge base from data/laws/
uv run python scripts/search_demo.py # W06 DoD verification (10 queries)
```

Embeddings are cached in `data/cache/` — re-running ingest does not re-bill the API. Full corpus embed costs <$0.01; never worry about it, but never bypass the cache either.

---

## 5. Conventions

The global CLAUDE.md covers general engineering principles. Project-specific additions:

### Code

- **Korean comments are OK** — legal/financial domain terms read better in Korean. Function and variable names stay in English.
- **Comments explain "why," not "what."**
- **No premature test introduction** — do not add pytest, ruff, mypy, or any tooling automatically. Discuss with the user first.
- **YAGNI, strictly by week** — do not build W07+ components while in W06, etc. Spec §14 lists explicit exclusions (no auth, no DB server, no OCR, no retraining pipeline).
- **Reproducibility (NFR-07)** — chunking parameters, embedding model, and prompt versions are recorded in index metadata / report meta. Keep this invariant when touching ingest or judge.

### Domain invariants (do not break)

- Verdict without evidence citation is forbidden → auto `NEEDS_REVIEW`.
- Cited articles must exist in the index (guardrails/verify.py is the enforcement point from W08).
- PII masking runs BEFORE any OpenAI API call; unmasked input never reaches logs.
- Only public/synthetic data. No real customer data, ever.
- Audit log is append-only JSONL with timestamp + input hash.

### Git

- **Conventional Commits**: `feat: / fix: / docs: / refactor: / chore: / test:`
- **English subject line preferred**, Korean body is fine.
- **AI collaboration disclosure** — append trailer when AI contributes meaningfully:
```
  Co-Authored-By: Claude <noreply@anthropic.com>
```
- **Be careful with direct `main` pushes.** Larger changes → feature branch + PR.
- Remote: `github.com/gwpark-dev/finlaw-regcheck-agent` (private).

### Security and secrets

Financial-domain project — the bar is high.

- **API keys NEVER appear in code, logs, or chat** — even temporarily. If the user pastes one by accident, tell them immediately.
- Credentials live in **Bitwarden**; runtime loads from `.env` only. `.env` is gitignored; only `.env.example` is tracked.
- Before installing an unfamiliar dependency, ask the user to verify it.

### Registered credentials

| Item | Storage |
|------|---------|
| OpenAI API key | Bitwarden + .env (`OPENAI_API_KEY`) |
| GitHub SSH key | ~/.ssh/ (passphrase + ssh-agent) |
