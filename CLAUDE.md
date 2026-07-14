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
| W08 ★ | Guardrails (PII masking, no-evidence hold, false-positive control) + audit logging | 진행 중 |
| W09 | On-premise architecture design doc | |
| W10 | Streamlit demo | |

**W07 → W08 인계 항목**

1. **재현성 (NFR-07 미충족)** — gpt-4.1도 동일 입력·seed 고정·temperature=0에서 2회
   실행 시 186셀 중 6셀(3.2%)이 뒤집힌다(v1.0은 5.6%). 2단을 코드로 옮겨 줄었으나 0은
   아니다 — 1단이 여전히 LLM이기 때문. 감사 대응 도구로서 "같은 문구를 같게 판정한다"가
   아직 보장되지 않는다. FR-09 신뢰도 임계값이 이를 흡수할 수 있는지 검토 중.
2. **classifier–게이트 결합 위험** — v2.1의 국면 게이트(제22조는 광고만)가 `input_type`에
   의존한다. 실험은 oracle(정답 주입)로 측정했으나 실서비스엔 oracle이 없다. 규칙 기반
   classifier가 광고/상담을 오분류하면 게이트가 **역작동**한다. classifier 자체의 기여는
   미미했으므로(+0.021), 개선보다 "유형을 사용자 입력으로 받는" 안을 검토 중.
3. **부당권유 원칙의 구성요건 오지정** — 6대 원칙 중 가장 약하다(정확도 0.74~0.75).
   E3(중대사항 미고지)·E8(적합성 회피)를 포괄 조항처럼 끌어다 쓰는 경향이 남아 있다.

측정 상세: `data/eval/w08_experiment_results.md`. 평가셋·라벨은 **동결** — FR-09 임계값
보정에 평가셋을 쓰면 과적합이므로 보정용 데이터는 별도 논의.

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
