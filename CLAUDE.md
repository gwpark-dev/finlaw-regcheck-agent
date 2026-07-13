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
| W07 | 6-principle verdict engine (Tool) + inspection-history Memory | **Done (일부 목표 미달)** — 평가셋 31건/186셀, 확정 라벨 기준: 스키마 준수 **100%**, 1건 **8.2초**(NFR-01 30초), 판정 정확도 **0.565**(목표 0.75 미달), 오탐률 **0.482**(목표 0.15 미달), 위반 재현율 0.95. **오탐 과다**가 미해결 (아래) |
| W08 ★ | Guardrails (PII masking, no-evidence hold, false-positive control) + audit logging | |
| W09 | On-premise architecture design doc | |
| W10 | Streamlit demo | |

**W07 미해결 이슈 — 판정 오탐 (W08 최우선)**

gpt-4o-mini가 문구에 문제가 하나라도 보이면 6대 원칙에 폭넓게 VIOLATION을 찍는다.
정상 라벨 166셀 중 80셀(48%)을 위반으로 판정했다. 재현율 0.95는 "거의 다 위반으로
찍은" 결과라 좋은 신호가 아니다. 프롬프트로 억제를 시도했으나(적용 국면 게이트, 조문
열거 행위 매칭 강제, 원칙별 few-shot) 한계가 있었다 — `agent/prompts/judge_v1.0.md`.

세 가지가 원인이 **아님**을 실측으로 배제했다:
- **라벨 주관성 아님** — 라벨 확정(제19조③ 국면 조건부 복수 라벨 등) 전후로 정확도
  0.578 → 0.578, 오탐률 0.469 → 0.469. 라벨을 고쳐도 수치가 움직이지 않는다.
- **근거 검색 실패 아님** — 인용 조항은 대체로 정확하고, evidence 미인용에 의한
  NEEDS_REVIEW 강등은 0건이었다.
- **스키마/파싱 문제 아님** — 준수율 100%.

남은 원인은 모델의 판단 자체다. 특히 심각한 것은 판단이 아니라 **독해 실패**다 —
w07-029/030은 "계약 체결 전 설명서·약관을 읽어보시기 바랍니다"를 문구에 그대로 담고
있는데도 모델은 그 항목이 "누락"이라 판정했다. 정상 문구의 **위험 고지 문장을 위반
근거로 삼는** 패턴도 있다(w07-015의 해지환급금 원금 미달 고지 → "단정적 판단" 판정).
컴플라이언스를 잘 지킨 문구일수록 더 걸리는 구조라 실무 도구로는 치명적이다
(명세서 §13 "오탐 과다 → 도구 불신").

또한 **temperature=0에서도 판정이 재현되지 않는다** — 동일 30문항 재실행 시 180셀 중
10셀(5.6%)이 뒤집혔다. NFR-07(재현성) 관점에서 별도 검토가 필요하다.

W08 후보: FR-09(신뢰도 임계값 보류), 모델 상향(명세서 §12 변경 → ADR 필요),
판정 근거를 조문 구성요건 체크리스트로 분해하는 2단 판정.

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
| LLM | OpenAI `gpt-4o-mini` (verdicts, W07+) |
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
