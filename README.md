# RegCheck — 금소법 컴플라이언스 점검 Agent

금융상품 광고/상담 문구의 금융소비자보호법 위반 소지를 근거 조항과 함께 점검하는 RAG Agent.
자세한 요구사항은 상위 폴더의 `RegCheck_프로젝트명세서.md` 참고.

**현재 진행: W06 — 법령 청킹/임베딩 → FAISS 벡터검색 RAG (FR-01, FR-02)**

## 셋업

```bash
# venv를 리눅스 파일시스템에 두어 /mnt/c(윈도우 마운트)의 느린 I/O를 피한다.
# 아래 export를 셸 프로필(~/.bashrc)에 넣어두면 매번 칠 필요가 없다.
export UV_PROJECT_ENVIRONMENT=~/venvs/regcheck

uv sync                       # pyproject.toml 기준 의존성 설치
cp .env.example .env          # OPENAI_API_KEY=sk-... 채우기 (.env는 git 추적 제외)
```

## 실행

```bash
# 1) 지식베이스 구축: data/laws/*.pdf → 조 단위 청킹 → 임베딩 → data/index/
uv run python -m rag.ingest
uv run python -m rag.ingest --dry-run   # 청킹만 확인 (API 호출/비용 없음)

# 2) 검색 확인
uv run python scripts/search_demo.py "원금 보장되고 수익률 확실한 상품입니다"
uv run python scripts/search_demo.py                    # 평가 질의 10건 + recall@5
uv run python scripts/search_demo.py "..." --principle "광고 규제"   # 원칙 태그 필터
```

`data/laws/`가 비어 있으면 어떤 파일을 넣어야 하는지 안내하고 종료한다.

## 구조

```
rag/config.py       공통 설정 — 청킹 파라미터, 임베딩 모델, 법령명·6대 원칙 매핑
rag/ingest.py       PDF/TXT → 조(條) 단위 청킹 → OpenAI 임베딩 → FAISS 인덱스
rag/retriever.py    코사인 유사도 Top-k 검색 (search)
agent/classifier.py 문구 유형·상품군 분류 (FR-03, 규칙 기반 최소 구현)
scripts/search_demo.py  검증용 CLI
data/laws/          법령 원문 (공개 자료)
data/index/         FAISS 인덱스 + meta.json (gitignore, ingest로 복원)
data/cache/         임베딩 캐시 (gitignore, 재실행 시 API 비용 방지)
data/eval/          평가 질의셋
```

## 검색 품질 (W06 DoD)

`data/eval/w06_queries.txt` 10건 기준 **recall@5 = 1.00** (목표 ≥ 0.80).

정답 조항은 해당 원칙의 *규제 체인*(본법 → 시행령 → 감독규정)을 모두 인정한다. 실무에서
근거로 드는 조항이 셋 중 어느 것이든 될 수 있기 때문이다 — 예컨대 꺾기(구속성 판매)의
구체적 판단 기준은 본법 제20조가 아니라 감독규정 제14조에 있다.

**알려진 한계**: 정답을 *본법 조항만*으로 좁히면 recall@5는 0.70으로 떨어진다. 제20조(꺾기),
제21조(재권유)는 각 호 6~7개가 한 덩어리라 임베딩이 희석되어, 같은 주제의 시행령·감독규정
조항에 밀린다. 판정 엔진(W07)은 원칙별로 `principle_filter`를 걸어 검색하므로 본법 조항을
항상 확보할 수 있어 실사용 경로에는 영향이 없다. 또 제46조(청약의 철회)·제69조(과태료)가
상위에 섞여 들어오는 노이즈가 남아 있다.

### 실험 기록 (recall@5, 평가셋 10건)

| 조건 | recall@5 |
|---|---|
| `3-small`, 조 단위 청킹 | 0.20 |
| `3-small` + 호(號) 세분화 | 0.20 |
| `3-small` + 어휘 하이브리드 | 0.20 |
| **`3-large`, 조 단위 청킹 (채택)** | **0.70** |
| `3-large` + 호 세분화 | 0.40 |
| `3-large` + 청크 측 LLM 해설 보강 | 0.70 |
| `3-large` + HyDE(질의를 법조문 문체로 변환) | 0.80 |

명세서 §7.1은 `text-embedding-3-small`을 지정했으나 실측 recall이 0.20에 그쳐 **`3-large`로
변경**했다. 한국어 법조문은 상용구("금융상품판매업자등은 … 대통령령으로 정하는 …")가 많아
`3-small`로는 조문 간 변별이 되지 않는다. 청크를 더 잘게 쪼개거나 키워드 매칭을 섞는 방법은
오히려 recall을 떨어뜨렸다. HyDE는 본법 조항 recall을 0.80까지 올리지만 질의마다 LLM을
호출해야 해 지연·비용이 붙고 결과가 흔들리므로(감사 추적성 NFR-05와 상충) 채택하지 않았다.

## 설계 메모

- **청킹**: 조(條) 단위가 기본. 조 전문이 1,200자를 넘으면 항(①②③) 단위로 분할한다.
  조 구조가 없는 문서(금융AI가이드라인)는 문자 윈도우(900자/오버랩 150자)로 폴백.
  부칙은 조항 번호가 제1조부터 다시 시작해 본문과 충돌하므로 제외한다.
- **원칙 태그**: 6대 판매규제는 금소법 *본법* 제17~22조에만 부여한다. 시행령·감독규정은
  조문 체계가 달라 같은 번호라도 내용이 다르므로 태그하지 않는다(`principle: null`).
- **유사도**: 벡터를 L2 정규화한 뒤 `IndexFlatIP` 내적 → 코사인 유사도와 동일.
  최고 유사도 < 0.35면 결과에 `low_confidence: True` — W08 Guardrail의 판정 보류 신호.
- **검색 단위**: `search()`는 청크를 반환한다(판정 엔진이 항 원문을 그대로 써야 하므로).
  다만 인용의 단위는 조항이라, 데모는 조마다 최고 점수 청크만 남겨 **Top-5 조항**을 보여준다
  (`scripts/search_demo.py: top_articles`). 안 그러면 제22조 7개 항이 5칸을 다 차지한다.
- **재현성(NFR-07)**: 임베딩 모델명·청킹 파라미터를 `data/index/meta.json`에 기록한다.
- **보안(NFR-06)**: API 키는 `.env`에서만 로드. 코드·로그에 노출하지 않는다.

## 다음 주차

W07 판정 엔진(`agent/judge.py`) · 점검 이력 Memory, W08 Guardrail · 감사 로깅,
W10 Streamlit UI. W06 범위에 없는 모듈은 아직 만들지 않는다(YAGNI).
