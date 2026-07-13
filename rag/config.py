"""RAG 파이프라인 공통 설정 (W06).

ingest(색인 구축)와 retriever(검색)가 같은 값을 봐야 하는 것들만 모아둔다.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LAWS_DIR = ROOT / "data" / "laws"
INDEX_DIR = ROOT / "data" / "index"
EVAL_DIR = ROOT / "data" / "eval"
# 임베딩 캐시를 index/와 분리해 둔다. data/index/를 지우고 재빌드해도 API를 다시 태우지 않기 위함.
CACHE_DIR = ROOT / "data" / "cache"

INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"
EMBED_CACHE_PATH = CACHE_DIR / "embeddings.json"

# 명세서 §7.1은 3-small을 적었으나, 평가셋 10건 실측에서 recall@5가 0.20에 그쳤다.
# 모델만 3-large로 바꾸면 0.70 — 한국어 법조문은 상용구가 많아 3-small로는 조문 간
# 변별이 되지 않는다. 청크 세분화·어휘 하이브리드는 오히려 recall을 떨어뜨렸다(실험 기록: README).
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH = 64

# NFR-07(재현성): 아래 파라미터는 meta.json에 그대로 기록된다.
CHUNK_PARAMS = {
    "strategy": "article_first_then_paragraph",
    "max_chunk_chars": 1200,  # 조(條) 전문이 이보다 길면 항(①②③) 단위로 분할
    "fallback_chunk_chars": 900,  # 조 구조가 없는 문서(가이드라인 등)용 문자 윈도우
    "fallback_overlap_chars": 150,
    "drop_supplementary": True,  # 부칙 제외 (조항 번호가 제1조부터 다시 시작해 본문과 충돌)
}

# §7.2 근거 부족 판단 임계값. W08 Guardrail에서 판정 보류(NEEDS_REVIEW) 트리거로 재사용.
LOW_CONFIDENCE_THRESHOLD = 0.35

# 파일명(확장자 제외) → 정식 법령명. data/laws/에 새 파일을 넣으면 여기에 추가한다.
LAW_NAMES = {
    "금소법": "금융소비자 보호에 관한 법률",
    "금소법_시행령": "금융소비자 보호에 관한 법률 시행령",
    "금소법_감독규정": "금융소비자 보호에 관한 감독규정",
    "AI기본법": "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법",
    "금융AI가이드라인": "금융분야 AI 가이드라인",
}

# 명세서 §2 6대 판매규제. 조항 번호는 금소법 '본법' 기준이므로
# 시행령·감독규정의 동일 번호 조항에는 태그하지 않는다(조문 체계가 달라 오태깅됨).
PRINCIPLE_SOURCE = "금소법"
PRINCIPLES = {
    17: "적합성 원칙",
    18: "적정성 원칙",
    19: "설명의무",
    20: "불공정영업행위 금지",
    21: "부당권유행위 금지",
    22: "광고 규제",
}
