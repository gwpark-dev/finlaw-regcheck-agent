"""입력 문구 유형 분류 (FR-03) — 인터페이스 정의 + 규칙 기반 최소 구현.

W07에서 W05 하이브리드(규칙+ML) 분류기를 이식하며 내부 구현만 교체한다.
호출부는 classify() 시그니처에만 의존하도록 유지할 것.
"""

from __future__ import annotations

# 상품군: 명세서 §7.3 product_category
PRODUCT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "보험": ("보험", "보장", "보험료", "해지환급금", "종신", "실손"),
    "펀드": ("펀드", "투자", "수익률", "주식", "채권", "파생", "ELS", "원금"),
    "대출": ("대출", "한도", "금리", "상환", "신용", "담보"),
    "예금": ("예금", "적금", "예치", "이자"),
    "카드": ("카드", "할부", "캐시백", "연회비"),
}

# 광고는 불특정 다수 대상 문구, 상담은 특정 고객과의 대화 문구로 구분한다.
AD_KEYWORDS = (
    "지금 신청",
    "이벤트",
    "특별 혜택",
    "출시",
    "무료",
    "1위",
    "최대",
    "누구나",
    "!",
    "%",
)
CONSULT_KEYWORDS = (
    "고객님",
    "권유",
    "상담",
    "가입하시",
    "설명드",
    "안내드",
    "하시겠",
    "문의",
)


def _match_count(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for kw in keywords if kw in text)


def classify(text: str) -> dict:
    """문구 유형과 상품군을 분류.

    반환: {"type": "광고" | "상담", "product": "보험" | "펀드" | ... | "기타"}
    """
    ad_score = _match_count(text, AD_KEYWORDS)
    consult_score = _match_count(text, CONSULT_KEYWORDS)
    # 동점이면 광고로 본다 — 광고 규제(제22조)가 더 엄격해 놓치는 쪽이 위험하다.
    input_type = "상담" if consult_score > ad_score else "광고"

    scores = {p: _match_count(text, kws) for p, kws in PRODUCT_KEYWORDS.items()}
    best = max(scores, key=lambda p: scores[p])
    product = best if scores[best] > 0 else "기타"

    return {"type": input_type, "product": product}
