"""개인정보 마스킹 — Guardrail 입력단 (FR-07, NFR-04).

OpenAI 호출 **전에** 반드시 통과시킨다. 마스킹되지 않은 원문은 API로도, 로그로도
나가지 않는다.

  mask("연락처는 010-1234-5678입니다")
  -> ("연락처는 [전화번호]입니다", [{"type": "전화번호", "count": 1}])

반환하는 findings에는 **탐지된 값 자체를 담지 않는다** — 유형과 건수만 남긴다.
로그에 개인정보를 남기지 않는다는 NFR-04를 지키려면 여기서부터 값을 버려야 한다.

한계: 정규식 기반이라 사람 이름·주소처럼 형태가 정해지지 않은 개인정보는 잡지 못한다.
명세서 §6.1은 NER 병용을 적었으나 새 라이브러리 도입 없이는 불가해 W08 범위에서는
패턴 기반만 구현한다. 이 프로젝트는 합성/공개 데이터만 다루므로(§8) 실사용 전 보완 지점.
"""

from __future__ import annotations

import re

# \b를 쓰면 안 된다. 파이썬 정규식에서 한글은 \w라 "5678로", "9012입니다"처럼 숫자 뒤에
# 조사가 붙으면 단어 경계가 성립하지 않아 매칭이 통째로 실패한다(실측). 한국어 문장에서는
# 거의 항상 조사가 붙으므로, 경계 대신 "숫자에 인접하지 않을 것"만 요구한다.
NOT_DIGIT_BEFORE = r"(?<!\d)"
NOT_DIGIT_AFTER = r"(?!\d)"

# 순서가 중요하다. 더 구체적인 패턴을 먼저 지워야 일반 패턴이 잘못 먹지 않는다.
# 예: 주민등록번호를 먼저 지우지 않으면 계좌번호 패턴이 앞부분을 갉아먹는다.
PATTERNS: list[tuple[str, re.Pattern]] = [
    # 주민등록번호 — 뒷자리 첫 숫자는 성별·세기 코드(내국인 1~4, 외국인 5~8)
    ("주민등록번호", re.compile(rf"{NOT_DIGIT_BEFORE}\d{{6}}\s*[-–]\s*[1-8]\d{{6}}{NOT_DIGIT_AFTER}")),
    # 카드번호 — 4-4-4-4
    ("카드번호", re.compile(rf"{NOT_DIGIT_BEFORE}\d{{4}}[-\s]\d{{4}}[-\s]\d{{4}}[-\s]\d{{4}}{NOT_DIGIT_AFTER}")),
    # 사업자등록번호 — 3-2-5
    ("사업자등록번호", re.compile(rf"{NOT_DIGIT_BEFORE}\d{{3}}\s*-\s*\d{{2}}\s*-\s*\d{{5}}{NOT_DIGIT_AFTER}")),
    # 휴대전화 / 유선전화
    ("전화번호", re.compile(rf"{NOT_DIGIT_BEFORE}01[016789][-\s.]?\d{{3,4}}[-\s.]?\d{{4}}{NOT_DIGIT_AFTER}")),
    ("전화번호", re.compile(rf"{NOT_DIGIT_BEFORE}0\d{{1,2}}[-\s.]\d{{3,4}}[-\s.]\d{{4}}{NOT_DIGIT_AFTER}")),
    # 이메일 — 문자군을 ASCII로 한정한다. \w를 쓰면 도메인 뒤의 한글까지 삼킨다.
    ("이메일", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # 계좌번호 — 은행마다 자릿수가 달라 범위로 잡는다. 위 패턴들을 먼저 지운 뒤라
    # 여기 걸리는 하이픈 숫자열은 계좌로 보는 것이 안전하다(놓치는 쪽이 더 위험).
    ("계좌번호", re.compile(rf"{NOT_DIGIT_BEFORE}\d{{2,6}}\s*-\s*\d{{2,6}}\s*-\s*\d{{2,8}}{NOT_DIGIT_AFTER}")),
]

TOKEN = "[{kind}]"


def mask(text: str) -> tuple[str, list[dict]]:
    """개인정보를 유형 토큰으로 치환. 반환: (마스킹된 텍스트, 탐지 요약)

    탐지 요약에는 유형과 건수만 담는다 — 값은 담지 않는다 (NFR-04).
    """
    counts: dict[str, int] = {}
    masked = text

    for kind, pattern in PATTERNS:
        masked, n = pattern.subn(TOKEN.format(kind=kind), masked)
        if n:
            counts[kind] = counts.get(kind, 0) + n

    findings = [{"type": k, "count": v} for k, v in counts.items()]
    return masked, findings


def has_pii(text: str) -> bool:
    """마스킹이 필요한 값이 남아 있는지. 파이프라인 사후 점검용."""
    return any(p.search(text) for _, p in PATTERNS)
