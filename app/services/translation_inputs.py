"""확실한 영문 수식 관계를 보호하고 앞 발언에 의존하는 번역 입력의 문맥을 선택한다."""
import re
from typing import Optional

from app.services.translation_validation import RAW_SYMBOL_PATTERN, RELATION_NUMBER, protected_symbols

SYMBOL = r'(?:u\d+|' + RAW_SYMBOL_PATTERN + r')'
PRIME = r"(?:\s+(?i:prime)|\s*[′'’])"
ATOM = r'(?:(?:' + RELATION_NUMBER + r')\s*)?' + SYMBOL + PRIME + r'?|' + RELATION_NUMBER
ATOM = r'(?:' + ATOM + r')'
LEFT = r'(?<![A-Za-z0-9_.])'
RIGHT = r'(?![A-Za-z0-9_])'
RELATION = re.compile(LEFT + r'(?P<a>' + ATOM + r')\s+(?P<op>(?i:of|over))\s+(?P<b>' + ATOM + r')' + RIGHT)
EQUALITY = re.compile(LEFT + r'(?P<a>' + ATOM + r')\s+(?i:equals)\s+(?P<b>' + ATOM + r')'
                      r'(?:\s+(?P<op>(?i:of|over))\s+(?P<c>' + ATOM + r'))?' + RIGHT)
MARKER = re.compile(r'\[MATH_[A-Z0-9_]+\]')
MARKER_HINT = re.compile(r'\bMATH_', re.IGNORECASE)
TRAILING_ATOM = re.compile(LEFT + ATOM + r'\s*$')
CONTEXT_CUE = re.compile(
    r"\b(?:this|that|these|those|it|its|they|them|their|he|she|him|her|his|here|there|above|below)\b|"
    r'\bthe\s+(?:equation|relation|term|expression|result|other\s+side)\b|'
    r'\b(?:left|end)\s+with\b|^\s*(?:and\s+)?(?:which|because)\b', re.IGNORECASE)


def _atom(text: str) -> Optional[str]:
    """단일 지원 기호·수치 계수·프라임을 값이나 인수를 추론하지 않고 표기한다.

    :param text: 정규식으로 찾은 단일 수식 항.
    :return: 정규화한 표기 또는 단위·복합 곱처럼 지원하지 않는 항이면 None.
    """
    if re.fullmatch(RELATION_NUMBER, text):
        return re.sub(r'\s+', '', text)
    if re.fullmatch(r'u\d+' + PRIME + r'?', text):
        return re.match(r'u\d+', text).group() + ('′' if re.search(PRIME + r'$', text) else '')
    symbols = protected_symbols(text)
    if len(symbols) != 1:
        return None
    number = re.match(RELATION_NUMBER, text)
    coefficient = re.sub(r'\s+', '', number.group()) if number else ''
    prime = '′' if re.search(PRIME + r'$', text) else ''
    return coefficient + symbols[0] + prime


def _relation(left: str, operator: str, right: str) -> Optional[str]:
    """지원되는 함수 또는 단순 분수의 두 항을 보존한 표현을 만든다.

    :param left: 왼쪽 원문 항.
    :param operator: of 또는 over. 대소문자를 허용한다.
    :param right: 오른쪽 원문 항.
    :return: 함수·분수 표기 또는 해석할 수 없는 항이면 None.
    """
    a, b = _atom(left), _atom(right)
    if a is None or b is None:
        return None
    if operator.lower() == 'of':
        if not protected_symbols(left) or re.match(RELATION_NUMBER, left):
            return None
        return f'{a}({b})'
    return f'{a}/({b})'


def _adjacent_relation(text: str, start: int, end: int, equality: bool) -> bool:
    """지원 범위 밖의 연쇄 관계·곱·괄호식에서 부분 등식을 떼어내는 것을 막는다.

    :param text: 전체 원문.
    :param start: 후보의 시작 문자 위치.
    :param end: 후보의 끝 문자 위치.
    :param equality: 등식 전체 후보인지 여부. 등식에는 더 엄격한 경계 검사를 적용한다.
    :return: 경계를 넘어서는 관계나 중단된 등식일 가능성이 있으면 True.
    """
    before, after = text[:start], text[end:]
    operators = 'of|over' + ('|equals|plus|minus' if equality else '')
    if (re.search(r'\b(?:' + operators + r')\s*$', before, re.IGNORECASE)
            or re.match(r'\s*(?:' + operators + r')\b', after, re.IGNORECASE)):
        return True
    if equality:
        return bool(TRAILING_ATOM.search(before)
                    or re.search(r'[+*/=^−-]\s*$', before)
                    or re.match(r'\s*(?:[(/+*=^]|' + ATOM + RIGHT + r')', after)
                    or re.match(r'\s*[,—-]?\s*(?:is|are|was|were|prime|a\s+constant)\b',
                                after, re.IGNORECASE))
    return bool(re.match(r'\s*\(', after))


def _marker(index: int) -> str:
    """숫자를 추가하지 않는 A·B·AA 방식의 수식 자리표시자를 만든다.

    :param index: 0부터 시작하는 수식 순번.
    :return: 대괄호로 감싼 고유한 MATH 자리표시자.
    """
    label = ''
    index += 1
    while index:
        index, digit = divmod(index - 1, 26)
        label = chr(65 + digit) + label
    return '[MATH_' + label + ']'


def protect_math(text: str) -> tuple[str, dict[str, str]]:
    """발언의 수식을 모두 표현할 수 있는 단순 함수·분수·원자 등식만 보호한다.

    :param text: 수정하지 않고 API 결과에 보존할 현재 영어 원문.
    :return: 모델용 입력과 등장 순서의 자리표시자·수식 사전. 원문은 변경하지 않는다.

    보호 뒤에 다른 기호·연산이 남으면 복합 발언 전체를 기존 처리에 남긴다.
    연쇄 연산·괄호·중단된 등식은 해석하지 않는다. 예약 표식이 이미 있는
    입력은 충돌을 피하기 위해 보호하지 않는다. 적분·로그 범위나 STT 오기를
    복원하지 않으며 복원된 표기는 별도로 원문 보존 검증을 거쳐야 한다.
    """
    if MARKER_HINT.search(text):
        return text, {}
    spans = []
    for match in EQUALITY.finditer(text):
        if _adjacent_relation(text, match.start(), match.end(), True):
            continue
        a = _atom(match['a'])
        b = (_relation(match['b'], match['op'], match['c']) if match['op'] else _atom(match['b']))
        if a is not None and b is not None and (protected_symbols(match['a']) or protected_symbols(match['b'])):
            spans.append((match.start(), match.end(), f'{a} = {b}'))
    for match in RELATION.finditer(text):
        if (any(start < match.end() and match.start() < end for start, end, _ in spans)
                or _adjacent_relation(text, match.start(), match.end(), False)):
            continue
        rendered = _relation(match['a'], match['op'], match['b'])
        if rendered is not None:
            spans.append((match.start(), match.end(), rendered))
    parts, replacements, offset = [], {}, 0
    for start, end, rendered in sorted(spans):
        marker = _marker(len(replacements))
        parts.extend((text[offset:start], marker))
        replacements[marker] = rendered
        offset = end
    parts.append(text[offset:])
    model_input = ''.join(parts)
    if (protected_symbols(model_input)
            or re.search(LEFT + r'u\d+' + RIGHT, model_input)
            or re.search(r'\b(?:plus|minus|over|equals)\b', model_input, re.IGNORECASE)):
        return text, {}
    return model_input, replacements


def restore_math(translation: str, text: str) -> tuple[str, Optional[str]]:
    """수식 표식의 개수·정체·순서를 검증한 뒤 원문의 보호 관계를 복원한다.

    :param translation: 모델이 생성하고 일반 출력 표식을 정리한 후보.
    :param text: 보호 전의 현재 영어 원문.
    :return: 복원 후보와 실패 사유. 표식 누락·복제·변경·순서 역전이면 math_placeholder_mismatch.
    """
    _, replacements = protect_math(text)
    if not replacements and MARKER_HINT.search(text):
        return translation, None
    if MARKER.findall(translation) != list(replacements):
        return translation, 'math_placeholder_mismatch'
    for marker, rendered in replacements.items():
        translation = translation.replace(marker, rendered)
    if MARKER_HINT.search(translation):
        return translation, 'math_placeholder_mismatch'
    return translation, None


def select_context(text: str, context: str) -> str:
    """앞 대상을 가리키거나 연산 단계를 이어가는 발언에만 제한된 원문 문맥을 남긴다.

    :param text: 현재 영어 원문. 대명사·지시어·이어지는 수학 설명을 보수적으로 찾는다.
    :param context: 호출자가 제공한 앞 영어 원문.
    :return: 외부 의존 단서나 정정 발언이면 마지막 256자, 그 외에는 빈 문자열.

    정정·대조와 지시어·대명사를 포함하는 발언은 문맥을 유지한다.
    일반적인 의존성 판별기가 아니며 모든 강의에서 의미가 유지됨을
    보장하지 않는다. API 응답 원문이나 STT 구간을 변경하지 않는다.
    """
    reference = CONTEXT_CUE.search(text)
    correction = re.match(r'^\s*(?:actually|instead|however|but|not\s+quite)\b', text, re.IGNORECASE)
    if correction:
        return context[-256:]
    if reference is None:
        return ''
    return context[-256:]
