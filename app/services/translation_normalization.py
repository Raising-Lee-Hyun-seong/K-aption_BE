"""원문에서 확인한 수식의 영어 이항 연산어만 기호로 바꾸고 괄호와 피연산자를 유지한다."""

import re
from typing import Optional

from app.services.translation_validation import RAW_SYMBOL_PATTERN, protected_symbols

NUMBER = r'\d+(?:\.\d+)?'
VARIABLE = r'(?:u\d+|' + RAW_SYMBOL_PATTERN + r')'
ATOM = r'(?:(?:' + NUMBER + r')\s*)?' + VARIABLE + r"(?:\s+prime|[′'’])?|" + NUMBER
ATOM = r'(?:' + ATOM + r')'
LEFT = re.compile(r'(?<![A-Za-z0-9_.,])(?P<atom>' + ATOM + r')\s*$', re.IGNORECASE)
RIGHT = re.compile(r'\s*(?P<atom>' + ATOM + r')(?![A-Za-z0-9_]|[.,]\d)', re.IGNORECASE)
WORDS = re.compile(r'\b(?:plus|minus|times|equals|over)\b', re.IGNORECASE)
SOURCE_OPERATORS = re.compile(r'\b(?:plus|minus|times|equals|over)\b|[+*/=×−-]', re.IGNORECASE)
OPERATORS = {'plus': '+', 'minus': '-', 'times': '×', 'equals': '=', 'over': '/',
             '+': '+', '-': '-', '−': '-', '*': '×', '×': '×', '=': '=', '/': '/'}
TOKEN = re.compile(r'\s*(?:(' + ATOM + r')(?![A-Za-z0-9_]|\.\d)|([()+*/×^−-]))', re.IGNORECASE)


def _key(atom: str) -> str:
    """수식 항의 공백·별칭·프라임 표기만 비교용 문자열로 정리한다.

    :param atom: 숫자·지원 변수·선택적 계수와 프라임으로 이루어진 항.
    :return: 비교용 문자열. 실제 원문이나 출력은 바꾸지 않는다.
    """
    return re.sub(r'[\s_]+', '', atom.lower()).replace('prime', '′').replace("'", '′').replace('’', '′')


def _known(atom: str, source: str) -> bool:
    """항의 변수가 현재 원문에 있고 수량 단위를 변수로 오인하지 않는지 확인한다.

    :param atom: 숫자 또는 지원 수식 항.
    :param source: 이전 문맥을 제외한 현재 영어 원문.
    :return: 숫자이거나 현재 원문에서 확인할 수 있는 수식 항이면 True.
    """
    if re.fullmatch(NUMBER, atom.strip()):
        return True
    indexed = re.fullmatch(r'u\d+(?:[′\'’])?', _key(atom))
    if indexed:
        return bool(re.search(r'(?<![A-Za-z0-9_])' + re.escape(_key(atom).rstrip('′'))
                              + r'(?![A-Za-z0-9_])', source, re.IGNORECASE))
    symbols = protected_symbols(atom)
    return bool(symbols) and set(symbols).issubset(protected_symbols(source))


def _group_end(text: str, start: int) -> Optional[int]:
    """이미 존재하는 소괄호의 짝을 찾고 과도한 길이·중첩은 지원 범위에서 제외한다.

    :param text: 모델 출력 문자열.
    :param start: 여는 소괄호 문자 위치.
    :return: 닫는 소괄호 다음 위치. 256자·중첩 8단계를 넘거나 짝이 없으면 None.
    """
    depth = 0
    for index in range(start, min(len(text), start + 256)):
        if text[index] == '(':
            depth += 1
            if depth > 8:
                return None
        elif text[index] == ')':
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _valid_group(text: str, source: str) -> bool:
    """명시된 괄호 안의 지원 항·기호와 연산자 배치를 계산이나 괄호 추가 없이 확인한다.

    :param text: 바깥 소괄호를 포함한 모델 출력의 수식 조각.
    :param source: 지원 변수를 확인할 현재 영어 원문.
    :return: 모든 문자를 소비하고 괄호·연산 배치가 지원 문법에 맞으면 True.

    의미·대수적 동치·로그 범위를 판정하지 않는다. 숫자·단위·관계 보존은
    정규화 뒤의 기존 검증기가 다시 검사한다.
    """
    position, depth, expect_atom, function = 0, 0, True, False
    while position < len(text):
        match = TOKEN.match(text, position)
        if match is None:
            return False
        position = match.end()
        atom, operator = match.groups()
        if function and operator != '(':
            return False
        if atom:
            if not expect_atom or not _known(atom, source):
                return False
            function = _key(atom) == 'ln'
            expect_atom = function
        elif operator == '(':
            if not expect_atom:
                return False
            depth += 1
            function, expect_atom = False, True
        elif operator == ')':
            if depth == 0 or expect_atom:
                return False
            depth -= 1
            expect_atom = False
        else:
            if expect_atom:
                return False
            expect_atom = True
    return depth == 0 and not expect_atom and not function


def _right(text: str, start: int, source: str, output: bool) -> Optional[tuple[str, bool]]:
    """연산어 오른쪽의 단일 항이나 명시적인 ln 괄호식의 비교 키를 읽는다.

    :param text: 원문 또는 모델 출력 문자열.
    :param start: 연산어 뒤의 문자 위치.
    :param source: 지원 변수의 기준 원문.
    :param output: 참이면 ln에 완성된 괄호식을 요구한다. 원문에서는 발화된 ln 표기를 허용한다.
    :return: 항 비교 키·괄호로 감싼 단일 항 여부 또는 지원하지 않으면 None.
    """
    offset = start + len(text[start:]) - len(text[start:].lstrip())
    if offset < len(text) and text[offset] == '(':
        end = _group_end(text, offset)
        if end is None:
            return None
        atom = text[offset + 1:end - 1].strip()
        if re.fullmatch(ATOM, atom, re.IGNORECASE) and _known(atom, source):
            return _key(atom), True
        return None
    match = RIGHT.match(text, start)
    if match is None or not _known(match['atom'], source):
        return None
    key = _key(match['atom'])
    if key == 'ln' and output:
        opening = match.end() + len(text[match.end():]) - len(text[match.end():].lstrip())
        if opening >= len(text) or text[opening] != '(':
            return None
        end = _group_end(text, opening)
        if end is None or not _valid_group(text[opening:end], source):
            return None
    return key, False


def _left(text: str, end: int, source: str) -> Optional[str]:
    """연산어 왼쪽의 알려진 단일 항만 읽고 복합 곱의 뒷부분은 제외한다.

    :param text: 원문 또는 모델 출력 문자열.
    :param end: 연산어 시작 문자 위치.
    :param source: 지원 변수의 기준 원문.
    :return: 비교용 항 또는 괄호식·복합 곱·알 수 없는 항이면 None.
    """
    match = LEFT.search(text[:end])
    if match is None or not _known(match['atom'], source):
        return None
    before = text[:match.start()].rstrip()
    if before.endswith(')') or LEFT.search(before):
        return None
    return _key(match['atom'])


def normalize_math_operators(translation: str, source: str) -> str:
    """원문과 피연산자·연산이 일치하는 영어 수식 연산어만 기호로 치환한다.

    :param translation: 수식 자리표시자를 복원한 모델 후보. 원본 문자열은 수정하지 않는다.
    :param source: 검증의 기준인 현재 영어 원문. 앞 문맥은 사용하지 않는다.
    :return: 연산어만 바꾼 후보. 확인할 수 없는 연산과 나머지 문자는 그대로 유지한다.

    u ln처럼 원문에 명시적으로 나란히 있는 단일 항과 ln 사이의 times만
    암묵 곱으로 허용한다. over 뒤 계수가 붙은 항은 이미 괄호가 있을 때만
    치환하며, 같은 항 사이에 여러 연산이 있거나 천 단위 쉼표가 있는
    항은 건너뛴다. 괄호·로그 범위·변수·숫자·분수 방향을 새로 만들지 않는다.
    반환값은 성공 판정이 아니며 기존 전체 출력 검증이 반드시 뒤따라야 한다.
    """
    if not protected_symbols(source) or WORDS.search(translation) is None:
        return translation
    depth = 0
    for char in translation:
        depth += (char == '(') - (char == ')')
        if depth < 0:
            return translation
    if depth:
        return translation
    allowed = set()
    for match in SOURCE_OPERATORS.finditer(source):
        left = _left(source, match.start(), source)
        right = _right(source, match.end(), source, False)
        if left is not None and right is not None:
            allowed.add((OPERATORS[match.group().lower()], left, right[0]))
    for match in re.finditer(r'\bln\b', source, re.IGNORECASE):
        left = _left(source, match.start(), source)
        if left is not None:
            allowed.add(('×', left, 'ln'))
    pair_operators = {}
    for operator, left, right in allowed:
        pair_operators.setdefault((left, right), set()).add(operator)
    replacements = []
    for match in WORDS.finditer(translation):
        left = _left(translation, match.start(), source)
        right = _right(translation, match.end(), source, True)
        operator = OPERATORS[match.group().lower()]
        if left is None or right is None or (operator, left, right[0]) not in allowed:
            continue
        if pair_operators[(left, right[0])] != {operator}:
            continue
        if operator == '/' and not right[1] and re.match(NUMBER + r'[A-Za-z]', right[0]):
            continue
        replacements.append((match.start(), match.end(), operator))
    for start, end, operator in reversed(replacements):
        translation = translation[:start] + operator + translation[end:]
    return translation


def has_math_operator_words(translation: str, source: str) -> bool:
    """현재 원문 변수가 있는 후보 수식에 영어 연산어가 남았는지 확인한다.

    :param translation: 정규화 뒤 기존 검증을 적용할 후보.
    :param source: 수식 변수의 기준 원문.
    :return: 영어 연산어 양옆 중 한쪽에 현재 원문의 변수가 있으면 True.

    일반 문장의 three times·over time은 숫자나 변수 사이 수식으로 판정하지 않는다.
    """
    for match in WORDS.finditer(translation):
        left = _left(translation, match.start(), source)
        right = _right(translation, match.end(), source, False)
        if (left is not None and re.fullmatch(NUMBER, left) is None
                or right is not None and re.fullmatch(NUMBER, right[0]) is None):
            return True
    return False
