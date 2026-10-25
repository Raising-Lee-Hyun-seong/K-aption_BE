"""번역의 반복·지시문·언어 혼입과 원문 숫자·수식 기호의 유실을 검출한다."""
import re
from decimal import Decimal
from typing import Optional


RAW_SYMBOL_PATTERN = (
    r"(?:d[\s_]*m[\s_]*[rR][\s_]*u|d[\s_]*[mv][\s_]*[rR]|"
    r"[mMvV][\s_]+[rR]|[mv][\s_]*[rR]|[mv]_?0|d[\s_]*[tmv]|ln|[mutvg])"
)
SYMBOL_LEFT_BOUNDARY = r"(?<![A-Za-z_])(?<![A-Za-z]['’])"
SYMBOL_PATTERN = re.compile(
    SYMBOL_LEFT_BOUNDARY + RAW_SYMBOL_PATTERN + r"(?![A-Za-z0-9_])"
)
SYMBOL_CANONICAL = {
    'dmr': 'dmR', 'dvr': 'dvR', 'mr': 'mR', 'vr': 'vR',
    'm0': 'm0', 'v0': 'v0', 'dt': 'dt', 'dm': 'dm', 'dv': 'dv',
    'ln': 'ln', 'm': 'm', 'u': 'u', 't': 't', 'v': 'v', 'g': 'g',
}
RELATION_NUMBER = r'[-−]?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
RELATION_ATOM = (
    r'(?:(?:' + RELATION_NUMBER + r')\s*)?(?:' + RAW_SYMBOL_PATTERN
    + r'|u\d+)|' + RELATION_NUMBER
)
SPOKEN_RELATION = re.compile(
    SYMBOL_LEFT_BOUNDARY + r'(?<![0-9.,])(?P<left>' + RELATION_ATOM
    + r')\s+(?P<operator>(?i:over|of))\s+(?P<right>' + RELATION_ATOM
    + r')(?![A-Za-z0-9_])'
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])[-−]?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?=$|[^A-Za-z0-9_]|(?:" + RAW_SYMBOL_PATTERN
    + r"|kg|km|cm|mm)(?![A-Za-z0-9_]))"
)
UNIT_PREFIX_PATTERN = re.compile(
    r'(?:\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten))\s*$',
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero": ("0", "영", "제로"), "one": ("1", "하나", "한"),
    "two": ("2", "둘", "두"), "three": ("3", "셋", "세"),
    "four": ("4", "넷", "네"), "five": ("5", "다섯"),
    "six": ("6", "여섯"), "seven": ("7", "일곱"),
    "eight": ("8", "여덟"), "nine": ("9", "아홉"),
    "ten": ("10", "열"),
}
LEAK_MARKERS = (
    "### Instruction", "### Input", "### 지시", "참고 문맥:",
    "이전 문맥(", "이전 영어 문맥:", "번역할 대상:", "용어:",
    "설명 없이 번역문만", "번역하거나 반복하지", "한국어로 번역하세요",
    "라는 영어 원문", "로 번역됩니다", "입력된 문장", "그대로 쓸 기호:",
    "(참고만 하세요.)", "원문을 한국어로 번역",
    "이전 영문 맥락", "Previous English context:",
)
FOREIGN_PROSE = re.compile(r'[\u0400-\u052f]{2,}|[\u3040-\u30ff]{2,}|[\u3400-\u9fff]{2,}')
UNIT_ALIASES = {
    'kilogram': (r'kg|kilograms?', '킬로그램'),
    'gram': (r'g|grams?', '그램'),
    'kilometer': (r'km|kilometers?|kilometres?', '킬로미터'),
    'centimeter': (r'cm|centimeters?|centimetres?', '센티미터'),
    'millimeter': (r'mm|millimeters?|millimetres?', '밀리미터'),
    'meter': (r'm|meters?|metres?', '미터'),
}


def _numbers(text: str) -> set:
    """변수의 첨자와 구별한 부호 포함 숫자를 표기와 무관한 값으로 반환한다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :return: 중복을 제거하고 Decimal로 정규화한 숫자 값 집합.
    """
    return {Decimal(re.sub(r'\s+', '', value).replace('−', '-').replace(',', ''))
            for value in NUMBER_PATTERN.findall(text)}


def _is_unit_symbol(text: str, start: int, symbol: str) -> bool:
    """숫자·영어 수사 바로 뒤의 m·g를 변수 대신 단위 약자로 구분한다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :param start: 검사할 기호가 텍스트에서 시작하는 문자 위치.
    :param symbol: 단위 약자인지 확인할 기호(m 또는 g).
    :return: 수량 뒤의 m·g를 단위로 판단하면 True.
    """
    return symbol in ('m', 'g') and bool(UNIT_PREFIX_PATTERN.search(text[:start]))


def _units(text: str, include_korean: bool = False) -> set:
    """지원하는 길이·질량 단위를 찾고 한국어 복합 단위는 긴 이름을 우선한다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :param include_korean: 참이면 한국어 단위 표기도 함께 인식한다.
    :return: 발견한 단위를 표준 이름으로 정규화한 집합.
    """
    result = set()
    compound_spans = [match.span() for match in SYMBOL_PATTERN.finditer(text)
                      if len(re.sub(r'\s+', '', match.group())) > 1]
    for key, (english, _) in UNIT_ALIASES.items():
        for match in re.finditer(r'(?<![A-Za-z_])(?:' + english + r')(?![A-Za-z0-9_])',
                                 text, re.IGNORECASE):
            if match.group().lower() in ('m', 'g'):
                if (any(start <= match.start() < end for start, end in compound_spans)
                        or not _is_unit_symbol(text, match.start(), match.group().lower())):
                    continue
            result.add(key)
    if include_korean:
        korean_to_key = {korean: key for key, (_, korean) in UNIT_ALIASES.items()}
        pattern = '|'.join(sorted(korean_to_key, key=len, reverse=True))
        result.update(korean_to_key[match] for match in re.findall(pattern, text))
    return result


def protected_symbols(text: str) -> list[str]:
    """원문 수식의 공백·밑줄·소문자 별칭을 정규화해 등장 순서대로 중복 없이 반환한다.

    dmRu는 dmR과 u로 보존한다. 복합 기호 속 단일 글자를 중복 추출하지 않으며,
    수량 뒤 m·g 단위 약자와 I'm·don't 같은 영어 축약형의 접미사는 독립 변수로
    취급하지 않는다. 따옴표 안의 변수와 변수 뒤 프라임은 보존하며 원문은 수정하지 않는다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :return: 원문에서 보존할 정규화된 수식 기호를 등장 순서로 담은 목록.
    """
    symbols = []
    for match in SYMBOL_PATTERN.finditer(text):
        compact = re.sub(r'[\s_]+', '', match.group()).lower()
        if _is_unit_symbol(text, match.start(), compact):
            continue
        values = ['dmR', 'u'] if compact == 'dmru' else [SYMBOL_CANONICAL[compact]]
        symbols.extend(values)
    return list(dict.fromkeys(symbols))


def _canonical_relation_atom(text: str) -> str:
    """관계식의 단일 기호·수치 계수에서 공백·밑줄 별칭을 정규화한다.

    :param text: 원문에서 추출한 지원 기호·숫자 또는 수치 계수와 기호.
    :return: 수치를 유지하고 mR·vR 등의 표기만 정규화한 문자열.
    """
    compact = re.sub(r'[\s_,]+', '', text).replace('−', '-')
    number = re.match(RELATION_NUMBER, compact)
    coefficient = number.group() if number else ''
    symbol = compact[len(coefficient):]
    if coefficient:
        value = Decimal(coefficient)
        coefficient = format(value.normalize(), 'f') if value else '0'
    return coefficient + SYMBOL_CANONICAL.get(symbol.lower(), symbol)


def spoken_math_relations(text: str) -> list[tuple[str, str, str]]:
    """지원하는 기호의 over 나눗셈과 of 함수 관계를 원문에서 보수적으로 추출한다.

    :param text: 관계를 보존할 영어 원문. 내용을 수정하거나 수식을 풀지 않는다.
    :return: (over 또는 of, 왼쪽 항, 오른쪽 항) 튜플 목록. ln 뒤 분모만 있으면 왼쪽 항은 빈 문자열.

    단위 약자와 일반 영어 구문은 제외한다. 지원 기호·숫자·u1 형태만 다루며
    일반 수식 파서나 STT 오류 복원기가 아니다. ln이 있는 발언의 고립된 over는
    분자·로그 범위를 추론하지 않고 분모 보존 단서만 반환한다.
    """
    relations, spans = [], []
    for match in SPOKEN_RELATION.finditer(text):
        left, right = match['left'], match['right']
        atoms = (left, right)
        if any(not (protected_symbols(atom) or re.fullmatch(RELATION_NUMBER, atom)
                    or re.fullmatch(r'u\d+', atom)) for atom in atoms):
            continue
        operator = match['operator'].lower()
        if operator == 'of' and not protected_symbols(left):
            continue
        relations.append((operator, _canonical_relation_atom(left), _canonical_relation_atom(right)))
        spans.append(match.span())
    if 'ln' in protected_symbols(text):
        for match in re.finditer(r'\b(?i:over)\s+(' + RELATION_ATOM + r')(?![A-Za-z0-9_])', text):
            if (not any(start <= match.start() < end for start, end in spans)
                    and protected_symbols(match[1])):
                relations.append(('over', '', _canonical_relation_atom(match[1])))
    return list(dict.fromkeys(relations))


def _relation_atom_pattern(atom: str) -> str:
    """알려진 관계 항의 공백을 허용하는 표기 정규식을 만든다.

    :param atom: 정규화된 숫자·기호 항.
    :return: 항 내부 공백을 허용하고 각 문자를 이스케이프한 정규식.
    """
    return r'\s*'.join(re.escape(char) for char in atom.lower())


def math_relation_problem(source: str, translation: str) -> Optional[str]:
    """명시적인 나눗셈의 항·방향과 함수의 인수 관계가 출력에 남는지 검사한다.

    :param source: 지원 기호로 된 over·of 관계를 포함할 수 있는 영어 원문.
    :param translation: 기호·숫자·언어 검사와 별도로 확인할 번역 후보.
    :return: missing_math_division 또는 missing_math_function. 지원 관계가 보존되거나 없으면 None.

    슬래시·분수 명령·분의·나누기와 함수 괄호·명시적 한국어 함수 표현을
    허용한다. 영문 over 보존은 관계 검사만 통과하며 기존 영어 연산어 검사에서
    별도로 거부한다. 일반 의미 정확도나 임의 대수적 동치까지 판정하지 않는다.
    """
    def normalize_atom(match):
        """출력의 지원 관계 항에서 같은 값의 소수·천 단위·기호 별칭을 정규화한다.

        :param match: 정규식이 찾은 단일 관계 항의 매치 객체.
        :return: 비교에만 사용할 정규화된 항. 실제 번역문은 수정하지 않는다.
        """
        return _canonical_relation_atom(match.group())

    output = re.sub(SYMBOL_LEFT_BOUNDARY + r'(?<![0-9.,])(?:' + RELATION_ATOM
                    + r')(?![A-Za-z0-9_])', normalize_atom, translation)
    output = output.lower().replace('_', '').replace('−', '-')
    for operator, left, right in spoken_math_relations(source):
        a, b = _relation_atom_pattern(left), _relation_atom_pattern(right)
        start, end = r'(?<![a-z0-9])', r'(?![a-z0-9])'
        wrapped_a = r'[([{]*\s*' + a + r'\s*[)\]}]*'
        wrapped_b = r'[([{]*\s*' + b + r'\s*[)\]}]*'
        if operator == 'of':
            forms = (
                start + a + r"\s*\(\s*" + b + r'\s*\)',
                start + b + r'\s*(?:의|에서의|에\s*대한)\s*' + a + end,
                start + a + r'\s*(?:은|는)?\s*' + b + r'\s*(?:의|에\s*대한)\s*함수',
            )
            reason = 'missing_math_function'
        elif left:
            forms = (
                start + wrapped_a + r'\s*(?:/|over)\s*' + wrapped_b + end,
                r'\\(?:d?frac)\s*\{\s*' + a + r'\s*\}\s*\{\s*' + b + r'\s*\}',
                start + b + r'\s*분의\s*' + a + end,
                start + a + r'\s*(?:을|를)?\s*' + b + r'\s*(?:으)?로\s*(?:나누|나눈|나눕|나눠)',
                start + b + r'\s*(?:으)?로\s*(?:나눈|나누어진|나누어\s*진|나뉜)\s*' + a + end,
            )
            reason = 'missing_math_division'
        else:
            forms = (
                r'(?:/|\bover)\s*' + wrapped_b + end,
                start + b + r'\s*분의',
                start + b + r'\s*(?:으)?로\s*(?:나누|나눈|나눕|나눠)',
                r'\\(?:d?frac)\s*\{[^{}]+\}\s*\{\s*' + b + r'\s*\}',
            )
            reason = 'missing_math_division'
        if not any(re.search(form, output) for form in forms):
            return reason
    return None


def _has_math_prime(text: str) -> bool:
    """프라임의 영문·한글·전용 기호 또는 변수 뒤 아포스트로피가 있는지 확인한다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :return: 수학 문맥의 프라임 표기를 발견하면 True.
    """
    if re.search(r'(?<![A-Za-z0-9_])primes?(?![A-Za-z0-9_])|프라임|′',
                 text, re.IGNORECASE):
        return True
    return bool(re.search(SYMBOL_LEFT_BOUNDARY + RAW_SYMBOL_PATTERN
                          + r"\s*['’](?![A-Za-z0-9_])", text))


def generation_problem(output: str) -> Optional[str]:
    """생성 중인 텍스트에 지시문 유출·손상 문자·뚜렷한 반복이 있는지 검사한다.

    :param output: 검사하거나 응답을 추출할 모델 생성 텍스트.
    :return: 생성 문제의 사유 코드. 지시 유출·손상·반복을 발견하지 못하면 None.
    """
    if any(marker in output for marker in LEAK_MARKERS):
        return "prompt_leak"
    if "\ufffd" in output:
        return "invalid_character"
    compact = re.sub(r"\s+", "", output)
    if re.search(r"(.{4,60}?)\1\1", compact) or re.search(r"(.)\1{11,}", compact):
        return "repetition"
    return None


def preservation_problem(source: str, translation: str) -> Optional[str]:
    """기호·숫자·단위·지원 함수와 분수 보존 및 원문에 없는 외국 문자 혼입을 검사한다.

    키릴·가나·한자 문자가 두 자 이상 이어지면 원문에도 같은 문자열이 있는지
    확인한다. 한글 바로 뒤 괄호의 한자 병기는 허용한다. 단일 수식 문자·그리스
    문자는 이 검사에서 제외하며 일반적인 언어·의미 판별을 보장하지 않는다.

    :param source: 숫자·수식·단위 보존의 기준이 되는 영어 원문.
    :param translation: 정리하거나 검증할 한국어 번역 후보 문자열.
    :return: 보존 실패 또는 unexpected_script 사유 코드. 문제를 발견하지 못하면 None.
    """
    prose = re.sub(r'(?<=[가-힣])\([\u3400-\u9fff]+\)', '', translation)
    if any(match.group() not in source for match in FOREIGN_PROSE.finditer(prose)):
        return 'unexpected_script'
    source_symbols = protected_symbols(source)
    output_symbols = protected_symbols(translation)
    for symbol in source_symbols:
        if symbol not in output_symbols:
            return "missing_math_symbol"
    if source_symbols and re.search(r'\bprimes?\b', source, re.IGNORECASE):
        if not _has_math_prime(translation):
            return "missing_math_prime"
    source_numbers = _numbers(source)
    output_numbers = _numbers(translation)
    if not source_numbers.issubset(output_numbers):
        return "missing_number"
    for word, alternatives in NUMBER_WORDS.items():
        for match in re.finditer(r"\b(?:(minus|negative)\s+)?" + word + r"\b", source, re.IGNORECASE):
            value = Decimal(alternatives[0]) * (-1 if match.group(1) else 1)
            source_numbers.add(value)
            numeric_match = value in output_numbers
            korean_match = any(re.search(r"(?<![가-힣])" + value
                                        + (r'(?=\s|$|개|명|번|의|이|입|인|에|으|과|와|를|만|도|은|는|로)'
                                           if len(value) == 1 else ''), translation)
                               for value in alternatives[1:])
            if match.group(1):
                korean_match = korean_match and bool(re.search(r'마이너스|음수|음의', translation))
            if not (numeric_match or korean_match):
                return "missing_number"
    if not output_numbers.issubset(source_numbers):
        return "invented_number"
    if not _units(translation, include_korean=True).issubset(_units(source)):
        return "invented_unit"
    return math_relation_problem(source, translation)
