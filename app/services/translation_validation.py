"""번역의 반복·지시문 유출과 원문 숫자·수식 기호의 유실을 검출한다."""
import re
from decimal import Decimal
from typing import Optional


RAW_SYMBOL_PATTERN = (
    r"(?:d\s*m\s*[rR]\s*u|d\s*[mv]\s*[rR]|[mMvV]\s+[rR]|[mv]\s*[rR]|[mv]0|"
    r"d\s*[tmv]|ln|[mutvg])"
)
SYMBOL_PATTERN = re.compile(
    r"(?<![A-Za-z_])" + RAW_SYMBOL_PATTERN + r"(?![A-Za-z0-9_])"
)
SYMBOL_CANONICAL = {
    'dmr': 'dmR', 'dvr': 'dvR', 'mr': 'mR', 'vr': 'vR',
    'm0': 'm0', 'v0': 'v0', 'dt': 'dt', 'dm': 'dm', 'dv': 'dv',
    'ln': 'ln', 'm': 'm', 'u': 'u', 't': 't', 'v': 'v', 'g': 'g',
}
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
)
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
    """원문 수식의 공백·소문자 별칭을 정규화해 등장 순서대로 중복 없이 반환한다.

    dmRu는 dmR과 u로 보존한다. 복합 기호 속 단일 글자를 중복 추출하지 않으며,
    수량 뒤 m·g 단위 약자는 독립 변수로 취급하지 않는다. 원문 자체는 수정하지 않는다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :return: 원문에서 보존할 정규화된 수식 기호를 등장 순서로 담은 목록.
    """
    symbols = []
    for match in SYMBOL_PATTERN.finditer(text):
        compact = re.sub(r'\s+', '', match.group()).lower()
        if _is_unit_symbol(text, match.start(), compact):
            continue
        values = ['dmR', 'u'] if compact == 'dmru' else [SYMBOL_CANONICAL[compact]]
        symbols.extend(values)
    return list(dict.fromkeys(symbols))


def _has_math_prime(text: str) -> bool:
    """프라임의 영문·한글·전용 기호 또는 변수 뒤 아포스트로피가 있는지 확인한다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :return: 수학 문맥의 프라임 표기를 발견하면 True.
    """
    if re.search(r'(?<![A-Za-z0-9_])primes?(?![A-Za-z0-9_])|프라임|′',
                 text, re.IGNORECASE):
        return True
    return bool(re.search(r'(?<![A-Za-z_])' + RAW_SYMBOL_PATTERN
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
    """수식 별칭·프라임·숫자의 유실과 원문에 없는 숫자·단위 추가를 검사한다.

    :param source: 숫자·수식·단위 보존의 기준이 되는 영어 원문.
    :param translation: 정리하거나 검증할 한국어 번역 후보 문자열.
    :return: 수식·숫자·단위 보존 실패 사유 코드. 문제를 발견하지 못하면 None.
    """
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
    return None
