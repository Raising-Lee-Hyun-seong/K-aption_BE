"""STT의 짧은 영어 조각을 문장과 시간 경계에 따라 자막 번역 단위로 묶는다."""
import re
from typing import Optional

from app.services.translation_validation import protected_symbols

SENTENCE_END = re.compile(r"[.!?][\"'’”\)\]]*$")
MAX_GAP_SECONDS = 2.0
INCOMPLETE_MATH_END = re.compile(
    r"\b(?:with|equals|plus|minus|over|of|from|to|is an?|(?:our|the|its) initial|"
    r"(?:we|you|they|I)(?: (?:can|will|would|should|must))?)\s*[,;:]*$",
    re.IGNORECASE,
)


def _math_link_reason(left: str, right: str) -> Optional[str]:
    """인접한 수학 설명에 명시적인 연결·정정 단서가 있는지 확인한다.

    :param left: 이미 묶인 앞 영어 원문. 수정하거나 수식을 복원하지 않는다.
    :param right: 다음 영어 원문.
    :return: 결합 근거 코드 또는 None. 수식의 완전성·정확성을 판정하지 않는다.
    """
    if re.search(r'[?!]["\'’”\)\]]*$', left) or re.search(r'[?!]["\'’”\)\]]*$', right):
        return None
    left_symbols, right_symbols = set(protected_symbols(left)), set(protected_symbols(right))
    if not left_symbols:
        return None
    previous = left.lower().replace('’', "'")
    following = right.lower().replace('’', "'")
    equation = bool(re.search(r'\b(?:equals|over|plus|minus)\b|=', previous))
    position = r'\b(?:goes?|comes?|moves?)\s+(?:up\s+)?(?:front|outside)\b'
    if (equation and re.match(r'^(?:(?:so|and|then)\s+)?(?:that|it|this)\b', following)
            and re.search(position, following) and len(right) <= 100):
        return 'factor_position'
    correction = r'\b(?:actually|rather|instead|correction|sorry)\b.*\b(?:not|quite|instead|rather)\b'
    if left_symbols & right_symbols and re.search(correction, following):
        return 'math_correction'
    continuation = r"^(?:(?:and|so)\s+(?:then\s+)?)?(?:we (?:have|got|get|are left with)|it(?:'s| is)|that(?:'s| is)|this is)\b"
    if (re.search(correction, previous) and left_symbols & right_symbols
            and re.search(continuation, following)):
        return 'correction_explanation'
    if re.search(r'\b(?:going|go|goes|runs?|ranges?)\s+from\b', previous):
        if re.search(r'\binitial condition\b', following) and len(right) <= 100:
            return 'limit_explanation'
        if (right_symbols and len(right) <= 100
                and re.match(r'^(?:to|\d+(?:\.\d+)?)\s+', following)):
            return 'limit_endpoint'
    if (right_symbols and not re.search(r'\bequals\b|=', following) and re.search(continuation, following)
            and (re.search(position, previous) or re.search(r'\bis a constant\b', previous))):
        return 'equation_continuation'
    if equation and 'ln' in right_symbols and re.search(r'\bintegrat(?:ed|ing|ion)\b', following):
        return 'integration_result'
    if ('ln' in left_symbols and re.search(r'\bintegrat(?:ed|ing|ion)\b', previous)
            and re.fullmatch(r'(?:and\s+)?(?:so\s+)?ln[. ]*', following)):
        return 'logarithm_continuation'
    if (re.search(r'\bof\s+[tv]\b', previous) and right_symbols
            and re.search(r'\b(?:equals|over|plus|minus)\b|=', following)
            and re.match(r'^and\s+(?:then\s+)?we\b', following)):
        return 'function_expression'
    return None


def merge_math_groups(groups: list[dict], max_duration: float = 18.0,
                      max_chars: int = 350) -> list[dict]:
    """시간·길이 제한 안에서 수식 설명·범위·정정 발언을 원문 그대로 합친다.

    :param groups: start·end·text·source_indices를 가진 기본 문장 묶음 목록.
    :param max_duration: 여러 묶음을 합칠 때 허용할 최대 시간(초).
    :param max_chars: 연결 공백을 포함한 최대 원문 문자 수.
    :return: 원래 시각·인덱스를 보존하고 결합된 항목에 merge_reasons를 기록한 목록.
    :raises ValueError: 시간·문자 상한이 양수가 아닌 경우.

    숫자·변수·문장부호를 고치거나 누락된 수식을 추론하지 않는다. 입력 목록과
    내부 인덱스 목록은 수정하지 않으며 2초를 초과하는 무음에서는 합치지 않는다.
    """
    if max_duration <= 0 or max_chars <= 0:
        raise ValueError('Subtitle grouping limits must be positive')
    merged = []
    for group in groups:
        current = {**group, 'source_indices': list(group['source_indices'])}
        if merged:
            previous = merged[-1]
            reason = _math_link_reason(previous['text'], current['text'])
            if (reason and current['start'] - previous['end'] <= MAX_GAP_SECONDS
                    and current['end'] - previous['start'] <= max_duration
                    and len(previous['text']) + 1 + len(current['text']) <= max_chars):
                previous['end'] = current['end']
                previous['text'] += ' ' + current['text']
                previous['source_indices'] += current['source_indices']
                previous['merge_reasons'] = [*previous.get('merge_reasons', []), reason]
                continue
        merged.append(current)
    return merged


def merge_incomplete_math_groups(groups: list[dict], max_duration: float = 36.0,
                                 max_chars: int = 350) -> list[dict]:
    """상한에서 끊긴 명확한 미완성 수학 발언을 다음 구간과 제한 안에서 연결한다.

    :param groups: start·end·text·source_indices를 가진 기존 문장 묶음 목록.
    :param max_duration: 미완성 발언을 이어 붙일 때의 최대 시간(초). 기본 36초.
    :param max_chars: 공백을 포함한 최대 원문 길이. 기본 350자.
    :return: 원문·시각·인덱스를 보존하고 결합 사유를 담은 새 묶음 목록.
    :raises ValueError: 시간·문자 상한이 양수가 아닐 때.

    수식 기호와 명확한 미완성 어미가 함께 있는 경우만 연결한다. 문장 종결,
    다음 질문, 2초 초과 무음과 상한을 넘어서는 결합은 하지 않는다.
    기존 긴 단일 묶음은 자르지 않으며 수식을 풀거나 원문을 보정하지 않는다.
    """
    if max_duration <= 0 or max_chars <= 0:
        raise ValueError('Continuation limits must be positive')
    merged = []
    for group in groups:
        current = {**group, 'source_indices': list(group['source_indices'])}
        if merged:
            previous = merged[-1]
            left, right = previous['text'], current['text']
            if (not SENTENCE_END.search(left)
                    and not re.search(r'[?!]["\'’”\)\]]*$', right)
                    and INCOMPLETE_MATH_END.search(left)
                    and (protected_symbols(left) or protected_symbols(right))
                    and current['start'] - previous['end'] <= MAX_GAP_SECONDS
                    and current['end'] - previous['start'] <= max_duration
                    and len(left) + 1 + len(right) <= max_chars):
                previous['end'] = current['end']
                previous['text'] = left + ' ' + right
                previous['source_indices'] += current['source_indices']
                previous['merge_reasons'] = [*previous.get('merge_reasons', []),
                                             'incomplete_math_continuation']
                continue
        merged.append(current)
    return merged


def group_segments(
    segments: list[dict], max_duration: float = 18.0, max_chars: int = 350,
    merge_math: bool = False,
) -> list[dict]:
    """순서를 유지하며 STT 조각을 묶고 첫·마지막 시각과 원본 인덱스를 반환한다.

    문장 끝의 마침표·물음표·느낌표, 두 구간 사이 2초를 초과하는 공백,
    합친 길이·시간 상한에서 그룹을 닫는다. 빈 원문은 제외한다. 원본 구간을
    분할하지 않으므로 단일 구간이 상한을 넘으면 시간과 원문을 그대로 보존한다.
    상한이 양수가 아니면 ValueError를 발생시킨다.

    :param segments: 시작·종료 시각과 text를 담은 STT 구간 목록.
    :param max_duration: 여러 구간을 합칠 때의 최대 시간(초). 긴 단일 구간은 보존한다.
    :param max_chars: 합친 원문에 허용할 최대 문자 수. 연결 공백을 포함한다.
    :param merge_math: 참이면 기본 묶음 뒤에 수학 설명의 보수적 결합을 적용한다. 기본 False.
    :return: start·end·text·source_indices와 선택적 merge_reasons를 담은 자막 묶음 목록.
    :raises ValueError: 최대 시간 또는 최대 문자 수가 양수가 아닐 때.
    """
    if max_duration <= 0 or max_chars <= 0:
        raise ValueError("Subtitle grouping limits must be positive")

    grouped = []
    pending = None
    for index, segment in enumerate(segments):
        text = " ".join(segment["text"].split())
        if not text:
            continue
        if pending is not None:
            gap = segment["start"] - pending["end"]
            duration = segment["end"] - pending["start"]
            chars = len(pending["text"]) + 1 + len(text)
            if gap > MAX_GAP_SECONDS or duration > max_duration or chars > max_chars:
                grouped.append(pending)
                pending = None

        if pending is None:
            pending = {
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
                "source_indices": [index],
            }
        else:
            pending["end"] = segment["end"]
            pending["text"] += " " + text
            pending["source_indices"].append(index)

        if SENTENCE_END.search(text):
            grouped.append(pending)
            pending = None

    if pending is not None:
        grouped.append(pending)
    return merge_math_groups(grouped, max_duration, max_chars) if merge_math else grouped
