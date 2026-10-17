"""저장된 자막 구간을 UTF-8 SRT·WebVTT 문서로 변환한다."""
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from math import isfinite
from operator import attrgetter
import re

from app.schemas import TranscriptionResponse

CONTROL_CHARACTERS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff]')


def _milliseconds(seconds: float) -> int:
    """초 단위 시각을 가장 가까운 밀리초로 반올림한다.

    :param seconds: 유한한 0 이상 초 단위 시각. 호출자가 미리 검증한다.
    :return: 정수 밀리초. 정확히 절반이면 올림한다.
    """
    return int((Decimal(str(seconds)) * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def _timestamp(milliseconds: int, separator: str) -> str:
    """정수 밀리초를 시간 올림이 반영된 자막 타임스탬프로 변환한다.

    :param milliseconds: 0 이상 정수 밀리초.
    :param separator: SRT는 쉼표, VTT는 마침표.
    :return: HH:MM:SS와 세 자리 밀리초로 구성한 문자열.
    """
    seconds, fraction = divmod(milliseconds, 1000)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f'{hour:02}:{minute:02}:{second:02}{separator}{fraction:03}'


def _plain_text(text: str) -> str:
    """원문의 제어 문자·빈 줄을 제거하고 자막의 한 줄 텍스트로 정규화한다.

    :param text: 영어 원문 또는 한국어 번역문.
    :return: 공백을 정리한 일반 텍스트. 마크업 이스케이프는 아직 하지 않는다.
    """
    return ' '.join(CONTROL_CHARACTERS.sub('', text).split())


def render_subtitles(
    result: TranscriptionResponse, file_format: str = 'srt', language: str = 'ko',
) -> str:
    """원문 시각을 밀리초로 변환해 한국어·영어·한영 자막을 생성한다.

    시작 시각순으로 안정 정렬하며 입력은 변경하지 않는다. 번역 오류·빈 번역은
    원문으로 대체하고 한영 모드에서도 중복 출력하지 않는다. 빈 줄·제어 문자를
    정리해 텍스트가 별도 cue로 해석되지 않게 한다.

    :param result: 기존 작업 API에서 조회한 자막 응답.
    :param file_format: srt 또는 vtt. 기본 srt.
    :param language: ko·en·both 중 하나. both는 한국어 다음 영어 순서.
    :return: 파일에 바로 UTF-8로 인코딩할 문서. 무음이면 SRT는 빈 문자열, VTT는 헤더만 반환한다.
    :raises ValueError: 형식·언어가 잘못됐거나 유한한 비음수 시작·양수 길이·비어 있지 않은 원문 조건을 위반할 때.

    시각은 반올림하며 밀리초에서 길이가 사라진 구간은 종료를 최소 1ms로 보정한다.
    VTT의 &, <, >는 엔티티로 이스케이프한다. SRT는 수식 기호를 보존하되
    본문의 -->는 타이밍 행 오인을 피하도록 →로 바꾼다.
    SRT의 HTML 형식 지원은 재생기에 따라 다르다. 파일 기록·모델 추론은 하지 않는다.
    """
    if file_format not in ('srt', 'vtt') or language not in ('ko', 'en', 'both'):
        raise ValueError('Unsupported subtitle format or language')
    for segment in result.segments:
        if not (isfinite(segment.start) and isfinite(segment.end)
                and 0 <= segment.start < segment.end):
            raise ValueError('Subtitle timestamps must be finite and have positive duration')
    newline, separator = ('\r\n', ',') if file_format == 'srt' else ('\n', '.')
    blocks = []
    for index, segment in enumerate(sorted(result.segments, key=attrgetter('start')), 1):
        source = _plain_text(segment.text)
        if not source:
            raise ValueError('Subtitle source text must not be empty')
        target = _plain_text(segment.translation or '') if not segment.translation_error else ''
        target = target or source
        lines = [source] if language == 'en' else [target]
        if language == 'both' and target != source:
            lines.append(source)
        start = _milliseconds(segment.start)
        end = max(start + 1, _milliseconds(segment.end))
        timing = f'{_timestamp(start, separator)} --> {_timestamp(end, separator)}'
        payload = newline.join(escape(line, quote=False) if file_format == 'vtt'
                               else line.replace('-->', '→') for line in lines)
        blocks.append(newline.join((str(index), timing, payload)))
    body = (newline * 2).join(blocks) + (newline * 2 if blocks else '')
    return 'WEBVTT\n\n' + body if file_format == 'vtt' else body
