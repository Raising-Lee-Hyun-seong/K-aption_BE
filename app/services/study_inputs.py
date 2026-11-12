"""완료된 자막을 근거 ID가 있는 학습자료 입력으로 변환하고 토큰 한도별로 묶는다."""

import json
import math
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from app.schemas import TranscriptionResponse


class StudySegment(BaseModel):
    """동일한 자막 결과 안에서 안정적인 ID와 원본 내용·분석 언어를 보존한다."""

    segment_id: str
    start: float
    end: float
    text: str
    translation: Optional[str]
    translation_error: Optional[Literal['translation_failed']]
    analysis_text: str
    analysis_language: Literal['en', 'ko']


class StudyChunk(BaseModel):
    """연속 자막의 원본 참조와 토큰 검사를 마친 실제 채팅 입력을 보관한다."""

    chunk_id: str
    segments: list[StudySegment]
    messages: list[dict[str, str]]
    prompt_tokens: int
    reserved_output_tokens: int


def _messages(segments: list[StudySegment], instruction: str) -> list[dict[str, str]]:
    """지시문과 분석용 자막 데이터를 분리해 손실 없는 JSON 채팅 입력을 만든다.

    :param segments: 현재 묶음에 포함할 시간순 자막.
    :param instruction: 호출자가 정한 챕터 또는 노트 생성용 시스템 지시문.
    :return: 시스템 지시와 ID·시각·선택 언어·본문 JSON을 담은 두 메시지.
    """
    rows = [{'segment_id': segment.segment_id, 'start': segment.start, 'end': segment.end,
             'language': segment.analysis_language, 'text': segment.analysis_text}
            for segment in segments]
    return [{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': json.dumps({'segments': rows}, ensure_ascii=False)}]


def prepare_study_inputs(
    transcription: TranscriptionResponse,
    count_tokens: Callable[[list[dict[str, str]]], int],
    *,
    instruction: str,
    context_window: int = 2048,
    output_tokens: int = 512,
) -> list[StudyChunk]:
    """자막을 수정하지 않고 참조 ID·번역 대체를 준비한 뒤 연속된 입력 묶음으로 나눈다.

    :param transcription: 기존 API의 완료 자막 결과. 빈 결과는 빈 묶음 목록으로 반환한다.
    :param count_tokens: 실제 서버의 채팅 템플릿·생성 시작 표식을 포함한 메시지 토큰 수 계산기.
        ServerTokenizer의 apply_chat_template과 encode를 순서대로 적용할 수 있다.
    :param instruction: 비어 있지 않은 분석용 시스템 지시문. 토큰 계산에 포함한다.
    :param context_window: 모델의 전체 문맥 한도(토큰). 기본값은 운영 Qwen의 2048이다.
    :param output_tokens: 생성 출력을 위해 예약할 양의 토큰 수. 문맥 한도보다 작아야 한다.
    :return: 모든 자막을 정확히 한 번 포함하는 시간순 묶음. ID는 동일 결과의 1부터 시작하는
        순서에 기반하며 다른 작업 사이의 전역 ID가 아니다. 파일 저장·모델 생성은 하지 않는다.
    :raises ValueError: 토큰 설정·계산 결과·자막 시각·원문이 유효하지 않거나 자막 하나가
        지시문 및 출력 예약과 함께 한도를 넘는 경우. 초과 자막 ID를 오류에 포함한다.

    번역 오류·누락·공백 번역은 영어 원문을 사용한다. 원문과 번역은 참조용으로 모두 보존한다.
    계산기가 로컬 서버에 접근하면 토큰화 HTTP 요청이 발생하며 해당 예외는 호출자에게 전달한다.
    반환된 messages를 변경하면 토큰 수를 다시 계산해야 한다. 문장 내 분할·시간 추정은 하지 않는다.
    """
    if (type(context_window) is not int or type(output_tokens) is not int
            or not 0 < output_tokens < context_window):
        raise ValueError('Require 0 < output_tokens < context_window in integer tokens')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('An analysis instruction is required')

    segments = []
    previous_start = -1.0
    for index, original in enumerate(transcription.segments, start=1):
        segment_id = f'segment-{index:06d}'
        if (not math.isfinite(original.start) or not math.isfinite(original.end)
                or original.start < 0 or original.end <= original.start
                or original.start < previous_start or not original.text.strip()):
            raise ValueError(f'Invalid source segment: {segment_id}')
        previous_start = original.start
        use_translation = (original.translation_error is None
                           and bool(original.translation and original.translation.strip()))
        segments.append(StudySegment(
            segment_id=segment_id, start=original.start, end=original.end,
            text=original.text, translation=original.translation,
            translation_error=original.translation_error,
            analysis_text=original.translation if use_translation else original.text,
            analysis_language='ko' if use_translation else 'en'))

    chunks = []
    pending = []
    pending_messages = []
    pending_tokens = 0
    for segment in segments:
        candidate = pending + [segment]
        messages = _messages(candidate, instruction)
        tokens = count_tokens(messages)
        if type(tokens) is not int or tokens < 0:
            raise ValueError('Token counter must return a nonnegative integer')
        if tokens + output_tokens > context_window:
            if pending:
                chunks.append(StudyChunk(
                    chunk_id=f'chunk-{len(chunks) + 1:06d}', segments=pending,
                    messages=pending_messages, prompt_tokens=pending_tokens,
                    reserved_output_tokens=output_tokens))
                candidate = [segment]
                messages = _messages(candidate, instruction)
                tokens = count_tokens(messages)
                if type(tokens) is not int or tokens < 0:
                    raise ValueError('Token counter must return a nonnegative integer')
            if tokens + output_tokens > context_window:
                raise ValueError(f'Segment exceeds token budget: {segment.segment_id}')
        pending, pending_messages, pending_tokens = candidate, messages, tokens
    if pending:
        chunks.append(StudyChunk(
            chunk_id=f'chunk-{len(chunks) + 1:06d}', segments=pending,
            messages=pending_messages, prompt_tokens=pending_tokens,
            reserved_output_tokens=output_tokens))
    return chunks
