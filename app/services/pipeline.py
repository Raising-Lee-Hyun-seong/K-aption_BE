"""동기 응답과 백그라운드 작업에서 공유하는 STT·문장 묶기·번역 파이프라인을 제공한다."""
from pathlib import Path
from typing import Awaitable, Callable, Optional

from fastapi.concurrency import run_in_threadpool

from app.schemas import TranscriptionResponse
from app.services.subtitles import group_segments
from timing import measure, segment_index

ProgressCallback = Callable[[str, int, Optional[int], int], Awaitable[None]]


class MediaValidationError(ValueError):
    """추론 파이프라인의 STT 입력 검증 실패를 다른 처리 오류와 구분한다."""


async def process_media(
    media_path: Path, transcriber, translator, language: str = 'en',
    progress: Optional[ProgressCallback] = None,
) -> TranscriptionResponse:
    """STT 결과를 문장으로 묶고 앞 원문 문맥으로 번역하며 실제 구간 진행을 알린다.

    :param media_path: 처리할 로컬 미디어 경로. 파일 삭제는 호출자가 담당한다.
    :param transcriber: transcribe 메서드를 제공하는 STT 서비스.
    :param translator: translate 메서드를 제공하는 번역 서비스.
    :param language: 음성 인식 언어 코드. API에서는 en만 허용한다.
    :param progress: 단계·완료 구간 수·전체 구간 수 또는 None·번역 실패 수를 받는 비동기 콜백.
    :return: 시각·원문·번역·부분 실패를 보존한 자막 응답.
    :raises MediaValidationError: STT의 오디오·길이 검증이 실패할 때.

    모델 추론을 실행하고 시간 로그를 남긴다. 콜백과 예기치 않은 서비스 오류는 전파한다.
    """
    if progress:
        await progress('transcribing', 0, None, 0)
    try:
        with measure('stt') as metrics:
            result = await run_in_threadpool(transcriber.transcribe, media_path, language)
            metrics['segment_count'] = len(result['segments'])
    except ValueError as exc:
        raise MediaValidationError(str(exc)) from exc
    if progress:
        await progress('grouping', 0, None, 0)
    with measure('sentence_grouping', source_segment_count=len(result['segments'])) as metrics:
        sentences = group_segments(result['segments'])
        metrics['sentence_count'] = len(sentences)
    previous_source, translated_segments, failed_segments = '', [], 0
    if progress:
        await progress('translating', 0, len(sentences), 0)
    with measure('translation_batch', source_segment_count=len(result['segments']),
                 sentence_count=len(sentences), failed_segments=0) as metrics:
        for index, sentence in enumerate(sentences):
            segment = {key: sentence[key] for key in ('start', 'end', 'text')}
            token = segment_index.set(index)
            try:
                with measure('segment_translation'):
                    translated = await run_in_threadpool(
                        translator.translate, segment['text'], previous_source, 128,
                    )
            except ValueError:
                translated_segments.append({**segment, 'translation': None,
                                            'translation_error': 'translation_failed'})
                failed_segments += 1
            else:
                translated_segments.append({**segment, 'translation': translated['translation'],
                                            'translation_error': None})
            finally:
                previous_source = segment['text'][-256:]
                segment_index.reset(token)
                metrics['failed_segments'] = failed_segments
            if progress:
                await progress('translating', index + 1, len(sentences), failed_segments)
    return TranscriptionResponse(segments=translated_segments)
