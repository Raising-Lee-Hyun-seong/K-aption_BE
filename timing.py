"""요청·자막 구간의 상관관계와 작업별 경과 시간을 Uvicorn 로그에 기록한다."""
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

request_id = ContextVar("kaption_request_id", default="-")
segment_index = ContextVar("kaption_segment_index", default=None)
logger = logging.getLogger("uvicorn.error.kaption.timing")


def log_duration(event: str, elapsed_seconds: float, **fields) -> None:
    """작업 시간과 요청·구간 식별자 및 제공된 메타데이터를 한 로그로 기록한다.

    :param event: 시간 로그에 기록할 작업 이름.
    :param elapsed_seconds: 작업 수행에 걸린 시간(초).
    :param fields: 시간 로그에 덧붙일 개수·상태 등의 메타데이터.
    :return: None.
    """
    metadata = {
        "event": event,
        "request_id": request_id.get(),
        "segment_index": segment_index.get(),
        "elapsed_seconds": elapsed_seconds,
        **fields,
    }
    logger.info(
        "event=%s request_id=%s segment_index=%s elapsed_seconds=%.3f %s",
        event,
        metadata["request_id"],
        metadata["segment_index"],
        elapsed_seconds,
        " ".join(f"{key}={value}" for key, value in fields.items()),
        extra={"timing": metadata},
    )


@contextmanager
def measure(event: str, **fields):
    """작업의 경과 시간을 성공·실패 모두 기록하며 본문에서 메타데이터를 추가하게 한다.

    :param event: 시간 로그에 기록할 작업 이름.
    :param fields: 시간 로그에 덧붙일 개수·상태 등의 메타데이터.
    :return: 컨텍스트 관리자. 진입 시 본문에서 필드를 추가할 수 있는 메타데이터 사전. 컨텍스트 종료 시 성공·실패 시간을 기록한다.
    :yield: 본문에서 필드를 추가할 수 있는 메타데이터 사전. 컨텍스트 종료 시 성공·실패 시간을 기록한다.

    본문의 예외·취소는 시간 기록 후 그대로 전파한다.
    """
    started = perf_counter()
    status = "ok"
    try:
        yield fields
    except BaseException as exc:
        status = "error"
        fields["error_type"] = type(exc).__name__
        raise
    finally:
        log_duration(event, perf_counter() - started, status=status, **fields)
