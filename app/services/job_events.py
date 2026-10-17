"""작업 상태 스냅샷을 SSE로 전달하고 종료·만료·연결 해제를 처리한다."""

import asyncio
import json
from time import monotonic
from typing import AsyncIterator

from app.schemas import JobStatusResponse
from app.services.jobs import JobManager

POLL_SECONDS = 1.0
HEARTBEAT_SECONDS = 15.0


async def job_events(manager: JobManager, initial: JobStatusResponse) -> AsyncIterator[str]:
    """최초 상태와 변경을 보내고 완료·실패·취소 완료·조회 불가 시 스트림을 끝낸다.

    :param manager: 기존 작업을 조회하는 관리자. 새 작업이나 모델 추론을 만들지 않는다.
    :param initial: 응답 헤더 전송 전에 조회·검증한 최초 작업 상태.
    :return: UTF-8로 전송할 SSE 문자열을 제공하는 비동기 생성기.
    :yield: status 이벤트의 JobStatusResponse JSON 또는 heartbeat 주석. 조회 불가는
        unavailable 이벤트의 안전한 code·message JSON을 제공한 후 종료한다.
    :raises asyncio.CancelledError: 연결 종료 등으로 응답 태스크가 취소된 경우. 작업은 계속된다.

    1초마다 조회하며 경과 시간만 바뀌면 상태 이벤트를 생략한다. 변경이 없어도
    15초마다 주석을 보낸다. 이벤트 이력·파일은 저장하지 않는다.
    """
    record = initial
    previous = None
    last_sent = monotonic()
    while True:
        comparable = record.model_dump(mode='json', exclude={'elapsed_seconds'})
        if comparable != previous:
            previous = comparable
            last_sent = monotonic()
            yield f'event: status\ndata: {record.model_dump_json()}\n\n'
        if record.status in ('completed', 'failed', 'cancelled'):
            return
        if monotonic() - last_sent >= HEARTBEAT_SECONDS:
            last_sent = monotonic()
            yield ': heartbeat\n\n'
        await asyncio.sleep(POLL_SECONDS)
        try:
            record = manager.get(initial.job_id)
        except (KeyError, OSError) as error:
            payload = {'code': 'job_not_found' if isinstance(error, KeyError) else 'storage_error',
                       'message': 'Job status is unavailable; query the job status endpoint'}
            yield 'event: unavailable\ndata: ' + json.dumps(payload) + '\n\n'
            return
