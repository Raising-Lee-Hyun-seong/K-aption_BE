"""로컬 파일 기반 작업 상태·결과를 보관하고 단일 백그라운드 작업자를 운영한다."""
import asyncio
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
from time import time
from typing import Callable
from uuid import uuid4

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import JobError, JobStatusResponse, TranscriptionResponse
from app.services.media import (
    ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES, UPLOAD_TOO_LARGE, UploadTooLargeError,
    copy_upload, validate_media_duration,
)
from app.services.pipeline import MediaValidationError, process_media
from timing import log_duration, measure, request_id

JOB_ID = re.compile(r'^[0-9a-f]{32}$')


class JobCapacityError(RuntimeError):
    """미완료 작업 한도에 도달했거나 작업자가 접수를 중단했음을 나타낸다."""


class UnsupportedMediaError(ValueError):
    """지원하지 않는 업로드 확장자를 나타낸다."""


class JobNotReadyError(RuntimeError):
    """작업이 완료되지 않아 결과를 반환할 수 없음을 나타낸다."""


class JobStorageError(RuntimeError):
    """저장된 작업 결과가 없거나 손상됐음을 나타낸다."""


class JobManager:
    """업로드 포함 미완료 작업 수를 제한하고 작업을 하나씩 처리·보관한다."""

    def __init__(self, root: Path, transcriber, translator, max_active: int = 4,
                 ttl_seconds: int = 86400, clock: Callable[[], float] = time):
        """저장 경로·공유 모델·용량·보관 시간과 내부 대기열을 준비한다.

        :param root: 작업 전용 로컬 디렉터리. 시작 시 생성하고 배타 잠금을 잡는다.
        :param transcriber: STT 서비스를 제공하는 공유 객체.
        :param translator: 번역 서비스를 제공하는 공유 객체.
        :param max_active: 업로드·대기·실행을 합친 최대 미완료 작업 수. 양의 정수.
        :param ttl_seconds: 완료·실패 이후 상태·결과 보관 시간(초). 양의 정수.
        :param clock: UTC epoch 초를 반환하는 시계. 테스트에서는 대체할 수 있다.
        :return: None.
        :raises ValueError: 작업 한도 또는 보관 시간이 양수가 아닐 때.
        """
        if max_active < 1 or ttl_seconds < 1:
            raise ValueError('Job capacity and TTL must be positive')
        self.root, self.transcriber, self.translator = Path(root), transcriber, translator
        self.max_active, self.ttl_seconds, self.clock = max_active, ttl_seconds, clock
        self.records = {}
        self.queue = asyncio.Queue()
        self._active = 0
        self._closing = False
        self._worker = self._janitor = self._lock_file = None
        self._stop = asyncio.Event()
        self._submissions = set()

    def _now(self) -> datetime:
        """현재 UTC 시각을 반환한다.

        :return: 시간대가 포함된 UTC datetime.
        """
        return datetime.fromtimestamp(self.clock(), timezone.utc)

    def _directory(self, job_id: str) -> Path:
        """유효한 서버 생성 작업 ID의 전용 디렉터리를 반환한다.

        :param job_id: 소문자 16진수 32자리 작업 ID.
        :return: 저장 루트 바로 아래 작업 경로.
        :raises KeyError: ID 형식이 잘못됐을 때.
        """
        if not JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def _write_json(self, path: Path, payload: dict) -> None:
        """JSON을 같은 디렉터리의 임시 파일에 기록한 뒤 원자적으로 교체한다.

        :param path: 기록할 작업 메타데이터 또는 결과 경로.
        :param payload: JSON 직렬화 가능한 응답 사전.
        :return: None.
        :raises OSError: 파일 기록 또는 교체에 실패할 때.
        """
        temporary = path.with_suffix('.tmp')
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _update(self, job_id: str, **changes) -> JobStatusResponse:
        """상태 변경을 검증·저장한 뒤 메모리의 현재 상태를 교체한다.

        :param job_id: 갱신할 작업 ID.
        :param changes: 상태·단계·구간 수·오류 등 덮어쓸 응답 필드.
        :return: 저장된 새 작업 상태.
        :raises OSError: 상태 저장에 실패할 때. 기존 메모리 상태는 유지한다.
        """
        data = self.records[job_id].model_dump()
        data.update(changes, updated_at=self._now())
        record = JobStatusResponse.model_validate(data)
        self._write_json(self._directory(job_id) / 'status.json', record.model_dump(mode='json'))
        self.records[job_id] = record
        return record

    def _fail(self, job_id: str, code: str, message: str) -> None:
        """작업을 실패로 표시하고 저장 실패 시에도 메모리에 안전한 오류를 유지한다.

        :param job_id: 실패 처리할 작업 ID.
        :param code: JobError에서 허용한 공개 오류 코드.
        :param message: 내부 예외·파일 경로를 포함하지 않는 공개 안내.
        :return: None.

        상태 JSON을 갱신한다. 디스크 오류가 나면 메모리만 갱신하며 재시작 시 복구한다.
        """
        changes = dict(status='failed', stage='failed', progress_percent=None,
                       error=JobError(code=code, message=message),
                       expires_at=self._now() + timedelta(seconds=self.ttl_seconds))
        try:
            self._update(job_id, **changes)
        except OSError:
            changes.update(updated_at=self._now(), error=JobError(
                code='storage_error', message='Job status could not be saved'))
            self.records[job_id] = self.records[job_id].model_copy(update=changes)

    def cleanup(self) -> None:
        """보관 기한이 끝난 완료·실패 작업의 디렉터리와 메모리를 삭제한다.

        :return: None.
        :raises OSError: 만료 파일을 삭제할 수 없을 때.
        """
        now = self._now()
        for job_id, record in list(self.records.items()):
            if record.expires_at is not None and now >= record.expires_at:
                directory = self._directory(job_id)
                if directory.exists():
                    shutil.rmtree(directory)
                del self.records[job_id]

    def _recover(self) -> None:
        """저장 상태를 읽고 중단 작업·손상 상태를 실패로 바꾸며 남은 입력을 정리한다.

        :return: None.
        :raises OSError: 복구에 필요한 파일 읽기·저장·정리가 실패할 때.

        이전 queued/running 작업은 자동 추론하지 않고 interrupted로 표시한다.
        """
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or not JOB_ID.fullmatch(directory.name):
                continue
            job_id = directory.name
            try:
                record = JobStatusResponse.model_validate_json((directory / 'status.json').read_text())
                if record.job_id != job_id:
                    raise ValueError('Mismatched job ID')
            except (ValueError, FileNotFoundError):
                record = JobStatusResponse(job_id=job_id, status='queued', stage='queued',
                                           created_at=self._now(), updated_at=self._now())
                self.records[job_id] = record
                self._fail(job_id, 'storage_error', 'Stored job status is invalid')
            else:
                self.records[job_id] = record
                if record.expires_at is not None and self._now() >= record.expires_at:
                    shutil.rmtree(directory)
                    del self.records[job_id]
                    continue
                if record.status in ('queued', 'running'):
                    self._fail(job_id, 'interrupted', 'Processing was interrupted; upload the file again')
                elif record.status == 'completed' and not (directory / 'result.json').is_file():
                    self._fail(job_id, 'storage_error', 'Stored job result is missing')
            (directory / 'input.media').unlink(missing_ok=True)
            for temporary in directory.glob('*.tmp'):
                temporary.unlink(missing_ok=True)
        self.cleanup()

    async def start(self) -> None:
        """저장소 배타 잠금과 복구를 완료한 뒤 작업자·만료 정리 태스크를 시작한다.

        :return: None.
        :raises RuntimeError: 같은 저장소를 다른 작업자가 사용하거나 이미 시작했을 때.
        :raises OSError: 저장소 생성·복구가 실패할 때.
        """
        if self._worker is not None or self._closing:
            raise RuntimeError('Job manager already started')
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / '.worker.lock').open('a')
        try:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Job storage requires a single server worker') from exc
            self._recover()
            self._worker = asyncio.create_task(self._run())
            self._janitor = asyncio.create_task(self._clean_periodically())
        except BaseException:
            self._lock_file.close()
            self._lock_file = None
            raise

    async def submit(self, upload: UploadFile, language: str = 'en') -> JobStatusResponse:
        """업로드·길이를 검증하고 파일과 접수 상태를 저장한 뒤 작업 ID를 반환한다.

        :param upload: 지원 확장자·1GiB 이하 업로드. 닫기는 호출자가 담당한다.
        :param language: 영어 en만 허용하는 인식 언어.
        :return: 디스크에 접수가 저장된 queued 상태 응답.
        :raises JobCapacityError: 미완료 한도 초과 또는 접수 중단 상태일 때.
        :raises UnsupportedMediaError: 허용하지 않는 확장자일 때.
        :raises UploadTooLargeError: 크기가 1GiB를 초과할 때.
        :raises MediaValidationError: 언어·알려진 길이·미디어 열기 검증이 실패할 때.
        :raises OSError: 입력·상태 저장이 실패할 때. 생성한 작업 파일은 정리한다.
        """
        if self._worker is None or self._worker.done() or self._closing:
            raise JobCapacityError('Job worker is unavailable')
        if language != 'en':
            raise MediaValidationError('Only English is supported')
        if Path(upload.filename or '').suffix.lower() not in ALLOWED_SUFFIXES:
            raise UnsupportedMediaError('Unsupported media file extension')
        if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError(UPLOAD_TOO_LARGE)
        if self._active >= self.max_active:
            raise JobCapacityError('Too many unfinished jobs; retry later')
        self._active += 1
        task = asyncio.current_task()
        self._submissions.add(task)
        job_id = uuid4().hex
        directory = self._directory(job_id)
        try:
            directory.mkdir()
            record = JobStatusResponse(job_id=job_id, status='queued', stage='queued',
                                       created_at=self._now(), updated_at=self._now())
            self.records[job_id] = record
            self._update(job_id)
            transfer = asyncio.create_task(run_in_threadpool(
                self._save_input, upload, directory / 'input.media'))
            try:
                await asyncio.shield(transfer)
            except asyncio.CancelledError:
                try:
                    await transfer
                except Exception:
                    pass
                raise
            if self._closing:
                raise JobCapacityError('Job worker is shutting down')
            self.queue.put_nowait((job_id, language))
            return self._snapshot(self.records[job_id])
        except BaseException:
            self.records.pop(job_id, None)
            self._active -= 1
            if directory.exists():
                shutil.rmtree(directory)
            raise
        finally:
            self._submissions.discard(task)

    def _save_input(self, upload: UploadFile, path: Path) -> None:
        """업로드 복사와 길이 사전 검사를 한 스레드에서 완료한다.

        :param upload: 읽을 업로드 스트림. 닫기는 호출자가 담당한다.
        :param path: 기록할 작업 입력 경로.
        :return: None.
        :raises UploadTooLargeError: 복사 중 크기 상한을 초과할 때.
        :raises MediaValidationError: 미디어 열기·알려진 길이 검증이 실패할 때.
        :raises OSError: 입력 파일 기록이 실패할 때.

        호출 태스크가 취소돼도 submit이 이 작업 종료를 기다린 후 파일을 정리한다.
        """
        with path.open('wb') as destination:
            copy_upload(upload.file, destination)
        try:
            validate_media_duration(path)
        except ValueError as exc:
            raise MediaValidationError(str(exc)) from exc

    def _snapshot(self, record: JobStatusResponse) -> JobStatusResponse:
        """저장된 상태를 바꾸지 않고 접수 이후 경과 시간을 계산한다.

        :param record: 조회할 현재 작업 상태.
        :return: 대기 포함 경과 초를 계산한 응답 복사본.
        """
        end = record.updated_at if record.status in ('completed', 'failed') else self._now()
        return record.model_copy(update={'elapsed_seconds': max(0, (end - record.created_at).total_seconds())})

    def get(self, job_id: str) -> JobStatusResponse:
        """만료 작업을 제거하고 작업 상태의 현재 스냅샷을 반환한다.

        :param job_id: 조회할 서버 생성 작업 ID.
        :return: 현재 상태와 경과 시간을 담은 응답.
        :raises KeyError: 작업이 없거나 만료됐거나 ID 형식이 잘못됐을 때.
        """
        self._directory(job_id)
        self.cleanup()
        return self._snapshot(self.records[job_id])

    def result(self, job_id: str) -> TranscriptionResponse:
        """완료 작업의 저장 결과를 검증해 반환하고 손상 결과는 실패 처리한다.

        :param job_id: 결과를 조회할 서버 생성 작업 ID.
        :return: 기존 동기 API와 같은 자막 응답.
        :raises KeyError: 작업이 없거나 만료됐을 때.
        :raises JobNotReadyError: 완료되지 않은 작업일 때.
        :raises JobStorageError: 결과 파일이 없거나 손상됐을 때.
        """
        if self.get(job_id).status != 'completed':
            raise JobNotReadyError('Job has not completed')
        try:
            return TranscriptionResponse.model_validate_json(
                (self._directory(job_id) / 'result.json').read_text())
        except (OSError, ValueError) as exc:
            self._fail(job_id, 'storage_error', 'Stored job result is unavailable')
            raise JobStorageError('Stored job result is unavailable') from exc

    async def _run(self) -> None:
        """대기 작업을 하나씩 실행하고 결과·실패를 저장한 뒤 입력 파일을 정리한다.

        :return: None. 종료 표식을 받으면 작업자 루프를 마친다.

        원문을 담은 입력은 성공·실패 후 삭제하며 공개 오류에 내부 예외를 넣지 않는다.
        """
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            job_id, language = item
            token = request_id.set(job_id)
            directory = self._directory(job_id)

            async def progress(stage, completed, total, failed):
                """현재 작업의 실제 단계·번역 구간 처리량을 저장한다.

                :param stage: transcribing·grouping·translating 중 하나.
                :param completed: 처리가 끝난 번역 구간 수. 번역 실패도 포함한다.
                :param total: 전체 번역 구간 수. STT 중에는 None.
                :param failed: 번역 검증 실패 구간 수.
                :return: None.
                """
                percent = 100 * completed / total if stage == 'translating' and total else None
                self._update(job_id, stage=stage, completed_segments=completed,
                             total_segments=total, failed_segments=failed, progress_percent=percent)

            try:
                with measure('job_processing', job_id=job_id):
                    self._update(job_id, status='running', stage='transcribing')
                    response = await process_media(directory / 'input.media', self.transcriber,
                                                   self.translator, language, progress)
                    self._write_json(directory / 'result.json', response.model_dump(mode='json'))
                    (directory / 'input.media').unlink(missing_ok=True)
                    self._update(job_id, status='completed', stage='completed', progress_percent=100,
                                 expires_at=self._now() + timedelta(seconds=self.ttl_seconds))
            except MediaValidationError:
                self._fail(job_id, 'invalid_media', 'Media audio or duration validation failed')
            except Exception as exc:
                self._fail(job_id, 'storage_error' if isinstance(exc, OSError) else 'processing_failed',
                           'Job processing failed; upload the file again')
            finally:
                try:
                    (directory / 'input.media').unlink(missing_ok=True)
                except OSError:
                    log_duration('job_cleanup', 0, job_id=job_id, status='error')
                self._active -= 1
                self.queue.task_done()
                request_id.reset(token)

    async def _clean_periodically(self) -> None:
        """종료 요청까지 60초 간격으로 만료 파일을 정리한다.

        :return: None.
        """
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=60)
            except asyncio.TimeoutError:
                try:
                    self.cleanup()
                except OSError:
                    log_duration('job_expiry_cleanup', 0, status='error')

    async def close(self) -> None:
        """접수를 막고 대기 작업을 실패 처리한 뒤 실행 중 추론을 기다리고 잠금을 해제한다.

        :return: None.

        실행 중 스레드를 취소하거나 입력을 먼저 지우지 않는다. 강제 종료 시에는 다음
        시작에서 interrupted로 복구한다. 대기 작업 입력은 정상 종료 시 삭제한다.
        """
        if self._worker is None:
            return
        self._closing = True
        self._stop.set()
        if self._submissions:
            await asyncio.gather(*list(self._submissions), return_exceptions=True)
        while not self.queue.empty():
            job_id, _ = self.queue.get_nowait()
            self._fail(job_id, 'interrupted', 'Server shut down before processing; upload again')
            try:
                (self._directory(job_id) / 'input.media').unlink(missing_ok=True)
            except OSError:
                log_duration('job_cleanup', 0, job_id=job_id, status='error')
            self._active -= 1
            self.queue.task_done()
        self.queue.put_nowait(None)
        try:
            await self._worker
            await self._janitor
        finally:
            self._lock_file.close()
            self._worker = None
