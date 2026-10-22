"""실제 작업 API·SSE·자막과 Docker Engine 통계를 측정하며 호스트 메모리와 구분한다."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import threading
from time import monotonic

import httpx


def timing_logs(raw: bytes, job_id: str) -> list:
    """Docker 로그 프레임에서 해당 시험 작업의 구조화된 수행 시간만 추출한다.

    :param raw: Docker Engine logs의 다중화 프레임 또는 일반 UTF-8 로그 bytes.
    :param job_id: 이번 시험에서 생성한 작업 ID.
    :return: 원문·번역문 없이 이벤트명·시간·토큰 수 등을 담은 로그 필드 목록.
    """
    chunks, offset = [], 0
    while offset + 8 <= len(raw) and raw[offset] in (1, 2) and raw[offset + 1:offset + 4] == b'\0\0\0':
        size = int.from_bytes(raw[offset + 4:offset + 8], 'big')
        chunks.append(raw[offset + 8:offset + 8 + size])
        offset += 8 + size
    text = (b''.join(chunks) if chunks else raw).decode('utf-8', errors='replace')
    result = []
    for line in text.splitlines():
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if fields.get('request_id') == job_id and 'event' in fields:
            result.append(fields)
    return result


def backend_timing(socket: str, container: str, job_id: str) -> list:
    """선택한 backend의 제한된 로그에서 시험 작업의 시간 이벤트를 읽는다.

    :param socket: Docker Engine Unix 소켓 경로.
    :param container: 시험 backend 컨테이너 ID 또는 이름.
    :param job_id: 측정할 작업 ID.
    :return: 구조화된 시간 로그 목록. 최근 10000줄 범위에 한정한다.
    :raises httpx.HTTPError: Docker 로그 요청 실패 시.
    """
    with httpx.Client(transport=httpx.HTTPTransport(uds=socket), base_url='http://docker',
                      timeout=10, trust_env=False) as client:
        response = client.get(f'/containers/{container}/logs',
                              params={'stdout': 'true', 'stderr': 'true', 'tail': '10000'})
        response.raise_for_status()
        return timing_logs(response.content, job_id)


def memory_sample(stats: dict) -> dict:
    """Docker 원시 통계에서 cgroup 사용량·비활성 파일 캐시 제외량·CPU 시간을 추출한다.

    :param stats: Docker Engine containers/stats JSON 응답.
    :return: bytes 단위 cgroup·working set·가능한 최고 사용량, 누적 CPU 나노초와
        메모리 회수·주요 페이지 폴트·CPU 제한 누적 카운터. 없는 값은 None이다.
    """
    memory = stats.get('memory_stats', {})
    usage = memory.get('usage')
    cache = memory.get('stats', {}).get('inactive_file',
                                      memory.get('stats', {}).get('total_inactive_file', 0))
    details = memory.get('stats', {})
    return {'cgroup_bytes': usage,
            'working_set_bytes': max(0, usage - cache) if usage is not None else None,
            'cgroup_max_bytes': memory.get('max_usage'),
            'limit_bytes': memory.get('limit'),
            'cpu_total_ns': stats.get('cpu_stats', {}).get('cpu_usage', {}).get('total_usage'),
            'memory_details': {key: details.get(key) for key in (
                'anon', 'file', 'kernel', 'pgmajfault', 'workingset_refault_file',
                'pgscan', 'pgsteal', 'pswpin', 'pswpout')},
            'cpu_throttling': {key: stats.get('cpu_stats', {}).get('throttling_data', {}).get(key)
                               for key in ('periods', 'throttled_periods', 'throttled_time')}}


def pipeline_succeeded(report: dict) -> bool:
    """작업 완료와 번역·자막 제공을 구분해 부분 실패를 시험 성공으로 처리하지 않는다.

    :param report: run_media가 반환한 작업·결과·자막 상태 보고서.
    :return: 구간이 존재하고 모든 번역·자막 다운로드가 성공했을 때만 True.
    """
    segments = report.get('result', {}).get('segments', [])
    final = report.get('final_job', {})
    return (report.get('final_job', {}).get('status') == 'completed'
            and not report.get('final_job', {}).get('failed_segments')
            and final.get('total_segments') == final.get('completed_segments') == len(segments)
            and bool(segments) and all(row.get('translation') and not row.get('translation_error') for row in segments)
            and all(report.get('subtitles', {}).get(fmt, {}).get('status') == 200
                    and report['subtitles'][fmt].get('download') for fmt in ('srt', 'vtt')))


class DockerSampler:
    """CLI 프로세스를 반복 생성하지 않고 Unix 소켓의 통계를 주기적으로 수집한다."""

    def __init__(self, socket: str, containers: list, interval: float = 1):
        """컨테이너 식별자·표본 간격·수집 목록을 준비한다.

        :param socket: Docker Engine Unix 소켓 경로. Docker Desktop 또는 Linux 소켓.
        :param containers: 측정할 backend·translator 컨테이너 ID 또는 이름 목록.
        :param interval: 표본 사이 대기 초. 양수.
        :return: None. 생성만으로 Docker에 접속하지 않는다.
        :raises ValueError: 컨테이너 목록이 비었거나 간격이 양수가 아닌 경우.
        """
        if not containers or interval <= 0:
            raise ValueError('Containers and a positive sample interval are required')
        self.socket, self.containers, self.interval = socket, containers, interval
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = None

    def start(self) -> None:
        """백그라운드 통계 수집을 시작한다.

        :return: None. 수집 스레드를 생성한다.
        """
        self.started = monotonic()
        self.initial_states = self.states()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def states(self) -> dict:
        """컨테이너 검사 응답에서 OOM·재시작 횟수만 추출한다.

        :return: 컨테이너별 OOMKilled·RestartCount. 실패는 errors에 기록하고 누락한다.
        """
        result = {}
        with httpx.Client(transport=httpx.HTTPTransport(uds=self.socket),
                          base_url='http://docker', timeout=3, trust_env=False) as client:
            for container in self.containers:
                try:
                    response = client.get(f'/containers/{container}/json')
                    response.raise_for_status()
                    data = response.json()
                    result[container] = {'oom_killed': data['State']['OOMKilled'],
                                         'restart_count': data['RestartCount']}
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    self.errors.append({'container': container, 'type': type(exc).__name__})
        return result

    def _run(self) -> None:
        """소켓 장애를 기록하되 표본 수집 스레드는 계속 실행한다.

        :return: None. samples와 errors에 결과를 추가한다.
        """
        previous = {}
        with httpx.Client(transport=httpx.HTTPTransport(uds=self.socket),
                          base_url='http://docker', timeout=3, trust_env=False) as client:
            while not self.stop.is_set():
                row = {'seconds': monotonic() - self.started, 'containers': {}}
                for container in self.containers:
                    try:
                        response = client.get(f'/containers/{container}/stats',
                                              params={'stream': 'false', 'one-shot': 'true'})
                        response.raise_for_status()
                        value = memory_sample(response.json())
                        now = monotonic()
                        value['sampled_seconds'] = now - self.started
                        old = previous.get(container)
                        cpu = value['cpu_total_ns']
                        value['cpu_percent'] = ((cpu - old[1]) / (now - old[0]) / 1e7
                                                if old and cpu is not None and old[1] is not None
                                                and cpu >= old[1] else None)
                        previous[container] = (now, cpu)
                        row['containers'][container] = value
                    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                        self.errors.append({'seconds': monotonic() - self.started,
                                            'container': container, 'type': type(exc).__name__})
                self.samples.append(row)
                self.stop.wait(self.interval)

    def close(self) -> dict:
        """수집을 멈추고 동시 관측 합계와 누락 여부를 반환한다.

        :return: 표본·오류·관측 메모리 합계 최대값·측정 범위를 담은 사전.
        """
        self.stop.set()
        if self.thread:
            self.thread.join(10)
        final_states = self.states() if self.thread else {}
        initial_states = getattr(self, 'initial_states', {})
        checked = len(initial_states) == len(self.containers) == len(final_states)
        restarted = (any(final_states[name]['oom_killed']
                         or final_states[name]['restart_count'] > initial_states[name]['restart_count']
                         for name in self.containers) if checked else None)
        totals = [sum(value['working_set_bytes'] for value in row['containers'].values())
                  for row in self.samples if len(row['containers']) == len(self.containers)
                  and all(value['working_set_bytes'] is not None for value in row['containers'].values())]
        return {'samples': self.samples, 'errors': self.errors,
                'initial_states': initial_states, 'final_states': final_states,
                'oom_or_restart': restarted,
                'sampled_combined_working_set_peak_bytes': max(totals, default=None),
                'complete': bool(totals) and len(totals) == len(self.samples)
                            and not self.errors and not bool(self.thread and self.thread.is_alive()),
                'host_memory_goal': 'unverified',
                'limits': ['Sampled container working sets exclude Docker/VM host overhead.',
                           'Containers are queried sequentially within each sampling round, not atomically.',
                           'Sampling can miss short peaks; separate host measurement is required.']}


def run_media(base_url: str, media: Path, deadline_seconds: float = 21600, report=None) -> dict:
    """실제 파일 작업을 접수하고 SSE·결과·SRT/VTT를 끝까지 확인한다.

    :param base_url: FastAPI backend 기본 HTTP 주소.
    :param media: 처리할 영상 또는 음성 파일. 스트리밍 업로드하며 전체를 메모리에 읽지 않는다.
    :param deadline_seconds: 접수 이후 작업 완료까지 기다릴 최대 초.
    :param report: 선택적 결과 사전. 접수 즉시 job_id·이벤트를 기록해 연결 종료 시에도 보존한다.
    :return: 작업 상태·단계별 관측 시각·결과·자막 HTTP 상태·소요 초.
    :raises ValueError: 제한 시간이 양수가 아닌 경우.
    :raises TimeoutError: 제한 시간 안에 작업이 끝나지 않은 경우. 해당 시험 작업에 취소를 요청한다.
    :raises httpx.HTTPError: backend 연결·HTTP 응답 처리 실패 시.
    :raises OSError: 미디어 파일을 읽을 수 없는 경우.
    """
    if deadline_seconds <= 0:
        raise ValueError('Deadline must be positive')
    started, events = monotonic(), []
    report = {} if report is None else report
    report['events'] = events
    with httpx.Client(base_url=base_url, timeout=180, trust_env=False) as client:
        initial = client.get('/health')
        initial.raise_for_status()
        with media.open('rb') as stream:
            response = client.post('/api/v1/jobs', files={'file': (media.name, stream)})
        response.raise_for_status()
        job_id = response.json()['job_id']
        report.update({'job_id': job_id, 'initial_health': initial.json()})
        with client.stream('GET', f'/api/v1/jobs/{job_id}/events') as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if monotonic() - started > deadline_seconds:
                    client.post(f'/api/v1/jobs/{job_id}/cancel')
                    raise TimeoutError('Benchmark job deadline exceeded; cancellation requested')
                if line.startswith('data:'):
                    state = json.loads(line[5:])
                    events.append({'seconds': monotonic() - started, 'state': state})
                    if state.get('status') in ('completed', 'failed', 'cancelled'):
                        break
        final = client.get(f'/api/v1/jobs/{job_id}')
        final.raise_for_status()
        report.update({'final_job': final.json(), 'elapsed_seconds': monotonic() - started})
        if report['final_job']['status'] == 'completed':
            response = client.get(f'/api/v1/jobs/{job_id}/result')
            response.raise_for_status()
            report['result'] = response.json()
            report['subtitles'] = {}
            for fmt in ('srt', 'vtt'):
                response = client.get(f'/api/v1/jobs/{job_id}/subtitles', params={'format': fmt, 'language': 'both'})
                response.raise_for_status()
                report['subtitles'][fmt] = {'status': response.status_code,
                                            'download': response.headers.get('content-disposition')}
        return report


def main() -> None:
    """명시한 시험 파일과 새 보고서 경로로 통합 시험을 실행한다.

    :return: None. 성공·실패를 모두 새 JSON 보고서에 저장한다.
    :raises FileExistsError: 기존 평가 결과를 덮어쓰려는 경우.
    :raises OSError: 미디어·보고서에 접근할 수 없는 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--docker-socket', default='/var/run/docker.sock')
    parser.add_argument('--container', action='append', default=[])
    parser.add_argument('--backend-container', help='STT·번역 시간 로그를 읽을 시험 backend')
    parser.add_argument('--macos-vm-pid', type=int,
                        help='macOS Docker.raw를 연 VM PID. footprint 조회 권한이 필요하다.')
    parser.add_argument('--deadline-seconds', type=float, default=21600)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new report path')
    sampler = DockerSampler(args.docker_socket, args.container) if args.container else None
    host_sampler = None
    if args.macos_vm_pid:
        if platform.system() != 'Darwin':
            parser.error('--macos-vm-pid requires macOS')
        if __package__:
            from .macos_docker_memory import MacDockerMemorySampler
        else:
            from macos_docker_memory import MacDockerMemorySampler
        host_sampler = MacDockerMemorySampler(args.macos_vm_pid)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'os': platform.platform(),
              'machine': platform.machine(), 'media': str(args.media), 'host_memory_goal': 'unverified'}
    started = monotonic()
    try:
        if host_sampler:
            host_sampler.start()
        if sampler:
            sampler.start()
        run_media(args.base_url, args.media, args.deadline_seconds, report)
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        report.setdefault('elapsed_seconds', monotonic() - started)
        if args.backend_container and report.get('job_id'):
            try:
                report['timing_logs'] = backend_timing(args.docker_socket, args.backend_container, report['job_id'])
            except httpx.HTTPError as exc:
                report['timing_logs_error'] = type(exc).__name__
        if host_sampler:
            report['host_macos'] = host_sampler.close()
        if sampler:
            report['docker'] = sampler.close()
        report['pipeline_succeeded'] = pipeline_succeeded(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    if report.get('error') or not pipeline_succeeded(report) or report.get('docker', {}).get('oom_or_restart'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
