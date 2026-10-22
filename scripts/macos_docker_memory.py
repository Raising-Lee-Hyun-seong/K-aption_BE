"""Docker 소유 VM을 확인하고 macOS의 Docker 관련 프로세스 물리 footprint를 표본 측정한다."""

import re
import subprocess
import threading
from time import monotonic


def docker_processes(output: str, vm_pid: int) -> list:
    """인자·환경변수 없는 ps 결과에서 Docker 실행 파일과 확인한 VM만 고른다.

    :param output: ps -axo pid=,rss=,comm= 결과. RSS 단위는 KiB이다.
    :param vm_pid: Docker.raw를 연 VM으로 사전 확인한 프로세스 ID.
    :return: pid·rss_bytes·executable 사전 목록.
    """
    rows = []
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3 or not all(value.isdigit() for value in fields[:2]):
            continue
        pid, rss, executable = int(fields[0]), int(fields[1]), fields[2]
        if (pid == vm_pid or '/Docker.app/' in executable
                or executable.startswith('/Library/PrivilegedHelperTools/com.docker.')):
            rows.append({'pid': pid, 'rss_bytes': rss * 1024, 'executable': executable})
    return rows


def physical_footprints(output: str) -> dict:
    """footprint bytes 출력에서 PID별 현재 물리 footprint만 추출한다.

    :param output: footprint --noCategories -f bytes 결과.
    :return: PID를 키로 하는 현재 phys_footprint bytes. 생애 최대값과 RSS는 포함하지 않는다.
    """
    return {int(pid): int(value) for pid, value in re.findall(
        r'^.*? \[(\d+)\]:.*?Footprint: (\d+) B', output, re.MULTILINE)}


class MacDockerMemorySampler:
    """컨테이너 합계와 구분해 Docker VM·데스크톱의 물리 footprint를 수집한다."""

    def __init__(self, vm_pid: int, interval: float = 5):
        """확인할 VM PID와 표본 간격을 저장한다.

        :param vm_pid: Docker.raw를 연 Virtualization 프로세스 ID. 양의 정수.
        :param interval: 표본 사이 대기 초. 양수.
        :return: None. 프로세스 목록·파일 내용·환경변수를 읽지 않는다.
        :raises ValueError: PID 또는 간격이 양수가 아닌 경우.
        """
        if vm_pid <= 0 or interval <= 0:
            raise ValueError('VM PID and interval must be positive')
        self.vm_pid, self.interval = vm_pid, interval
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = None

    def start(self) -> None:
        """VM의 Docker 소유 증거를 확인한 뒤 표본 스레드를 시작한다.

        :return: None. lsof는 열린 파일 경로만 조회하며 파일 내용을 읽지 않는다.
        :raises RuntimeError: 지정 PID가 Docker.raw를 연 VM인지 확인할 수 없는 경우.
        :raises subprocess.SubprocessError: 메타데이터 조회 실패 또는 시간 초과 시.
        """
        result = subprocess.run(['lsof', '-p', str(self.vm_pid), '-Fn'],
                                capture_output=True, text=True, timeout=10, check=True)
        paths = [line[1:] for line in result.stdout.splitlines() if line.startswith('n')
                 and '/com.docker.docker/' in line and line.endswith('/Docker.raw')]
        if not paths:
            raise RuntimeError('The selected VM has no Docker.raw ownership evidence')
        self.attribution = {'vm_pid': self.vm_pid, 'method': 'lsof open Docker.raw',
                            'verified': True}
        self.started = monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        """명령 인자 없는 프로세스 목록과 물리 footprint를 반복 수집한다.

        :return: None. 실패를 기록하고 다음 표본을 계속 시도한다.
        """
        while not self.stop.is_set():
            try:
                listing = subprocess.run(['ps', '-axo', 'pid=,rss=,comm='],
                                         capture_output=True, text=True, timeout=5, check=True)
                processes = docker_processes(listing.stdout, self.vm_pid)
                if not any(row['pid'] == self.vm_pid for row in processes):
                    raise RuntimeError('Docker VM process disappeared')
                result = subprocess.run(['footprint', '--noCategories', '-f', 'bytes',
                                         *[str(row['pid']) for row in processes]],
                                        capture_output=True, text=True, timeout=15, check=True)
                values = physical_footprints(result.stdout)
                if set(values) != {row['pid'] for row in processes}:
                    self.errors.append({'seconds': monotonic() - self.started,
                                        'type': 'IncompleteFootprint',
                                        'missing_pids': sorted({row['pid'] for row in processes} - set(values))})
                for row in processes:
                    row['physical_footprint_bytes'] = values.get(row['pid'])
                if self.vm_pid not in values:
                    raise RuntimeError('Docker VM footprint unavailable')
                self.samples.append({'seconds': monotonic() - self.started,
                                     'processes': processes, 'sum_physical_footprint_bytes': sum(values.values())})
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                self.errors.append({'seconds': monotonic() - self.started, 'type': type(exc).__name__})
            self.stop.wait(self.interval)

    def close(self) -> dict:
        """표본 수집을 멈추고 호스트 측정 범위를 명시해 반환한다.

        :return: PID별 표본·오류·현재 footprint 합계 관측 최대값과 한계 사전.
        """
        self.stop.set()
        if self.thread:
            self.thread.join(22)
        return {'samples': self.samples, 'errors': self.errors,
                'attribution': getattr(self, 'attribution', {'verified': False}),
                'sampled_process_footprint_peak_bytes': max(
                    (row['sum_physical_footprint_bytes'] for row in self.samples), default=None),
                'complete': bool(self.samples) and not self.errors
                            and not bool(self.thread and self.thread.is_alive()),
                'host_memory_goal': 'unverified',
                'limits': ['macOS process physical footprint sum includes the verified Docker VM.',
                           'Unavailable process footprints are null and excluded from the observed sum.',
                           'Shared memory may be counted more than once; kernel/driver attribution is incomplete.',
                           'Sampling can miss short peaks and is not a Windows/WSL measurement.']}
