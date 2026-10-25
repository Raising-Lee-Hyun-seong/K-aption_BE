"""기존 Qwen CPU 서버의 4·6·8 CPU와 추론 스레드를 교차 비교하고 원래 설정을 복구한다."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_cpu_stt_options import save_new
from scripts.evaluate_cpu_translation_models import RecordingQwen, load_cases, run_qwen

SERVER = 'k-aption_be-translator-1'
ORDER = (4, 6, 8, 8, 6, 4)
PILOT_INDICES = (2, 10, 16, 18)


def docker(arguments: list, env=None, check: bool = True) -> subprocess.CompletedProcess:
    """Docker CLI를 실행하고 비밀 환경 설정을 출력하지 않은 결과를 반환한다.

    :param arguments: docker 뒤에 전달할 인자 목록. 셸로 해석하지 않는다.
    :param env: 선택적 CLI 환경. None이면 현재 환경을 상속한다. 저장·출력하지 않는다.
    :param check: 참이면 0이 아닌 종료 코드를 예외로 처리한다.
    :return: stdout·stderr·종료 코드가 있는 프로세스 결과.
    :raises subprocess.CalledProcessError: check가 참이고 명령이 실패한 경우.
    """
    return subprocess.run(['docker', *arguments], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=check)


def server_state() -> dict:
    """서버의 지정된 실행 명령·이미지·자원 제한·상태 필드만 조회한다.

    :return: 비밀 환경 변수와 전체 Docker 설정을 제외한 실행 상태 사전.
    """
    fmt = ('{"image":{{json .Image}},"cmd":{{json .Config.Cmd}},'
           '"nano_cpus":{{.HostConfig.NanoCpus}},"memory":{{.HostConfig.Memory}},'
           '"memory_swap":{{.HostConfig.MemorySwap}},"oom":{{.State.OOMKilled}},'
           '"restart_count":{{.RestartCount}},"id":{{json .Id}}}')
    return json.loads(docker(['inspect', SERVER, '--format', fmt]).stdout)


def configure(threads: int, original: dict) -> dict:
    """서버만 재생성해 CPU 제한·두 추론 스레드를 바꾸고 동일한 나머지 설정을 확인한다.

    :param threads: CPU 개수와 threads·threads-batch에 함께 적용할 4·6·8 중 하나.
    :param original: 비교 전 저장한 기존 서버 상태. 이미지·메모리·명령 비교에 사용한다.
    :return: 준비된 서버 상태와 재생성·로딩에 걸린 초 사전. 측정 번역 시간에 포함하지 않는다.
    :raises RuntimeError: 모델 서버가 준비되지 않거나 의도하지 않은 설정 차이가 생긴 경우.

    빈 공개 기본값 파일을 지정해 .env를 읽지 않고 translator만 재생성한다.
    """
    started = monotonic()
    environment = dict(os.environ, KAPTION_CPU_THREADS=str(threads),
                       KAPTION_TRANSLATOR_CPUS=str(threads))
    docker(['compose', '--env-file', 'docker/cpu.defaults', 'up', '-d', '--no-deps',
            '--no-build', '--pull', 'never', '--force-recreate', 'translator'], env=environment)
    deadline = monotonic() + 120
    while monotonic() < deadline:
        ready = docker(['exec', SERVER, 'curl', '--fail', '--silent',
                        'http://localhost:8080/health'], check=False)
        if ready.returncode == 0:
            break
        sleep(1)
    else:
        raise RuntimeError('CPU translator did not become ready')
    state = server_state()
    expected = list(original['cmd'])
    for flag in ('--threads', '--threads-batch'):
        expected[expected.index(flag) + 1] = str(threads)
    if (state['cmd'] != expected or state['image'] != original['image']
            or state['nano_cpus'] != threads * 1_000_000_000
            or any(state[key] != original[key] for key in ('memory', 'memory_swap'))
            or state['oom'] or state['restart_count']):
        raise RuntimeError('Unexpected translator configuration or restart')
    return {'state': state, 'configure_seconds': monotonic() - started}


def replay(args) -> None:
    """동일 원문을 기존 번역 서비스로 재생하고 시도별 시간·출력·검증 결과를 저장한다.

    :param args: output·threads·선택적 indices를 가진 CLI 인자 객체.
    :return: None. 워밍업 뒤 원문별 결과와 합계를 새 JSON에 저장한다.
    :raises FileExistsError: 최종 파일이나 사례별 기록이 이미 존재하는 경우.
    :raises RuntimeError: 서버나 워밍업이 정상적으로 준비되지 않은 경우.
    """
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = load_cases()
    if args.indices:
        cases = [cases[index] for index in args.indices]
    service = RecordingQwen('http://translator:8080', ROOT / 'templates/korean.json', cache_prompt=True)
    report = {'threads': args.threads, 'started_at': datetime.now(timezone.utc).isoformat(),
              'cases_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
              'results': [], 'limits': ['Saved STT replay; excludes STT, upload and subtitles.',
                                       'Same cache-disabled warmup per run, excluded from measured total.',
                                       'Heuristic acceptance does not guarantee semantic accuracy.']}
    try:
        if not service.loaded():
            raise RuntimeError('Translator unavailable before replay')
        service.cache_prompt = False
        report['warmup'] = run_qwen(service, {'text': 'This is the initial mass.', 'context': ''})
        if not report['warmup']['accepted']:
            raise RuntimeError('Warmup failed')
        service.cache_prompt = True
        for index, case in enumerate(cases):
            row = {**case, **run_qwen(service, case)}
            report['results'].append(row)
            save_new(args.output.with_name(args.output.stem + f'-case{index}.json'), row)
            print(json.dumps({'threads': args.threads, 'id': case['id'],
                              'seconds': row['seconds'], 'accepted': row['accepted']}), flush=True)
        report['seconds'] = sum(row['seconds'] for row in report['results'])
        report['failed'] = sum(not row['accepted'] for row in report['results'])
        save_new(args.output, report)
    finally:
        service.close()


def run_condition(directory: Path, label: str, threads: int, original: dict, indices: tuple) -> dict:
    """실제 CPU 설정을 확인한 뒤 분리된 클라이언트 컨테이너로 한 조건을 측정한다.

    :param directory: 새 평가 결과와 로그를 저장할 호스트 디렉터리.
    :param label: 기존 파일과 겹치지 않는 조건 이름.
    :param threads: 적용할 CPU 개수·추론 스레드 수.
    :param original: 복구·설정 동등성 확인에 사용할 기존 서버 상태.
    :param indices: 재생할 사례 인덱스 튜플. 비면 19개 전체를 재생한다.
    :return: 원시 재생 결과와 서버 상태·재생성 시간이 있는 사전.
    :raises RuntimeError: 측정 중 서버 재시작·OOM·설정 차이가 발견된 경우.
    """
    environment = configure(threads, original)
    command = ['run', '--rm', '--name', 'kaption-thread-replay-client',
               '--network', 'k-aption_be_inference', '--cpus', '1', '--memory', '268435456',
               '--memory-swap', '268435456', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '--mount', f'type=bind,source={ROOT},target=/app,readonly',
               '--mount', f'type=bind,source={directory},target=/reports',
               '--entrypoint', 'python', 'kaption-backend:local',
               'scripts/evaluate_cpu_translation_threads.py', 'replay',
               '--threads', str(threads), '--output', f'/reports/{label}.json']
    if indices:
        command.extend(['--indices', *(str(index) for index in indices)])
    result = docker(command)
    save_new(directory / f'{label}-client-log.json', {'stdout': result.stdout, 'stderr': result.stderr})
    report = json.loads((directory / f'{label}.json').read_text())
    state = server_state()
    if state != environment['state']:
        raise RuntimeError('Translator state changed during measurement')
    row = {'label': label, 'threads': threads, 'seconds': report['seconds'],
           'failed': report['failed'], 'server': environment, 'report': report}
    save_new(directory / f'{label}-environment.json', row)
    print(json.dumps({key: row[key] for key in ('label', 'threads', 'seconds', 'failed')}), flush=True)
    return row


def control(args) -> None:
    """대표 입력을 교차 비교하고 유망한 경우 전체 입력을 확인한 뒤 기존 설정을 복구한다.

    :param args: 새 호스트 평가 디렉터리 directory를 가진 CLI 인자 객체.
    :return: None. 서버를 재생성해 측정하고 마지막에 원래 4 CPU 설정을 복구한다.
    :raises FileExistsError: 지정한 평가 디렉터리가 이미 존재하는 경우.
    :raises RuntimeError: 실행 중인 작업·예상 밖의 설정·서버 준비 실패가 발견된 경우.
    :raises subprocess.CalledProcessError: Docker 명령이 실패한 경우.
    """
    args.directory.mkdir(parents=True, exist_ok=False)
    original = server_state()
    if original['nano_cpus'] != 4_000_000_000:
        raise RuntimeError('Expected original 4 CPU setting')
    active_code = ("from pathlib import Path; import json; "
                   "print(sum(json.loads(p.read_text()).get('status') in "
                   "('queued','running','cancelling') for p in Path('/data/jobs').glob('*/status.json')))")
    if int(docker(['exec', 'k-aption_be-backend-1', 'python', '-c', active_code]).stdout):
        raise RuntimeError('Active jobs must finish before benchmarking')
    report = {'original': original, 'pilot_order': ORDER, 'pilot_indices': PILOT_INDICES,
              'pilot': [], 'full': [], 'source': Path(__file__).read_text(),
              'limits': ['Mac ARM64 Docker only; Windows amd64 untested.',
                         'Fixed counterbalanced pilot order, two passes per condition; not statistical proof.',
                         'CPU allocation and both thread counts change together; effects are not separated.',
                         'VM stays at 8 vCPU and 6GB; model, prompts and memory limits stay fixed.',
                         'No CPU affinity, thermal control or fixed host frequency.']}
    try:
        for index, threads in enumerate(ORDER):
            report['pilot'].append(run_condition(args.directory, f'pilot-{index + 1}-{threads}',
                                                 threads, original, PILOT_INDICES))
        means = {threads: sum(row['seconds'] for row in report['pilot'] if row['threads'] == threads) / 2
                 for threads in (4, 6, 8)}
        report['pilot_means'] = means
        report['pilot_fastest'] = min(means, key=means.get)
        if min(means.values()) < means[4] * .95:
            for threads in (6, 8, 4):
                report['full'].append(run_condition(args.directory, f'full-{threads}',
                                                    threads, original, ()))
    finally:
        report['restoration'] = configure(4, original)
        save_new(args.directory / 'comparison.json', report)


def main() -> None:
    """평가 컨트롤러 또는 Docker 내부 원문 재생 모드를 실행한다.

    :return: None. 명시한 새 경로에 결과를 저장하고 컨트롤러는 서버 설정을 복구한다.
    :raises FileExistsError: 평가 디렉터리나 출력 파일이 이미 존재하는 경우.
    :raises RuntimeError: 서버 준비·설정 확인·워밍업·측정 조건이 충족되지 않는 경우.
    :raises subprocess.CalledProcessError: Docker 명령이 실패한 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    controller = commands.add_parser('control')
    controller.add_argument('--directory', type=Path, required=True)
    client = commands.add_parser('replay')
    client.add_argument('--output', type=Path, required=True)
    client.add_argument('--threads', type=int, choices=(4, 6, 8), required=True)
    client.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args()
    if args.mode == 'control':
        control(args)
    else:
        replay(args)


if __name__ == '__main__':
    main()
