"""짧은 동일 입력으로 프롬프트 캐시의 시간·실제 재사용 토큰·검증 결과를 비교한다."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path.cwd() if __file__ == '<stdin>' else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.translation_http import HttpTranslationService

CASES = (
    {'id': 'velocity', 'text': 'The velocity is constant.'},
    {'id': 'integrate', 'text': 'Now we can integrate the equation.'},
    {'id': 'mass-symbol', 'text': 'That is the initial mass, 2 m0.'},
)
CONTEXT = 'We are deriving the rocket equation.'


class RecordingTranslationService(HttpTranslationService):
    """기존 생성·검증·재시도를 그대로 사용하면서 서버 시간과 원시 출력을 보존한다."""

    def __init__(self, base_url: str, template_path: Path):
        """짧은 제한 시간과 시도별 기록 목록을 준비한다.

        :param base_url: 실행 중인 CPU 번역 서버의 기본 HTTP 주소.
        :param template_path: 기존 텍스트 설정 JSON 경로.
        :return: None. 모델을 직접 로드하지 않는다.
        """
        super().__init__(base_url, template_path, timeout=60)
        self.attempts = []

    def _stream_output(self, prompt: str, max_tokens: int):
        """실제 SSE를 그대로 제공하고 서버가 보낸 시간·재사용 토큰을 기록한다.

        :param prompt: 운영 서비스가 만든 현재 메시지 JSON.
        :param max_tokens: 시도별 출력 토큰 상한.
        :return: 원래 생성 조각을 제공하는 생성기.
        :yield: text·generation_tokens·finish_reason·server_timings 속성의 원래 조각.
        :raises ValueError: 원래 HTTP 생성기가 연결 또는 응답 오류를 전달할 때.
        """
        row = {'messages': json.loads(prompt), 'cache_prompt': self.cache_prompt,
               'raw_output': '', 'server_timings': None}
        self.attempts.append(row)
        stream = super()._stream_output(prompt, max_tokens)
        try:
            for part in stream:
                row['raw_output'] += part.text
                row['generated_tokens'] = part.generation_tokens
                row['finish_reason'] = part.finish_reason
                if part.server_timings is not None:
                    row['server_timings'] = part.server_timings
                yield part
        finally:
            stream.close()


def benchmark(service, cache_order=(False, True), context_mode: str = 'rolling') -> list:
    """동일한 워밍업·원문·문맥·생성 설정으로 두 캐시 조건을 순차 비교한다.

    :param service: 서버 시간을 기록하는 실제 서비스 또는 시험 대역.
    :param cache_order: 측정 순서. False·True를 각각 한 번 포함한다.
    :param context_mode: rolling이면 직전 원문을 다음 문맥으로 사용하고 fixed이면 배경을 고정한다.
    :return: 조건별 워밍업·시도·출력·클라이언트 시간·실패 수 목록.
    :raises ValueError: 두 조건을 정확히 한 번씩 지정하지 않거나 문맥 모드가 유효하지 않은 경우.
    """
    if len(cache_order) != 2 or {type(value) for value in cache_order} != {bool} or set(cache_order) != {False, True}:
        raise ValueError('Compare False and True exactly once')
    if context_mode not in ('rolling', 'fixed'):
        raise ValueError('Use rolling or fixed context')
    modes = []
    for enabled in cache_order:
        service.cache_prompt = False
        warmup_start = len(service.attempts)
        try:
            warmup = service.translate('The acceleration is constant.', CONTEXT, 128)
        except ValueError as exc:
            warmup = {'error': type(exc).__name__, 'message': str(exc)}
        warmup_attempts = service.attempts[warmup_start:]
        service.cache_prompt = enabled
        rows, started, context = [], perf_counter(), CONTEXT
        for case in CASES:
            first, before = len(service.attempts), perf_counter()
            row = {**case, 'context': context}
            try:
                row['result'] = service.translate(case['text'], context, 128)
            except ValueError as exc:
                row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            row['client_seconds'] = perf_counter() - before
            row['attempts'] = service.attempts[first:]
            rows.append(row)
            if context_mode == 'rolling':
                context = case['text'][-256:]
        modes.append({'cache_prompt': enabled, 'warmup': warmup,
                      'warmup_attempts': warmup_attempts, 'results': rows,
                      'client_seconds': perf_counter() - started,
                      'failed_cases': sum('error' in row for row in rows)})
    return modes


def main() -> None:
    """명시한 서버에서 짧은 비교를 실행하고 기존 결과를 덮어쓰지 않고 저장한다.

    :return: None. stdout 또는 새 JSON 파일에 출력·시간·캐시 근거·소스 해시를 기록한다.
    :raises FileExistsError: 출력 경로가 이미 있는 경우.
    :raises ValueError: 서버가 준비되지 않았거나 지정한 모델 파일명과 다른 경우.
    :raises OSError: 출력 또는 선택적 Whisper 캐시에 접근하지 못한 경우.

    resident-stt를 지정하면 CPU Small INT8 모델을 지연 로드해 유지하지만 STT 추론은 하지 않는다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads-label', type=int, required=True,
                        help='외부에서 실제 적용한 스레드 수. 이 도구가 서버 스레드를 변경하지는 않는다.')
    parser.add_argument('--cache-order', choices=('off-first', 'on-first'), default='off-first')
    parser.add_argument('--context-mode', choices=('rolling', 'fixed'), default='rolling')
    parser.add_argument('--resident-stt', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new report path')
    service = RecordingTranslationService(args.base_url, ROOT / 'templates/korean.json')
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
              'threads_label': args.threads_label, 'context': CONTEXT,
              'context_mode': args.context_mode,
              'input_sha256': hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
              'resident_stt': args.resident_stt,
              'limits': ['Three short cases, one pass: not a statistical speed or general quality benchmark.',
                         'STT weights may be resident, but STT decoding/inference and pipeline peaks are excluded.',
                         'Thread count is a label; verify the actual server command separately.',
                         'VM memory is not changed; this cannot prove a benefit from increasing VM RAM.']}
    try:
        if not service.loaded():
            raise ValueError('Translator is not ready')
        props = service.client.get('/props')
        props.raise_for_status()
        props = props.json()
        if Path(props.get('model_path', '')).name != 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf':
            raise ValueError('Unexpected loaded model')
        report['server'] = {key: props.get(key) for key in ('model_path', 'model_alias', 'total_slots')}
        paths = ('app/services/translation.py', 'app/services/translation_qwen.py',
                 'app/services/translation_http.py', 'app/services/translation_validation.py',
                 'data/physics_glossary.json', 'data/qwen_translation_terms.json')
        report['implementation_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                            for name in paths}
        resident_stt = None
        if args.resident_stt:
            from transcribe import TranscriptionService
            started = perf_counter()
            resident_stt = TranscriptionService()
            resident_stt._get_model()
            report['stt_model_load_seconds'] = perf_counter() - started
        report['modes'] = benchmark(service, (False, True) if args.cache_order == 'off-first' else (True, False),
                                    args.context_mode)
    finally:
        service.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'modes': [
        {key: mode[key] for key in ('cache_prompt', 'client_seconds', 'failed_cases')}
        for mode in report['modes']]}), flush=True)


if __name__ == '__main__':
    main()
