"""기존 4bit KoreanLM의 CPU 연산·짧은 번역 시간과 프로세스 최대 RSS를 측정한다."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.translation import TranslationService
from app.services.translation_llamacpp import LlamaCppTranslationService


def probe_cpu() -> dict:
    """작은 FP16·4bit 행렬 곱과 활성화 JIT를 실제 CPU에서 실행해 필수 연산을 확인한다.

    :return: 실제 MLX 장치·출력 모양·정확한 계산값·JIT 활성화 결과를 담은 사전.
    :raises ValueError: CPU 4bit 연산 결과가 기대값 64와 다를 때.
    :raises RuntimeError: MLX CPU 연산 또는 실행 시 C++ 컴파일러를 사용할 수 없을 때.
    """
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.cpu)
    inputs = mx.ones((1, 64), dtype=mx.float16)
    weights = mx.ones((32, 64), dtype=mx.float16)
    quantized = mx.quantize(weights, group_size=64, bits=4)
    output = mx.quantized_matmul(inputs, *quantized, transpose=True, group_size=64, bits=4)
    mx.eval(output)
    value = output[0, 0].item()
    if output.shape != (1, 32) or value != 64:
        raise ValueError('CPU quantized matrix multiplication returned an unexpected result')
    activated = mx.compile(nn.silu)(mx.array([0.0, 1.0], dtype=mx.float16))
    mx.eval(activated)
    activation = activated.tolist()
    if activation[0] != 0 or not 0.7 < activation[1] < 0.8:
        raise ValueError('CPU compiled activation returned an unexpected result')
    return {'device': str(mx.default_device()), 'shape': list(output.shape), 'value': value,
            'compiled_activation': activation}


def probe_llamacpp() -> dict:
    """CPU 네이티브 바인딩 로드와 메모리 매핑 지원을 확인한다.

    :return: CPU 백엔드·네이티브 라이브러리 정보·mmap 지원 여부 사전. 모델 추론은 아니다.
    :raises ImportError: CPU 바인딩을 설치하지 않은 경우.
    :raises RuntimeError: 네이티브 라이브러리를 사용할 수 없는 경우.
    """
    from llama_cpp import llama_cpp
    return {'device': 'cpu', 'engine': 'llama-cpp',
            'native_library': llama_cpp.llama_print_system_info().decode('utf-8'),
            'mmap_supported': bool(llama_cpp.llama_supports_mmap()), 'gpu_layers': 0}


def benchmark(model_dir: Path, max_tokens: int, probe_only: bool = False,
              backend: str = 'mlx', threads: int = 4) -> dict:
    """CPU 연산 또는 실제 모델의 고정 영어 한 문장 번역을 실행하고 측정값을 반환한다.

    :param model_dir: 기존 MLX 4bit KoreanLM 모델 경로. 다운로드나 변환은 하지 않는다.
    :param max_tokens: 시도당 최대 출력 토큰 수. CLI에서는 1 이상 128 이하.
    :param probe_only: 참이면 작은 연산만 검사하고 모델을 로드하지 않는다.
    :param backend: mlx 또는 llama-cpp. 후자는 CPU 바인딩 로드만 예비 검사한다.
    :param threads: llama-cpp의 양의 CPU 스레드 수. 기본 4.
    :return: 장비·패키지·연산·시간·최대 RSS·번역 검증 결과 사전. 의미 정확도 평가는 아니다.
    :raises OSError: 로컬 모델 파일을 읽을 수 없을 때.
    :raises RuntimeError: 모델 또는 CPU 연산을 사용할 수 없을 때.

    MLX CPU를 기본 장치로 설정하며 모델 추론은 현재 프로세스에서 실행한다.
    모델 로딩·번역 시작은 표준 오류에 기록하고 보고서 JSON은 반환값으로 구분한다.
    """
    if backend not in ('mlx', 'llama-cpp') or threads < 1:
        raise ValueError('Expected mlx or llama-cpp and a positive thread count')
    packages = ('mlx', 'mlx-lm') if backend == 'mlx' else ('llama-cpp-python', 'gguf')
    report = {'system': platform.system(), 'machine': platform.machine(), 'backend': backend,
              'python': platform.python_version(),
              'versions': {name: version(name) for name in packages},
              'probe': probe_cpu() if backend == 'mlx' else probe_llamacpp(),
              'mode': 'probe' if probe_only else 'model'}
    if not probe_only:
        print('Loading the existing 4bit model on CPU...', file=sys.stderr, flush=True)
        started = perf_counter()
        service = (TranslationService(model_dir, ROOT / 'templates/korean.json', device='cpu')
                   if backend == 'mlx' else LlamaCppTranslationService(
                       model_dir, ROOT / 'templates/korean.json', threads=threads))
        report['load_seconds'] = perf_counter() - started
        print('Translating one fixed sentence on CPU...', file=sys.stderr, flush=True)
        started = perf_counter()
        try:
            result = service.translate('The velocity is constant.', '', max_tokens)
        except ValueError as error:
            report['translation'] = {'status': 'rejected', 'message': str(error)}
        else:
            report['translation'] = {'status': 'accepted', **result}
        report['translation_wall_seconds'] = perf_counter() - started
        report['max_tokens'] = max_tokens
        if backend == 'llama-cpp':
            report['threads'] = threads
            service.close()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_mib'] = rss / (1024 ** 2 if platform.system() == 'Darwin' else 1024)
    return report


def main() -> int:
    """CPU 검증을 실행하고 측정 JSON을 표시하거나 저장한 뒤 종료 코드를 반환한다.

    :return: 성공은 0, 실행 오류는 1, 번역 검증 실패는 2.
    :raises SystemExit: CLI 인자가 잘못된 경우.

    --output 지정 시 부모 디렉터리를 만들고 측정 JSON 파일을 덮어쓴다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=Path(os.getenv('KAPTION_MODEL_DIR', ROOT / 'models/koreanlm-4bit')))
    parser.add_argument('--backend', choices=['mlx', 'llama-cpp'],
                        default=os.getenv('KAPTION_TRANSLATION_BACKEND', 'mlx'))
    parser.add_argument('--threads', type=int, default=int(os.getenv('KAPTION_CPU_THREADS', '4')))
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 128:
        parser.error('--max-tokens must be between 1 and 128')
    if args.threads < 1:
        parser.error('--threads must be positive')
    try:
        report = benchmark(args.model_dir, args.max_tokens, args.probe_only,
                           backend=args.backend, threads=args.threads)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        report = {'error': str(error), 'system': platform.system(), 'machine': platform.machine()}
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    if 'error' in report:
        return 1
    return 2 if report.get('translation', {}).get('status') == 'rejected' else 0


if __name__ == '__main__':
    raise SystemExit(main())
