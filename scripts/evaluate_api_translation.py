"""실제 API 번역 서비스로 저장 강의를 기본·수식 결합 조건에서 재생하고 출력을 기록한다."""

import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_translation_service, TEMPLATE_PATH
from app.services.subtitles import merge_math_groups
from scripts.evaluate_candidates import save_report, sha256_file
from scripts.evaluate_lecture_quality import baseline_for_indices, evaluate_groups, full_groups, summarize


def evaluate(model_path: Path, backend: str, evaluated_on: date, merge_math: bool = False) -> dict:
    """API와 같은 팩토리·실제 모델로 저장 STT를 번역하고 원시 생성 시도를 보존한다.

    :param model_path: Qwen MLX 디렉터리 또는 Qwen GGUF 파일 경로.
    :param backend: mlx 또는 llama-cpp.
    :param evaluated_on: 실제 평가를 수행한 날짜. 명령행에서 명시한다.
    :param merge_math: 참이면 기존 36구간에 API와 같은 수식 결합을 적용한다.
    :return: 원문·번역·기존 구간 정렬·검증 실패·생성 시도와 출처를 담은 보고서.
    :raises OSError: 모델·평가 입력 파일을 읽지 못한 경우.
    :raises ValueError: 모델이나 입력 구성이 유효하지 않은 경우.

    현재 프로세스의 번역 모델 환경을 Qwen으로 설정하고 모델을 로드한다.
    저장 STT를 재사용하므로 음성 인식과 HTTP 전체 성능 평가는 아니다.
    """
    reference = json.loads((ROOT / 'evaluation/mit_rocket_reference.json').read_text())
    baseline = json.loads((ROOT / 'evaluation/lecture.json').read_text())
    _, groups = full_groups(reference, baseline)
    if len(groups) != 36:
        raise ValueError('Expected 36 saved STT groups')
    original_groups = groups
    if merge_math:
        groups = merge_math_groups(groups)
    for index, group in enumerate(groups):
        group['id'] = f'group-{index}'
        indices = set(group['source_indices'])
        group['original_group_ids'] = [f'stt-{number}' for number, original in enumerate(original_groups)
                                       if indices.intersection(original['source_indices'])]
        group['baseline'] = baseline_for_indices(baseline['results'], group['source_indices'])
    os.environ['KAPTION_TRANSLATION_BACKEND'] = backend
    os.environ['KAPTION_TRANSLATION_MODEL'] = 'qwen3'
    started = perf_counter()
    service = create_translation_service(model_path, TEMPLATE_PATH)
    report = {'evaluated_on': evaluated_on.isoformat(), 'backend': backend,
              'model_path': str(model_path), 'service': type(service).__name__,
              'load_seconds': perf_counter() - started,
              'lecture_source': reference['source'],
              'input_sha256': sha256_file(ROOT / 'evaluation/lecture.json'),
              'merge_math': merge_math, 'original_group_count': len(original_groups),
              'raw_attempts': [],
              'limits': ['Saved STT replay, not new transcription or end-to-end HTTP timing.',
                         'Validation success is not semantic accuracy; independent review pending.']}
    implementations = ('app/services/translation.py', 'app/services/translation_qwen.py',
                       'app/services/translation_llamacpp.py', 'app/services/translation_validation.py',
                       'app/services/subtitles.py',
                       'data/physics_glossary.json', 'data/qwen_translation_terms.json')
    report['implementation_sha256'] = {name: sha256_file(ROOT / name) for name in implementations}
    weight = model_path / 'model.safetensors' if model_path.is_dir() else model_path
    report['model_sha256'] = sha256_file(weight)
    generate = service._generate_output

    def collect(prompt: str, max_tokens: int) -> dict:
        """실제 생성기를 실행하고 정리·검증 전 출력과 프롬프트를 수집한다.

        :param prompt: 모델별 채팅 템플릿으로 만든 실제 프롬프트.
        :param max_tokens: 한 시도에서 생성할 최대 토큰 수.
        :return: 실제 생성기의 text·tokens·종료 사유 사전.
        """
        result = generate(prompt, max_tokens)
        report['raw_attempts'].append({'prompt': prompt, **result})
        return result

    service._generate_output = collect
    try:
        report['results'] = evaluate_groups(service, groups)
        report['summary'] = summarize(report['results'])
        report['summary']['failed_original_group_ids'] = [original_id for row in report['results']
                                                         if row['error'] for original_id in row['original_group_ids']]
    finally:
        close = getattr(service, 'close', None)
        if callable(close):
            close()
    if backend == 'mlx':
        import mlx.core as mx
        report['mlx_peak_gb'] = mx.get_peak_memory() / 1e9
    return report


def main() -> None:
    """명시한 모델과 새 보고서 경로로 강의 재생 평가를 실행한다.

    :return: None. 평가 JSON을 새 경로에 저장한다.
    :raises FileExistsError: 기존 평가 결과를 덮어쓰려는 경우.
    :raises OSError: 모델·입출력 파일에 접근할 수 없는 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('mlx', 'llama-cpp'), default='mlx')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluated-on', type=date.fromisoformat, required=True)
    parser.add_argument('--merge-math', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new report path: {args.output}')
    report = evaluate(args.model, args.backend, args.evaluated_on, args.merge_math)
    save_report(args.output, report)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
