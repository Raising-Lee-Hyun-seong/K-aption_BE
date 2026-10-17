"""평가 전용 Qwen 서비스로 저장 STT의 문맥·검증·재시도 번역을 실행하고 근거를 저장한다."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.translation import TranslationService
from scripts.evaluate_candidates import generate_once, prepare_candidate, save_report, sha256_file
from scripts.evaluate_lecture_quality import evaluate_groups, full_groups, summarize
from scripts.evaluate_prompt_matrix import GLOSSARY_ADDITIONS, prompt_service, variant_prompt


class EvaluationQwenService(TranslationService):
    """API를 바꾸지 않고 기존 출력 검사·재시도에 Qwen 전용 프롬프트를 연결한다."""

    def __init__(self, model, tokenizer, context_format: str = 'system'):
        """외부에서 로드한 Qwen과 기존 API 용어 검증·잠금을 평가용으로 준비한다.

        :param model: 검증된 로컬 Qwen3 MLX 모델 인스턴스.
        :param tokenizer: 같은 모델의 채팅 템플릿과 인코더를 제공하는 토크나이저.
        :param context_format: 참고 문맥을 넣는 방식. inline 또는 system.
        :return: None.
        """
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tokenizer = model, tokenizer
        self.context_format = context_format
        self.sampler = make_sampler(temp=0)
        self.glossary = prompt_service().glossary
        self._lock = Lock()
        self.raw_attempts = []

    def _build_prompt(self, text: str, context: str, retry: bool = False) -> str:
        """Qwen 보완 조건에 앞 영어 문맥과 재시도 지시를 연결한다.

        :param text: 현재 번역할 영어 원문.
        :param context: 앞 영어 원문. 마지막 256자만 참고로 제공한다.
        :param retry: 문맥 없는 재시도에서 완결된 한국어 문장 지시를 추가할지 여부.
        :return: Qwen 채팅 템플릿으로 구성한 프롬프트 문자열.
        """
        return variant_prompt('qwen-relations-v2', self.tokenizer, self, text, context, retry,
                              self.context_format)

    def _generate_output(self, prompt: str, max_tokens: int) -> dict:
        """기존 스트림 조기 종료를 수행하고 정리 전 출력과 프롬프트를 평가 근거로 보존한다.

        :param prompt: 모델에 전달할 프롬프트 문자열.
        :param max_tokens: 한 시도에서 생성할 최대 토큰 수.
        :return: 기존 생성기의 text·tokens·finish_reason·stop_reason 사전.
        """
        result = super()._generate_output(prompt, max_tokens)
        self.raw_attempts.append({'prompt': prompt, **result})
        return result


def main() -> None:
    """Qwen을 로컬에서만 로드해 36개 저장 STT 묶음을 평가하고 결과·원시 시도를 저장한다.

    :return: None.
    :raises FileExistsError: 출력 파일이 이미 있어 이전 결과를 덮어쓸 위험이 있을 때.
    :raises ValueError: 모델 출처 검증에 실패하거나 입력 묶음 수가 36개가 아닐 때.

    평가 JSON을 저장하며 실제 STT·API 서버는 실행하거나 변경하지 않는다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluated-on', type=date.fromisoformat, required=True)
    parser.add_argument('--context-format', choices=('inline', 'system'), default='system')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new output path: {args.output}')
    reference = json.loads((ROOT / 'evaluation/mit_rocket_reference.json').read_text())
    baseline = json.loads((ROOT / 'evaluation/lecture.json').read_text())
    _, groups = full_groups(reference, baseline)
    if len(groups) != 36:
        raise ValueError('Expected the existing 36 saved STT groups')
    manifest = json.loads((ROOT / 'evaluation/model_candidates.json').read_text())
    candidate = next(row for row in manifest['selected'] if row['original_repository'].startswith('Qwen/'))
    source = prepare_candidate(candidate, download=False)
    import mlx.core as mx
    from mlx_lm import load

    mx.reset_peak_memory()
    started = perf_counter()
    model, tokenizer = load(str(ROOT / candidate['local_directory']),
                            tokenizer_config={'token': False, 'local_files_only': True})
    mx.synchronize()
    service = EvaluationQwenService(model, tokenizer, args.context_format)
    inputs = ('evaluation/lecture.json', 'evaluation/mit_rocket_reference.json')
    implementations = ('scripts/evaluate_qwen_lecture.py', 'scripts/evaluate_prompt_matrix.py',
                       'scripts/evaluate_candidates.py', 'scripts/evaluate_lecture_quality.py',
                       'app/services/translation.py', 'app/services/translation_validation.py',
                       'app/services/subtitles.py', 'data/physics_glossary.json',
                       str(GLOSSARY_ADDITIONS.relative_to(ROOT)))
    report = {
        'evaluated_on': args.evaluated_on.isoformat(), 'variant': 'qwen-relations-v2',
        'source_verification': source, 'lecture_source': reference['source'],
        'model_load_seconds': perf_counter() - started,
        'settings': {'temperature': 0, 'max_tokens': 128, 'context_characters': 256,
                     'context_format': args.context_format,
                     'max_attempts': 2, 'retry_context': False, 'output_cleaning': True,
                     'early_stopping_rules': True, 'prompt_plus_output_limit': 2048,
                     'validation_glossary': 'unchanged API glossary (28 entries)'},
        'input_sha256': {name: sha256_file(ROOT / name) for name in inputs},
        'implementation_sha256': {name: sha256_file(ROOT / name) for name in implementations},
        'limits': ['Saved STT is reused; decoding and end-to-end API time are not measured.',
                   'This differs from the 22-case raw generation experiment.',
                   'Validation success is not semantic accuracy; expert review remains pending.',
                   'This is an evaluation subclass; the API remains KoreanLM.'],
        'complete': False,
    }
    report['warmup'] = generate_once(model, tokenizer, service._build_prompt(
        'Hello, welcome to the lecture.', ''), 128)
    save_report(args.output, report)
    report['results'] = evaluate_groups(service, groups)
    report['summary'] = summarize(report['results'])
    report['raw_attempts'] = service.raw_attempts
    report['mlx_process_peak_gb'] = mx.get_peak_memory() / 1e9
    report['complete'] = True
    save_report(args.output, report)


if __name__ == '__main__':
    main()
