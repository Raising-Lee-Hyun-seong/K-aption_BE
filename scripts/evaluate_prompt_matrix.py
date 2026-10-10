"""같은 22문장으로 두 로컬 모델의 네 프롬프트를 재시도 없이 비교하고 원시 결과를 저장한다."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.translation import TranslationService
from scripts.evaluate_candidates import (
    build_prompt, generate_once, inspect_output, load_cases, prepare_candidate,
    save_report, sha256_file,
)

VARIANTS = ('koreanlm-current', 'koreanlm-simple', 'qwen-base', 'qwen-glossary')
QWEN_INSTRUCTION = (
    'Translate the following English text into Korean. Preserve negation, numbers '
    'and mathematical relations. Output only the translation.'
)
QWEN_PRESERVATION = (
    ' Keep mathematical symbols, operations, inequalities and corrections unchanged. '
    'Preserve the subject and whether a sentence is a statement or question. '
    'Do not add facts, units, explanations or instructions.'
)


def prompt_service() -> TranslationService:
    """모델 로딩 없이 현재 API의 프롬프트·용어 매칭 로직만 준비한다.

    :return: 템플릿·용어집만 설정된 실험 전용 TranslationService 객체.
    """
    service = TranslationService.__new__(TranslationService)
    service.template = json.loads((ROOT / 'templates/korean.json').read_text())
    service.glossary = json.loads((ROOT / 'data/physics_glossary.json').read_text())
    return service


def variant_prompt(variant: str, tokenizer, service: TranslationService, text: str) -> str:
    """기준 번역을 사용하지 않고 지정된 실험 조건의 번역 프롬프트를 구성한다.

    :param variant: VARIANTS의 네 조건 중 하나.
    :param tokenizer: Qwen 채팅 템플릿을 제공하는 모델 토크나이저.
    :param service: 현재 템플릿·용어집·원문 매칭 로직을 가진 실험 객체.
    :param text: 번역할 영어 원문. 이전 문맥을 추가하지 않는다.
    :return: 모델에 그대로 전달할 프롬프트 문자열.
    :raises ValueError: 알려지지 않은 실험 조건일 때.
    """
    if variant == 'koreanlm-current':
        return service._build_prompt(text, '')
    if variant == 'koreanlm-simple':
        return service.template['prompt_input'].format(
            instruction='다음 영어 원문을 한국어로 번역하세요. 번역문만 출력하세요.', input=text)
    if variant == 'qwen-base':
        return build_prompt(tokenizer, {'original_repository': 'Qwen/Qwen3-4B-Instruct-2507'}, text)
    if variant == 'qwen-glossary':
        instruction = QWEN_INSTRUCTION + QWEN_PRESERVATION
        terms = service._matching_terms(text)
        if terms:
            instruction += ' Terminology: ' + '; '.join(
                f'{entry["en"]}={entry["ko"]}' for entry in terms) + '.'
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': instruction + '\n\n' + text}],
            tokenize=False, add_generation_prompt=True)
    raise ValueError(f'Unknown variant: {variant}')


def verify_koreanlm(directory: Path) -> dict:
    """기존 KoreanLM의 설정·가중치 인덱스와 파일을 확인하고 재현용 해시를 기록한다.

    :param directory: 기존 KoreanLM 4bit 디렉터리.
    :return: 원본 모델 출처와 로컬 파일별 크기·SHA256 사전.
    :raises FileNotFoundError: 필수 파일이나 인덱스가 참조한 가중치가 없을 때.
    :raises ValueError: 4bit가 아니거나 가중치가 비어 있거나 모델 경로를 벗어날 때.

    원본 가중치의 공개 해시와 대조한 검증은 아니며 로컬 실행 파일의 동일성만 기록한다.
    """
    config = json.loads((directory / 'config.json').read_text())
    if config.get('quantization', {}).get('bits') != 4:
        raise ValueError('KoreanLM must use the existing 4bit configuration')
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    if not names:
        raise ValueError('Empty KoreanLM weight index')
    metadata = ['config.json', 'model.safetensors.index.json', 'source.json',
                'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                'generation_config.json']
    files = []
    for name in names + metadata:
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Model file must remain inside the model directory')
        if path.stat().st_size == 0:
            raise ValueError(f'Empty model file: {name}')
        files.append({'name': name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return {'original_source': json.loads((directory / 'source.json').read_text()),
            'files': files, 'verification_scope': 'local fingerprints, not upstream LFS verification'}


def diagnose_output(service: TranslationService, source: str, output: str, finish_reason: str) -> dict:
    """원시 출력을 바꾸지 않고 형식·보존 검사와 현재 용어 검사를 진단 정보로 기록한다.

    :param service: 원문 용어 매칭과 현재 검증 로직을 제공하는 실험 객체.
    :param source: 검사 기준인 영어 원문.
    :param output: 정리하지 않은 원시 생성 텍스트.
    :param finish_reason: MLX 생성 종료 사유.
    :return: 기본 진단과 원시 텍스트에 적용한 API 검증 사유의 사전. 의미 판정이 아니다.
    """
    return {'validation_problems': inspect_output(source, output, finish_reason),
            'raw_validator_rejection_reason': service._rejection_reason(output, source, '')}


def main() -> None:
    """조건 하나를 별도 프로세스에서 두 번 평가하고 중간 결과를 계속 저장한다.

    :return: None.
    :raises ValueError: CLI 생성 조건이나 모델 파일 검증이 잘못되었을 때.

    로컬 모델만 로드하고 평가 JSON을 저장한다. API·가중치를 변경하거나 재시도하지 않는다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        raise ValueError('Repeats and max_tokens must be positive')
    service = prompt_service()
    if args.variant.startswith('koreanlm-'):
        directory = ROOT / 'models/koreanlm-4bit'
        source = verify_koreanlm(directory)
    else:
        manifest = json.loads((ROOT / 'evaluation/model_candidates.json').read_text())
        candidate = next(row for row in manifest['selected'] if row['original_repository'].startswith('Qwen/'))
        directory = ROOT / candidate['local_directory']
        source = prepare_candidate(candidate, download=False)
    print(json.dumps({'event': 'source_verified', 'variant': args.variant}), flush=True)
    import mlx.core as mx
    from mlx_lm import load

    mx.reset_peak_memory()
    started = perf_counter()
    model, tokenizer = load(str(directory), tokenizer_config={'token': False, 'local_files_only': True})
    mx.synchronize()
    output = args.output or ROOT / f'evaluation/prompt-matrix-{args.variant}.json'
    report = {
        'evaluated_on': '2026-10-10', 'variant': args.variant, 'model_directory': str(directory.relative_to(ROOT)),
        'source_verification': source, 'model_load_seconds': perf_counter() - started,
        'platform': {'system': platform.system(), 'machine': platform.machine()},
        'versions': {name: version(name) for name in ('mlx-lm', 'mlx', 'transformers')},
        'settings': {'temperature': 0, 'max_tokens': args.max_tokens, 'context': False,
                     'retries': 0, 'early_stopping_rules': False, 'output_cleaning': False,
                     'repeats': args.repeats, 'prompt_plus_output_limit': 2048},
        'eos_token_ids': sorted(tokenizer.eos_token_ids),
        'input_sha256': {name: sha256_file(ROOT / name) for name in
                         ('evaluation/mit_rocket_cases.json', 'evaluation/physics_cases.json')},
        'implementation_sha256': {name: sha256_file(ROOT / name) for name in
                                  ('scripts/evaluate_prompt_matrix.py', 'scripts/evaluate_candidates.py',
                                   'app/services/translation.py', 'app/services/translation_validation.py',
                                   'templates/korean.json', 'data/physics_glossary.json')},
        'limits': ['Raw validator diagnostics are not semantic accuracy or the actual API retry outcome.',
                   'Qwen glossary changes terminology and preservation instructions together.',
                   'Current KoreanLM prompt uses no preceding context; this is not the 36-group API run.'],
        'results': [], 'complete': False, 'semantic_review': 'pending',
    }
    report['warmup'] = generate_once(model, tokenizer, variant_prompt(
        args.variant, tokenizer, service, 'Hello, welcome to the lecture.'), args.max_tokens)
    save_report(output, report)
    for repeat in range(1, args.repeats + 1):
        for case in load_cases():
            row = {**case, 'repeat': repeat}
            started = perf_counter()
            try:
                prompt = variant_prompt(args.variant, tokenizer, service, case['text'])
                row['prompt'] = prompt
                if len(tokenizer.encode(prompt)) + args.max_tokens > 2048:
                    raise ValueError('Prompt and output exceed the shared 2048-token limit')
                row.update(generate_once(model, tokenizer, prompt, args.max_tokens))
                row.update(diagnose_output(service, case['text'], row['translation'], row['finish_reason']))
                row['error'] = None
            except Exception as exc:
                row.update(translation=None, error=type(exc).__name__ + ': ' + str(exc),
                           elapsed_seconds=perf_counter() - started, validation_problems=['generation_error'])
            report['results'].append(row)
            save_report(output, report)
            print(json.dumps({'event': 'case_complete', 'variant': args.variant, 'id': case['id'],
                              'repeat': repeat, 'seconds': row['elapsed_seconds'],
                              'finish_reason': row.get('finish_reason'), 'error': row['error']}), flush=True)
    report['complete'] = True
    report['mlx_process_peak_gb'] = mx.get_peak_memory() / 1e9
    save_report(output, report)


if __name__ == '__main__':
    main()
