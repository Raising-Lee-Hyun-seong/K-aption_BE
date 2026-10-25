"""저장된 동일 강의 입력으로 Qwen CPU와 평가 전용 M2M100 INT8의 시간·출력을 비교한다."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.translation_http import HttpTranslationService
from app.services.translation_inputs import protect_math, restore_math
from scripts.evaluate_cpu_stt_options import save_new
from scripts.evaluate_lecture_quality import collect_attempts


class RecordingQwen(HttpTranslationService):
    """기존 Qwen 요청·검증을 그대로 실행하면서 원시 출력과 서버 시간을 보존한다."""

    def _stream_output(self, prompt: str, max_tokens: int):
        """기존 생성 스트림을 전달하고 시도별 원시 출력·서버 시간을 복사한다.

        :param prompt: Qwen 시스템·사용자 메시지의 JSON 문자열.
        :param max_tokens: 한 시도의 출력 토큰 상한.
        :return: 원래 생성 조각을 제공하는 생성기.
        :yield: text·generation_tokens·finish_reason·server_timings를 포함한 생성 조각.
        """
        row = {'raw_output': '', 'server_timings': None}
        self.recorded.append(row)
        stream = super()._stream_output(prompt, max_tokens)
        try:
            for chunk in stream:
                row['raw_output'] += chunk.text
                if chunk.server_timings is not None:
                    row['server_timings'] = chunk.server_timings
                yield chunk
        finally:
            stream.close()


def load_cases() -> list:
    """기존 영상의 17개 묶음과 별도 강의의 두 문장을 원문 그대로 불러온다.

    :return: id·suite·text·context가 있는 19개 사례 목록. 정답 번역은 포함하지 않는다.
    """
    rows = json.loads((ROOT / 'evaluation/rocket-docker-inputs-v4-full-261025.json').read_text())
    cases = [{'id': f'rocket-{i}', 'suite': 'rocket', 'text': row['text'],
              'context': rows['result']['segments'][i - 1]['text'][-256:] if i else ''}
             for i, row in enumerate(rows['result']['segments'])]
    other = json.loads((ROOT / 'evaluation/holdout-linear-algebra-261021.json').read_text())
    cases.extend({'id': row['id'], 'suite': 'linear_algebra', 'text': row['text'], 'context': ''}
                 for row in other['cases'])
    return cases


def run_qwen(service: RecordingQwen, case: dict) -> dict:
    """기존 캐시·검증·최대 두 번 재시도로 한 사례의 Qwen 번역을 실행한다.

    :param service: 평가 전용으로 생성한 기존 Qwen HTTP 서비스.
    :param case: 현재 영어 원문과 이전 영어 문맥이 있는 사례.
    :return: 번역 또는 오류, 원시 시도, 검증 로그와 소요 초를 가진 사전.
    """
    service.recorded = []
    start = perf_counter()
    with collect_attempts() as decisions:
        try:
            result = service.translate(case['text'], case['context'], 128)
            row = {'translation': result['translation'], 'accepted': True, 'response': result}
        except ValueError as exc:
            row = {'translation': None, 'accepted': False,
                   'error': {'type': type(exc).__name__, 'message': str(exc)}}
    return {**row, 'seconds': perf_counter() - start, 'attempts': service.recorded,
            'decisions': decisions}


def run_m2m(model, tokenizer, validator: HttpTranslationService, case: dict,
            protect: bool, beam: int) -> dict:
    """별도 지시문 없이 영어를 한국어로 번역하고 기존 원문 기준 검증을 기록한다.

    :param model: CPU INT8로 로드한 CTranslate2 Translator.
    :param tokenizer: src_lang이 en인 로컬 M2M100Tokenizer.
    :param validator: 기존 Qwen과 같은 용어·수식·존댓말 검사 제공 객체. 생성에는 사용하지 않는다.
    :param case: 영어 원문과 검사에만 사용할 이전 문맥이 있는 사례.
    :param protect: 참이면 기존 단순 수식 보호와 표식 복원을 사용한다.
    :param beam: 실제 번역 탐색 폭. 양의 정수.
    :return: 원시 출력·복원 출력·형식 검사 결과·소요 초 사전. 재시도·말투 보정은 하지 않는다.
    """
    start = perf_counter()
    text = protect_math(case['text'])[0] if protect else case['text']
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    result = model.translate_batch([source], target_prefix=[['__ko__']], beam_size=beam,
                                   max_decoding_length=128, return_scores=False)[0]
    tokens = result.hypotheses[0]
    raw = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)
    translated, restoration_problem = restore_math(raw, case['text']) if protect else (raw, None)
    reason = restoration_problem or validator._rejection_reason(translated, case['text'], case['context'])
    if len(tokens) >= 128:
        reason = reason or 'possible_token_limit'
    return {'model_input': text, 'raw_output': raw, 'translation': translated,
            'accepted': reason is None, 'rejection_reason': reason,
            'input_tokens': len(source), 'output_tokens': len(tokens),
            'seconds': perf_counter() - start}


def memory_observation() -> dict:
    """현재 평가 프로세스와 해당 컨테이너의 메모리 최대치를 읽는다.

    :return: Linux ru_maxrss의 KiB 값과 cgroup memory.peak·memory.events 값 사전.
    """
    values = {'process_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    for name in ('memory.peak', 'memory.events'):
        path = Path('/sys/fs/cgroup') / name
        values[name] = path.read_text().strip() if path.exists() else None
    return values


def main() -> None:
    """한 모델 조건을 실행하고 새 원시 기록과 집계 보고서를 저장한다.

    :return: None. 모델을 로드하거나 기존 CPU 서버를 호출하고 새 평가 파일을 생성한다.
    :raises FileExistsError: 최종 또는 사례별 평가 경로가 이미 존재하는 경우.
    :raises OSError: 모델·입력 자료·출력 경로에 접근할 수 없는 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['qwen', 'm2m'], required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--protect-math', action='store_true')
    parser.add_argument('--beam', type=int, choices=[1, 5], default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--translator-url', default='http://translator:8080')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = load_cases()
    service = RecordingQwen(args.translator_url, ROOT / 'templates/korean.json', cache_prompt=True)
    report = {'mode': args.mode,
              'math_protection': args.protect_math if args.mode == 'm2m' else 'existing_qwen_service',
              'beam': args.beam if args.mode == 'm2m' else None,
              'architecture': platform.machine(), 'platform': platform.platform(),
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
              'results': [], 'limits': [
                  'Saved STT replay only; excludes upload, STT, grouping and subtitles.',
                  'Development lecture and two previously inspected different-lecture sentences.',
                  'One warm pass per model option; not a statistically controlled speed comparison.',
                  'M2M has no preceding context, glossary enforcement or politeness instruction.',
                  'M2M gets the same post-generation checks; no retry or manual output repair.',
                  'Passing heuristic checks does not imply semantic correctness.',
                  'Container memory excludes the separate Qwen server and existing API process.',
                  'Mac ARM64 Docker only; Windows amd64 is untested.',
              ]}
    try:
        if args.mode == 'm2m':
            import ctranslate2
            from transformers import M2M100Tokenizer

            start = perf_counter()
            tokenizer = M2M100Tokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
            tokenizer.src_lang = 'en'
            assert tokenizer.lang_code_to_token['ko'] == '__ko__'
            model = ctranslate2.Translator(str(args.model_dir), device='cpu', compute_type='int8',
                                          inter_threads=1, intra_threads=4)
            report['load_seconds'] = perf_counter() - start
            report['compute_type'] = model.compute_type
            report['packages'] = {name: importlib.metadata.version(name)
                                  for name in ('ctranslate2', 'transformers', 'sentencepiece')}
            warm = {'text': 'This is the initial mass.', 'context': ''}
            report['warmup'] = run_m2m(model, tokenizer, service, warm, args.protect_math, args.beam)
        else:
            service.cache_prompt = False
            report['warmup'] = run_qwen(service, {'text': 'This is the initial mass.', 'context': ''})
            service.cache_prompt = True
        for index, case in enumerate(cases):
            result = (run_qwen(service, case) if args.mode == 'qwen' else
                      run_m2m(model, tokenizer, service, case, args.protect_math, args.beam))
            row = {**case, **result}
            report['results'].append(row)
            save_new(args.output.with_name(args.output.stem + f'-case{index}.json'), row)
            print(json.dumps({'mode': args.mode, 'id': case['id'], 'seconds': row['seconds'],
                              'accepted': row['accepted']}), flush=True)
        report['seconds'] = sum(row['seconds'] for row in report['results'])
        report['accepted'] = sum(row['accepted'] for row in report['results'])
        report['memory'] = memory_observation()
        save_new(args.output, report)
    finally:
        service.close()


if __name__ == '__main__':
    main()
