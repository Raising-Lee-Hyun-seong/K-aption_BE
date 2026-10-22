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

from app.main import TEMPLATE_PATH
from app.services.translation_http import HttpTranslationService
from scripts.download_cpu_model import load_manifest, file_digest
from app.services.subtitles import merge_math_groups
from scripts.evaluate_candidates import save_report, sha256_file
from scripts.evaluate_lecture_quality import baseline_for_indices, evaluate_groups, full_groups, summarize


def evaluate(base_url: str, manifest_path: Path, model_path: Path, evaluated_on: date,
             merge_math: bool = False, holdout: Path = None) -> dict:
    """같은 HTTP 서비스로 저장 STT와 별도 강의 구간을 재생하고 실제 출력을 보존한다.

    :param base_url: 로컬 CPU llama.cpp 서버의 기본 HTTP 주소.
    :param manifest_path: 평가할 모델의 고정 배포·양자화·SHA256 명세.
    :param model_path: 실제 서버에 로드한 GGUF 원본 파일. 평가 전 크기·해시를 검사한다.
    :param evaluated_on: 실제 평가일. 명령행에서 명시한다.
    :param merge_math: 기존 36구간에 운영 API와 같은 수식 결합을 적용할지 여부.
    :param holdout: 선택적 독립 강의 사례 JSON. cases 목록을 저장 입력 뒤에 추가한다.
    :return: 원문·출력·원시 생성·서버 설정·출처와 검증 요약 보고서.
    :raises ValueError: 모델 무결성·서버 경로·입력 구성이 유효하지 않은 경우.
    :raises OSError: 입력 파일이나 서버에 접근하지 못한 경우.

    STT는 재실행하지 않으며 서버 모델을 교체하지 않는다. 보고서는 검증 통과를
    의미 정확도로 해석하지 않고 각 구간에 수동 평가가 필요함을 명시한다.
    """
    manifest = load_manifest(manifest_path)
    if model_path.stat().st_size != manifest['bytes'] or file_digest(model_path) != manifest['sha256']:
        raise ValueError('Evaluation model failed size/SHA256 verification')
    reference = json.loads((ROOT / 'evaluation/mit_rocket_reference.json').read_text())
    baseline = json.loads((ROOT / 'evaluation/lecture.json').read_text())
    _, groups = full_groups(reference, baseline)
    if len(groups) != 36:
        raise ValueError('Expected 36 saved STT groups')
    original_groups = list(groups)
    if merge_math:
        groups = merge_math_groups(groups)
    for index, group in enumerate(groups):
        group['id'] = f'group-{index}'
        indices = set(group['source_indices'])
        group['original_group_ids'] = [f'stt-{number}' for number, original in enumerate(original_groups)
                                       if indices.intersection(original['source_indices'])]
        group['baseline'] = baseline_for_indices(baseline['results'], group['source_indices'])
    if holdout:
        cases = json.loads(holdout.read_text())['cases']
        groups.extend({**case, 'original_group_ids': [], 'source_indices': [], 'holdout': True}
                      for case in cases)
    started = perf_counter()
    service = HttpTranslationService(base_url, TEMPLATE_PATH, timeout=180)
    try:
        response = service.client.get('/props')
        response.raise_for_status()
        props = response.json()
        if Path(props.get('model_path', '')).name != model_path.name:
            raise ValueError('Server loaded model does not match the evaluation model')
    except Exception:
        service.close()
        raise
    report = {'evaluated_on': evaluated_on.isoformat(), 'backend': 'llama.cpp-server-cpu',
              'base_url': base_url, 'model_manifest': manifest, 'server_props': props,
              'service': type(service).__name__, 'connect_seconds': perf_counter() - started,
              'lecture_source': reference['source'],
              'input_sha256': sha256_file(ROOT / 'evaluation/lecture.json'),
              'holdout': json.loads(holdout.read_text()) if holdout else None,
              'merge_math': merge_math, 'original_group_count': len(original_groups),
              'raw_attempts': [],
              'limits': ['Saved STT replay, not STT or end-to-end HTTP pipeline timing.',
                         'Validation success is not semantic accuracy; manual review required.']}
    implementations = ('app/services/translation.py', 'app/services/translation_qwen.py',
                       'app/services/translation_http.py', 'app/services/translation_validation.py',
                       'app/services/subtitles.py', 'data/physics_glossary.json',
                       'data/qwen_translation_terms.json')
    report['implementation_sha256'] = {name: sha256_file(ROOT / name) for name in implementations}
    report['model_sha256'] = manifest['sha256']
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
    return report


def main() -> None:
    """명시한 모델과 새 보고서 경로로 강의 재생 평가를 실행한다.

    :return: None. 평가 JSON을 새 경로에 저장한다.
    :raises FileExistsError: 기존 평가 결과를 덮어쓰려는 경우.
    :raises OSError: 모델·입출력 파일에 접근할 수 없는 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'docker/qwen3-model.json')
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluated-on', type=date.fromisoformat, required=True)
    parser.add_argument('--merge-math', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new report path: {args.output}')
    report = evaluate(args.base_url, args.manifest, args.model, args.evaluated_on, args.merge_math, args.holdout)
    save_report(args.output, report)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
