"""변경 파일과 관련된 단위 테스트만 선택하여 로컬과 CI에서 실행한다."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
API_TESTS = ('test_api_contract', 'test_pipeline', 'test_jobs',
             'test_media_limits', 'test_subtitle_export', 'test_job_events', 'test_job_cancellation',
             'test_capabilities')
TRANSLATION_TESTS = ('test_translation_validation', 'test_pipeline',
                     'test_prompt_matrix', 'test_qwen_lecture')
CPU_TESTS = ('test_http_translation', 'test_model_download', 'test_qwen_translation', 'test_cpu_configuration')
OWNERS = {
    'app/main.py': API_TESTS + ('test_http_translation', 'test_qwen_translation'),
    'app/schemas.py': API_TESTS,
    'app/middleware.py': ('test_timing', 'test_api_contract'),
    'timing.py': ('test_timing',),
    'transcribe.py': ('test_pipeline', 'test_media_limits', 'test_stt_hints', 'test_stt_features'),
    'app/services/transcription.py': ('test_pipeline', 'test_media_limits'),
    'app/services/media.py': ('test_media_limits', 'test_jobs', 'test_pipeline', 'test_capabilities'),
    'app/services/jobs.py': ('test_jobs', 'test_subtitle_export', 'test_job_events', 'test_job_cancellation'),
    'app/services/job_events.py': ('test_job_events', 'test_job_cancellation'),
    'app/services/pipeline.py': ('test_pipeline', 'test_jobs', 'test_job_cancellation', 'test_math_grouping'),
    'app/services/subtitles.py': ('test_subtitles', 'test_pipeline', 'test_jobs', 'test_math_grouping'),
    'app/services/subtitle_export.py': ('test_subtitle_export',),
    'app/services/translation.py': TRANSLATION_TESTS + ('test_http_translation', 'test_qwen_translation'),
    'app/services/translation_llamacpp.py': ('test_llamacpp_translation', 'test_qwen_translation'),
    'app/services/translation_qwen.py': ('test_qwen_translation', 'test_http_translation'),
    'app/services/translation_http.py': ('test_http_translation', 'test_api_contract'),
    'data/qwen_translation_terms.json': ('test_qwen_translation',),
    'data/qwen3_mlx.json': ('test_qwen_translation',),
    'Makefile': ('test_qwen_translation',),
    'app/services/translation_validation.py': TRANSLATION_TESTS + ('test_candidate_evaluation',),
    'data/physics_glossary.json': TRANSLATION_TESTS,
    'scripts/evaluate_translation.py': ('test_translation_validation',),
    'scripts/evaluate_candidates.py': ('test_candidate_evaluation',),
    'scripts/evaluate_prompt_matrix.py': ('test_prompt_matrix', 'test_qwen_lecture'),
    'scripts/evaluate_qwen_lecture.py': ('test_qwen_lecture',),
    'scripts/evaluate_lecture_quality.py': ('test_lecture_evaluation',),
    'scripts/prepare_koreanlm_lora.py': ('test_lora_conversion',),
    'scripts/prepare_koreanlm.py': ('test_lora_conversion',),
    'scripts/run_changed_tests.py': ('test_changed_tests',),
    'scripts/benchmark_cpu.py': ('test_cpu_benchmark', 'test_macos_docker_memory'),
    'scripts/macos_docker_memory.py': ('test_macos_docker_memory',),
    'app/services/stt_features.py': ('test_stt_features', 'test_stt_hints', 'test_pipeline'),
    'scripts/prepare_koreanlm_gguf.py': ('test_gguf_conversion',),
    'scripts/prepare_qwen.py': ('test_qwen_translation',),
    'scripts/evaluate_api_translation.py': ('test_qwen_translation', 'test_math_grouping'),
    'scripts/evaluate_stt_hints.py': ('test_stt_hints',),
    'scripts/download_cpu_model.py': ('test_model_download',),
    'docker/koreanlm-model.json': ('test_model_download',),
    'docker/qwen3-model.json': ('test_qwen_translation', 'test_model_download'),
    'Dockerfile.cpu': CPU_TESTS,
    '.dockerignore': CPU_TESTS,
    'compose.cpu.yaml': CPU_TESTS,
    'compose.yaml': CPU_TESTS,
    'Dockerfile.translator': CPU_TESTS,
    'docker/cpu.defaults': CPU_TESTS,
    'requirements-cpu.txt': CPU_TESTS,
    '.github/workflows/backend-ci.yml': ('test_changed_tests',),
    'requirements-ci.txt': ('test_changed_tests', 'test_pipeline',
                            'test_candidate_evaluation', 'test_cpu_configuration', 'test_cpu_benchmark'),
    'requirements-api.txt': API_TESTS,
    'requirements.txt': API_TESTS,
    'tests/test_cpu_runtime.py': ('test_cpu_benchmark', 'test_http_translation'),
    'tests/test_llamacpp_translation.py': ('test_http_translation',),
    'requirements-stt.txt': ('test_pipeline', 'test_media_limits', 'test_jobs'),
    'requirements-mlx.txt': TRANSLATION_TESTS + ('test_lora_conversion',),
}


def changed_files(base, head, root=ROOT, merge_base=False):
    """Git 비교 범위에서 추가·수정·삭제된 경로를 읽는다.

    :param base: 비교 기준 커밋의 SHA 또는 최초 push의 영(0) SHA.
    :param head: 비교 대상 커밋 SHA.
    :param root: Git 저장소 경로.
    :param merge_base: PR처럼 공통 조상 이후의 대상 브랜치 변경만 비교할지 여부.
    :return: 저장소 기준 변경 경로 목록. 이름 변경은 삭제와 추가로 반환한다.
    :raises ValueError: SHA가 40자리 또는 64자리 16진수가 아닌 경우.
    :raises subprocess.CalledProcessError: 커밋 조회나 Git 비교가 실패한 경우.
    """
    for revision in (base, head):
        if not re.fullmatch(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})', revision):
            raise ValueError('비교 기준과 대상에는 전체 커밋 SHA를 지정하세요.')
    if set(base) == {'0'}:
        command = ['git', 'ls-tree', '-r', '--name-only', '-z', head]
    else:
        command = ['git', 'diff', '--no-renames', '--name-only', '-z']
        if merge_base:
            command.append('--merge-base')
        command.extend([base, head, '--'])
    output = subprocess.run(command, cwd=root, check=True, capture_output=True).stdout
    return [path.decode('utf-8') for path in output.split(b'\0') if path]


def select_tests(paths, root=ROOT):
    """변경 경로를 테스트 소유권 표에 연결하고 중복과 삭제된 테스트를 제외한다.

    :param paths: 저장소 기준 변경 경로 목록.
    :param root: 테스트 파일 존재 여부를 확인할 저장소 경로.
    :return: 정렬된 tests/test_이름.py 경로 목록. 문서만 바뀌면 빈 목록.
    :raises ValueError: 실행 코드나 의존성 파일에 테스트 연결이 없는 경우.
    """
    selected = set()
    unknown = []
    for path in paths:
        path = Path(path).as_posix().removeprefix('./')
        if path in OWNERS:
            selected.update(OWNERS[path])
        elif path.startswith('tests/test_') and path.endswith('.py') and '/' not in path[6:]:
            selected.add(Path(path).stem)
        elif path.startswith('evaluation/') and path.endswith('.json'):
            selected.update(('test_candidate_evaluation', 'test_lecture_evaluation',
                             'test_prompt_matrix', 'test_qwen_lecture'))
        elif path.endswith('.py') or Path(path).name.startswith('requirements'):
            unknown.append(path)
    if unknown:
        raise ValueError('테스트 연결을 OWNERS에 추가하세요: ' + ', '.join(sorted(unknown)))
    return sorted(f'tests/{name}.py' for name in selected
                  if (root / 'tests' / f'{name}.py').is_file())


def run_tests(paths):
    """지정한 테스트 파일만 수집하고 실행하며 전체 탐색은 수행하지 않는다.

    :param paths: 실행할 tests/test_이름.py 경로 목록.
    :return: 성공 또는 실행 대상 없음은 0, 테스트 실패는 1.
    """
    sys.path.insert(0, str(ROOT))
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for path in paths:
        suite.addTests(loader.discover(str(ROOT / 'tests'), pattern=Path(path).name))
    if not paths:
        print('변경 범위에 해당하는 테스트가 없습니다.')
        return 0
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main():
    """명시한 파일 또는 Git SHA 범위로 테스트를 선택하고 조회하거나 실행한다.

    :return: 테스트 또는 입력 검증의 종료 코드.
    :raises SystemExit: 잘못된 CLI 인자가 전달된 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--files', nargs='+', help='구현한 파일의 저장소 기준 경로')
    source.add_argument('--base', help='비교 기준의 전체 커밋 SHA')
    parser.add_argument('--head', help='비교 대상의 전체 커밋 SHA (--base와 함께 사용)')
    parser.add_argument('--merge-base', action='store_true', help='공통 조상 기준으로 비교 (PR용)')
    parser.add_argument('--list', action='store_true', help='실행 없이 선택된 경로만 출력')
    args = parser.parse_args()
    if bool(args.base) != bool(args.head):
        parser.error('--base와 --head를 함께 지정하세요.')
    if args.merge_base and not args.base:
        parser.error('--merge-base는 --base와 함께 지정하세요.')
    try:
        paths = args.files if args.files is not None else changed_files(
            args.base, args.head, merge_base=args.merge_base)
        tests = select_tests(paths)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.list:
        print('\n'.join(tests), end='\n' if tests else '')
        return 0
    return run_tests(tests)


if __name__ == '__main__':
    raise SystemExit(main())
