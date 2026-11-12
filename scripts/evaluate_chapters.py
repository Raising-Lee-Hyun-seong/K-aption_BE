"""저장된 실제 영상 자막으로 로컬 Qwen 챕터 생성 결과·시도·시간을 새 보고서에 기록한다."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import TranscriptionResponse
from app.services.chapters import ChapterService
from app.services.translation_http import HttpTranslationService


def evaluate(inputs: list[Path], base_url: str, evaluated_on: date) -> dict:
    """로컬 서버의 고정 모델 이름을 확인한 뒤 저장 자막을 순차 분석한다.

    :param inputs: 기존 TranscriptionResponse 형식의 자막 JSON 파일 경로 목록.
    :param base_url: localhost·루프백 또는 Docker 내부 translator 서버의 HTTP 기본 주소.
    :param evaluated_on: 실제 평가일. 호출자가 명시하는 날짜.
    :return: 입력·구현 해시, 서버 속성, 챕터·시간·원시 출력 또는 실패 기록을 담은 보고서.
    :raises ValueError: 주소·원본 자막·모델 이름이 유효하지 않은 경우.
    :raises OSError: 입력 파일을 읽지 못한 경우.
    :raises httpx.HTTPError: 서버 준비 상태나 모델 속성을 조회하지 못한 경우.

    모델·컨테이너·운영 설정은 변경하지 않는다. 별도 프로세스의 공유 잠금은 운영 작업을
    직렬화하지 않으므로 활성 STT·번역 작업이 없는 상태에서 실행해야 한다.
    실제 오디오·판서와 대조하지 않으며 구조 검증 통과를 의미 정확도로 해석하지 않는다.
    """
    if urlsplit(base_url).hostname not in ('localhost', '127.0.0.1', '::1', 'translator'):
        raise ValueError('Only the local Qwen server is supported')
    loaded = [(path, path.read_bytes()) for path in inputs]
    transcriptions = [(path, hashlib.sha256(raw).hexdigest(),
                       TranscriptionResponse.model_validate_json(raw)) for path, raw in loaded]
    manifest = json.loads((ROOT / 'docker/qwen3-model.json').read_text(encoding='utf-8'))
    service = HttpTranslationService(base_url, ROOT / 'templates/korean.json', timeout=180)
    try:
        service.client.get('/health').raise_for_status()
        response = service.client.get('/props')
        response.raise_for_status()
        props = response.json()
        if Path(props.get('model_path', '')).name != manifest['filename']:
            raise ValueError('Server model filename does not match the fixed Q4_K_M model')
        report = {'evaluated_on': evaluated_on.isoformat(), 'base_url': base_url,
                  'model_manifest': manifest, 'server_props': props, 'results': [],
                  'limitations': [
                      'Saved subtitle replay; excludes upload, STT and video decoding.',
                      'Model filename checked; live weight SHA256 not independently verified.',
                      'Semantic accuracy and chapter boundaries require manual review.',
                      'Single observations are not repeatable performance averages.'],
                  'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ('app/services/chapters.py', 'app/services/study_inputs.py',
                                   'app/services/translation_http.py', 'scripts/evaluate_chapters.py')}}
        for path, digest, transcription in transcriptions:
            started = perf_counter()
            row = {'input': str(path), 'input_sha256': digest,
                   'source_segments': len(transcription.segments)}
            try:
                result = ChapterService(service).generate(transcription)
                row.update(status='completed', **result.model_dump())
            except ValueError as exc:
                row.update(status='failed', error=str(exc), attempts=getattr(exc, 'attempts', []),
                           elapsed_seconds=perf_counter() - started)
            report['results'].append(row)
        return report
    finally:
        service.close()


def main() -> int:
    """평가 인자를 받아 기존 결과를 덮어쓰지 않고 새 JSON 보고서를 저장한다.

    :return: 모든 입력이 성공하면 0, 생성 실패가 있으면 1.
    :raises SystemExit: 필수 인자나 날짜가 유효하지 않은 경우.
    :raises FileExistsError: 출력 파일이 이미 존재하는 경우.
    :raises OSError: 입출력 파일에 접근하지 못한 경우.
    :raises ValueError: 주소·입력·서버 모델이 평가 조건에 맞지 않는 경우.
    :raises httpx.HTTPError: 서버 준비 상태나 모델 속성을 조회하지 못한 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080')
    parser.add_argument('--evaluated-on', type=date.fromisoformat, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = evaluate(args.inputs, args.base_url, args.evaluated_on)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps([{'input': row['input'], 'status': row['status'],
                       'chapter_count': len(row.get('chapters', []))}
                      for row in report['results']], ensure_ascii=False))
    return 0 if all(row['status'] == 'completed' for row in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
