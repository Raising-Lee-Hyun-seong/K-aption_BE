"""같은 원본 강의에서 STT 기본·일반 수학 힌트 조건을 비교하고 원시 결과를 보존한다."""

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_candidates import sha256_file, save_report
from transcribe import TranscriptionService

MATH_PROMPT = ('A university physics lecture about rocket sleds, differential equations, '
               'separation of variables, integration limits and prime notation. '
               'Symbols include mR, vR, m0, v0, u and t.')
MATH_HOTWORDS = 'rocket sled, differential, separation of variables, integration, prime'


def main() -> None:
    """동일 CPU small 모델을 재사용해 기본 전사와 두 가지 용어 힌트를 비교한다.

    :return: None. 새 JSON 경로에 입력 해시·조건·구간별 전사를 기록한다.
    :raises FileExistsError: 이전 평가 결과를 덮어쓰려는 경우.
    :raises OSError: 원본 강의·출력 파일에 접근할 수 없는 경우.

    모델을 로드하고 실제 음성 인식을 세 번 실행한다. 기준 자막·정답 수식은
    프롬프트에 넣지 않는다. 모델 종류·VAD·beam size는 기본 API와 같다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluated-on', type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new report path: {args.output}')
    service = TranscriptionService()
    report = {'evaluated_on': args.evaluated_on.isoformat(), 'media_sha256': sha256_file(args.media),
              'model': 'small', 'device': 'cpu', 'compute_type': 'int8', 'conditions': [],
              'limits': ['Development lecture, not an independent transcription accuracy benchmark.',
                         'Hints contain vocabulary only, not reference captions or equations.']}
    for label, prompt, hotwords in (('baseline', None, None), ('math_prompt', MATH_PROMPT, None),
                                    ('math_hotwords', None, MATH_HOTWORDS)):
        service.initial_prompt, service.hotwords = prompt, hotwords
        result = service.transcribe(args.media)
        report['conditions'].append({'condition': label, 'initial_prompt': prompt,
                                     'hotwords': hotwords, **result})
        print(json.dumps({'condition': label, 'seconds': result['elapsed_seconds'],
                          'segments': len(result['segments'])}), flush=True)
    save_report(args.output, report)


if __name__ == '__main__':
    main()
