"""운영 설정을 바꾸지 않고 CPU STT 탐색 폭과 강의 용어 힌트를 교차 비교한다."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_lecture_quality import compare_stt
from transcribe import TranscriptionService

HINT = 'A university physics lecture about a rocket sled, velocity, mass, and integration.'
WORDS = ('rocket sled, differential, separation of variables, integration limits, '
         'prime, natural logarithm, dry mass, fuel mass, mR, vR, m0, v0, dt, dmR, dvR')
ORDER = ['baseline', 'beam1', 'beam1_hints', 'beam5_hints', 'beam1_hints', 'baseline']


def save_new(path: Path, report: dict) -> None:
    """이전 결과를 덮어쓰지 않고 새 평가 기록을 배타적으로 저장한다.

    :param path: 새 JSON 파일 경로. 부모 디렉터리는 존재해야 한다.
    :param report: 저장할 JSON 직렬화 가능한 평가 사전.
    :return: None. 지정 경로에 UTF-8 JSON 파일을 생성한다.
    :raises FileExistsError: 같은 경로가 이미 존재하는 경우.
    """
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main() -> None:
    """오디오를 한 번 디코딩하고 고정한 조건 순서대로 실제 CPU STT를 측정한다.

    :return: None. 모델을 로드하고 조건별 원시 전사·시간·자막 차이를 새 JSON에 저장한다.
    :raises FileExistsError: 최종 파일이나 조건별 기록이 이미 존재하는 경우.
    :raises OSError: 미디어·기준 자막·출력 파일에 접근할 수 없는 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from faster_whisper.audio import decode_audio

    service = TranscriptionService()
    start = perf_counter()
    model = service._get_model()
    loading = perf_counter() - start
    start = perf_counter()
    audio = decode_audio(str(args.media))
    decoding = perf_counter() - start
    reference = json.loads(args.reference.read_text())
    report = {
        'model': 'small', 'device': 'cpu', 'compute_type': 'int8',
        'platform': platform.platform(), 'architecture': platform.machine(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('faster-whisper', 'ctranslate2', 'numpy')},
        'media_sha256': hashlib.sha256(args.media.read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_load_seconds': loading, 'decode_seconds': decoding,
        'audio_seconds': len(audio) / 16000, 'order': ORDER, 'runs': [],
        'limits': [
            'STT only; excludes upload, grouping, translation and subtitle generation.',
            'Baseline and combined option repeated twice; ablations once; not statistical proof.',
            'One reused model and decoded audio, fixed order; CPU frequency/load not controlled.',
            'Caption WER includes known caption errors; not a semantic accuracy score.',
            'Lecture-specific hints contain no reference sentences or equations.',
            'No production settings changed; no Windows amd64 measurement.',
        ],
    }
    for index, name in enumerate(ORDER):
        beam = 1 if name.startswith('beam1') else 5
        hints = name.endswith('_hints')
        start = perf_counter()
        segments, info = model.transcribe(
            audio, language='en', task='transcribe', beam_size=beam, vad_filter=True,
            initial_prompt=HINT if hints else None, hotwords=WORDS if hints else None,
        )
        rows = [{'start': max(0.0, s.start), 'end': min(len(audio) / 16000, s.end),
                 'text': s.text.strip()} for s in segments if s.text.strip()]
        elapsed = perf_counter() - start
        comparison = compare_stt(reference, {'results': rows})
        row = {'condition': name, 'beam_size': beam, 'initial_prompt': HINT if hints else None,
               'hotwords': WORDS if hints else None, 'seconds': elapsed, 'segments': rows,
               'language_probability': info.language_probability, 'caption_comparison': comparison}
        report['runs'].append(row)
        save_new(args.output.with_name(args.output.stem + f'-run{index + 1}.json'), row)
        print(json.dumps({'condition': name, 'seconds': elapsed, 'segments': len(rows),
                          'caption_wer': comparison['word_error_rate']}), flush=True)
    save_new(args.output, report)


if __name__ == '__main__':
    main()
