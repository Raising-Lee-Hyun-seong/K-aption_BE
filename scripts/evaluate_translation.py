"""고정 물리학 문장 또는 실제 미디어로 번역 품질 대리지표와 처리 시간을 기록한다."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.services.translation import TranslationService
from app.services.subtitles import group_segments
from transcribe import TranscriptionService


def evaluate(service, cases, sequential=False, context_mode='previous_source'):
    """문장별 번역과 용어 포함·금지어·문맥 복사 여부를 기록하며 결과를 순차 반환한다.

    :param service: translate 메서드를 제공하는 번역 서비스.
    :param cases: 영어 원문과 선택적 문맥·검증 기준을 담은 평가 사례 목록.
    :param sequential: 참이면 앞 사례를 다음 사례의 문맥으로 사용한다.
    :param context_mode: previous_source면 앞 영어 원문, 그 외에는 앞 번역문을 문맥으로 사용한다.
    :return: 아래 값을 순차 제공하는 생성기.
    :yield: 각 사례의 원문·번역·문맥·오류·검증 결과·처리 시간을 담은 사전.
    """
    previous = ''
    for case in cases:
        context = previous if sequential else case.get('context', '')
        start = perf_counter()
        try:
            result = service.translate(case['text'], context, 128)
            output = result['translation']
            error = None
        except ValueError as exc:
            result, output, error = {}, '', str(exc)
        term_pass = bool(output) and all(term in output for term in case.get('required_terms', []))
        forbidden = [term for term in case.get('forbidden_terms', []) if term in output]
        copied = bool(context and output.strip() == context.strip())
        row = {**case, **result, 'translation': output, 'context': context, 'error': error,
               'term_pass': term_pass, 'forbidden_matches': forbidden,
               'context_copied': copied, 'total_seconds': perf_counter() - start}
        previous = (case['text'][-256:] if context_mode == 'previous_source' else output)
        print(json.dumps({'id': case['id'], 'translation': output, 'error': error}, ensure_ascii=False), flush=True)
        yield row


def main():
    """평가 문장 또는 미디어를 읽어 번역 결과와 요약 통계를 JSON 파일에 저장한다.

    :return: None.

    CLI 출력 경로에 평가 JSON을 저장하고 진행 상황·요약을 표준 출력에 기록한다. 설정에 따라 STT·번역 모델을 실행한다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--service-file', type=Path, help='비교할 서비스 구현 스냅샷')
    parser.add_argument('--media', type=Path)
    parser.add_argument('--raw-segments', action='store_true', help='과거 조각별 실행 재현용: 문장 묶기 생략')
    parser.add_argument('--context-mode', choices=('previous_source', 'previous_translation'),
                        default='previous_source', help='미디어 평가 문맥: 기본은 앞 영어 원문')
    args = parser.parse_args()
    service_type = TranslationService
    if args.service_file:
        spec = importlib.util.spec_from_file_location('comparison_service', args.service_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        service_type = module.TranslationService
    started = perf_counter()
    service = service_type(ROOT / 'models/koreanlm-4bit', ROOT / 'templates/korean.json')
    load_seconds = perf_counter() - started
    stt_seconds = None
    if args.media:
        started = perf_counter()
        transcription = TranscriptionService().transcribe(args.media)
        stt_seconds = perf_counter() - started
        segments = transcription['segments']
        if not args.raw_segments:
            segments = group_segments(segments)
        cases = [{'id': str(i), **segment} for i, segment in enumerate(segments)]
    else:
        cases = json.loads((ROOT / 'evaluation/physics_cases.json').read_text())
    rows = list(evaluate(service, cases, sequential=bool(args.media), context_mode=args.context_mode))
    summary = {'case_count': len(rows), 'model_load_seconds': load_seconds,
               'stt_seconds': stt_seconds,
               'translation_seconds': sum(row['total_seconds'] for row in rows),
               'valid_outputs': sum(not row['error'] for row in rows),
               'term_passes': sum(row['term_pass'] for row in rows) if not args.media else None,
               'forbidden_term_cases': sum(bool(row['forbidden_matches']) for row in rows),
               'context_copies': sum(row['context_copied'] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary,
                                      'settings': {'context_mode': args.context_mode,
                                                   'raw_segments': args.raw_segments},
                                      'results': rows},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
