"""저장된 STT와 MIT 공식 자막을 대조하고 동일 모델의 번역·시간·실패를 기록한다."""
import argparse
from contextlib import contextmanager
from difflib import SequenceMatcher
import json
import logging
from operator import itemgetter
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AttemptCollector(logging.Handler):
    """원문 없이 번역 시도의 시간·토큰 수·검증 사유만 수집하는 평가용 핸들러다."""
    def __init__(self):
        """시도 로그를 저장할 빈 목록을 초기화한다.

        :return: None.
        """
        super().__init__()
        self.attempts = []

    def emit(self, record):
        """구조화된 시간 로그 중 번역 시도 항목만 복사한다.

        :param record: 번역 시도 메타데이터를 포함할 수 있는 로그 레코드.
        :return: None.
        """
        timing = getattr(record, 'timing', {})
        if timing.get('event') == 'translation_attempt':
            self.attempts.append(dict(timing))


@contextmanager
def collect_attempts():
    """한 번역 호출의 시도 로그를 수집한 뒤 로거 설정과 핸들러를 복구한다.

    :return: 컨텍스트 관리자. 진입 시 본문 실행 중 번역 시도 로그가 추가되는 목록.
    :yield: 본문 실행 중 번역 시도 로그가 추가되는 목록.
    """
    from timing import logger

    collector, previous_level = AttemptCollector(), logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.INFO)
    try:
        yield collector.attempts
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)


def normalize_words(text: str) -> list[str]:
    """영어 대소문자·문장부호·공백 차이를 제거하되 단어와 숫자 표기는 보존한다.

    :param text: 검사하거나 번역할 입력 텍스트.
    :return: 등장 순서대로 정규화한 영어 단어·숫자 문자열 목록.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace("’", "'"))


def word_error_counts(reference: list[str], hypothesis: list[str]) -> dict:
    """단어 편집 거리에 따른 대체·삭제·삽입 수와 공식 자막 대비 WER을 계산한다.

    :param reference: 공식 자막의 기준 단어 목록.
    :param hypothesis: 비교할 STT 단어 목록.
    :return: 기준·대상 단어 수, 대체·삭제·삽입 수와 WER 사전. 기준이 비면 WER은 None.
    """
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_word in enumerate(reference, 1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_word in enumerate(hypothesis, 1):
            if ref_word == hyp_word:
                current.append(previous[hyp_index - 1])
                continue
            distance, substitutions, deletions, insertions = previous[hyp_index - 1]
            substitute = (distance + 1, substitutions + 1, deletions, insertions)
            distance, substitutions, deletions, insertions = previous[hyp_index]
            delete = (distance + 1, substitutions, deletions + 1, insertions)
            distance, substitutions, deletions, insertions = current[hyp_index - 1]
            insert = (distance + 1, substitutions, deletions, insertions + 1)
            current.append(min((substitute, delete, insert), key=itemgetter(0)))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {
        'reference_words': len(reference), 'hypothesis_words': len(hypothesis),
        'substitutions': substitutions, 'deletions': deletions, 'insertions': insertions,
        'word_errors': distance,
        'word_error_rate': distance / len(reference) if reference else None,
    }


def compare_stt(reference: dict, baseline: dict) -> dict:
    """저장된 전체 STT를 공식 자막과 대조하고 대표 차이와 기준 자료의 한계를 반환한다.

    :param reference: 공식 자막의 구간과 출처 정보를 담은 사전.
    :param baseline: 기존 STT·번역 결과와 요약을 담은 평가 사전.
    :return: 단어 오류 집계·시퀀스 유사도·표기 차이·공식 자막의 불확실성 사전.
    """
    reference_words = normalize_words(' '.join(cue['text'] for cue in reference['segments']))
    stt_words = normalize_words(' '.join(row['text'] for row in baseline['results']))
    matcher = SequenceMatcher(a=reference_words, b=stt_words, autojunk=False)
    differences = [
        {'operation': operation, 'reference_word_range': [ref_start, ref_end],
         'stt_word_range': [stt_start, stt_end],
         'reference': ' '.join(reference_words[ref_start:ref_end]),
         'stt': ' '.join(stt_words[stt_start:stt_end])}
        for operation, ref_start, ref_end, stt_start, stt_end in matcher.get_opcodes()
        if operation != 'equal'
    ]
    return {
        **word_error_counts(reference_words, stt_words),
        'sequence_match_ratio': matcher.ratio(), 'differences': differences,
        'official_caption_uncertainties': reference['source'].get('known_uncertainties', []),
        'interpretation': '공식 자막 대비 표기 차이의 대리지표이며 음성 인식 정확도가 아니다. '
                          '공식 자막의 오기와 수식 띄어쓰기 차이도 오류 수에 포함된다.',
    }


def baseline_for_indices(baseline_rows: list[dict], indices: list[int]) -> dict:
    """새 STT 묶음과 동일한 원래 구간의 기존 번역·실패·시간을 함께 보존한다.

    :param baseline_rows: 원래 STT 구간별 번역·실패·시간 기록 목록.
    :param indices: 선택할 원래 구간의 0부터 시작하는 인덱스 목록.
    :return: 선택한 인덱스·기존 번역 목록·오류 수·번역 시간 합계 사전.
    """
    rows = [baseline_rows[index] for index in indices]
    return {
        'source_indices': indices,
        'translations': [row.get('translation') for row in rows],
        'error_count': sum(bool(row.get('error')) for row in rows),
        'translation_seconds': sum(row.get('total_seconds', 0) for row in rows),
        'interpretation': '이전 조각별 결과를 묶음과 나란히 비교하기 위한 자료이며 정확도 판정은 아니다.',
    }


def case_groups(cases: list[dict], stt_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """수동 기준 문장을 공식 입력과 같은 시간대의 STT 입력으로 각각 구성한다.

    :param cases: 영어 원문과 선택적 문맥·검증 기준을 담은 평가 사례 목록.
    :param stt_rows: 시작·종료 시각과 영어 원문을 담은 기존 STT 결과 목록.
    :return: 공식 기준 입력 묶음 목록과 시간이 겹치는 STT 입력 묶음 목록의 튜플.
    """
    official, stt = [], []
    for case in cases:
        official.append({**case, 'source_indices': case['source_cue_indices']})
        indices = [index for index, row in enumerate(stt_rows)
                   if row['start'] < case['end'] and row['end'] > case['start']]
        rows = [stt_rows[index] for index in indices]
        stt.append({
            **case,
            'start': rows[0]['start'] if rows else case['start'],
            'end': rows[-1]['end'] if rows else case['end'],
            'text': ' '.join(row['text'] for row in rows),
            'source_indices': indices, 'baseline': baseline_for_indices(stt_rows, indices),
            'alignment_note': '시간이 겹치는 전체 STT 구간을 선택했다. 경계의 앞뒤 문장이 포함될 수 있다.',
        })
    return official, stt


def full_groups(reference: dict, baseline: dict) -> tuple[list[dict], list[dict]]:
    """공식 자막과 저장된 STT에 동일한 문장 묶기 규칙을 적용하고 기존 결과를 연결한다.

    :param reference: 공식 자막의 구간과 출처 정보를 담은 사전.
    :param baseline: 기존 STT·번역 결과와 요약을 담은 평가 사전.
    :return: 공식 자막 묶음 목록과 기존 결과를 연결한 STT 묶음 목록의 튜플.
    """
    from app.services.subtitles import group_segments

    official = group_segments(reference['segments'], max_duration=18, max_chars=350)
    stt = group_segments(baseline['results'], max_duration=18, max_chars=350)
    for index, group in enumerate(official):
        group['id'] = f'official-{index}'
    for index, group in enumerate(stt):
        group['id'] = f'stt-{index}'
        group['baseline'] = baseline_for_indices(baseline['results'], group['source_indices'])
    return official, stt


def evaluate_groups(service, groups: list[dict], context_mode: str = 'previous_source') -> list[dict]:
    """원문 문맥으로 구간별 번역을 수행하며 실패도 보존하고 기준 번역은 수동 비교용으로 남긴다.

    :param service: translate 메서드를 제공하는 번역 서비스.
    :param groups: 시각·원문·식별자·선택적 기준 번역이 있는 평가 묶음 목록.
    :param context_mode: previous_source면 앞 영어 원문을 사용하고 그 외에는 문맥을 비운다.
    :return: 번역·문맥·오류·시도 로그·처리 시간을 담은 구간별 결과 목록.
    """
    results, previous_source = [], ''
    for group in groups:
        context = previous_source if context_mode == 'previous_source' else ''
        started = perf_counter()
        with collect_attempts() as attempts:
            try:
                generated = service.translate(group['text'], context, 128)
                translation, error = generated['translation'], None
            except ValueError as exc:
                generated, translation, error = {}, None, str(exc)
        row = {**group, **generated, 'translation': translation, 'context': context,
               'error': error, 'attempts': attempts, 'total_seconds': perf_counter() - started,
               'semantic_review': 'pending_manual_review'}
        results.append(row)
        previous_source = group['text'][-256:]
        print(json.dumps({'id': group['id'], 'translation': translation, 'error': error,
                          'total_seconds': row['total_seconds']}, ensure_ascii=False), flush=True)
    return results


def summarize(rows: list[dict]) -> dict:
    """모델 검증 통과·실패와 처리 시간을 집계하되 의미 정확도와 구분한다.

    :param rows: 번역 결과·오류·시간·시도 정보를 담은 평가 행 목록.
    :return: 묶음·수락·실패·시도 수, 시간 합계와 거부 사유 집계 사전. 의미 정확도는 None.
    """
    return {
        'group_count': len(rows), 'validated_outputs': sum(row['error'] is None for row in rows),
        'failed_outputs': sum(row['error'] is not None for row in rows),
        'translation_seconds': sum(row['total_seconds'] for row in rows),
        'attempt_count': sum(len(row.get('attempts', [])) for row in rows),
        'rejection_reasons': {
            reason: sum(attempt.get('rejection_reason') == reason for row in rows
                        for attempt in row.get('attempts', []))
            for reason in sorted({attempt['rejection_reason'] for row in rows
                                  for attempt in row.get('attempts', [])
                                  if attempt.get('rejection_reason')})
        },
        'semantic_accuracy': None,
        'interpretation': '검증 통과는 의미 정확성의 판정이 아니다. 기준 번역·수식·부정·비교 관계를 수동 확인해야 한다.',
    }


def main():
    """공식·STT 대조와 번역 평가를 실행하고 JSON에 출처·기준·결과·시간을 저장한다.

    :return: None.

    CLI 출력 경로에 평가 JSON을 저장한다. 저장 STT를 재사용하며 stt-only가 아니면 번역 모델을 실행한다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases-only', action='store_true', help='수동 기준 문장 10개와 같은 시간대 STT만 평가')
    parser.add_argument('--context-mode', choices=('previous_source', 'none'), default='previous_source')
    parser.add_argument('--stt-only', action='store_true', help='번역 모델을 로딩하지 않고 STT 대조만 저장')
    parser.add_argument('--source', choices=('both', 'official', 'stt'), default='both',
                        help='번역 평가할 영어 입력. STT 대조는 항상 기록')
    args = parser.parse_args()
    reference = json.loads((ROOT / 'evaluation/mit_rocket_reference.json').read_text(encoding='utf-8'))
    cases = json.loads((ROOT / 'evaluation/mit_rocket_cases.json').read_text(encoding='utf-8'))['cases']
    baseline = json.loads((ROOT / 'evaluation/lecture.json').read_text(encoding='utf-8'))
    output = {
        'source': reference['source'], 'stt_source': 'evaluation/lecture.json',
        'stt_reexecuted': False, 'stt_comparison': compare_stt(reference, baseline),
        'manual_reference_cases': cases, 'baseline_summary': baseline['summary'],
        'settings': {'cases_only': args.cases_only, 'context_mode': args.context_mode,
                     'input_source': args.source,
                     'max_tokens': 128, 'model': 'models/koreanlm-4bit'},
    }
    if not args.stt_only:
        from app.services.translation import TranslationService

        started = perf_counter()
        service = TranslationService(ROOT / 'models/koreanlm-4bit', ROOT / 'templates/korean.json')
        output['model_load_seconds'] = perf_counter() - started
        official, stt = case_groups(cases, baseline['results']) if args.cases_only else full_groups(reference, baseline)
        if args.source in ('both', 'official'):
            output['official_results'] = evaluate_groups(service, official, args.context_mode)
        if args.source in ('both', 'stt'):
            output['stt_results'] = evaluate_groups(service, stt, args.context_mode)
        output['summary'] = {category: summarize(output[category + '_results'])
                             for category in ('official', 'stt')
                             if category + '_results' in output}
        output['comparison_limits'] = [
            '공식과 STT는 구간 경계가 다를 수 있으므로 그룹별 시간으로 직접 성능 우열을 판정하지 않는다.',
            'both에서는 공식을 먼저 실행해 이후 STT에 워밍업 효과가 있을 수 있다. stt 단독 실행과 조건이 다르다.',
            '이전 저장 결과와 실행 조건·문맥·출력 검증이 달라 정확도 개선율로 해석하지 않는다.',
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    comparison_summary = {key: value for key, value in output['stt_comparison'].items()
                          if key not in ('differences', 'official_caption_uncertainties')}
    print(json.dumps(output.get('summary', comparison_summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
