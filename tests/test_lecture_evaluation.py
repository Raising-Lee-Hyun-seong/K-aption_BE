"""공식 자막 평가의 단어 대조·시간 정렬·기존 결과 보존·영어 문맥 전달을 검증한다."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from scripts.evaluate_lecture_quality import (
    baseline_for_indices, case_groups, compare_stt, evaluate_groups, normalize_words,
    summarize, word_error_counts,
)

ROOT = Path(__file__).resolve().parents[1]


class LectureEvaluationTests(unittest.TestCase):
    """모델 실행 없이 평가 지표와 기준 자료가 의도한 의미로 사용되는지 검증한다."""

    def test_word_error_counts_preserve_distinct_edit_types(self):
        """대체·삽입·삭제를 정확히 세고 빈 공식 원문은 비율을 판정하지 않는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        counts = word_error_counts(['a', 'b', 'c'], ['a', 'x', 'c', 'y'])
        self.assertEqual(counts['substitutions'], 1)
        self.assertEqual(counts['insertions'], 1)
        self.assertEqual(counts['deletions'], 0)
        self.assertAlmostEqual(counts['word_error_rate'], 2 / 3)
        self.assertEqual(word_error_counts(['a', 'b', 'c'], ['a', 'c'])['deletions'], 1)
        self.assertEqual(word_error_counts([], ['a'])['insertions'], 1)
        self.assertIsNone(word_error_counts([], ['a'])['word_error_rate'])

    def test_normalization_does_not_repair_formula_or_caption(self):
        """문장부호·대소문자만 정리하고 변수의 띄어쓰기와 공식 자막 오기를 보정하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(normalize_words("We’re AT 2m0, m r!"), ["we're", 'at', '2m0', 'm', 'r'])
        self.assertNotEqual(normalize_words('m r'), normalize_words('mR'))
        compared = compare_stt(
            {'segments': [{'text': 'We arrived.'}],
             'source': {'known_uncertainties': [{'text': 'arrived'}]}},
            {'results': [{'text': 'We derived.'}]},
        )
        self.assertEqual(compared['substitutions'], 1)
        self.assertEqual(compared['official_caption_uncertainties'], [{'text': 'arrived'}])
        self.assertIn('정확도가 아니다', compared['interpretation'])

    def test_case_alignment_uses_overlap_and_preserves_baseline_failures(self):
        """시간이 실제로 겹치는 구간만 선택하고 기존 번역 실패·시간을 보존하는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        rows = [
            {'start': 0, 'end': 1, 'text': 'Before.', 'translation': '이전.', 'error': None},
            {'start': 1, 'end': 2, 'text': 'Target.', 'translation': '', 'error': 'failed', 'total_seconds': 3},
            {'start': 2, 'end': 3, 'text': 'After.', 'translation': '이후.', 'error': None},
        ]
        case = {'id': 'test', 'start': 1, 'end': 2, 'source_cue_indices': [4], 'text': 'Official.'}
        official, stt = case_groups([case], rows)
        self.assertEqual(official[0]['text'], 'Official.')
        self.assertEqual(stt[0]['text'], 'Target.')
        self.assertEqual(stt[0]['source_indices'], [1])
        self.assertEqual(stt[0]['baseline']['error_count'], 1)
        self.assertEqual(stt[0]['baseline']['translation_seconds'], 3)
        self.assertEqual(baseline_for_indices(rows, [])['translations'], [])

    def test_failed_translation_still_passes_english_context(self):
        """번역 실패 후에도 앞 영어 원문의 마지막 256자를 전달하고 번역문을 문맥으로 쓰지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        source = 'English ' * 40
        service = Mock()
        service.translate.side_effect = [ValueError('invalid output'), {'translation': '다음 번역.'}]
        with patch('builtins.print'):
            rows = evaluate_groups(service, [{'id': 'first', 'text': source},
                                             {'id': 'next', 'text': 'Next.'}])
        self.assertEqual(service.translate.call_args_list[0].args, (source, '', 128))
        self.assertEqual(service.translate.call_args_list[1].args, ('Next.', source[-256:], 128))
        self.assertIsNone(rows[0]['translation'])
        self.assertEqual(rows[0]['error'], 'invalid output')
        self.assertEqual(rows[1]['semantic_review'], 'pending_manual_review')
        summary = summarize(rows)
        self.assertEqual(summary['failed_outputs'], 1)
        self.assertIsNone(summary['semantic_accuracy'])

    def test_reference_cases_use_traceable_unmodified_official_cues(self):
        """기준 문장 입력과 시각이 출처 cue를 그대로 묶고 라이선스·오기 정보를 보존하는지 확인한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        reference = json.loads((ROOT / 'evaluation/mit_rocket_reference.json').read_text(encoding='utf-8'))
        cases = json.loads((ROOT / 'evaluation/mit_rocket_cases.json').read_text(encoding='utf-8'))
        self.assertEqual(len(reference['segments']), 42)
        self.assertEqual(len(cases['cases']), 10)
        self.assertTrue(reference['source']['caption_url'].startswith('https://ocw.mit.edu/'))
        self.assertIn('by-nc-sa/4.0', reference['source']['license']['url'])
        self.assertTrue(reference['source']['known_uncertainties'])
        for case in cases['cases']:
            with self.subTest(case=case['id']):
                cues = [reference['segments'][index] for index in case['source_cue_indices']]
                self.assertEqual(case['text'], ' '.join(cue['text'] for cue in cues))
                self.assertEqual(case['start'], cues[0]['start'])
                self.assertEqual(case['end'], cues[-1]['end'])
                self.assertTrue(case['reference_ko'])
                self.assertTrue(case['semantic_checks'])


if __name__ == '__main__':
    unittest.main()
