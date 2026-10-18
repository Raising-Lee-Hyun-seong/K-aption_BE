"""수식 문장의 연결·범위·정정 결합과 원문·시각·구간 추적 보존을 검증한다."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from app.services.subtitles import group_segments, merge_math_groups


class MathGroupingTests(unittest.TestCase):
    """명시적인 연결 단서가 있는 인접 수학 설명만 제한 안에서 묶는지 검사한다."""

    def segments(self, texts):
        """각 영어 문장을 연속된 3초짜리 STT 구간으로 만든다.

        :param texts: 순서대로 배치할 영어 원문 목록.
        :return: start·end·text를 가진 테스트 STT 구간 목록.
        """
        return [{'start': index * 3, 'end': (index + 1) * 3, 'text': text}
                for index, text in enumerate(texts)]

    def test_constant_position_and_division_are_one_translation_unit(self):
        """상수의 위치 설명과 뒤 나눗셈을 합치되 원문·시각·인덱스를 보존한다.

        :return: None.
        """
        texts = ["And we're left with dvR equals u is a constant.",
                 'So that goes up front.', 'And then we have dmR over mR.']
        segments = self.segments(texts)
        original = deepcopy(segments)
        self.assertEqual(len(group_segments(segments)), 3)
        groups = group_segments(segments, merge_math=True)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['text'], ' '.join(texts))
        self.assertEqual(groups[0]['source_indices'], [0, 1, 2])
        self.assertEqual((groups[0]['start'], groups[0]['end']), (0, 9))
        self.assertEqual(segments, original)

    def test_integration_limits_keep_uncertain_two_without_repair(self):
        """적분 시작·초기 조건·끝점 파편을 묶으며 STT의 2를 to로 고치지 않는다.

        :return: None.
        """
        texts = ['So we have vR going from v0.', 'That is our initial condition here.',
                 '2 vR prime of t.']
        groups = group_segments(self.segments(texts), merge_math=True)
        self.assertEqual(groups[0]['text'], ' '.join(texts))
        self.assertEqual(groups[0]['source_indices'], [0, 1, 2])

    def test_mass_correction_preserves_both_original_and_revised_values(self):
        """질량 정정 설명을 합치고 m0·2m0와 부정 원문을 모두 유지한다.

        :return: None.
        """
        texts = ['And for the mass, we have mR prime equals m0.', 'Actually, not quite m0.',
                 "It's 2 m0, because the initial mass is dry mass and fuel mass."]
        groups = group_segments(self.segments(texts), merge_math=True)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['text'], ' '.join(texts))

    def test_formula_and_integration_explanation_merge_without_solving(self):
        """명시 식과 적분 설명을 연결하되 u1 등의 불확실한 원문은 수정하지 않는다.

        :return: None.
        """
        texts = ["We're going to get vR minus v0 equals u1 over m.",
                 'Integrated gives us an ln.', 'So ln.']
        groups = group_segments(self.segments(texts), merge_math=True)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['text'], ' '.join(texts))

    def test_independent_equations_normal_sentences_and_questions_stay_separate(self):
        """독립 등식·일반 문장·물음표 경계는 결합 단서와 혼동하지 않는다.

        :return: None.
        """
        cases = [
            ['We have vR equals u is a constant.', 'And then we have mR equals m0.'],
            ['We have vR equals v0.', 'We have mR equals m0.'],
            ['We have dvR equals u.', 'The instructor goes outside.'],
            ['We derived a relation.', 'That goes up front.'],
            ['Does vR equal u?', 'So that goes up front.'],
        ]
        for texts in cases:
            with self.subTest(texts=texts):
                self.assertEqual(len(group_segments(self.segments(texts), merge_math=True)), 2)

    def test_silence_time_and_character_limits_are_still_hard_boundaries(self):
        """수식 연결 단서가 있어도 긴 무음·시간·문자 상한을 넘으면 합치지 않는다.

        :return: None.
        """
        segments = self.segments(['We have dvR equals u.', 'So that goes up front.'])
        for options in ({'max_duration': 5}, {'max_chars': 30}):
            with self.subTest(options=options):
                self.assertEqual(len(group_segments(segments, merge_math=True, **options)), 2)
        segments[1].update(start=5.1, end=8)
        self.assertEqual(len(group_segments(segments, merge_math=True)), 2)

    def test_base_group_metadata_is_not_mutated_and_empty_input_is_supported(self):
        """기존 묶음의 내부 인덱스를 수정하지 않고 빈 입력과 잘못된 상한을 처리한다.

        :return: None.
        """
        groups = group_segments(self.segments(['We have dvR equals u.', 'So that goes up front.']))
        original = deepcopy(groups)
        merged = merge_math_groups(groups)
        self.assertEqual(groups, original)
        self.assertEqual(merged[0]['source_indices'], [0, 1])
        self.assertEqual(merge_math_groups([]), [])
        with self.assertRaises(ValueError):
            merge_math_groups([], max_duration=0)

    def test_saved_lecture_keeps_all_original_text_indices_and_time_limits(self):
        """저장 강의 전체에서 원문·원본 인덱스가 빠지거나 중복되지 않는지 확인한다.

        :return: None.
        """
        root = Path(__file__).resolve().parents[1]
        source = json.loads((root / 'evaluation/lecture.json').read_text())['results']
        original = group_segments(source)
        merged = group_segments(source, merge_math=True)
        self.assertLess(len(merged), len(original))
        self.assertEqual(' '.join(item['text'] for item in original),
                         ' '.join(item['text'] for item in merged))
        self.assertEqual([index for item in original for index in item['source_indices']],
                         [index for item in merged for index in item['source_indices']])
        for group in merged:
            if group.get('merge_reasons'):
                self.assertLessEqual(group['end'] - group['start'], 18)
                self.assertLessEqual(len(group['text']), 350)
