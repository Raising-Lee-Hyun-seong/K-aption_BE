"""STT 조각의 문장 묶기, 시간·길이 제한 및 원본 인덱스 보존을 검증한다."""
import unittest

from app.services.subtitles import group_segments


class SubtitleGroupingTests(unittest.TestCase):
    """문장 경계에서 조각을 합치되 원문 순서와 시각을 잃지 않는지 확인한다."""

    def test_fragments_merge_until_sentence_end(self):
        """중간 STT 조각을 이어 붙이고 문장 끝에서 닫아 첫·마지막 시각을 보존한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [
            {"start": 3.6, "end": 7.28, "text": "The relation between the speed of"},
            {"start": 7.28, "end": 10.08, "text": " the rocket and the mass of "},
            {"start": 10.08, "end": 13.24, "text": "the rocket."},
            {"start": 13.24, "end": 17.76, "text": "Now separate the variables."},
        ]
        result = group_segments(segments)
        self.assertEqual(result, [
            {"start": 3.6, "end": 13.24,
             "text": "The relation between the speed of the rocket and the mass of the rocket.",
             "source_indices": [0, 1, 2]},
            {"start": 13.24, "end": 17.76, "text": "Now separate the variables.",
             "source_indices": [3]},
        ])
        self.assertEqual(segments[1]["text"], " the rocket and the mass of ")

    def test_empty_text_is_removed_without_changing_source_indices(self):
        """빈 원문을 제외하면서 나머지 구간의 인덱스와 순서를 유지한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [
            {"start": 0, "end": 1, "text": " "},
            {"start": 1, "end": 2, "text": "The mass is"},
            {"start": 2, "end": 3, "text": "\n\t"},
            {"start": 3, "end": 4, "text": "constant."},
        ]
        self.assertEqual(group_segments(segments), [
            {"start": 1, "end": 4, "text": "The mass is constant.", "source_indices": [1, 3]},
        ])
        self.assertEqual(group_segments([]), [])
        self.assertEqual(group_segments(segments[:1]), [])

    def test_duration_limit_keeps_fragments_in_separate_groups(self):
        """합친 구간이 시간 상한을 넘으면 다음 조각부터 새 그룹을 만든다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [
            {"start": 0, "end": 10, "text": "First fragment"},
            {"start": 10, "end": 18, "text": "at the boundary"},
            {"start": 18, "end": 20, "text": "final fragment."},
        ]
        result = group_segments(segments)
        self.assertEqual([item["source_indices"] for item in result], [[0, 1], [2]])
        self.assertEqual([(item["start"], item["end"]) for item in result], [(0, 18), (18, 20)])

    def test_character_limit_counts_joining_space(self):
        """두 조각 사이 공백까지 포함해 문자 상한을 적용하고 텍스트를 자르지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [
            {"start": 0, "end": 1, "text": "abcde"},
            {"start": 1, "end": 2, "text": "fghij"},
        ]
        self.assertEqual(len(group_segments(segments, max_chars=11)), 1)
        result = group_segments(segments, max_chars=10)
        self.assertEqual([item["text"] for item in result], ["abcde", "fghij"])

    def test_long_silence_closes_unfinished_group(self):
        """문장부호가 없어도 긴 무음 전후의 조각을 별도 자막으로 반환한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [
            {"start": 0, "end": 1, "text": "Before the pause"},
            {"start": 3.1, "end": 4, "text": "after the pause"},
        ]
        self.assertEqual([item["source_indices"] for item in group_segments(segments)], [[0], [1]])

    def test_terminal_punctuation_and_closing_quotes(self):
        """문장 끝의 물음표·느낌표·닫는 따옴표를 인식해 뒤 문장과 합치지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [
            {"start": 0, "end": 1, "text": 'Why?"'},
            {"start": 1, "end": 2, "text": "Go!"},
            {"start": 2, "end": 3, "text": "It ends.”"},
            {"start": 3, "end": 4, "text": "Next"},
        ]
        self.assertEqual([item["source_indices"] for item in group_segments(segments)],
                         [[0], [1], [2], [3]])

    def test_decimal_does_not_close_sentence(self):
        """소수 내부의 마침표를 문장 끝으로 처리하지 않아 숫자 뒤 조각을 합친다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segments = [
            {"start": 0, "end": 1, "text": "The mass is 2.0"},
            {"start": 1, "end": 2, "text": "kilograms."},
        ]
        self.assertEqual(group_segments(segments)[0]["text"], "The mass is 2.0 kilograms.")

    def test_single_long_source_segment_is_not_truncated(self):
        """상한보다 긴 단일 STT 조각은 임의 시각 분할이나 원문 손실 없이 보존한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        segment = {"start": 0, "end": 30, "text": "x" * 400}
        self.assertEqual(group_segments([segment]), [{**segment, "source_indices": [0]}])

    def test_non_positive_limits_are_rejected(self):
        """양수가 아닌 시간·문자 상한을 입력 오류로 거부한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        for limits in ({"max_duration": 0}, {"max_duration": -1}, {"max_chars": 0}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                group_segments([], **limits)
