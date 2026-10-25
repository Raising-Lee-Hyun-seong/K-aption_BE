"""실제 8 CPU 실패의 연산어 복원과 일반 문장·불명확한 수식의 비수정 경계를 검사한다."""

import unittest

from app.services.translation_normalization import has_math_operator_words, normalize_math_operators
from app.services.translation_validation import preservation_problem

SOURCE = ('And we ultimately want this, so this is vr of t, of course, '
          'and then we get v0 plus u ln mr over 2m0.')
RAW = ('그리고 결국 우리가 원하는 것은 바로 이 식이므로, 이는 t에 대한 vr의 값이며, '
       '그 결과는 v0 plus u times ln(mR / 2m0)입니다.')


class TranslationNormalizationTests(unittest.TestCase):
    """원문으로 확인한 연산어만 바꾸고 기존 보존 검사를 통과하는지 검사한다."""

    def test_recorded_eight_cpu_failure_preserves_every_non_operator_character(self):
        """실제 실패 출력의 plus·times만 바꾸고 로그 괄호·변수·숫자·문장을 보존한다.

        :return: None.
        """
        result = normalize_math_operators(RAW, SOURCE)
        self.assertEqual(result, RAW.replace('plus', '+').replace('times', '×'))
        self.assertIsNone(preservation_problem(SOURCE, result))
        self.assertFalse(has_math_operator_words(result, SOURCE))
        self.assertEqual(normalize_math_operators(result, SOURCE), result)

    def test_source_confirmed_binary_operations_and_explicit_denominator(self):
        """같은 항의 이항 연산과 이미 괄호가 있는 분모만 치환한다.

        :return: None.
        """
        for source, raw, expected in (
            ('mR minus m0.', 'mR minus m0입니다.', 'mR - m0입니다.'),
            ('mR equals m0.', 'mR equals m0입니다.', 'mR = m0입니다.'),
            ('u times vR.', 'u times vR입니다.', 'u × vR입니다.'),
            ('mR over m0.', 'mR over m0입니다.', 'mR / m0입니다.'),
            ('mR over 2m0.', 'mR over (2m0)입니다.', 'mR / (2m0)입니다.'),
            ('v0 + u.', 'v0 PLUS u입니다.', 'v0 + u입니다.'),
        ):
            with self.subTest(source=source):
                self.assertEqual(normalize_math_operators(raw, source), expected)

    def test_plain_prose_times_units_and_partial_words_are_unchanged(self):
        """일반 빈도·시간 표현·단위·긴 단어의 일부를 수학 연산자로 바꾸지 않는다.

        :return: None.
        """
        for source, raw in (
            ('It changes over time.', '시간은 over time으로 변합니다.'),
            ('We read m three times.', 'm을 three times 읽습니다.'),
            ('Read m 3 times.', 'm을 3 times 읽습니다.'),
            ('5 m times 3 m.', '5 m times 3 m입니다.'),
            ('u is constant.', 'u의 timestamp를 봅니다.'),
        ):
            with self.subTest(source=source):
                self.assertEqual(normalize_math_operators(raw, source), raw)
                self.assertFalse(has_math_operator_words(raw, source))

    def test_wrong_operators_and_reversed_operands_still_have_rejection_evidence(self):
        """원문과 다른 연산이나 뒤바뀐 나눗셈을 기호로 숨기지 않는다.

        :return: None.
        """
        for source, raw in (
            ('v0 plus u.', 'v0 minus u입니다.'),
            ('mR over m0.', 'm0 over mR입니다.'),
            ('u times vR.', 'vR times u입니다.'),
            ('mR equals m0.', 'm0 equals mR입니다.'),
            ('u ln mR.', 'u times ln(m0)입니다.'),
        ):
            with self.subTest(source=source):
                self.assertEqual(normalize_math_operators(raw, source), raw)
                self.assertTrue(has_math_operator_words(raw, source))

    def test_does_not_add_parentheses_or_choose_logarithm_scope(self):
        """괄호 없는 로그·계수 분모·괄호식 왼쪽 항·복합 곱을 새로 해석하지 않는다.

        :return: None.
        """
        for source, raw in (
            ('mR over 2m0.', 'mR over 2m0입니다.'),
            ('u ln mR.', 'u times ln mR입니다.'),
            ('mR dvR equals dmRu.', 'mR dvR equals dmRu입니다.'),
            ('mR minus m0.', '(mR) minus m0입니다.'),
        ):
            with self.subTest(source=source):
                self.assertEqual(normalize_math_operators(raw, source), raw)

    def test_same_operands_in_conflicting_source_relations_are_not_repaired(self):
        """같은 항을 더하고 빼는 두 절에서 연산이 뒤바뀐 오역을 정규화로 숨기지 않는다.

        :return: None.
        """
        source = 'The initial mass is mR plus m0; the final mass is mR minus m0.'
        raw = '초기 질량은 mR minus m0이고 최종 질량은 mR plus m0입니다.'
        self.assertEqual(normalize_math_operators(raw, source), raw)
        self.assertTrue(has_math_operator_words(raw, source))

    def test_thousands_are_not_matched_as_partial_numeric_operands(self):
        """미지원 천 단위 숫자의 일부를 별도 피연산자로 인식해 치환하지 않는다.

        :return: None.
        """
        for source, raw in (('mR plus 1,000.', 'mR plus 1,000입니다.'),
                            ('1,000 plus mR.', '1,000 plus mR입니다.'),
                            ('mR plus 1,000.', 'mR plus 1입니다.')):
            with self.subTest(source=source):
                self.assertEqual(normalize_math_operators(raw, source), raw)

    def test_broken_or_unsupported_function_arguments_are_not_normalized(self):
        """괄호 손상·빈 인수·연속 연산자·미지원 변수가 있는 곱은 정규화하지 않는다.

        :return: None.
        """
        for argument in ('ln(mR / 2m0', 'ln(mR / 2m0))', 'ln()', 'ln(mR // 2m0)',
                         'ln(mR / / 2m0)', 'ln(2 2m0)', 'ln(x / 2m0)',
                         'ln(' + '(' * 9 + 'mR' + ')' * 10):
            raw = '이 값은 u times ' + argument + '입니다.'
            with self.subTest(argument=argument):
                self.assertEqual(normalize_math_operators(raw, SOURCE), raw)
                self.assertTrue(has_math_operator_words(raw, SOURCE))

    def test_valid_nested_parentheses_keep_the_existing_scope(self):
        """이미 존재하는 지원 중첩 괄호를 그대로 유지하면서 연산어만 바꾼다.

        :return: None.
        """
        raw = '이 값은 u times ln((mR / 2m0))입니다.'
        self.assertEqual(normalize_math_operators(raw, SOURCE), raw.replace('times', '×'))

    def test_normalization_does_not_hide_missing_numbers_symbols_or_reversed_fraction(self):
        """연산어를 정리한 뒤에도 숫자·기호 누락과 분수 반전은 기존 검사에서 거부된다.

        :return: None.
        """
        for raw in (RAW.replace('2m0', '3m0'), RAW.replace('vr의 값', '값'),
                    RAW.replace('mR / 2m0', '2m0 / mR')):
            with self.subTest(raw=raw):
                self.assertIsNotNone(preservation_problem(SOURCE, normalize_math_operators(raw, SOURCE)))


if __name__ == '__main__':
    unittest.main()
