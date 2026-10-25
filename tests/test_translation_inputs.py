"""수식 보호의 의미 경계·표식 손상과 문맥 선택의 입력 계약을 검증한다."""
import unittest

from app.services.translation_inputs import protect_math, restore_math, select_context
from app.services.translation_validation import preservation_problem


class TranslationInputTests(unittest.TestCase):
    """모델 없이 원문 수식과 모델용 보호 입력의 왕복 관계를 검증한다."""

    def test_function_equality_and_fraction_survive_restoration(self):
        """함수 인수·프라임·분수 방향을 원문 기준 보존 검사까지 유지한다.

        :return: None.
        """
        cases = [('mR prime equals m of vr.', 'mR′ = m(vR)'),
                 ('And then dmR over mR.', 'dmR/(mR)'),
                 ('mR prime equals m0.', 'mR′ = m0'),
                 ('u1 over 2m0.', 'u1/(2m0)')]
        for source, expression in cases:
            with self.subTest(source=source):
                original = source
                model_input, replacements = protect_math(source)
                self.assertEqual(source, original)
                self.assertEqual(list(replacements.values()), [expression])
                self.assertIn('[MATH_A]', model_input)
                restored, problem = restore_math('이제 [MATH_A]입니다.', source)
                self.assertIsNone(problem)
                self.assertIn(expression, restored)
                self.assertIsNone(preservation_problem(source, restored))

    def test_ambiguous_chains_units_and_existing_functions_are_not_reinterpreted(self):
        """연쇄 나눗셈·단위·복합 곱의 부분 등식과 기존 괄호식을 새 식으로 만들지 않는다.

        :return: None.
        """
        for source in ('m over m of vR.', 'mR dvR equals dmRu.', '5 m of vR.',
                       'The mass of a rocket.', 'dvR equals u is a constant.',
                       'mR equals m(vR).', 'mR equals m0 plus u.', 'vR - v0 equals m0.'):
            with self.subTest(source=source):
                self.assertEqual(protect_math(source), (source, {}))

    def test_mixed_math_is_left_to_existing_translation_without_solving_logarithm(self):
        """남은 복합 연산·로그의 범위를 추론하거나 부분 표식으로 고정하지 않는다.

        :return: None.
        """
        source = 'vR minus v0 equals u1 over m. Then v0 plus u ln mR over 2m0.'
        self.assertEqual(protect_math(source), (source, {}))
        for text in ('u1 plus m of vR.', '2 plus m of vR.', 'dvR equals u is a constant, and dmR over mR.'):
            with self.subTest(text=text):
                self.assertEqual(protect_math(text), (text, {}))

    def test_missing_duplicate_swapped_or_invented_markers_are_rejected(self):
        """표식 누락·복제·교환·새 표식이나 표식 문자 변경을 성공으로 복원하지 않는다.

        :return: None.
        """
        source = 'm of vR and dmR over mR.'
        for candidate in ('수식입니다.', '[MATH_A]와 [MATH_A]와 [MATH_B]입니다.',
                          '[MATH_B]와 [MATH_A]입니다.', '[MATH_A]와 [MATH_C]입니다.',
                          '[math_a]와 [MATH_B]입니다.', '[MATH_A]와 [MATH_B]와 [MATH_C입니다.'):
            with self.subTest(candidate=candidate):
                self.assertEqual(restore_math(candidate, source)[1], 'math_placeholder_mismatch')
        self.assertEqual(restore_math('[MATH_A]입니다.', 'Hello.')[1], 'math_placeholder_mismatch')

    def test_reserved_source_markers_do_not_collide(self):
        """원문에 예약 표식이 있으면 이를 다른 수식으로 대체하지 않는다.

        :return: None.
        """
        source = 'Explain [MATH_A] and m of vR.'
        self.assertEqual(protect_math(source), (source, {}))

    def test_context_is_kept_for_references_and_removed_for_complete_statements(self):
        """지시 대상·이어지는 관계는 문맥을 유지하고 독립적인 명시 문장은 문맥을 생략한다.

        :return: None.
        """
        context = 'An earlier physics statement. ' * 20
        for source in ('This is the initial mass.', 'We move it to the other side.',
                       'And we are left with dvR.', 'Which means the sled stops.',
                       'Actually, not quite m0, it is 2m0.', 'She wrote the expression.',
                       'We multiply by dt, so that cancels.',
                       'mR decreases, which means this term is less than 0.'):
            with self.subTest(source=source):
                self.assertEqual(select_context(source, context), context[-256:])
        for source in ('The rocket mass is constant.', 'We want to get the speed of the rocket.',
                       'mR prime equals m of vR.', 'And then we have dmR over mR.'):
            with self.subTest(source=source):
                self.assertEqual(select_context(source, context), '')


if __name__ == '__main__':
    unittest.main()
