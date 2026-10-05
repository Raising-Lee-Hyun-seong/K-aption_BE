"""수식·숫자 보존과 반복·지시문 유출 조기 종료 및 스트림 자원 정리를 검증한다."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.translation_validation import (
    generation_problem, preservation_problem, protected_symbols,
)
from tests import test_pipeline


class PreservationTests(unittest.TestCase):
    """실제 강의에서 관찰된 수식 삭제·수치 변경·단위 첨가를 검증한다."""
    def test_symbols_match_whole_identifiers_with_korean_particles(self):
        """수식 식별자는 한국어 조사와 붙어도 인식하고 다른 식별자의 일부와 혼동하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        source = 'mR dvR = dmR u, from v0 to vR, ln mR and dt.'
        self.assertEqual(protected_symbols(source), ['mR', 'dvR', 'dmR', 'u', 'v0', 'vR', 'ln', 'dt'])
        self.assertIsNone(preservation_problem(source, 'mR은 dvR = dmR u이며 v0부터 vR까지 ln mR와 dt입니다.'))
        self.assertEqual(preservation_problem('mR', 'dmR입니다.'), 'missing_math_symbol')
        self.assertEqual(preservation_problem('vR', 'VR 기술입니다.'), 'missing_math_symbol')
        self.assertEqual(protected_symbols('The mass is 2m0.'), ['m0'])
        self.assertEqual(preservation_problem('The mass is 2m0.', '질량은 2입니다.'), 'missing_math_symbol')
        self.assertIsNone(preservation_problem('The mass is 2m0.', '질량은 2m0입니다.'))

    def test_number_loss_and_normalized_decimal(self):
        """명시 숫자의 변경은 거부하고 같은 값의 소수 표기는 허용한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(preservation_problem('At t = 0, mass = 2 m0.', 't = 1에서 질량 = 2 m0.'), 'missing_number')
        self.assertIsNone(preservation_problem('Value is 2.0.', '값은 2입니다.'))
        self.assertEqual(preservation_problem('Value is -1.', '값은 1입니다.'), 'missing_number')
        self.assertEqual(preservation_problem('m0', 'm0이며 5입니다.'), 'invented_number')
        self.assertIsNone(preservation_problem('Value is - 1.', '값은 -1입니다.'))
        self.assertIsNone(preservation_problem('1,000 and .5', '1000과 0.5입니다.'))
        self.assertEqual(preservation_problem('2dmR', '3dmR입니다.'), 'missing_number')

    def test_number_words_do_not_match_substring_of_another_number(self):
        """영어 수사 1을 10으로 바꾸는 결과를 숫자 부분 일치로 허용하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(preservation_problem('One kilogram.', '10킬로그램입니다.'), 'missing_number')
        self.assertIsNone(preservation_problem('Two kilograms.', '두 킬로그램입니다.'))
        self.assertIsNone(preservation_problem('zero', '영입니다.'))
        self.assertIsNone(preservation_problem('two', '두개의 값입니다.'))
        self.assertIsNone(preservation_problem('minus two', '-2입니다.'))
        self.assertEqual(preservation_problem('minus two', '2입니다.'), 'missing_number')

    def test_invented_mass_and_distance_units(self):
        """2m0를 킬로그램·미터로 바꾼 오류를 거부하고 원문에 있는 단위는 허용한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(preservation_problem('2 m0', '2 m0 킬로그램입니다.'), 'invented_unit')
        self.assertEqual(preservation_problem('2 m0', '2 m0 미터입니다.'), 'invented_unit')
        self.assertIsNone(preservation_problem('Two kilograms.', '2킬로그램입니다.'))
        self.assertIsNone(preservation_problem('Two kilometers.', '2킬로미터입니다.'))
        self.assertIsNone(preservation_problem('2 cm', '2센티미터입니다.'))
        self.assertEqual(preservation_problem('2 meters', '2센티미터입니다.'), 'invented_unit')
        self.assertIsNone(preservation_problem('2 kilograms', '2kg입니다.'))
        self.assertIsNone(preservation_problem('2kg', '2킬로그램입니다.'))
        self.assertEqual(preservation_problem('2kg', '20kg입니다.'), 'missing_number')
        self.assertIsNone(preservation_problem('2 m and 2 g', '2미터와 2그램입니다.'))
        self.assertEqual(preservation_problem('m', '미터입니다.'), 'missing_math_symbol')

    def test_spaced_and_lowercase_math_aliases_are_canonical(self):
        """공식 자막의 띄어쓴·소문자 기호를 보존하고 원표기와 붙인 표기를 모두 허용한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        source = 'd v r, d m r, m r, v r; dvr, dmr, mr, vr.'
        self.assertEqual(protected_symbols(source), ['dvR', 'dmR', 'mR', 'vR'])
        self.assertEqual(protected_symbols('M r and V r'), ['mR', 'vR'])
        self.assertIsNone(preservation_problem(source, 'dvR, dmR, mR, vR입니다.'))
        self.assertIsNone(preservation_problem('dvR dmR mR vR',
                                              'd v r과 d m r, m r, v r입니다.'))
        self.assertEqual(preservation_problem('The change is d v r.', '데이터 버스입니다.'),
                         'missing_math_symbol')
        self.assertEqual(preservation_problem('The value is m r.', '메트릭 값입니다.'),
                         'missing_math_symbol')
        self.assertEqual(preservation_problem('2mr', '3 m r입니다.'), 'missing_number')

    def test_dmru_preserves_both_differential_and_velocity_symbol(self):
        """STT의 붙인 dmRu를 dmR u와 동일하게 취급하고 어느 기호의 누락도 거부한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(protected_symbols('dmRu'), ['dmR', 'u'])
        self.assertEqual(protected_symbols('d m r u'), ['dmR', 'u'])
        self.assertIsNone(preservation_problem('dmRu', 'dmR u입니다.'))
        self.assertIsNone(preservation_problem('dmR u', 'dmRu입니다.'))
        self.assertEqual(preservation_problem('dmRu', 'dmR입니다.'), 'missing_math_symbol')
        self.assertEqual(preservation_problem('dmRu', '메트릭 버스입니다.'), 'missing_math_symbol')
        self.assertEqual(preservation_problem('2dmRu', '3dmR u입니다.'), 'missing_number')
        self.assertEqual(protected_symbols('dmRux imaginaryVariable'), [])

    def test_independent_variables_do_not_match_compound_parts_or_words(self):
        """독립 변수를 보호하되 복합 기호 내부 글자와 일반 영어 단어를 중복 추출하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(protected_symbols('m u t v g'), ['m', 'u', 't', 'v', 'g'])
        self.assertEqual(protected_symbols('dmR dvR mR vR dt dm dv'),
                         ['dmR', 'dvR', 'mR', 'vR', 'dt', 'dm', 'dv'])
        self.assertEqual(protected_symbols('summary mutual program meter grammar'), [])
        self.assertIsNone(preservation_problem('m u t', 'm은 u와 t에 관련됩니다.'))
        self.assertEqual(preservation_problem('m u t', '질량은 u와 t에 관련됩니다.'),
                         'missing_math_symbol')

    def test_independent_variables_and_unit_abbreviations_are_distinguished(self):
        """수량 뒤 m·g는 단위로 허용하고 독립 변수 및 띄어쓴 복합 기호와 혼동하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(protected_symbols('2 m, 2 g and 2kg'), [])
        self.assertIsNone(preservation_problem('2m and 2g', '2미터와 2그램입니다.'))
        self.assertIsNone(preservation_problem('m = 2 and g = 2', 'm은 2이고 g는 2입니다.'))
        self.assertEqual(preservation_problem('g', '그램입니다.'), 'missing_math_symbol')
        self.assertIsNone(preservation_problem('2 m r', '2mR입니다.'))
        self.assertEqual(preservation_problem('2 m r', '2mR 미터입니다.'), 'invented_unit')

    def test_math_prime_markers_are_required_and_aliases_are_accepted(self):
        """수학 문맥의 prime을 프라임 표기로 보존하고 소수·프리미엄 오역을 거부한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        source = 'All primes are at t = 0.'
        for translation in ('모든 prime은 t = 0입니다.', '모든 프라임은 t = 0입니다.',
                            't′은 0입니다.', "t'은 0입니다.", 't’은 0입니다.'):
            with self.subTest(translation=translation):
                self.assertIsNone(preservation_problem(source, translation))
        for translation in ('소수는 t = 0입니다.', '프리미엄은 t = 0입니다.',
                            "t는 0이며 it's라는 단어가 있습니다."):
            with self.subTest(translation=translation):
                self.assertEqual(preservation_problem(source, translation), 'missing_math_prime')
        self.assertIsNone(preservation_problem('The primes are small.', '소수는 작습니다.'))
        self.assertIsNone(preservation_problem('u prime', "u'입니다."))

    def test_prompt_leak_repetition_and_invalid_character(self):
        """지시문 복사·짧은 반복 루프·손상 문자를 검출하되 정상 응답 표식은 허용한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertEqual(generation_problem('이전 영어 문맥: hello'), 'prompt_leak')
        self.assertEqual(generation_problem('설명입니다. (참고만 하세요.)'), 'prompt_leak')
        self.assertEqual(generation_problem('원문을 한국어로 번역합니다.'), 'prompt_leak')
        self.assertEqual(generation_problem('속도를 계산합니다. ' * 3), 'repetition')
        self.assertEqual(generation_problem('2.0000000000000000'), 'repetition')
        self.assertEqual(generation_problem('잘못된\ufffd'), 'invalid_character')
        self.assertIsNone(generation_problem('### Response:\n출력: "한국어 번역."'))


class StreamingTests(unittest.TestCase):
    """실제 MLX 없이 스트림 검증과 재시도·종료·토큰 통계를 확인한다."""
    def translator(self):
        """기존 서비스 테스트 도우미로 모델 없는 번역 서비스를 구성한다.

        :return: 모델을 로드하지 않는 TranslationService 테스트 인스턴스.
        """
        return test_pipeline.ServiceTests().translator()

    def test_repetition_stops_stream_and_retry_closes_both_generators(self):
        """반복 발견 즉시 남은 토큰을 소비하지 않고 재시도하며 두 생성기를 모두 닫는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        closed, consumed, prompts = [], [], []

        def stream_generate(*args, **kwargs):
            """첫 시도는 반복을 만들고 두 번째 시도는 정상 문장을 반환한다.

            :param args: 생성 API와 같은 호출 형태를 받기 위한 위치 인자. 대역에서는 사용하지 않는다.
            :param kwargs: 대상 함수에 전달할 추가 키워드 인자 사전.
            :return: 아래 값을 순차 제공하는 생성기.
            :yield: 종료 사유·생성 토큰 수·부분 텍스트를 담은 테스트용 스트림 응답.
            """
            attempt = len(prompts)
            prompts.append(kwargs['prompt'])
            try:
                chunks = ['같은 문장.' * 3, '소비하면 안 되는 토큰'] if attempt == 0 else ['정상 문장.']
                for index, chunk in enumerate(chunks, 1):
                    consumed.append((attempt, index))
                    yield SimpleNamespace(text=chunk, generation_tokens=index * 7,
                                          finish_reason='stop' if attempt else None)
            finally:
                closed.append(attempt)

        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=stream_generate)}):
            result = self.translator().translate('Sentence.', 'Earlier source.', 128)
        self.assertEqual(consumed, [(0, 1), (1, 1)])
        self.assertEqual(closed, [0, 1])
        self.assertIn('Earlier source.', prompts[0])
        self.assertNotIn('Earlier source.', prompts[1])
        self.assertEqual(result['translation'], '정상 문장.')
        self.assertEqual(result['generated_tokens'], 7)

    def test_truncated_output_is_rejected(self):
        """최대 토큰 수로 끝난 미완성 출력은 한글이 있어도 성공으로 반환하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        def stream_generate(*args, **kwargs):
            """토큰 상한에서 종료된 응답을 제공한다.

            :param args: 생성 API와 같은 호출 형태를 받기 위한 위치 인자. 대역에서는 사용하지 않는다.
            :param kwargs: 대상 함수에 전달할 추가 키워드 인자 사전.
            :return: 아래 값을 순차 제공하는 생성기.
            :yield: 종료 사유·생성 토큰 수·부분 텍스트를 담은 테스트용 스트림 응답.
            """
            yield SimpleNamespace(text='속도는', generation_tokens=128, finish_reason='length')

        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=stream_generate)}):
            with self.assertRaises(ValueError):
                self.translator().translate('The velocity is constant.', '', 128)

    def test_leak_is_rejected_before_cleaning(self):
        """정리 과정에서 지워질 수 있는 지시문도 원시 출력 단계에서 검출한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        def stream_generate(*args, **kwargs):
            """입력 지시문과 그럴듯한 응답이 섞인 결과를 제공한다.

            :param args: 생성 API와 같은 호출 형태를 받기 위한 위치 인자. 대역에서는 사용하지 않는다.
            :param kwargs: 대상 함수에 전달할 추가 키워드 인자 사전.
            :return: 아래 값을 순차 제공하는 생성기.
            :yield: 종료 사유·생성 토큰 수·부분 텍스트를 담은 테스트용 스트림 응답.
            """
            yield SimpleNamespace(text='### Instruction: translate\n### Response: 정상 문장.',
                                  generation_tokens=20, finish_reason='stop')

        with patch.dict(sys.modules, {'mlx_lm': SimpleNamespace(stream_generate=stream_generate)}):
            with self.assertRaises(ValueError):
                self.translator().translate('Sentence.', '', 128)


if __name__ == '__main__':
    unittest.main()
