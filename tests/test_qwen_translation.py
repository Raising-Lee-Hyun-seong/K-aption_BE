"""Qwen 기본 선택·존댓말·문맥 분리·대상 보존 재시도와 CPU 채팅 구성을 검증한다."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.main import create_translation_service
from app.services.translation_http import HttpTranslationService
from app.services.translation_qwen import INSTRUCTION, informal_ending, render_chatml
from scripts.download_cpu_model import load_manifest

ROOT = Path(__file__).resolve().parents[1]


class QwenTranslationTests(unittest.TestCase):
    """모델 생성 대역으로 실제 API에 연결된 Qwen 계약을 검사한다."""

    def service(self):
        """공통 설정과 Qwen 용어를 갖춘 모델 없는 서비스를 만든다.

        :return: 실제 프롬프트·검증·재시도 구현을 사용하는 Qwen 서비스 대역.
        """
        service = HttpTranslationService.__new__(HttpTranslationService)
        service.model_family = 'qwen3'
        service._configure_text(ROOT / 'templates/korean.json')
        service._configure_qwen()
        service.tokenizer = SimpleNamespace(encode=Mock(return_value=[1]),
                                           apply_chat_template=Mock(side_effect=render_chatml))
        return service


    def test_context_is_only_in_bounded_system_message(self):
        """앞 문맥은 시스템에만 넣고 현재 원문은 사용자 메시지에 한 번 넣는다.

        :return: None.
        """
        service = self.service()
        context = 'old context ' * 40
        source = 'This is the initial mass.'
        prompt = service._build_prompt(source, context)
        messages = service.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual([message['role'] for message in messages], ['system', 'user'])
        self.assertTrue(messages[0]['content'].endswith(context[-256:]))
        self.assertNotIn(context[-256:], messages[1]['content'])
        self.assertEqual(prompt.count(source), 1)
        self.assertIn('division expressed by "over"', prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))

    def test_object_mistranslation_is_retried_without_previous_context(self):
        """썰매를 다른 대상으로 바꾼 결과를 거부하고 문맥 없는 두 번째 번역을 쓴다.

        :return: None.
        """
        service = self.service()
        service._generate_output = Mock(side_effect=[
            {'text': '슬라이드는 결국 멈춥니다.', 'tokens': 10, 'finish_reason': 'stop', 'stop_reason': None},
            {'text': '썰매는 결국 멈춥니다.', 'tokens': 10, 'finish_reason': 'stop', 'stop_reason': None}])
        result = service.translate('This sled will eventually come to a stop.', 'Earlier English.', 128)
        self.assertEqual(result['translation'], '썰매는 결국 멈춥니다.')
        self.assertEqual(service._generate_output.call_count, 2)
        self.assertIn('Earlier English.', service._generate_output.call_args_list[0].args[0])
        self.assertNotIn('Earlier English.', service._generate_output.call_args_list[1].args[0])

    def test_common_instructions_precede_changing_context_and_terms(self):
        """문맥·용어·변수가 달라도 고정 지시는 같은 접두부에 한 번만 넣는다.

        :return: None.
        """
        service = self.service()
        prefixes = []
        for source, context in (('This rocket mass is constant.', 'Earlier source.'),
                                ('That is the velocity.', 'Different source.')):
            prompt = service._build_prompt(source, context)
            messages = service.tokenizer.apply_chat_template.call_args.args[0]
            system, user = (message['content'] for message in messages)
            self.assertIn(INSTRUCTION, system)
            self.assertNotIn(INSTRUCTION, user)
            self.assertLess(system.index(INSTRUCTION), system.index(context))
            self.assertNotIn(context, user)
            self.assertTrue(user.startswith('Translate the following English text into Korean.'))
            self.assertTrue(user.endswith(source))
            self.assertEqual(prompt.count(INSTRUCTION), 1)
            prefixes.append(system.split('Previous English context:')[0])
        self.assertEqual(prefixes[0], prefixes[1])

    def test_contraction_translation_does_not_retry_or_request_fake_symbols(self):
        """축약형을 가진 정상 존댓말 번역은 가짜 변수 지시나 추가 생성 없이 반환한다.

        :return: None.
        """
        service = self.service()
        service._generate_output = Mock(return_value={
            'text': '그럴 필요는 없습니다.', 'tokens': 8, 'finish_reason': 'stop', 'stop_reason': None})
        result = service.translate("We don't have to.", 'Earlier English.', 128)
        self.assertEqual(result['translation'], '그럴 필요는 없습니다.')
        self.assertEqual(service._generate_output.call_count, 1)
        self.assertNotIn('Keep these source symbols:', service._generate_output.call_args.args[0])


    def test_spoken_math_operators_cannot_remain_untranslated(self):
        """수학 변수의 영어 연산어 복사는 거부하되 연산 기호·한글 표기는 허용한다.

        :return: None.
        """
        service = self.service()
        source = 'And then we have dmR over mR.'
        self.assertEqual(service._rejection_reason('그리고 dmR over mR입니다.', source, ''),
                         'untranslated_math_operator')
        self.assertIsNone(service._rejection_reason('그리고 dmR / mR입니다.', source, ''))
        self.assertIsNone(service._rejection_reason('그리고 dmR을 mR로 나눕니다.', source, ''))
        self.assertIsNone(service._rejection_reason('적분하면 ln을 얻습니다.', 'Integrated gives us an ln.', ''))

    def test_function_relation_guides_both_attempts_and_rejects_multiplication(self):
        """수식 표식을 잃은 곱셈 오역을 거부하고 복원한 함수·등식 관계만 반환한다.

        :return: None.
        """
        service = self.service()
        service._generate_output = Mock(side_effect=[
            {'text': 'mR 프라임은 m에 vR을 곱한 값입니다.', 'tokens': 20,
             'finish_reason': 'stop', 'stop_reason': None},
            {'text': '이제 [MATH_A]입니다.', 'tokens': 15,
             'finish_reason': 'stop', 'stop_reason': None}])
        result = service.translate('mR prime equals m of vr.', 'Earlier English.', 128)
        self.assertEqual(result['translation'], '이제 mR′ = m(vR)입니다.')
        self.assertEqual(service._generate_output.call_count, 2)
        for call in service._generate_output.call_args_list:
            self.assertIn('[MATH_A]', call.args[0])
            self.assertNotIn('m of vr', call.args[0])
        prompt = service._build_prompt('v0 plus u ln mr over 2m0.', '')
        self.assertNotIn('[MATH_A]', prompt)
        self.assertIn('mR/(2m0)', prompt)

    def test_formal_style_is_requested_in_first_attempt_and_retry(self):
        """첫 생성과 재시도 모두 존댓말을 지시하고 원문의 질문·진술을 보존한다.

        :return: None.
        """
        service = self.service()
        for retry in (False, True):
            prompt = service._build_prompt('What does this equation tell us?', '', retry)
            self.assertIn('polite Korean', prompt)
            self.assertIn('합니다', prompt)
            self.assertIn('statement or question', prompt)

    def test_plain_speech_is_retried_without_rewriting_the_output(self):
        """반말 출력은 기계적으로 고치지 않고 실제 모델의 존댓말 재시도 결과를 쓴다.

        :return: None.
        """
        service = self.service()
        service._generate_output = Mock(side_effect=[
            {'text': '이것이 초기 질량이다.', 'tokens': 8, 'finish_reason': 'stop', 'stop_reason': None},
            {'text': '이것이 초기 질량입니다.', 'tokens': 9, 'finish_reason': 'stop', 'stop_reason': None}])
        result = service.translate('This is the initial mass.', '', 128)
        self.assertEqual(result['translation'], '이것이 초기 질량입니다.')
        self.assertEqual(service._generate_output.call_count, 2)
        self.assertEqual(service._rejection_reason('이것이 초기 질량이다.', 'This is the initial mass.', ''),
                         'informal_speech')

    def test_formal_statements_questions_and_math_nouns_are_not_rejected(self):
        """존댓말 진술·질문과 람다·캐나다 같은 명사구를 반말로 오인하지 않는다.

        :return: None.
        """
        for text in ('속도는 일정합니다.', '초기 질량입니다.', '무엇을 뜻합니까?', '적분합시다.',
                     '람다.', '캐나다.', '우간다.', '판다.'):
            with self.subTest(text=text):
                self.assertFalse(informal_ending(text))
        for text in ('속도는 일정하다.', '초기 질량이다.', '그것을 얻었다.', '해보자.', '그 값이 사라진다.',
                     '질량입니다. 속도가 감소한다.', '왜 그런 거야?'):
            with self.subTest(text=text):
                self.assertTrue(informal_ending(text))

    def test_repeated_plain_speech_stops_at_two_attempts(self):
        """두 번 모두 반말이면 시도 한도를 지키고 번역 실패로 처리한다.

        :return: None.
        """
        service = self.service()
        service._generate_output = Mock(return_value={
            'text': '이것이 초기 질량이다.', 'tokens': 8,
            'finish_reason': 'stop', 'stop_reason': None})
        with self.assertRaisesRegex(ValueError, 'valid Korean translation'):
            service.translate('This is the initial mass.', '', 128)
        self.assertEqual(service._generate_output.call_count, 2)


    def test_cpu_manifest_pins_public_qwen_artifact(self):
        """공개 CPU 파일의 모델·리비전·크기·SHA256을 고정한 명세를 검사한다.

        :return: None.
        """
        manifest = load_manifest(ROOT / 'docker/qwen3-model.json')
        self.assertEqual(manifest['model'], 'Qwen/Qwen3-4B-Instruct-2507')
        self.assertEqual(manifest['quantization'], 'Q4_K_M')
        self.assertEqual(manifest['bytes'], 2497281120)
        self.assertEqual(manifest['revision'], 'a06e946bb6b655725eafa393f4a9745d460374c9')
