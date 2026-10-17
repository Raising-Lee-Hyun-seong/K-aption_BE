"""평가 전용 Qwen 프롬프트의 문맥 경계·재시도·기존 API 용어집 격리를 검증한다."""
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.evaluate_prompt_matrix import prompt_service, variant_prompt
from scripts.evaluate_qwen_lecture import EvaluationQwenService
from test_prompt_matrix import MessageTokenizer


class QwenLectureTests(unittest.TestCase):
    """모델 없이 문맥 전달과 실제 상속된 검증·재시도 흐름을 확인한다."""

    def test_lecture_without_context_matches_raw_refined_prompt(self):
        """문맥 없는 최초 시도는 22문장 실험의 보완 프롬프트와 정확히 일치한다.

        :return: None.
        :raises AssertionError: 강의 평가가 보완 조건과 다른 기본 지시를 사용할 때.
        """
        service = EvaluationQwenService.__new__(EvaluationQwenService)
        service.tokenizer = MessageTokenizer()
        service.context_format = 'system'
        service.variant = 'qwen-relations-v2'
        service.glossary = prompt_service().glossary
        text = 'The rocket has constant speed.'
        self.assertEqual(service._build_prompt(text, ''), variant_prompt(
            'qwen-relations-v2', service.tokenizer, service, text))
        self.assertEqual(len(service.glossary), 28)

    def test_failed_attempt_retries_without_previous_context(self):
        """실제 번역 검증이 실패하면 다음 시도에서 문맥을 제거하고 출력 지시를 추가한다.

        :return: None.
        :raises AssertionError: 문맥이 재시도로 누출되거나 256자 제한·실패 복구가 다를 때.
        """
        service = EvaluationQwenService.__new__(EvaluationQwenService)
        service.tokenizer = SimpleNamespace(
            apply_chat_template=MessageTokenizer().apply_chat_template,
            encode=Mock(return_value=[1]),
        )
        service.glossary = prompt_service().glossary
        service._lock = Lock()
        service.context_format = 'system'
        service.variant = 'qwen-relations-v2'
        service._generate_output = Mock(side_effect=[
            {'text': '', 'tokens': 1, 'finish_reason': 'stop', 'stop_reason': None},
            {'text': '다음 문장입니다.', 'tokens': 5, 'finish_reason': 'stop', 'stop_reason': None},
        ])
        context = 'PRIVATE_PREFIX' + 'x' * 256
        result = service.translate('Next sentence.', context, 128)
        prompts = [call.args[0] for call in service._generate_output.call_args_list]
        self.assertEqual(result['translation'], '다음 문장입니다.')
        self.assertEqual(len(prompts), 2)
        self.assertIn('x' * 256, prompts[0])
        self.assertNotIn('PRIVATE_PREFIX', prompts[0])
        self.assertNotIn('Previous English context', prompts[1])
        self.assertNotIn('x' * 256, prompts[1])
        self.assertIn('complete Korean translation sentences', prompts[1])

    def test_context_is_separate_from_current_source(self):
        """참고 문맥은 시스템 메시지에만 두고 현재 원문은 사용자 메시지에만 넣는다.

        :return: None.
        :raises AssertionError: 참고 문맥과 현재 번역 대상의 메시지 경계가 섞일 때.
        """
        import json

        prompt = variant_prompt('qwen-relations-v2', MessageTokenizer(), prompt_service(),
                                'Current source.', 'Previous source.', context_format='system')
        messages = json.loads(prompt.removesuffix('<assistant>'))
        self.assertEqual([message['role'] for message in messages], ['system', 'user'])
        self.assertIn('Previous source.', messages[0]['content'])
        self.assertNotIn('Previous source.', messages[1]['content'])
        self.assertIn('Current source.', messages[1]['content'])
        self.assertNotIn('Current source.', messages[0]['content'])
