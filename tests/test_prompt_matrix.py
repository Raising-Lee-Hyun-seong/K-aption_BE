"""실험 조건별 프롬프트의 원문 보존·기준 번역 분리·가중치 누락과 의미 검사 한계를 검증한다."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_candidates import build_prompt, load_cases
from scripts.evaluate_prompt_matrix import (
    VARIANTS, diagnose_output, prompt_service, variant_prompt, verify_koreanlm,
)


class MessageTokenizer:
    """모델 없이 채팅 역할과 내용을 보존해 프롬프트 구성을 검사한다."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        """입력 메시지를 직렬화하고 생성 시작 표식을 제공한다.

        :param messages: 원문과 번역 지시가 담긴 채팅 메시지 목록.
        :param tokenize: 이 대역에서는 거짓이어야 하는 토큰화 옵션.
        :param add_generation_prompt: 생성 시작 표식 추가 여부.
        :return: 메시지 문자열과 선택적인 생성 시작 표식.
        :raises AssertionError: 토큰화를 요청할 때.
        """
        assert not tokenize
        return json.dumps(messages, ensure_ascii=False) + ('<assistant>' if add_generation_prompt else '')


class PromptMatrixTests(unittest.TestCase):
    """동일 입력 비교를 유지하고 진단 통과를 의미 정확도로 오인하지 않도록 검증한다."""

    def test_every_variant_preserves_source_without_reference_leak(self):
        """기준·보완 조건 모두 22개 원문을 한 번만 포함하고 기준 번역을 포함하지 않는다.

        :return: None.
        :raises AssertionError: 원문이 중복·누락되거나 기준 번역이 프롬프트에 들어갈 때.
        """
        tokenizer, service = MessageTokenizer(), prompt_service()
        for variant in VARIANTS:
            for case in load_cases():
                with self.subTest(variant=variant, case=case['id']):
                    prompt = variant_prompt(variant, tokenizer, service, case['text'])
                    self.assertEqual(prompt.count(case['text']), 1)
                    self.assertNotIn(case.get('reference_ko', case.get('reference')), prompt)

    def test_current_and_qwen_baselines_match_existing_builders(self):
        """기준 조건은 현재 API와 기존 후보 평가의 프롬프트를 그대로 사용한다.

        :return: None.
        :raises AssertionError: 기준 프롬프트를 임의로 바꿀 때.
        """
        tokenizer, service = MessageTokenizer(), prompt_service()
        for case in load_cases():
            text = case['text']
            self.assertEqual(variant_prompt('koreanlm-current', tokenizer, service, text),
                             service._build_prompt(text, ''))
            self.assertEqual(variant_prompt('qwen-base', tokenizer, service, text), build_prompt(
                tokenizer, {'original_repository': 'Qwen/Qwen3-4B-Instruct-2507'}, text))

    def test_glossary_uses_only_matching_terms_and_preserves_relations(self):
        """용어 조건은 실제 원문 용어만 붙이고 복합 용어와 연산·문장 종류 보존을 지시한다.

        :return: None.
        :raises AssertionError: 무관한 용어나 겹친 단순 용어가 들어가거나 보존 지시가 없을 때.
        """
        prompt = variant_prompt('qwen-glossary', MessageTokenizer(), prompt_service(),
                                'The centripetal force is zero.')
        self.assertIn('centripetal force=구심력', prompt)
        self.assertNotIn('force=힘', prompt)
        self.assertNotIn('mass=질량', prompt)
        self.assertIn('operations, inequalities and corrections', prompt)
        self.assertIn('statement or question', prompt)

    def test_wrong_operator_is_preserved_as_a_semantic_review_case(self):
        """기호를 유지해도 연산이 틀린 출력은 진단 통과와 별개로 원시 검토 대상에 남긴다.

        :return: None.
        :raises AssertionError: 원시 출력이 바뀌거나 검사 통과를 의미 정답으로 기록할 때.
        """
        output = 'vR와 v0를 곱하고 u1을 m로 나눕니다.'
        result = diagnose_output(prompt_service(),
                                 'We get vR minus v0 equals u1 over m.', output, 'stop')
        self.assertEqual(result['validation_problems'], [])
        self.assertIsNone(result['raw_validator_rejection_reason'])
        self.assertEqual(output, 'vR와 v0를 곱하고 u1을 m로 나눕니다.')
        self.assertNotIn('semantic_pass', result)
        self.assertIn('token_limit', diagnose_output(prompt_service(), 'Hello.', '안녕하세요.',
                                                   'length')['validation_problems'])

    def test_refined_terms_are_isolated_from_baseline_and_api_glossary(self):
        """보완 용어는 원문에 있을 때만 사용하고 기준 조건과 API 용어집을 바꾸지 않는다.

        :return: None.
        :raises AssertionError: 실험 간 용어가 누출되거나 무관한 용어가 입력될 때.
        """
        service, tokenizer = prompt_service(), MessageTokenizer()
        original = list(service.glossary)
        for variant in ('qwen-glossary-v2', 'qwen-relations-v2'):
            prompt = variant_prompt(variant, tokenizer, service, 'The rocket has constant speed.')
            self.assertIn('rocket=로켓', prompt)
            self.assertIn('constant=상수', prompt)
            self.assertNotIn('magnitude=크기', prompt)
            self.assertNotIn('rocket=로켓', variant_prompt(
                variant, tokenizer, service, 'Velocity is constant.'))
        self.assertEqual(service.glossary, original)
        self.assertNotIn('rocket=로켓', variant_prompt(
            'qwen-glossary', tokenizer, service, 'The rocket has constant speed.'))

    def test_missing_shard_is_rejected_before_model_loading(self):
        """부분 가중치 인덱스의 누락 샤드는 모델 로딩 전에 거부한다.

        :return: None.
        :raises AssertionError: 누락된 샤드를 무시할 때.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({'quantization': {'bits': 4}}))
            (path / 'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'weight': 'missing.safetensors'}}))
            with self.assertRaises(FileNotFoundError):
                verify_koreanlm(path)

    def test_unknown_variant_is_rejected(self):
        """알 수 없는 조건을 다른 모델 프롬프트로 조용히 대체하지 않는다.

        :return: None.
        :raises AssertionError: 알 수 없는 실험 조건을 허용할 때.
        """
        with self.assertRaises(ValueError):
            variant_prompt('unknown', MessageTokenizer(), prompt_service(), 'Hello.')


if __name__ == '__main__':
    unittest.main()
