"""모델 실행 없이 후보 가중치의 손상 검출·전용 번역 입력·원시 실패 보존을 검증한다."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from transformers.utils.chat_template_utils import _compile_jinja_template
from scripts.evaluate_candidates import build_prompt, configure_stops, inspect_output, load_cases, model_overrides, verify_file


class TemplateTokenizer:
    """공개된 실제 채팅 템플릿을 토큰화 없이 렌더링하는 테스트용 객체다."""

    def __init__(self, metadata):
        """모델의 공개 템플릿과 특수 토큰 설정을 저장한다.

        :param metadata: chat_template과 tokenizer_config를 담은 후보 메타데이터.
        :return: None.
        """
        self.config = metadata['tokenizer_config']
        self.template = _compile_jinja_template(metadata.get('chat_template') or self.config['chat_template'])

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        """실제 템플릿의 형식·언어 제약을 적용해 프롬프트를 만든다.

        :param messages: user 역할과 번역할 입력을 담은 메시지 목록.
        :param tokenize: 테스트에서는 거짓이어야 하는 토큰화 옵션.
        :param add_generation_prompt: 모델 응답 시작 표식을 추가할지 여부.
        :return: 렌더링한 채팅 문자열.
        :raises AssertionError: 토큰화가 요청되면 발생한다.
        """
        assert not tokenize
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, **self.config)


class CandidateEvaluationTests(unittest.TestCase):
    """평가 전에 손상 파일을 막고 모델별 올바른 입력과 진단의 한계를 확인한다."""

    def test_same_size_corruption_is_rejected_before_loading(self):
        """파일 크기가 같아도 SHA256이 다른 가중치는 로딩 전 검증에서 거부한다.

        :return: None.
        :raises AssertionError: 손상된 파일이 정상 파일로 통과하면 발생한다.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            expected = {'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
            path.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                verify_file(path, expected)

    def test_real_templates_preserve_all_sources_and_korean_target(self):
        """두 실제 템플릿에 22개 원문이 중복·손실 없이 한국어 대상으로 들어간다.

        :return: None.
        :raises AssertionError: 언어·템플릿·원문 보존 조건이 맞지 않으면 발생한다.
        """
        root = Path(__file__).resolve().parents[1]
        sources = json.loads((root / 'evaluation/model_candidates_sources.json').read_text())['models']
        models = {row['repository']: row for row in sources}
        candidates = json.loads((root / 'evaluation/model_candidates.json').read_text())['selected']
        cases = load_cases()
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({row['id'] for row in cases}), 22)
        for candidate in candidates:
            tokenizer = TemplateTokenizer(models[candidate['repository']])
            for case in cases:
                prompt = build_prompt(tokenizer, candidate, case['text'])
                self.assertEqual(prompt.count(case['text']), 1)
                self.assertIn('Korean', prompt)
                self.assertNotIn(case.get('reference_ko', case.get('reference')), prompt)

    def test_truncated_and_invented_units_are_diagnostics_not_rewrites(self):
        """잘린 응답·추가 단위를 진단하되 원시 응답과 의미 오류는 따로 보존한다.

        :return: None.
        :raises AssertionError: 진단이 원시 응답을 바꾸거나 오류를 놓치면 발생한다.
        """
        output = '초기 질량은 2m0 킬로그램이다.'
        problems = inspect_output('The initial mass is 2m0.', output, 'length')
        self.assertIn('token_limit', problems)
        self.assertIn('invented_unit', problems)
        self.assertEqual(output, '초기 질량은 2m0 킬로그램이다.')
        # 문자열 보존 검사만으로 속력·속도 관계의 역전을 검출하지 못함을 명시한다.
        self.assertEqual(inspect_output('Speed is the magnitude of velocity.',
                                        '속도는 속력의 크기이다.', 'stop'), [])

    def test_gemma_turn_end_is_added_without_replacing_eos(self):
        """실제 실행에서 반복된 턴 종료 표식을 기존 EOS와 함께 중단 조건으로 보존한다.

        :return: None.
        :raises AssertionError: Gemma 종료 토큰이 누락되거나 기존 EOS가 제거되면 발생한다.
        """
        class Stops:
            """EOS ID와 종료 표식 사전만 제공하는 토크나이저 대역이다."""
            eos_token_ids = None

            def get_vocab(self):
                """검증된 Gemma 종료 표식의 ID를 제공한다.

                :return: 종료 표식과 토큰 ID 106의 매핑.
                """
                return {'<end_of_turn>': 106}

            def add_eos_token(self, token):
                """지정한 표식의 실제 ID를 종료 조건에 추가한다.

                :param token: 사전에 있는 종료 표식 문자열.
                :return: None.
                """
                self.eos_token_ids.add(self.get_vocab()[token])

        tokenizer = Stops()
        tokenizer.eos_token_ids = {1}
        result = configure_stops(tokenizer, {'original_repository': 'google/translategemma-4b-it'})
        self.assertEqual(result['before'], [1])
        self.assertEqual(result['after'], [1, 106])

    def test_new_rope_factor_is_not_silently_lost_or_conflicted(self):
        """실제 파일에 명시된 위치 배율을 전달하며 원본 불변성과 충돌 거부를 확인한다.

        :return: None.
        :raises AssertionError: 배율이 유실되거나 원본이 바뀌거나 충돌을 놓치면 발생한다.
        """
        root = Path(__file__).resolve().parents[1]
        models = json.loads((root / 'evaluation/model_candidates_sources.json').read_text())['models']
        original = next(row['config'] for row in models if row['repository'] == 'mlx-community/translategemma-4b-it-4bit')
        candidate = {'original_repository': 'google/translategemma-4b-it'}
        result = model_overrides(candidate, original)
        self.assertEqual(result['text_config']['rope_scaling'], {'type': 'linear', 'factor': 8.0})
        self.assertIsNone(original['text_config']['rope_scaling'])
        result['text_config']['rope_scaling']['factor'] = 2
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            model_overrides(candidate, result)


if __name__ == '__main__':
    unittest.main()
