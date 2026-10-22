"""모델 대역으로 llama.cpp CPU 선택·토큰 경계·공통 번역 재시도를 검증한다."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.main import create_translation_service
from app.services.translation_llamacpp import LlamaCppTranslationService, LlamaTokenizer


class TokenModel:
    """부분 UTF-8 토큰과 EOS·생성기 종료를 제어하는 CPU 모델 대역이다."""

    def __init__(self, pieces, sequence):
        """토큰 바이트와 생성 순서를 준비한다.

        :param pieces: EOS를 제외한 토큰 ID와 바이트의 대응 사전.
        :param sequence: 차례로 제공할 생성 토큰 ID 목록.
        :return: None.
        """
        self.pieces, self.sequence = pieces, sequence
        self.closed = False
        self.reset_count = 0
        self.options = None

    def reset(self):
        """요청별 KV 초기화 횟수를 기록한다.

        :return: None.
        """
        self.reset_count += 1

    def tokenize(self, text, add_bos, special):
        """BOS 포함 정책의 토큰 ID 대역을 제공한다.

        :param text: UTF-8로 인코딩한 프롬프트 바이트.
        :param add_bos: BOS 포함 여부.
        :param special: 특수 토큰 인식 여부.
        :return: 고정 프롬프트 토큰 ID 목록.
        """
        return [1, 5, 6]

    def generate(self, tokens, **options):
        """선택한 토큰을 제공하고 중간 종료에서도 닫힘을 기록한다.

        :param tokens: 프롬프트 토큰 ID 목록.
        :param options: 탐욕 생성과 반복 패널티 등 샘플링 설정.
        :return: 토큰 ID를 제공하는 생성기.
        :yield: 다음 생성 토큰 ID 정수.
        """
        self.options = options
        try:
            yield from self.sequence
        finally:
            self.closed = True

    def detokenize(self, tokens):
        """현재까지의 토큰을 부분 UTF-8 바이트를 포함해 연결한다.

        :param tokens: EOS를 제외한 생성 토큰 ID 목록.
        :return: 디코딩 전 바이트.
        """
        return b''.join(self.pieces[token] for token in tokens)

    def token_eos(self):
        """대역의 EOS ID를 반환한다.

        :return: EOS 토큰 ID 99.
        """
        return 99


class LlamaCppTranslationTests(unittest.TestCase):
    """기존 API 계약을 유지하며 CPU 네이티브 생성 경계를 처리하는지 확인한다."""

    def service(self, model):
        """모델 로딩 없이 실제 공통 텍스트 설정을 가진 서비스를 구성한다.

        :param model: 생성·토큰화 메서드를 제공하는 대역.
        :return: 초기화한 LlamaCppTranslationService 대역.
        """
        service = LlamaCppTranslationService.__new__(LlamaCppTranslationService)
        service.model, service.tokenizer = model, LlamaTokenizer(model)
        service._configure_text(Path(__file__).resolve().parents[1] / 'templates/korean.json')
        return service

    def test_partial_utf8_is_buffered_until_complete_and_eos_is_not_text(self):
        """UTF-8 중간 바이트를 이상 문자로 오인하지 않고 EOS를 본문에 넣지 않는다.

        :return: None.
        """
        korean = '속도'.encode()
        model = TokenModel({10: korean[:1], 11: korean[1:4], 12: korean[4:]}, [10, 11, 12, 99])
        result = self.service(model)._generate_output('prompt', 10)
        self.assertEqual(result, {'text': '속도', 'tokens': 4, 'finish_reason': 'stop', 'stop_reason': None})
        self.assertEqual(model.reset_count, 1)
        self.assertEqual(model.options, {'temp': 0.0, 'top_k': 1, 'repeat_penalty': 1.0})
        self.assertTrue(model.closed)

    def test_token_limit_with_partial_utf8_is_rejected(self):
        """출력 한도에서 끝난 불완전 UTF-8은 길이·이상 문자 정보로 보존한다.

        :return: None.
        """
        model = TokenModel({10: '속'.encode()[:1]}, [10, 10])
        result = self.service(model)._generate_output('prompt', 1)
        self.assertEqual(result['tokens'], 1)
        self.assertEqual(result['finish_reason'], 'length')
        self.assertEqual(result['stop_reason'], 'invalid_character')
        self.assertTrue(model.closed)

    def test_prompt_leak_closes_generator_before_remaining_tokens(self):
        """공통 지시문 유출 검사가 발동하면 남은 토큰 생성 전에 닫는다.

        :return: None.
        """
        model = TokenModel({10: '참고 문맥:'.encode(), 11: b'extra'}, [10, 11, 99])
        result = self.service(model)._generate_output('prompt', 10)
        self.assertEqual(result['tokens'], 1)
        self.assertEqual(result['stop_reason'], 'prompt_leak')
        self.assertTrue(model.closed)

    def test_existing_validation_retries_without_context(self):
        """출력 이상은 기존 정책대로 문맥 없이 한 번 재시도하고 기존 응답을 반환한다.

        :return: None.
        """
        service = self.service(TokenModel({}, []))
        service._generate_output = Mock(side_effect=[
            {'text': 'bad', 'tokens': 1, 'finish_reason': None, 'stop_reason': 'invalid_character'},
            {'text': '속도는 일정합니다.', 'tokens': 9, 'finish_reason': 'stop', 'stop_reason': None}])
        result = service.translate('The velocity is constant.', 'An earlier English sentence.', 64)
        self.assertEqual(result['translation'], '속도는 일정합니다.')
        self.assertEqual(result['generated_tokens'], 9)
        self.assertEqual(service._generate_output.call_count, 2)
        self.assertIn('An earlier English sentence.', service._generate_output.call_args_list[0].args[0])
        self.assertNotIn('An earlier English sentence.', service._generate_output.call_args_list[1].args[0])

    def test_model_initialization_disables_every_gpu_offload(self):
        """GPU 없이 mmap·2048 문맥·4스레드를 사용하고 명시적 종료로 자원을 반환한다.

        :return: None.
        """
        model = Mock(metadata={'general.architecture': 'llama'})
        constructor = Mock(return_value=model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            path.write_bytes(b'fixture')
            with patch.dict('sys.modules', {'llama_cpp': SimpleNamespace(Llama=constructor)}):
                service = LlamaCppTranslationService(path, Path('templates/korean.json'))
                self.assertEqual(constructor.call_args.kwargs['n_gpu_layers'], 0)
                self.assertFalse(constructor.call_args.kwargs['offload_kqv'])
                self.assertFalse(constructor.call_args.kwargs['op_offload'])
                self.assertEqual(constructor.call_args.kwargs['n_ctx'], 2048)
                self.assertEqual(constructor.call_args.kwargs['n_threads'], 4)
                self.assertTrue(constructor.call_args.kwargs['use_mmap'])
                service.close()
        model.close.assert_called_once()

    def test_missing_model_and_invalid_threads_fail_before_binding_load(self):
        """잘못된 모델 경로·스레드 수를 네이티브 로딩 전에 거부한다.

        :return: None.
        """
        with self.assertRaises(FileNotFoundError):
            LlamaCppTranslationService(Path('/missing.gguf'), Path('/missing.json'))
        with self.assertRaises(ValueError):
            LlamaCppTranslationService(Path('/missing.gguf'), Path('/missing.json'), threads=0)

    def test_backend_factory_is_explicit_and_keeps_default_mlx(self):
        """환경 설정으로만 엔진을 선택하며 알 수 없는 백엔드를 조용히 대체하지 않는다.

        :return: None.
        """
        model, template = Path('/model.gguf'), Path('/template.json')
        with patch.dict('os.environ', {'KAPTION_TRANSLATION_BACKEND': 'llama-cpp', 'KAPTION_CPU_THREADS': '3'}), \
                patch('app.services.translation_llamacpp.LlamaCppTranslationService') as constructor:
            create_translation_service(model, template)
            constructor.assert_called_once_with(model, template, threads=3)
        with patch.dict('os.environ', {'KAPTION_TRANSLATION_BACKEND': 'unknown'}), self.assertRaises(ValueError):
            create_translation_service(model, template)
