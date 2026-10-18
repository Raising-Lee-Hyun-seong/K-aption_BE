"""Qwen API 기본 선택·문맥 분리·대상 보존 재시도와 CPU 채팅 구성을 검증한다."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.main import create_translation_service
from app.services.translation_llamacpp import LlamaCppTranslationService
from app.services.translation_qwen import QwenTranslationService, render_chatml
from scripts.download_cpu_model import load_manifest
from scripts.prepare_qwen import prepare

ROOT = Path(__file__).resolve().parents[1]


class QwenTranslationTests(unittest.TestCase):
    """모델 생성 대역으로 실제 API에 연결된 Qwen 계약을 검사한다."""

    def service(self):
        """공통 설정과 Qwen 용어를 갖춘 모델 없는 서비스를 만든다.

        :return: 실제 프롬프트·검증·재시도 구현을 사용하는 Qwen 서비스 대역.
        """
        service = QwenTranslationService.__new__(QwenTranslationService)
        service.model_family = 'qwen3'
        service._configure_text(ROOT / 'templates/korean.json')
        service._configure_qwen()
        service.tokenizer = SimpleNamespace(encode=Mock(return_value=[1]),
                                           apply_chat_template=Mock(side_effect=render_chatml))
        return service

    def test_default_mlx_factory_selects_qwen_and_keeps_legacy_explicit(self):
        """기본 Mac 모델은 Qwen이며 기존 KoreanLM은 명시적으로 선택하는지 확인한다.

        :return: None.
        """
        model, template = Path('/model'), ROOT / 'templates/korean.json'
        for family, target in (('qwen3', 'app.services.translation_qwen.QwenTranslationService'),
                               ('koreanlm', 'app.main.TranslationService')):
            with self.subTest(family=family), patch.dict('os.environ', {
                    'KAPTION_TRANSLATION_BACKEND': 'mlx', 'KAPTION_TRANSLATION_MODEL': family,
                    'KAPTION_TRANSLATION_DEVICE': 'auto'}), patch(target) as constructor:
                create_translation_service(model, template)
                constructor.assert_called_once_with(model, template, device='auto')
        with patch.dict('os.environ', {'KAPTION_TRANSLATION_MODEL': 'unknown'}):
            with self.assertRaisesRegex(ValueError, 'Translation model'):
                create_translation_service(model, template)

    def test_context_is_only_in_bounded_system_message(self):
        """앞 문맥은 시스템에만 넣고 현재 원문은 사용자 메시지에 한 번 넣는다.

        :return: None.
        """
        service = self.service()
        context = 'old context ' * 40
        source = 'And then we have dmR over mR.'
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
        result = service.translate('The sled will eventually come to a stop.', 'Earlier English.', 128)
        self.assertEqual(result['translation'], '썰매는 결국 멈춥니다.')
        self.assertEqual(service._generate_output.call_count, 2)
        self.assertIn('Earlier English.', service._generate_output.call_args_list[0].args[0])
        self.assertNotIn('Earlier English.', service._generate_output.call_args_list[1].args[0])

    def test_qwen_cpu_uses_chatml_without_extra_bos_and_closes_model(self):
        """Qwen GGUF 구조에 맞는 템플릿·BOS 정책과 CPU 해제를 적용한다.

        :return: None.
        """
        model = Mock(metadata={'general.architecture': 'qwen3'})
        constructor = Mock(return_value=model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            path.write_bytes(b'fixture')
            with patch.dict('sys.modules', {'llama_cpp': SimpleNamespace(Llama=constructor)}):
                service = LlamaCppTranslationService(path, ROOT / 'templates/korean.json')
                prompt = service._build_prompt('The rocket moves.', '')
                service.tokenizer.encode(prompt)
                model.tokenize.assert_called_once_with(prompt.encode(), add_bos=False, special=True)
                self.assertIn('rocket=로켓', prompt)
                self.assertEqual(service.model_family, 'qwen3')
                service.close()
        model.close.assert_called_once()

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

    def test_wrong_mlx_architecture_is_rejected_before_native_loading(self):
        """기존 모델 경로를 Qwen으로 잘못 지정하면 모델 로딩 전에 명시적으로 거부한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text('{"model_type":"llama"}')
            with self.assertRaisesRegex(ValueError, 'Qwen3'):
                QwenTranslationService(path, ROOT / 'templates/korean.json')

    def test_cpu_manifest_pins_public_qwen_artifact(self):
        """공개 CPU 파일의 모델·리비전·크기·SHA256을 고정한 명세를 검사한다.

        :return: None.
        """
        manifest = load_manifest(ROOT / 'docker/qwen3-model.json')
        self.assertEqual(manifest['model'], 'Qwen/Qwen3-4B-Instruct-2507')
        self.assertEqual(manifest['quantization'], 'Q8_0')
        self.assertEqual(manifest['bytes'], 4280403520)
        self.assertEqual(manifest['revision'], 'e6f794d44f9395d0184a966c27b5ae99ea356fcb')

    def test_mlx_preparation_reuses_verified_files_and_rejects_traversal(self):
        """준비된 MLX 파일을 네트워크 없이 검증하고 파일명 경로 이탈을 거부한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model = path / 'model'
            model.mkdir()
            payload = b'fixture'
            (model / 'model.safetensors').write_bytes(payload)
            source = {'repository': 'mlx-community/model', 'revision': 'fixed', 'files': [
                {'name': 'model.safetensors', 'bytes': len(payload),
                 'sha256': hashlib.sha256(payload).hexdigest()}]}
            manifest = path / 'manifest.json'
            manifest.write_text(json.dumps(source))
            with patch('scripts.download_cpu_model.urlopen') as download:
                prepare(model, manifest)
                download.assert_not_called()
            source['files'][0]['name'] = '../outside'
            manifest.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'flat filenames'):
                prepare(model, manifest)
