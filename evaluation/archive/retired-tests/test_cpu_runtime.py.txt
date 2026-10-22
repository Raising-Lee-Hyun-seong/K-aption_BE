"""CPU 장치 선택 순서·서버 전달·벤치마크 결과와 메모리 단위를 검증한다."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.main import app, lifespan, MODEL_DIR, TEMPLATE_PATH
from app.services.translation import TranslationService
from scripts import benchmark_cpu


class CpuRuntimeTests(unittest.TestCase):
    """기본 Mac 실행을 유지하며 CPU 선택을 모델 로드 이전에 적용하는지 확인한다."""

    def load_service(self, device):
        """작은 모델 파일 대역과 MLX 대역으로 초기화 호출 순서를 수집한다.

        :param device: auto 또는 cpu 장치 선택값.
        :return: 초기화 중 MLX 장치 설정·모델 로드의 호출 순서 목록.
        """
        calls = []
        cpu = object()

        def set_device(device):
            """CPU 설정 호출을 기록하고 장치 인자를 확인한다.

            :param device: MLX CPU 장치 대역.
            :return: None.
            """
            self.assertIs(device, cpu)
            calls.append('cpu')

        def load(path):
            """모델 로딩 호출을 기록하고 모델·토크나이저 대역을 반환한다.

            :param path: 검증된 4bit 모델 디렉터리 문자열.
            :return: 모델·토크나이저 대역 튜플.
            """
            calls.append('load')
            return Mock(), Mock()

        core = SimpleNamespace(cpu=cpu, set_default_device=set_device)
        modules = {'mlx': SimpleNamespace(core=core), 'mlx.core': core,
                   'mlx_lm': SimpleNamespace(load=load),
                   'mlx_lm.sample_utils': SimpleNamespace(make_sampler=Mock())}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'quantization': {'bits': 4}}))
            (root / 'model.safetensors').write_bytes(b'fixture')
            (root / 'template.json').write_text('{}')
            with patch.dict(sys.modules, modules):
                TranslationService(root, root / 'template.json', device=device)
        return calls

    def test_cpu_is_selected_before_loading_model_and_generation_stream(self):
        """CPU 강제 설정이 mlx_lm 로딩 전에 적용되는지 확인한다.

        :return: None.
        """
        self.assertEqual(self.load_service('cpu'), ['cpu', 'load'])

    def test_auto_keeps_existing_default_device(self):
        """기본 auto 실행은 기존 MLX 기본 장치를 덮어쓰지 않는지 검사한다.

        :return: None.
        """
        self.assertEqual(self.load_service('auto'), ['load'])

    def test_unknown_device_is_rejected_before_model_access(self):
        """잘못된 장치 설정을 모델 경로 확인 전에 명시적으로 거부한다.

        :return: None.
        """
        with self.assertRaisesRegex(ValueError, 'auto or cpu'):
            TranslationService(Path('/not-a-model'), Path('/not-a-template'), device='cuda')


class CpuLifespanTests(unittest.IsolatedAsyncioTestCase):
    """서버 환경 설정이 실제 번역 서비스 초기화에 전달되는지 검사한다."""

    async def test_cpu_configuration_reaches_translation_service(self):
        """CPU 환경 설정을 서비스에 전달하고 서버 종료 시 공유 참조를 해제한다.

        :return: None.
        """
        manager = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        with patch.dict('os.environ', {'KAPTION_TRANSLATION_DEVICE': 'cpu',
                                      'KAPTION_TRANSLATION_MODEL': 'koreanlm',
                                      'KAPTION_TRANSLATION_BACKEND': 'mlx'}), \
                patch('app.main.JobManager', return_value=manager), \
                patch('app.main.TranslationService') as service, \
                patch('app.main.TranscriptionService'):
            async with lifespan(app):
                service.assert_called_once_with(MODEL_DIR, TEMPLATE_PATH, device='cpu')
                self.assertIs(manager.translator, service.return_value)
        manager.close.assert_awaited_once()
        self.assertIsNone(app.state.translator)

    async def test_native_model_is_closed_only_after_worker_shutdown(self):
        """실행 중 추론을 가진 작업자가 종료된 뒤에만 네이티브 모델을 해제한다.

        :return: None.
        """
        order = []

        async def stop_worker():
            """추론 종료를 기다린 작업자 정리를 기록한다.

            :return: None.
            """
            order.append('worker')

        def close_model():
            """작업자 종료 여부를 확인한 뒤 모델 해제를 기록한다.

            :return: None.
            """
            self.assertEqual(order, ['worker'])
            order.append('model')

        manager = SimpleNamespace(start=AsyncMock(), close=stop_worker)
        translator = SimpleNamespace(close=close_model)
        with patch('app.main.JobManager', return_value=manager), \
                patch('app.main.create_translation_service', return_value=translator), \
                patch('app.main.TranscriptionService'):
            async with lifespan(app):
                self.assertEqual(order, [])
        self.assertEqual(order, ['worker', 'model'])


class CpuBenchmarkTests(unittest.TestCase):
    """실제 모델을 로드하지 않고 측정 보고서의 성공·실패·단위 계약을 검증한다."""

    def test_probe_mode_never_loads_model_and_converts_rss_units(self):
        """연산 검사 모드는 모델을 로드하지 않고 Linux·Mac 최대 RSS 단위를 맞춘다.

        :return: None.
        """
        for system, rss in (('Linux', 1024 ** 2), ('Darwin', 1024 ** 3)):
            with self.subTest(system=system), \
                    patch.object(benchmark_cpu, 'version', return_value='fixture'), \
                    patch.object(benchmark_cpu, 'probe_cpu', return_value={'device': 'cpu'}), \
                    patch.object(benchmark_cpu.platform, 'system', return_value=system), \
                    patch.object(benchmark_cpu.resource, 'getrusage', return_value=SimpleNamespace(ru_maxrss=rss)), \
                    patch.object(benchmark_cpu, 'TranslationService') as service:
                result = benchmark_cpu.benchmark(Path('/unused'), 64, probe_only=True)
            self.assertEqual(result['peak_rss_mib'], 1024)
            self.assertEqual(result['mode'], 'probe')
            service.assert_not_called()

    def test_translation_rejection_is_measured_and_not_reported_as_success(self):
        """번역 형식 검증 실패도 시간을 기록하며 성공 출력으로 보고하지 않는다.

        :return: None.
        """
        with patch.object(benchmark_cpu, 'version', return_value='fixture'), \
                patch.object(benchmark_cpu, 'probe_cpu', return_value={'device': 'cpu'}), \
                patch.object(benchmark_cpu, 'TranslationService') as service:
            service.return_value.translate.side_effect = ValueError('invalid translation')
            result = benchmark_cpu.benchmark(Path('/model'), 32)
        self.assertEqual(result['translation']['status'], 'rejected')
        self.assertGreaterEqual(result['translation_wall_seconds'], 0)
        self.assertGreaterEqual(result['load_seconds'], 0)
        service.assert_called_once_with(Path('/model'), benchmark_cpu.ROOT / 'templates/korean.json', device='cpu')

    def test_llamacpp_report_uses_cpu_engine_and_releases_model(self):
        """GGUF 측정은 MLX를 호출하지 않고 CPU 스레드를 전달하며 모델을 닫는다.

        :return: None.
        """
        with patch.object(benchmark_cpu, 'version', return_value='fixture'), \
                patch.object(benchmark_cpu, 'probe_llamacpp', return_value={'device': 'cpu'}), \
                patch.object(benchmark_cpu, 'TranslationService') as mlx, \
                patch.object(benchmark_cpu, 'LlamaCppTranslationService') as native:
            native.return_value.translate.return_value = {'translation': '속도는 일정합니다.',
                                                         'generated_tokens': 13}
            result = benchmark_cpu.benchmark(Path('/model.gguf'), 64, backend='llama-cpp', threads=3)
        self.assertEqual(result['backend'], 'llama-cpp')
        self.assertEqual(result['translation']['status'], 'accepted')
        self.assertEqual(result['threads'], 3)
        native.assert_called_once_with(Path('/model.gguf'), benchmark_cpu.ROOT / 'templates/korean.json', threads=3)
        native.return_value.close.assert_called_once()
        mlx.assert_not_called()

    def test_cli_saves_report_and_returns_rejected_exit_code(self):
        """측정 JSON을 명시한 파일에 저장하고 번역 검증 실패를 종료 코드 2로 반환한다.

        :return: None.
        """
        report = {'translation': {'status': 'rejected'}, 'peak_rss_mib': 2048}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'nested/report.json'
            with patch.object(sys, 'argv', ['benchmark_cpu.py', '--output', str(target)]), \
                    patch.object(benchmark_cpu, 'benchmark', return_value=report), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark_cpu.main(), 2)
            self.assertEqual(json.loads(target.read_text()), report)

    def test_cli_error_and_invalid_token_limit_have_failure_exit_codes(self):
        """실행 오류는 JSON 오류와 종료 코드 1, 잘못된 출력 한도는 실행 전 오류로 표시한다.

        :return: None.
        """
        output = io.StringIO()
        with patch.object(sys, 'argv', ['benchmark_cpu.py']), \
                patch.object(benchmark_cpu, 'benchmark', side_effect=RuntimeError('unsupported CPU')), \
                redirect_stdout(output):
            self.assertEqual(benchmark_cpu.main(), 1)
        self.assertEqual(json.loads(output.getvalue())['error'], 'unsupported CPU')
        with patch.object(sys, 'argv', ['benchmark_cpu.py', '--max-tokens', '0']), \
                patch.object(benchmark_cpu, 'benchmark') as run, \
                patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                benchmark_cpu.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
