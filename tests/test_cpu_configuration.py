"""공통 Compose의 CPU 강제·내부 모델 서버·메모리 예산과 런타임 의존성을 검증한다."""

from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


class CpuConfigurationTests(unittest.TestCase):
    """실기기 검증을 대신하지 않고 배포 설정의 필수 제약을 회귀 검사한다."""

    def test_native_platforms_and_internal_cpu_server(self):
        """플랫폼 강제 없이 CPU 서버를 내부 네트워크에만 제공한다.

        :return: None.
        """
        config = yaml.safe_load((ROOT / 'compose.yaml').read_text())
        self.assertTrue(config['networks']['inference']['internal'])
        for service in config['services'].values():
            self.assertNotIn('platform', service)
            self.assertNotIn('gpus', service)
        server = config['services']['translator']
        self.assertNotIn('ports', server)
        args = server['command']
        for option, value in (('--device', 'none'), ('--n-gpu-layers', '0'),
                              ('--parallel', '1'), ('--ctx-size', '2048'), ('--cache-ram', '0')):
            self.assertEqual(args[args.index(option) + 1], value)
        self.assertIn('--no-kv-offload', args)
        self.assertIn('--no-op-offload', args)
        self.assertIn('--no-repack', args)

    def test_decimal_memory_budget_and_health_dependency(self):
        """GB 예산을 정확한 bytes로 제한하고 모델 준비 뒤에 backend를 시작한다.

        :return: None.
        """
        services = yaml.safe_load((ROOT / 'compose.yaml').read_text())['services']
        self.assertEqual(services['backend']['mem_limit'] + services['translator']['mem_limit'], 6500000000)
        self.assertEqual(services['backend']['depends_on']['translator']['condition'], 'service_healthy')
        self.assertEqual(services['translator']['depends_on']['model-init']['condition'],
                         'service_completed_successfully')

    def test_backend_has_no_local_inference_dependencies(self):
        """backend 의존성에 GPU·로컬 번역 엔진이 다시 추가되지 않게 검사한다.

        :return: None.
        """
        requirements = '\n'.join((ROOT / name).read_text() for name in
                                 ('requirements-api.txt', 'requirements-cpu.txt', 'requirements-stt.txt'))
        packages = [line.strip().split('=')[0].split('>')[0].split('<')[0]
                    for line in requirements.splitlines() if line and not line.startswith(('#', '-r'))]
        for name in ('mlx', 'mlx-lm', 'mlx-metal', 'torch', 'llama-cpp-python', 'bitsandbytes'):
            self.assertNotIn(name, packages)
