"""CPU 비교 도구가 실행 중 작업을 보호하고 실패 후 기존 서버 설정을 복구하는지 검사한다."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_cpu_translation_threads as evaluation


def original_state() -> dict:
    """외부 Docker를 호출하지 않는 4 CPU 서버의 검사용 상태를 만든다.

    :return: CPU 제한·추론 스레드·메모리·이미지·재시작 상태 사전.
    """
    return {'cmd': ['--threads', '4', '--threads-batch', '4'], 'image': 'same-image',
            'nano_cpus': 4_000_000_000, 'memory': 4_000_000_000,
            'memory_swap': 4_000_000_000, 'oom': False, 'restart_count': 0}


class CpuThreadSafetyTests(unittest.TestCase):
    """운영 서버를 재생성하는 평가 경로의 중단·복구·설정 검증을 검사한다."""

    def test_active_job_prevents_server_recreation(self):
        """진행 중인 작업이 있으면 서버를 재생성하지 않는지 검사한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(directory=Path(directory) / 'new')
            with (patch.object(evaluation, 'server_state', return_value=original_state()),
                  patch.object(evaluation, 'docker', return_value=subprocess.CompletedProcess([], 0, '1\n', '')),
                  patch.object(evaluation, 'configure') as configure):
                with self.assertRaisesRegex(RuntimeError, 'Active jobs'):
                    evaluation.control(args)
                configure.assert_not_called()

    def test_failed_measurement_restores_original_cpu_and_saves_audit(self):
        """측정 도중 오류가 나도 4 CPU 복구와 실패 전 감사 기록이 남는지 검사한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(directory=Path(directory) / 'new')
            original = original_state()
            with (patch.object(evaluation, 'server_state', return_value=original),
                  patch.object(evaluation, 'docker', return_value=subprocess.CompletedProcess([], 0, '0\n', '')),
                  patch.object(evaluation, 'run_condition', side_effect=RuntimeError('measurement failed')),
                  patch.object(evaluation, 'configure', return_value={'restored': True}) as configure):
                with self.assertRaisesRegex(RuntimeError, 'measurement failed'):
                    evaluation.control(args)
                configure.assert_called_once_with(4, original)
                report = json.loads((args.directory / 'comparison.json').read_text())
                self.assertEqual(report['restoration'], {'restored': True})
                self.assertEqual(report['pilot'], [])

    def test_cpu_limit_mismatch_cannot_be_reported_as_six_cpu(self):
        """스레드만 바뀌고 실제 CPU 제한이 그대로면 측정 조건으로 수락하지 않는지 검사한다.

        :return: None.
        """
        original = original_state()
        wrong = {**original, 'cmd': ['--threads', '6', '--threads-batch', '6']}
        with (patch.object(evaluation, 'docker', return_value=subprocess.CompletedProcess([], 0, '', '')),
              patch.object(evaluation, 'server_state', return_value=wrong)):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected translator configuration'):
                evaluation.configure(6, original)


if __name__ == '__main__':
    unittest.main()
