"""호스트 프로세스 선택과 footprint 해석 시 VM 누락·RSS 혼동·과거 최대값 혼입을 방지한다."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.macos_docker_memory import docker_processes, physical_footprints, MacDockerMemorySampler


class MacDockerMemoryTests(unittest.TestCase):
    """실제 프로세스·파일 조회 없이 메모리 수집 범위를 검증한다."""

    def test_docker_process_selection_includes_only_verified_vm(self):
        """같은 이름의 다른 VM은 제외하고 Docker·확인된 VM만 선택한다.

        :return: None.
        """
        listing = ('1 100 /Applications/Docker.app/Contents/MacOS/com.docker.backend\n'
                   '2 200 /System/com.apple.Virtualization.VirtualMachine\n'
                   '3 300 /System/com.apple.Virtualization.VirtualMachine\n'
                   '4 400 /Applications/Editor.app/helper\n'
                   '5 500 /Applications/Docker.app/Docker Desktop\n')
        rows = docker_processes(listing, 2)
        self.assertEqual([row['pid'] for row in rows], [1, 2, 5])
        self.assertEqual(rows[1]['rss_bytes'], 200 * 1024)

    def test_current_physical_footprint_excludes_lifetime_peak(self):
        """현재 물리 footprint를 사용하고 과거 peak·요약 합계를 중복 더하지 않는다.

        :return: None.
        """
        output = ('VM [2]: 64-bit    Footprint: 4000 B (16384 bytes per page)\n'
                  '    phys_footprint: 4000 B\n    phys_footprint_peak: 9000 B\n'
                  'Docker Desktop [5]: 64-bit    Footprint: 500 B\n'
                  'Summary Footprint: 4500 B\n')
        self.assertEqual(physical_footprints(output), {2: 4000, 5: 500})

    def test_vm_without_docker_ownership_is_rejected(self):
        """무관한 VM의 footprint를 Docker 사용량으로 보고하지 않는다.

        :return: None.
        """
        sampler = MacDockerMemorySampler(2)
        with patch('scripts.macos_docker_memory.subprocess.run',
                   return_value=SimpleNamespace(stdout='n/other/VirtualMachine.raw\n')):
            with self.assertRaises(RuntimeError):
                sampler.start()
        self.assertFalse(sampler.close()['attribution']['verified'])

    def test_sampled_host_memory_does_not_prove_goal(self):
        """표본 합계가 작아도 호스트 전체·Windows 메모리 목표를 합격 처리하지 않는다.

        :return: None.
        """
        sampler = MacDockerMemorySampler(2)
        sampler.samples = [{'sum_physical_footprint_bytes': 4500}]
        result = sampler.close()
        self.assertEqual(result['sampled_process_footprint_peak_bytes'], 4500)
        self.assertEqual(result['host_memory_goal'], 'unverified')
        sampler.errors.append({'type': 'TimeoutExpired'})
        self.assertFalse(sampler.close()['complete'])

    def test_unreadable_helper_preserves_partial_vm_sample(self):
        """특권 helper 조회 실패를 누락으로 기록하면서 실제 VM 표본은 보존한다.

        :return: None.
        """
        sampler = MacDockerMemorySampler(2)
        sampler.started = 0
        listing = ('2 100 /System/com.apple.Virtualization.VirtualMachine\n'
                   '3 20 /Library/PrivilegedHelperTools/com.docker.vmnetd\n')
        footprint = 'VM [2]: 64-bit    Footprint: 4000 B\n'
        with patch('scripts.macos_docker_memory.subprocess.run', side_effect=[
            SimpleNamespace(stdout=listing), SimpleNamespace(stdout=footprint)
        ]), patch.object(sampler.stop, 'is_set', side_effect=[False, True]), \
                patch.object(sampler.stop, 'wait'):
            sampler._run()
        result = sampler.close()
        self.assertEqual(result['sampled_process_footprint_peak_bytes'], 4000)
        self.assertEqual(result['samples'][0]['processes'][1]['physical_footprint_bytes'], None)
        self.assertEqual(result['errors'][0]['missing_pids'], [3])
        self.assertFalse(result['complete'])
