"""CPU 통합 측정에서 메모리 단위·불완전 표본·호스트 목표 오판을 방지한다."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import httpx

from scripts.benchmark_cpu import DockerSampler, memory_sample, timing_logs, pipeline_succeeded, run_media


class InterruptedEvents(httpx.SyncByteStream):
    """실제 SSE 전송 도중 연결이 끊기는 시험 응답을 제공한다."""

    def __iter__(self):
        """진행 이벤트 뒤에 연결 종료를 발생시킨다.

        :return: SSE bytes를 제공하는 생성기.
        :yield: 진행 상태 한 개를 담은 SSE 프레임 bytes.
        :raises httpx.RemoteProtocolError: 첫 프레임을 보낸 뒤 연결이 끊길 때.
        """
        yield b'data: {"status":"running"}\n\n'
        raise httpx.RemoteProtocolError('Connection interrupted')


class CpuBenchmarkTests(unittest.TestCase):
    """실제 Docker 연결 없이 원시 통계 해석과 측정 범위 표시를 확인한다."""

    def test_cgroup_usage_and_working_set_are_separate(self):
        """파일 캐시 포함 사용량과 비활성 파일 캐시 제외량을 구분한다.

        :return: None.
        """
        result = memory_sample({'memory_stats': {'usage': 1000, 'stats': {'inactive_file': 300},
                                                'limit': 4000},
                                'cpu_stats': {'cpu_usage': {'total_usage': 900}}})
        self.assertEqual(result['cgroup_bytes'], 1000)
        self.assertEqual(result['working_set_bytes'], 700)
        self.assertEqual(result['cpu_total_ns'], 900)
        self.assertIsNone(result['cgroup_max_bytes'])
        self.assertEqual(memory_sample({})['working_set_bytes'], None)

    def test_combined_peak_uses_same_sample_and_never_proves_host_goal(self):
        """서비스별 서로 다른 최대값을 더하지 않고 같은 시점의 합계를 사용한다.

        :return: None.
        """
        sampler = DockerSampler('/unused', ['backend', 'translator'])
        sampler.samples = [{'containers': {'backend': {'working_set_bytes': 100},
                                           'translator': {'working_set_bytes': 900}}},
                           {'containers': {'backend': {'working_set_bytes': 500},
                                           'translator': {'working_set_bytes': 200}}}]
        result = sampler.close()
        self.assertEqual(result['sampled_combined_working_set_peak_bytes'], 1000)
        self.assertEqual(result['host_memory_goal'], 'unverified')
        self.assertTrue(result['complete'])

    def test_memory_pressure_and_cpu_throttling_counters_are_preserved(self):
        """OOM 원인 분석용 파일 회수·페이지 폴트·CPU 제한 카운터를 손실 없이 보존한다.

        :return: None.
        """
        result = memory_sample({
            'memory_stats': {'usage': 1000, 'stats': {
                'anon': 600, 'file': 300, 'pgmajfault': 12,
                'workingset_refault_file': 42, 'pgscan': 55}},
            'cpu_stats': {'throttling_data': {'periods': 10, 'throttled_periods': 2,
                                             'throttled_time': 123}}})
        self.assertEqual(result['memory_details']['anon'], 600)
        self.assertEqual(result['memory_details']['file'], 300)
        self.assertEqual(result['memory_details']['pgmajfault'], 12)
        self.assertEqual(result['memory_details']['workingset_refault_file'], 42)
        self.assertIsNone(result['memory_details']['pswpout'])
        self.assertEqual(result['cpu_throttling']['throttled_time'], 123)
        self.assertIsNone(memory_sample({})['cpu_throttling']['throttled_time'])

    def test_sampling_error_does_not_report_complete_measurement(self):
        """빈 표본·실패 표본은 완전한 메모리 측정으로 표시하지 않는다.

        :return: None.
        """
        sampler = DockerSampler('/unused', ['backend'])
        self.assertFalse(sampler.close()['complete'])
        sampler.samples = [{'containers': {'backend': {'working_set_bytes': 100}}}]
        sampler.errors.append({'type': 'ReadTimeout'})
        self.assertFalse(sampler.close()['complete'])

    def test_incomplete_round_is_not_hidden_by_one_good_sample(self):
        """정상 표본이 하나 있어도 다른 회차의 컨테이너 누락을 완전 측정으로 보지 않는다.

        :return: None.
        """
        sampler = DockerSampler('/unused', ['backend', 'translator'])
        sampler.samples = [{'containers': {'backend': {'working_set_bytes': 100},
                                           'translator': {'working_set_bytes': 200}}},
                           {'containers': {'backend': {'working_set_bytes': 100}}}]
        self.assertFalse(sampler.close()['complete'])

    def test_invalid_sampling_configuration_is_rejected(self):
        """측정 대상이나 간격이 없으면 잘못된 시험 구성을 거부한다.

        :return: None.
        """
        with self.assertRaises(ValueError):
            DockerSampler('/unused', [])
        with self.assertRaises(ValueError):
            DockerSampler('/unused', ['backend'], 0)

    def test_backend_log_frames_keep_only_the_current_job_timing(self):
        """다중화 로그에서 다른 작업·HTTP 로그를 제외하고 STT 실측 시간을 추출한다.

        :return: None.
        """
        text = (b'INFO: event=stt request_id=abc elapsed_seconds=2.5 status=ok\n'
                b'INFO: event=stt request_id=other elapsed_seconds=9 status=ok\n'
                b'INFO: GET /health HTTP/1.1 200 OK\n')
        frame = b'\1\0\0\0' + len(text).to_bytes(4, 'big') + text
        result = timing_logs(frame, 'abc')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['event'], 'stt')
        self.assertEqual(result[0]['elapsed_seconds'], '2.5')

    def test_completed_job_with_missing_translation_is_not_success(self):
        """completed라도 누락 번역·빈 결과·다운로드 누락이 있으면 시험 성공에서 제외한다.

        :return: None.
        """
        report = {'final_job': {'status': 'completed', 'failed_segments': 0,
                                'total_segments': 1, 'completed_segments': 1},
                  'result': {'segments': [{'translation': '번역입니다.', 'translation_error': None}]},
                  'subtitles': {fmt: {'status': 200, 'download': 'attachment'} for fmt in ('srt', 'vtt')}}
        self.assertTrue(pipeline_succeeded(report))
        report['final_job']['total_segments'] = 2
        self.assertFalse(pipeline_succeeded(report))
        report['final_job']['total_segments'] = 1
        report['result']['segments'][0]['translation'] = None
        self.assertFalse(pipeline_succeeded(report))
        report['result']['segments'] = []
        self.assertFalse(pipeline_succeeded(report))

    def test_job_id_and_events_survive_sse_disconnect(self):
        """OOM 등으로 SSE가 끊겨도 접수 작업 ID·관측 이벤트를 실패 보고서에 남긴다.

        :return: None.
        """
        def handler(request):
            """실제 httpx 경로로 접수와 끊어지는 이벤트 응답을 제공한다.

            :param request: health·작업 접수·SSE HTTP 요청.
            :return: 시험용 JSON 또는 SSE 응답.
            """
            if request.url.path == '/health':
                return httpx.Response(200, json={'translation_model_loaded': True})
            if request.url.path == '/api/v1/jobs':
                return httpx.Response(202, json={'job_id': 'interrupted-job'})
            return httpx.Response(200, stream=InterruptedEvents())

        client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
        report = {}
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / 'audio.wav'
            media.write_bytes(b'fixture')
            with patch('scripts.benchmark_cpu.httpx.Client', return_value=client):
                with self.assertRaises(httpx.RemoteProtocolError):
                    run_media('http://test', media, report=report)
        self.assertEqual(report['job_id'], 'interrupted-job')
        self.assertEqual(report['events'][0]['state'], {'status': 'running'})
