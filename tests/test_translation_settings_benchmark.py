"""짧은 캐시 비교에서 동일 입력·워밍업 제외·실패 기록과 실제 SSE 시간 수집을 검증한다."""

import json
from pathlib import Path
import unittest

import httpx

from scripts.benchmark_translation_settings import benchmark, CASES, CONTEXT, RecordingTranslationService

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkServiceDouble:
    """비교 순서와 시도 목록을 기록하며 한 조건의 실패를 재현한다."""

    def __init__(self):
        """호출·시도 목록과 캐시 기본값을 준비한다.

        :return: None.
        """
        self.calls, self.attempts = [], []
        self.cache_prompt = False

    def translate(self, text, context, max_tokens):
        """한 입력을 기록하고 선택한 캐시 조건의 거부를 재현한다.

        :param text: 현재 원문.
        :param context: 고정 영어 배경 문맥.
        :param max_tokens: 비교 조건의 출력 토큰 상한.
        :return: 시험용 번역 사전.
        :raises ValueError: 캐시를 켠 두 번째 사례가 거부될 때.
        """
        self.calls.append((text, context, max_tokens, self.cache_prompt))
        self.attempts.append({'raw_output': text})
        if self.cache_prompt and text == CASES[1]['text']:
            raise ValueError('Rejected output')
        return {'translation': '번역입니다.'}


class TranslationSettingsBenchmarkTests(unittest.TestCase):
    """실제 모델 없이 비교 조건의 공정성과 실패 누락을 검사한다."""

    def test_same_inputs_and_limits_with_separate_uncached_warmups(self):
        """양쪽 조건이 같은 입력·문맥·상한을 쓰고 워밍업은 측정 사례에서 제외한다.

        :return: None.
        """
        service = BenchmarkServiceDouble()
        modes = benchmark(service, (True, False))
        self.assertEqual([mode['cache_prompt'] for mode in modes], [True, False])
        self.assertEqual([row['text'] for row in modes[0]['results']],
                         [row['text'] for row in modes[1]['results']])
        self.assertEqual([mode['failed_cases'] for mode in modes], [1, 0])
        self.assertEqual([len(mode['results']) for mode in modes], [3, 3])
        self.assertFalse(service.calls[0][3])
        self.assertFalse(service.calls[4][3])
        self.assertTrue(all(call[2] == 128 for call in service.calls))
        self.assertEqual([row['context'] for row in modes[0]['results']],
                         [CONTEXT, CASES[0]['text'], CASES[1]['text']])
        self.assertEqual([row['context'] for row in modes[0]['results']],
                         [row['context'] for row in modes[1]['results']])
        self.assertEqual(len(modes[0]['warmup_attempts']), 1)
        self.assertEqual(len(modes[0]['results'][1]['attempts']), 1)

    def test_bad_comparison_order_is_rejected(self):
        """한 조건 누락·중복·정수 플래그가 비교 결과로 저장되는 것을 막는다.

        :return: None.
        """
        for order in ((False,), (True, True), (0, 1)):
            with self.assertRaises(ValueError):
                benchmark(BenchmarkServiceDouble(), order)
        with self.assertRaises(ValueError):
            benchmark(BenchmarkServiceDouble(), context_mode='unknown')

    def test_fixed_context_is_explicit(self):
        """배경 고정 시험은 별도 모드로 선택하며 실제 문맥 갱신 조건과 혼동하지 않는다.

        :return: None.
        """
        modes = benchmark(BenchmarkServiceDouble(), context_mode='fixed')
        self.assertTrue(all(row['context'] == CONTEXT for mode in modes for row in mode['results']))

    def test_recording_preserves_actual_raw_sse_and_cache_counters(self):
        """HTTP 생성기를 대체하지 않고 실제 전달된 출력과 서버 시간 필드를 수집한다.

        :return: None.
        """
        def handler(request):
            """채팅 SSE와 재사용 토큰 수를 반환한다.

            :param request: 실제 httpx 채팅 요청.
            :return: 정상 종료·시간·사용량을 포함하는 SSE 응답.
            """
            payload = json.loads(request.content)
            self.assertTrue(payload['cache_prompt'])
            events = [{'choices': [{'delta': {'content': '속도는 일정합니다.'}, 'finish_reason': 'stop'}]},
                      {'choices': [], 'usage': {'completion_tokens': 8},
                       'timings': {'cache_n': 180, 'prompt_n': 20, 'prompt_ms': 2}}]
            body = ''.join('data: ' + json.dumps(row, ensure_ascii=False) + '\n\n' for row in events)
            return httpx.Response(200, text=body + 'data: [DONE]\n\n')

        service = RecordingTranslationService('http://test', ROOT / 'templates/korean.json')
        service.client.close()
        service.client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
        self.addCleanup(service.close)
        service.cache_prompt = True
        output = service._generate_output(service._build_prompt(CASES[0]['text'], CONTEXT), 128)
        self.assertEqual(output['text'], '속도는 일정합니다.')
        self.assertEqual(service.attempts[0]['server_timings']['cache_n'], 180)
        self.assertEqual(service.attempts[0]['raw_output'], output['text'])
