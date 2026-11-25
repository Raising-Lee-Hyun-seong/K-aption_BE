"""모델 실행 없이 시간 로그 파싱·요청별 토큰 합산·비용 계산을 검증한다."""
import unittest

from scripts.estimate_cost import estimate, parse_events, summarize_requests

LOG = [
    'INFO: event=request_started request_id=abc segment_index=None elapsed_seconds=0.000 method=POST',
    'INFO: event=stt_inference request_id=abc segment_index=None elapsed_seconds=30.000 status=ok audio_seconds=120.0',
    'INFO: event=translation_attempt request_id=abc segment_index=0 elapsed_seconds=2.000 status=ok attempt=1 '
    'prompt_tokens=200 max_tokens=128 raw_output_tokens=20 accepted=False',
    'INFO: event=translation_attempt request_id=abc segment_index=0 elapsed_seconds=2.000 status=ok attempt=2 '
    'prompt_tokens=180 max_tokens=128 raw_output_tokens=20 accepted=True',
    'unrelated line',
]
PRICES = {'translation': {'m': {'input_per_million': 1.0, 'output_per_million': 10.0}},
          'stt': {'s': {'per_minute': 0.01}}}


class CostEstimateTests(unittest.TestCase):
    """재시도 토큰까지 합산하고 1분당 값과 1시간 비용을 정확히 계산한다."""

    def test_summary_includes_retries(self):
        """재시도를 포함한 모든 시도의 입력·출력 토큰을 요청별로 합한다.

        :return: None.
        """
        rows = summarize_requests(parse_events(LOG))
        self.assertEqual(rows, [{'request_id': 'abc', 'audio_minutes': 2.0, 'attempts': 2,
                                 'prompt_tokens': 380, 'output_tokens': 40}])

    def test_estimate_per_minute_and_hourly_cost(self):
        """2분 영상의 380·40토큰을 1분당 190·20, 1시간 비용으로 환산한다.

        :return: None.
        """
        result = estimate(summarize_requests(parse_events(LOG))[0], PRICES)
        self.assertEqual(result['input_tokens_per_minute'], 190)
        self.assertEqual(result['output_tokens_per_minute'], 20)
        # 60분 × (190×1 + 20×10) / 1e6 = 0.0234
        self.assertEqual(result['translation_usd_per_hour'], {'m': 0.0234})
        self.assertEqual(result['stt_usd_per_hour'], {'s': 0.6})

    def test_request_without_duration_is_skipped(self):
        """영상 길이 로그가 없으면 1분당 값을 계산하지 않고 제외한다.

        :return: None.
        """
        self.assertEqual(summarize_requests(parse_events(LOG[2:4])), [])


if __name__ == '__main__':
    unittest.main()
