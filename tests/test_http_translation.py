"""HTTP 번역 서비스의 템플릿·직렬 검증·재시도·SSE 종료·서버 장애를 검증한다."""

import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app.services.translation import TranslationTransportError
from app.services.translation_http import HttpTranslationService
from app.services.translation_qwen import render_chatml
from app.main import app, health, lifespan, create_translation_service

ROOT = Path(__file__).resolve().parents[1]


class HttpTranslationTests(unittest.TestCase):
    """실제 HTTP 전송 형식을 대역 서버로 확인하며 모델은 로드하지 않는다."""

    def service(self, answers: list, token_count: int = 100, cache_prompt: bool = False):
        """템플릿·토큰화·채팅 엔드포인트를 가진 HTTP 전송 대역을 구성한다.

        :param answers: 생성 결과 문자열 또는 HTTP 오류 응답·예외 목록.
        :param token_count: 토큰화 엔드포인트가 반환할 입력 토큰 수.
        :param cache_prompt: 공통 입력 KV를 재사용하도록 요청할지 여부.
        :return: 서비스와 요청 기록 목록. 테스트 종료 시 연결을 닫는다.
        """
        requests = []
        remaining = iter(answers)

        def handle(request):
            """서버 계약의 JSON·SSE 응답을 반환하고 전송 내용을 기록한다.

            :param request: httpx가 생성한 HTTP 요청.
            :return: HTTP 응답 대역.
            """
            payload = json.loads(request.content) if request.content else None
            requests.append((request.url.path, payload))
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': render_chatml(payload['messages'])})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': list(range(token_count))})
            answer = next(remaining)
            if isinstance(answer, Exception):
                raise answer
            if isinstance(answer, httpx.Response):
                return answer
            events = [
                {'choices': [{'delta': {'content': answer}, 'finish_reason': None}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                {'choices': [], 'usage': {'completion_tokens': 17},
                 'timings': {'cache_n': 80 if payload['cache_prompt'] else 0, 'prompt_ms': 3}}]
            body = ''.join('data: ' + json.dumps(event, ensure_ascii=False) + '\n\n' for event in events)
            return httpx.Response(200, text=body + 'data: [DONE]\n\n')

        service = HttpTranslationService('http://translator:8080', ROOT / 'templates/korean.json',
                                         cache_prompt=cache_prompt)
        service.client.close()
        service.client = httpx.Client(base_url='http://translator:8080', transport=httpx.MockTransport(handle))
        service.tokenizer.client = service.client
        self.addCleanup(service.close)
        return service, requests

    def test_messages_context_terms_and_exact_token_usage(self):
        """동일한 메시지를 토큰화·채팅에 보내고 문맥·용어·존댓말·사용량을 보존한다.

        :return: None.
        """
        service, requests = self.service(['로켓의 질량은 일정합니다.'])
        result = service.translate('This rocket mass is constant.', 'Earlier English.', 128)
        self.assertEqual(result['prompt_tokens'], 100)
        self.assertEqual(result['generated_tokens'], 17)
        apply = requests[0][1]['messages']
        chat = requests[-1][1]
        self.assertEqual(apply, chat['messages'])
        self.assertIn('Earlier English.', apply[0]['content'])
        self.assertIn('존댓말', apply[0]['content'])
        self.assertIn('rocket=로켓', apply[1]['content'])
        self.assertEqual(chat['temperature'], 0)
        self.assertFalse(chat['cache_prompt'])
        self.assertFalse(requests[1][1]['add_special'])
        self.assertTrue(requests[1][1]['parse_special'])

    def test_independent_source_omits_context_and_math_is_restored_before_validation(self):
        """독립 문장의 문맥은 보내지 않고 HTTP 생성 표식은 원문 수식으로 복원한다.

        :return: None.
        """
        service, requests = self.service(['이제 [MATH_A]입니다.'])
        result = service.translate('mR prime equals m of vR.', 'Private earlier context.', 128)
        self.assertEqual(result['translation'], '이제 mR′ = m(vR)입니다.')
        chat = requests[-1][1]
        self.assertNotIn('Private earlier context.', json.dumps(chat))
        self.assertIn('[MATH_A]', chat['messages'][1]['content'])
        self.assertNotIn('[MATH_A]', result['translation'])

    def test_missing_math_marker_has_bounded_retry(self):
        """수식 표식이 사라진 두 응답은 한 번만 재시도한 뒤 기존 실패 계약으로 전달한다.

        :return: None.
        """
        service, requests = self.service(['함수입니다.', '함수입니다.'])
        with self.assertRaises(ValueError):
            service.translate('m of vR.', '', 128)
        self.assertEqual(sum(path == '/v1/chat/completions' for path, _ in requests), 2)

    def test_recorded_math_operator_output_returns_without_second_generation(self):
        """실제 8 CPU 실패 출력은 연산어만 정리하고 기존 검증 후 첫 생성으로 반환한다.

        :return: None.
        """
        source = ('And we ultimately want this, so this is vr of t, of course, '
                  'and then we get v0 plus u ln mr over 2m0.')
        raw = ('이는 t에 대한 vr의 값이며, 그 결과는 '
               'v0 plus u times ln(mR / 2m0)입니다.')
        service, requests = self.service([raw])
        result = service.translate(source, 'Earlier English.', 128)
        self.assertEqual(result['translation'], raw.replace('plus', '+').replace('times', '×'))
        self.assertEqual(sum(path == '/v1/chat/completions' for path, _ in requests), 1)

    def test_normalized_bad_numbers_and_wrong_operators_keep_bounded_failure(self):
        """정규화로 부정확한 숫자·원문과 다른 연산·반말을 수락하지 않고 두 번으로 제한한다.

        :return: None.
        """
        source = 'We get v0 plus u ln mR over 2m0.'
        for raw in ('이 값은 v0 plus u times ln(mR / 3m0)입니다.',
                    '이 값은 v0 minus u times ln(mR / 2m0)입니다.',
                    '이 값은 v0 plus u times ln(mR / 2m0)이다.'):
            with self.subTest(raw=raw):
                service, requests = self.service([raw, raw])
                with self.assertRaises(ValueError):
                    service.translate(source, '', 128)
                self.assertEqual(sum(path == '/v1/chat/completions' for path, _ in requests), 2)

    def test_invalid_output_retries_without_context(self):
        """반말은 재시도하고 이전 문맥을 제거한 존댓말 결과만 반환한다.

        :return: None.
        """
        service, requests = self.service(['초기 질량이다.', '초기 질량입니다.'])
        result = service.translate('This is the initial mass.', 'Earlier English.', 128)
        self.assertEqual(result['translation'], '초기 질량입니다.')
        chats = [body for path, body in requests if path == '/v1/chat/completions']
        self.assertEqual(len(chats), 2)
        self.assertNotIn('Earlier English.', json.dumps(chats[1]))

    def test_cache_is_opt_in_and_keeps_server_timing_evidence(self):
        """선택적 캐시를 실제 요청에 전달하고 사용량 이벤트의 재사용 토큰 근거를 보존한다.

        :return: None.
        """
        service, requests = self.service(['속도는 일정합니다.'], cache_prompt=True)
        prompt = service._build_prompt('The velocity is constant.', '')
        parts = list(service._stream_output(prompt, 128))
        self.assertTrue(requests[-1][1]['cache_prompt'])
        self.assertEqual(parts[-1].server_timings['cache_n'], 80)
        self.assertEqual(parts[-1].generation_tokens, 17)

    def test_factory_rejects_ambiguous_cache_flags(self):
        """문자열 false를 참으로 오해하지 않고 true·false만 설정으로 수락한다.

        :return: None.
        """
        for value, expected in (('false', False), ('true', True), (' FALSE ', False)):
            with patch.dict('os.environ', {'KAPTION_TRANSLATOR_PROMPT_CACHE': value}):
                service = create_translation_service()
                self.addCleanup(service.close)
                self.assertIs(service.cache_prompt, expected)
        with patch.dict('os.environ', {'KAPTION_TRANSLATOR_PROMPT_CACHE': 'yes'}):
            with self.assertRaises(ValueError):
                create_translation_service()
        with self.assertRaises(ValueError):
            HttpTranslationService('http://translator', ROOT / 'templates/korean.json', cache_prompt='false')

    def test_application_factory_enables_prompt_cache_by_default(self):
        """환경변수 미지정 시 운영 서비스에서 공통 프롬프트 캐시를 활성화한다.

        :return: None.
        """
        def defaults(name, default=None):
            """실제 환경값 대신 호출부의 기본값을 제공한다.

            :param name: 조회할 환경변수 이름.
            :param default: 환경변수 미지정 시 사용할 값.
            :return: 전달받은 기본값.
            """
            return default

        with patch('app.main.os.getenv', side_effect=defaults):
            service = create_translation_service()
        self.addCleanup(service.close)
        self.assertTrue(service.cache_prompt)

    def test_timeout_and_bad_gateway_have_bounded_retry(self):
        """시간 초과·HTTP 장애는 한 번만 재시도하고 기존 부분 실패 계약으로 전달한다.

        :return: None.
        """
        for failure in (httpx.ReadTimeout('timeout'), httpx.Response(503)):
            with self.subTest(failure=failure):
                service, requests = self.service([failure, failure])
                with self.assertRaises(TranslationTransportError):
                    service.translate('Hello.', '', 128)
                self.assertEqual(sum(path == '/v1/chat/completions' for path, _ in requests), 2)
                self.assertTrue(service._lock.acquire(blocking=False))
                service._lock.release()

    def test_timeout_can_recover_on_second_attempt(self):
        """첫 생성 연결이 실패해도 두 번째 정상 결과를 반환한다.

        :return: None.
        """
        service, _ = self.service([httpx.ReadTimeout('timeout'), '안녕하세요.'])
        self.assertEqual(service.translate('Hello.', '', 128)['translation'], '안녕하세요.')

    def test_malformed_or_empty_stream_cannot_succeed(self):
        """빈 스트림·잘못된 JSON·종료 누락은 성공 응답으로 취급하지 않는다.

        :return: None.
        """
        for text in ('', 'data: not-json\n\n', 'data: {"choices": []}\n\ndata: [DONE]\n\n'):
            with self.subTest(text=text):
                service, _ = self.service([httpx.Response(200, text=text), httpx.Response(200, text=text)])
                with self.assertRaises(TranslationTransportError):
                    service.translate('Hello.', '', 128)

    def test_length_finish_cannot_be_overwritten_by_usage(self):
        """마지막 usage 이벤트가 토큰 상한 종료를 정상 종료로 덮어쓰지 않는다.

        :return: None.
        """
        events = [{'choices': [{'delta': {'content': '초기 질량입니다.'}, 'finish_reason': 'length'}]},
                  {'choices': [], 'usage': {'completion_tokens': 128}}]
        body = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events) + 'data: [DONE]\n\n'
        service, _ = self.service([httpx.Response(200, text=body), httpx.Response(200, text=body)])
        with self.assertRaisesRegex(ValueError, 'valid Korean translation'):
            service.translate('This is the initial mass.', '', 128)

    def test_input_limit_prevents_generation(self):
        """서버의 실제 입력 토큰과 출력 합계가 2048을 넘으면 생성을 보내지 않는다.

        :return: None.
        """
        service, requests = self.service([], token_count=2000)
        with self.assertRaisesRegex(ValueError, '2048'):
            service.translate('Hello.', '', 128)
        self.assertNotIn('/v1/chat/completions', [path for path, _ in requests])

    def test_math_and_number_loss_still_fail_validation(self):
        """HTTP 전환 후에도 숫자·수식 기호 누락을 성공으로 반환하지 않는다.

        :return: None.
        """
        service, _ = self.service(['초기 질량입니다.', '초기 질량입니다.'])
        with self.assertRaises(ValueError):
            service.translate('The initial mass is 2 m0.', '', 128)

    def test_settings_reject_invalid_origin_and_timeout(self):
        """비HTTP·사용자 정보 포함 주소와 비유한 타임아웃을 거부한다.

        :return: None.
        """
        for url, timeout in (('file:///models', 1), ('http://user:secret@translator', 1),
                             ('http://translator/v1', 1), ('http://translator', float('inf'))):
            with self.subTest(url=url), self.assertRaises(ValueError):
                HttpTranslationService(url, ROOT / 'templates/korean.json', timeout)


class HttpLifespanTests(unittest.IsolatedAsyncioTestCase):
    """공유 추론 잠금과 CPU STT 강제 설정 및 HTTP 클라이언트 수명을 검사한다."""

    async def test_cpu_stt_shares_lock_and_client_closes_after_worker(self):
        """STT는 CPU INT8을 쓰며 작업자가 종료된 뒤 HTTP 클라이언트를 닫는다.

        :return: None.
        """
        order = []

        async def close_worker():
            """진행 작업 정리의 완료 순서를 기록한다.

            :return: None.
            """
            order.append('worker')

        def close_client():
            """작업 정리 이후에 HTTP 연결이 종료되는지 확인한다.

            :return: None.
            """
            self.assertEqual(order, ['worker'])
            order.append('client')

        manager = SimpleNamespace(start=AsyncMock(), close=close_worker)
        translator = SimpleNamespace(_lock=Lock(), close=close_client)
        with patch('app.main.JobManager', return_value=manager), \
                patch('app.main.create_translation_service', return_value=translator), \
                patch('app.main.TranscriptionService') as stt, \
                patch.dict('os.environ', {'KAPTION_STT_DEVICE': 'cuda', 'KAPTION_STT_COMPUTE_TYPE': 'float16'}):
            async with lifespan(app):
                self.assertEqual(stt.call_args.kwargs['device'], 'cpu')
                self.assertEqual(stt.call_args.kwargs['compute_type'], 'int8')
                self.assertIs(app.state.transcriber._inference_lock, translator._lock)
        self.assertEqual(order, ['worker', 'client'])
        self.assertIsNone(app.state.translator)

    async def test_health_tracks_remote_model_failure(self):
        """HTTP 클라이언트 존재와 원격 모델 준비 여부를 구분한다.

        :return: None.
        """
        translator = SimpleNamespace(loaded=Mock(return_value=False))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            translator=translator, transcriber=SimpleNamespace(loaded=False))))
        result = health(request)
        self.assertFalse(result.translation_model_loaded)
        self.assertEqual(result.status, 'starting')
