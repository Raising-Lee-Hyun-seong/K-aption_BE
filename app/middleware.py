"""API 요청별 식별자를 생성하고 본문 수신부터 응답 전송까지의 시간을 기록한다."""
from time import perf_counter
from uuid import uuid4

from timing import log_duration, request_id, segment_index


class TimingMiddleware:
    """동시 요청과 스레드풀 추론 로그를 연결하고 API 요청의 HTTP 상태·시간을 기록한다."""

    def __init__(self, app):
        """시간을 측정할 하위 ASGI 애플리케이션을 저장한다.

        :param app: 서비스를 저장할 FastAPI 또는 호출할 하위 ASGI 애플리케이션.
        :return: None.
        """
        self.app = app

    async def __call__(self, scope, receive, send):
        """API 요청을 실행하고 오류가 나도 요청 시간을 기록한 뒤 문맥 변수를 복구한다.

        :param scope: 요청 종류·경로·헤더 등을 담은 ASGI 문맥.
        :param receive: ASGI 요청 이벤트를 비동기로 수신하는 함수.
        :param send: ASGI 응답 이벤트를 비동기로 전송하는 함수.
        :return: None.

        하위 앱의 예외·취소를 시간 기록과 문맥 복구 후 그대로 전파한다.
        """
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        request_token = request_id.set(uuid4().hex[:12])
        segment_token = segment_index.set(None)
        status_code = 500
        error_type = None
        log_duration("request_started", 0.0, method=scope["method"], path=scope["path"])

        async def send_with_status(message):
            """HTTP 응답 시작 이벤트에서 실제 상태 코드를 저장하고 응답을 전달한다.

            :param message: 전송하거나 수집할 ASGI 응답 이벤트 사전.
            :return: None.
            """
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            try:
                log_duration(
                    "http_request",
                    perf_counter() - started,
                    method=scope["method"],
                    path=scope["path"],
                    status_code=status_code,
                    status="error" if error_type or status_code >= 400 else "ok",
                    error_type=error_type,
                )
            finally:
                segment_index.reset(segment_token)
                request_id.reset(request_token)
