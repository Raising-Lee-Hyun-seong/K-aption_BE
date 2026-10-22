"""llama.cpp 서버의 채팅·토큰화 API를 호출하고 제한된 SSE 생성 스트림을 제공한다."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx

from app.services.translation import TranslationService, TranslationTransportError
from app.services.translation_qwen import QwenPromptMixin


class ServerTokenizer:
    """메시지는 JSON으로 보존하고 실제 서버 템플릿과 토크나이저로 길이를 검사한다."""

    def __init__(self, client: httpx.Client):
        """재사용할 HTTP 클라이언트를 보관한다.

        :param client: 번역 서버 기본 주소와 타임아웃이 설정된 클라이언트.
        :return: None.
        """
        self.client = client

    def apply_chat_template(self, messages: list, tokenize=False, add_generation_prompt=True) -> str:
        """채팅 역할과 원문을 손실 없이 직렬화한다.

        :param messages: 시스템 지시와 현재 사용자 원문 메시지.
        :param tokenize: False만 지원한다. 실제 토큰화는 encode에서 수행한다.
        :param add_generation_prompt: True만 지원한다. 서버 채팅 API가 생성 시작을 추가한다.
        :return: HTTP 채팅 API에 그대로 보낼 메시지의 JSON 문자열.
        :raises ValueError: 지원하지 않는 옵션을 사용한 경우.
        """
        if tokenize or not add_generation_prompt:
            raise ValueError('Use untokenized messages with a generation prompt')
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, prompt: str) -> list:
        """서버 채팅 템플릿의 실제 토큰 수를 계산한다.

        :param prompt: 직렬화된 시스템·사용자 메시지.
        :return: 서버 템플릿을 적용하고 BOS 중복 없이 토큰화한 ID 목록.
        :raises TranslationTransportError: 서버 연결·토큰화 응답이 실패한 경우.
        """
        try:
            response = self.client.post('/apply-template', json={'messages': json.loads(prompt)})
            response.raise_for_status()
            rendered = response.json()['prompt']
            response = self.client.post('/tokenize', json={
                'content': rendered, 'add_special': False, 'parse_special': True})
            response.raise_for_status()
            tokens = response.json()['tokens']
            if not isinstance(tokens, list) or not all(type(value) is int for value in tokens):
                raise ValueError('Invalid token IDs')
            return tokens
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            raise TranslationTransportError('Translator tokenization unavailable') from exc


class HttpTranslationService(QwenPromptMixin, TranslationService):
    """Qwen 프롬프트와 검증을 유지하며 원격 CPU 서버에서만 생성한다."""

    def __init__(self, base_url: str, template_path: Path, timeout: float = 120):
        """제한된 HTTP 연결 풀과 공통 텍스트 설정을 초기화한다.

        :param base_url: 신뢰하는 llama.cpp 서버의 http 또는 https 기본 주소.
        :param template_path: 공통 텍스트 설정 파일 경로.
        :param timeout: HTTP 연결·읽기·쓰기 제한 초. 양의 유한값.
        :return: None. Qwen 가중치나 토크나이저를 로컬에 로드하지 않는다.
        :raises ValueError: 주소·타임아웃 설정이 유효하지 않은 경우.
        :raises OSError: 공통 텍스트 설정을 읽지 못한 경우.
        """
        url = urlsplit(base_url)
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
                or url.path not in ('', '/') or url.query or url.fragment
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Translator URL must be an HTTP origin and timeout must be positive')
        super().__init__(template_path)
        self._configure_qwen()
        self.client = httpx.Client(base_url=base_url.rstrip('/'), trust_env=False,
                                   timeout=httpx.Timeout(timeout, connect=min(timeout, 10)),
                                   limits=httpx.Limits(max_connections=2, max_keepalive_connections=1))
        self.tokenizer = ServerTokenizer(self.client)

    def loaded(self) -> bool:
        """서버의 현재 모델 준비 상태를 제한된 HTTP 요청으로 확인한다.

        :return: 서버 health가 200일 때만 True.
        """
        try:
            return self.client.get('/health', timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        """앱 작업자 종료 후 HTTP 연결 풀을 해제한다.

        :return: None.
        """
        self.client.close()

    def _stream_output(self, prompt: str, max_tokens: int):
        """OpenAI 호환 채팅 SSE를 읽고 검증 가능한 생성 조각을 제공한다.

        :param prompt: 원문과 문맥을 구분한 메시지 JSON.
        :param max_tokens: 최대 출력 토큰 수. 서버 컨텍스트 상한은 별도로 검사한다.
        :return: HTTP 응답을 수명 동안 관리하는 생성기.
        :yield: text·generation_tokens·finish_reason 속성을 가진 생성 조각.
        :raises TranslationTransportError: 연결·시간 초과·빈 응답·손상 응답·조기 종료 시.

        생성기 종료 시 HTTP 응답을 닫아 반복 감지나 취소 이후 서버 생성을 중단한다.
        """
        payload = {'model': 'qwen3-4b-instruct-2507', 'messages': json.loads(prompt),
                   'temperature': 0, 'top_k': 1, 'top_p': 1, 'repeat_penalty': 1.0,
                   'seed': 0, 'max_tokens': max_tokens, 'stream': True,
                   'stream_options': {'include_usage': True}, 'cache_prompt': False}
        count, ended, last_finish = 0, False, None
        try:
            with self.client.stream('POST', '/v1/chat/completions', json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    event = json.loads(data)
                    if 'error' in event:
                        raise ValueError('Server reported a generation error')
                    usage = event.get('usage') or {}
                    if usage:
                        count = usage['completion_tokens']
                        if type(count) is not int or count < 0:
                            raise ValueError('Invalid completion token count')
                    choices = event.get('choices') or []
                    if not choices:
                        if usage:
                            yield SimpleNamespace(text='', generation_tokens=count, finish_reason=last_finish)
                        continue
                    choice = choices[0]
                    text = choice.get('delta', {}).get('content') or ''
                    finish = choice.get('finish_reason')
                    if text:
                        count += 1
                    if finish:
                        ended = True
                        last_finish = finish
                    yield SimpleNamespace(text=text, generation_tokens=count, finish_reason=finish)
                if not ended:
                    raise ValueError('Generation ended without a finish reason')
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            raise TranslationTransportError('Translator generation unavailable') from exc
