"""학습자료 입력의 자막 보존·영어 대체·토큰 경계와 로컬 토큰화 연동을 검증한다."""

import json
import unittest
from unittest.mock import Mock

import httpx

from app.schemas import TranscriptionResponse, TranscriptionSegment
from app.services.study_inputs import prepare_study_inputs
from app.services.translation_http import ServerTokenizer


def source(count=3):
    """시간과 원문이 구별되는 자막 결과를 만든다.

    :param count: 생성할 자막 개수.
    :return: 원본 보존과 묶음 분할 시험용 자막 응답.
    """
    return TranscriptionResponse(segments=[TranscriptionSegment(
        start=index * 5, end=index * 5 + 4, text=f'Equation {index}: x = 2.',
        translation=f'식 {index}: x = 2입니다.', translation_error=None)
        for index in range(count)])


def token_count(messages):
    """채팅 지시·자막 데이터·템플릿 비용을 반영하는 결정적 시험 계산기를 제공한다.

    :param messages: 시스템 지시와 JSON 자막 메시지.
    :return: 자막 한 개당 10에 지시문 길이와 가상 템플릿 비용 5를 더한 시험 토큰 수.
    """
    return 5 + len(messages[0]['content']) + 10 * len(json.loads(messages[1]['content'])['segments'])


class StudyInputTests(unittest.TestCase):
    """모델 생성 없이 근거 연결·입력 예산·원본 무변경 계약을 확인한다."""

    def test_order_ids_and_originals_survive_chunking(self):
        """분할 후에도 모든 자막이 순서대로 한 번씩 포함되고 원본은 변하지 않는다.

        :return: None.
        """
        original = source(5)
        before = original.model_dump()
        chunks = prepare_study_inputs(original, token_count, instruction='분석',
                                      context_window=37, output_tokens=10)
        self.assertEqual([len(chunk.segments) for chunk in chunks], [2, 2, 1])
        flattened = [segment for chunk in chunks for segment in chunk.segments]
        self.assertEqual([segment.segment_id for segment in flattened],
                         [f'segment-{index:06d}' for index in range(1, 6)])
        for old, new in zip(original.segments, flattened):
            self.assertEqual({key: getattr(new, key) for key in type(old).model_fields}, old.model_dump())
        self.assertEqual(original.model_dump(), before)
        repeated = prepare_study_inputs(original, token_count, instruction='분석',
                                        context_window=37, output_tokens=10)
        self.assertEqual(chunks, repeated)
        chunks[0].segments[0].text = '수정된 복사본'
        self.assertEqual(original.model_dump(), before)

    def test_translation_fallback_and_json_round_trip(self):
        """정상 번역은 한국어로 분석하고 실패·빈 번역은 원문을 손실 없이 사용한다.

        :return: None.
        """
        original = source(5)
        original.segments[0].translation = '수식 "x" = 2\n다음 줄입니다.'
        original.segments[1].translation = None
        original.segments[2].translation = '  '
        original.segments[3].translation_error = 'translation_failed'
        chunks = prepare_study_inputs(original, token_count, instruction='분석')
        segments = chunks[0].segments
        self.assertEqual([item.analysis_language for item in segments], ['ko', 'en', 'en', 'en', 'ko'])
        self.assertEqual(segments[0].analysis_text, original.segments[0].translation)
        for index in (1, 2, 3):
            self.assertEqual(segments[index].analysis_text, original.segments[index].text)
        rows = json.loads(chunks[0].messages[1]['content'])['segments']
        self.assertEqual([row['text'] for row in rows], [item.analysis_text for item in segments])
        self.assertEqual([row['segment_id'] for row in rows], [item.segment_id for item in segments])

    def test_exact_budget_includes_instruction_and_output(self):
        """한도와 정확히 같은 입력은 허용하고 지시문이 길어지면 더 작은 묶음으로 나눈다.

        :return: None.
        """
        chunks = prepare_study_inputs(source(2), token_count, instruction='분석',
                                      context_window=37, output_tokens=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].prompt_tokens + chunks[0].reserved_output_tokens, 37)
        chunks = prepare_study_inputs(source(2), token_count, instruction='긴 분석',
                                      context_window=37, output_tokens=10)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk.prompt_tokens, token_count(chunk.messages))
            self.assertLessEqual(chunk.prompt_tokens + chunk.reserved_output_tokens, 37)

    def test_oversize_reports_source_id_without_truncation(self):
        """중간의 단일 초과 자막은 ID가 있는 오류로 알리고 원문을 자르지 않는다.

        :return: None.
        """
        counter = Mock(side_effect=[10, 100, 100])
        original = source(2)
        before = original.model_dump()
        with self.assertRaisesRegex(ValueError, 'segment-000002'):
            prepare_study_inputs(original, counter, instruction='분석',
                                 context_window=50, output_tokens=10)
        self.assertEqual(original.model_dump(), before)
        with self.assertRaisesRegex(ValueError, 'segment-000001'):
            prepare_study_inputs(source(1), token_count, instruction='분석',
                                 context_window=15, output_tokens=10)

    def test_empty_result_does_not_call_tokenizer(self):
        """음성이 없는 결과는 서버 토큰화 없이 빈 목록을 반환한다.

        :return: None.
        """
        counter = Mock()
        self.assertEqual(prepare_study_inputs(source(0), counter, instruction='분석'), [])
        counter.assert_not_called()

    def test_invalid_settings_and_counter_results_are_rejected(self):
        """유효하지 않은 예산·지시문과 계산기의 잘못된 토큰 수를 거부한다.

        :return: None.
        """
        for context, output in ((10, 10), (10, 0), (0, 1), (True, 1), (20, 1.5)):
            with self.subTest(context=context, output=output), self.assertRaises(ValueError):
                prepare_study_inputs(source(1), token_count, instruction='분석',
                                     context_window=context, output_tokens=output)
        for instruction in ('', '  ', None):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                prepare_study_inputs(source(1), token_count, instruction=instruction)
        for result in (-1, True, 1.2, None):
            with self.subTest(result=result), self.assertRaises(ValueError):
                prepare_study_inputs(source(1), Mock(return_value=result), instruction='분석')
        with self.assertRaisesRegex(RuntimeError, 'tokenization failed'):
            prepare_study_inputs(source(1), Mock(side_effect=RuntimeError('tokenization failed')),
                                 instruction='분석')

    def test_invalid_source_is_rejected_before_tokenization(self):
        """유한하지 않은 시각·역전 구간·시간순 위반·빈 원문을 토큰화 전에 거부한다.

        :return: None.
        """
        for field, value in (('start', float('nan')), ('end', float('inf')),
                             ('end', 0), ('start', -1), ('text', '  ')):
            original = source(1)
            setattr(original.segments[0], field, value)
            counter = Mock()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                prepare_study_inputs(original, counter, instruction='분석')
            counter.assert_not_called()
        original = source(2)
        original.segments.reverse()
        with self.assertRaisesRegex(ValueError, 'segment-000002'):
            prepare_study_inputs(original, token_count, instruction='분석')

    def test_server_tokenizer_receives_exact_chunk_messages(self):
        """기존 로컬 서버 토큰화 어댑터가 최종 메시지의 템플릿과 토큰을 계산한다.

        :return: None.
        """
        requests = []

        def handle(request):
            """실제 외부 접속 없이 서버 템플릿·토큰화 응답을 제공한다.

            :param request: 로컬 토큰화 클라이언트의 HTTP 요청.
            :return: 템플릿 또는 토큰 ID 목록을 담은 HTTP 응답.
            """
            payload = json.loads(request.content)
            requests.append((request.url.path, payload))
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': 'rendered chat with generation marker'})
            return httpx.Response(200, json={'tokens': [1, 2, 3]})

        with httpx.Client(base_url='http://translator:8080',
                          transport=httpx.MockTransport(handle), trust_env=False) as client:
            tokenizer = ServerTokenizer(client)

            def count(messages):
                """기존 서버 토크나이저로 생성 시작 표식을 포함한 채팅 길이를 센다.

                :param messages: 준비된 시스템·자막 메시지.
                :return: 실제 서버 형식 토큰 ID 목록의 길이.
                """
                return len(tokenizer.encode(tokenizer.apply_chat_template(messages)))

            chunks = prepare_study_inputs(source(1), count, instruction='분석')
        self.assertEqual(requests[0], ('/apply-template', {'messages': chunks[0].messages}))
        self.assertEqual(requests[1][1], {'content': 'rendered chat with generation marker',
                                         'add_special': False, 'parse_special': True})
        self.assertEqual(chunks[0].prompt_tokens, 3)


if __name__ == '__main__':
    unittest.main()
