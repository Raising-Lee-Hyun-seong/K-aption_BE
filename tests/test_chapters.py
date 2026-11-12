"""로컬 챕터 생성의 JSON·근거·재시도·시간 매핑·분할 경계와 평가 저장 계약을 검증한다."""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from app.schemas import TranscriptionResponse
from app.services.chapters import ChapterGenerationError, ChapterService, _parse_boundaries
from app.services.study_inputs import prepare_study_inputs
from app.services.translation_http import HttpTranslationService
from scripts import evaluate_chapters

ROOT = Path(__file__).resolve().parents[1]


def transcription(count=3):
    """시각과 원문이 구별되는 간단한 미적분 자막을 만든다.

    :param count: 만들 자막 수.
    :return: 기존 API 계약의 완료 자막 응답.
    """
    return TranscriptionResponse.model_validate({'segments': [
        {'start': index * 10, 'end': index * 10 + 8, 'text': f'The derivative {index}.',
         'translation': f'도함수 {index}에 대한 설명입니다.', 'translation_error': None}
        for index in range(count)]})


def boundaries(*indices):
    """지정한 자막 순서에서 시작하는 경계 JSON을 만든다.

    :param indices: 1부터 시작하는 자막 번호들.
    :return: 한국어 제목·설명이 있는 원시 챕터 JSON.
    """
    return json.dumps({'chapters': [
        {'start_segment_id': f'segment-{index:06d}', 'title': f'도함수 주제 {index}',
         'description': '도함수의 기하학적 의미를 설명합니다.'} for index in indices]}, ensure_ascii=False)


class ChapterTests(unittest.TestCase):
    """실제 HTTP 전송 형식의 모의 서버로 외부 모델 없이 생성 흐름을 검사한다."""

    def service(self, answers, split=False, merge_tokens=100, wrong_model=False):
        """토큰화·생성·상태·모델 조회가 가능한 모의 로컬 Qwen 서비스를 만든다.

        :param answers: 원시 문자열, 문자열·종료 사유 튜플 또는 전송 예외의 순서 목록.
        :param split: 참이면 입력 자막 한 개씩 묶이도록 시험 토큰 수를 계산한다.
        :param merge_tokens: 경계 병합 입력의 시험 토큰 수.
        :param wrong_model: 참이면 고정 모델과 다른 모델 이름을 반환한다.
        :return: HTTP 번역 서비스와 요청 기록 목록. 테스트 종료 시 연결을 닫는다.
        """
        requests = []
        remaining = iter(answers)

        def handle(request):
            """로컬 채팅 SSE와 템플릿·토큰화 응답을 재현한다.

            :param request: 모의 서버에 전달된 요청.
            :return: HTTP JSON 또는 SSE 응답.
            """
            payload = json.loads(request.content) if request.content else None
            requests.append((request.url.path, payload))
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            if request.url.path == '/props':
                return httpx.Response(200, json={'model_path': 'other.gguf' if wrong_model else
                    '/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'})
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': json.dumps(payload['messages'])})
            if request.url.path == '/tokenize':
                messages = json.loads(payload['content'])
                data = json.loads(messages[1]['content'])
                count = (200 + 700 * len(data['segments']) if split else 100) if 'segments' in data else merge_tokens
                return httpx.Response(200, json={'tokens': list(range(count))})
            self.assertTrue(service._lock.locked())
            answer = next(remaining)
            if isinstance(answer, Exception):
                raise answer
            raw, finish = answer if isinstance(answer, tuple) else (answer, 'stop')
            events = [{'choices': [{'delta': {'content': raw}, 'finish_reason': None}]},
                      {'choices': [{'delta': {}, 'finish_reason': finish}]},
                      {'choices': [], 'usage': {'completion_tokens': 40}}]
            return httpx.Response(200, text=''.join(
                'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n' for event in events)
                + 'data: [DONE]\n\n')

        service = HttpTranslationService('http://translator:8080', ROOT / 'templates/korean.json')
        service.client.close()
        service.client = httpx.Client(base_url='http://translator:8080',
                                     transport=httpx.MockTransport(handle), trust_env=False)
        service.tokenizer.client = service.client
        self.addCleanup(service.close)
        return service, requests

    def test_chapters_derive_complete_coverage_and_times_from_sources(self):
        """모델은 시작 ID만 선택하고 시각과 전체 근거는 원본에서 계산한다.

        :return: None.
        """
        original = transcription()
        before = original.model_dump()
        service, requests = self.service([boundaries(1, 3)])
        result = ChapterService(service).generate(original)
        self.assertEqual([(item.start, item.end) for item in result.chapters], [(0, 18), (20, 28)])
        self.assertEqual(result.chapters[0].segment_ids, ['segment-000001', 'segment-000002'])
        self.assertEqual(result.chapters[1].chapter_id, 'chapter-000002')
        self.assertEqual(original.model_dump(), before)
        self.assertGreaterEqual(result.elapsed_seconds, 0)
        generation = [payload for path, payload in requests if path == '/v1/chat/completions']
        self.assertEqual(len(generation), 1)
        self.assertEqual(generation[0]['model'], 'qwen3-4b-instruct-2507')
        self.assertEqual(generation[0]['max_tokens'], 768)
        schema = generation[0]['response_format']['json_schema']['schema']
        self.assertEqual(generation[0]['response_format']['type'], 'json_schema')
        variants = schema['properties']['chapters']['anyOf']
        self.assertEqual([item['maxItems'] for item in variants], [1, 2, 3])
        for variant in variants:
            self.assertEqual(variant['items'][0]['properties']['start_segment_id']['const'],
                             'segment-000001')
        self.assertEqual(variants[1]['items'][1]['properties']['start_segment_id']['enum'],
                         ['segment-000002', 'segment-000003'])
        self.assertFalse(service._lock.locked())
        self.assertEqual(result.attempts[0]['finish_reason'], 'stop')

    def test_invalid_boundaries_are_rejected(self):
        """존재하지 않는 ID·누락·중복·역전·빈 목록·추가 필드·잘못된 언어를 거부한다.

        :return: None.
        """
        segments = prepare_study_inputs(transcription(), Mock(return_value=100),
                                        instruction='시험')[0].segments
        bad = ['not JSON', '```json\n' + boundaries(1) + '\n```',
               boundaries(2), boundaries(1, 1), boundaries(1, 3, 2), boundaries(1, 99),
               '{"chapters":[]}', boundaries(1, 2, 3, 3),
               boundaries(1).replace('도함수 주제 1', 'derivative'),
               boundaries(1).replace('"title":', '"start":123,"title":'),
               boundaries(1).replace('segment-000001', 'segment-000001\n')]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_boundaries(raw, segments)

    def test_retry_and_failure_preserve_diagnostics_and_release_lock(self):
        """출력·전송 실패는 두 번까지만 재시도하고 원시 기록을 보존한다.

        :return: None.
        """
        for first in ('invalid', (boundaries(1), 'length'), (boundaries(1), None),
                      httpx.ConnectError('offline')):
            service, requests = self.service([first, boundaries(1)])
            with self.subTest(first=first):
                result = ChapterService(service).generate(transcription())
                self.assertEqual(len(result.attempts), 2)
                self.assertIsNotNone(result.attempts[0]['error'])
                self.assertIsNone(result.attempts[1]['error'])
                self.assertFalse(service._lock.locked())
        service, requests = self.service(['invalid', 'invalid'])
        with self.assertRaises(ChapterGenerationError) as caught:
            ChapterService(service).generate(transcription())
        self.assertEqual(len(caught.exception.attempts), 2)
        self.assertEqual(caught.exception.attempts[-1]['raw'], 'invalid')
        self.assertFalse(service._lock.locked())

    def test_boundary_merge_preserves_ids_and_original_start(self):
        """분할 경계의 같은 주제는 합치고 다른 주제는 그대로 보존한다.

        :return: None.
        """
        for merge in (True, False):
            decision = json.dumps({'merge': merge, 'title': '도함수' if merge else '',
                                   'description': '접선과 도함수를 설명합니다.' if merge else ''})
            service, requests = self.service([boundaries(1), boundaries(2), decision], split=True)
            result = ChapterService(service).generate(transcription(2))
            self.assertEqual(len(result.chapters), 1 if merge else 2)
            self.assertEqual([item for chapter in result.chapters for item in chapter.segment_ids],
                             ['segment-000001', 'segment-000002'])
            self.assertEqual(result.chapters[0].start, 0)
            self.assertEqual(result.chapters[-1].end, 18)
            for path, payload in requests:
                if path == '/v1/chat/completions':
                    self.assertIn(payload['max_tokens'], (512, 768))

    def test_merge_budget_and_invalid_decision_fail_without_partial_success(self):
        """병합 예산 초과·잘못된 병합 판단은 부분 성공을 반환하지 않는다.

        :return: None.
        """
        service, requests = self.service([boundaries(1), boundaries(2)], split=True, merge_tokens=1600)
        with self.assertRaisesRegex(ChapterGenerationError, 'budget') as caught:
            ChapterService(service).generate(transcription(2))
        self.assertEqual(len(caught.exception.attempts), 2)
        self.assertEqual(sum(path == '/v1/chat/completions' for path, _ in requests), 2)
        for raw in ('{"merge":"yes","title":"","description":""}',
                    '{"merge":true,"title":"","description":""}'):
            service, requests = self.service([boundaries(1), boundaries(2), raw, raw], split=True)
            with self.subTest(raw=raw), self.assertRaises(ChapterGenerationError):
                ChapterService(service).generate(transcription(2))

    def test_empty_and_oversize_inputs_do_not_generate(self):
        """빈 자막은 빈 목차를 반환하고 단일 초과 자막은 생성 전에 실패한다.

        :return: None.
        """
        service, requests = self.service([])
        result = ChapterService(service).generate(transcription(0))
        self.assertEqual(result.chapters, [])
        self.assertEqual(requests, [])
        with patch.object(service.tokenizer, 'encode', return_value=list(range(2000))):
            with self.assertRaisesRegex(ValueError, 'segment-000001'):
                ChapterService(service).generate(transcription(1))
        self.assertFalse(any(path == '/v1/chat/completions' for path, _ in requests))

    def test_real_saved_subtitle_shapes_and_evaluation_report(self):
        """실제 두 영상의 저장 자막을 모의 모델로 처리해 전체 근거 연결을 검사한다.

        :return: None.
        """
        paths = [ROOT / 'evaluation' / name for name in (
            'video-rocket-calculus-fix-api-261101.json', 'video-calculus-calculus-fix-api-261101.json')]
        service, requests = self.service([boundaries(1), boundaries(1)])
        with patch.object(evaluate_chapters, 'HttpTranslationService', return_value=service):
            report = evaluate_chapters.evaluate(paths, 'http://translator:8080', date(2026, 11, 12))
        for path, row in zip(paths, report['results']):
            original = TranscriptionResponse.model_validate_json(path.read_bytes())
            self.assertEqual(row['status'], 'completed')
            self.assertEqual(len(row['chapters'][0]['segment_ids']), len(original.segments))
            self.assertEqual(row['chapters'][0]['start'], original.segments[0].start)
            self.assertEqual(row['chapters'][0]['end'], max(item.end for item in original.segments))
            self.assertEqual(len(row['input_sha256']), 64)

    def test_evaluation_rejects_external_server_and_wrong_model(self):
        """외부 서비스 주소·다른 모델은 생성 전에 거부한다.

        :return: None.
        """
        with self.assertRaisesRegex(ValueError, 'local'):
            evaluate_chapters.evaluate([], 'https://api.openai.com', date(2026, 11, 12))
        service, requests = self.service([], wrong_model=True)
        with patch.object(evaluate_chapters, 'HttpTranslationService', return_value=service):
            with self.assertRaisesRegex(ValueError, 'filename'):
                evaluate_chapters.evaluate([], 'http://translator:8080', date(2026, 11, 12))
        self.assertFalse(any(path == '/v1/chat/completions' for path, _ in requests))

    def test_evaluation_cli_does_not_overwrite_results(self):
        """기존 평가 보고서는 모델을 호출하기 전에 덮어쓰기를 거부한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            output.write_text('preserved', encoding='utf-8')
            with patch('sys.argv', ['evaluate_chapters', '--inputs', 'unused.json',
                                   '--evaluated-on', '2026-11-12', '--output', str(output)]), \
                    patch.object(evaluate_chapters, 'evaluate') as evaluate:
                with self.assertRaises(FileExistsError):
                    evaluate_chapters.main()
                evaluate.assert_not_called()
                self.assertEqual(output.read_text(), 'preserved')

    def test_evaluation_records_failed_generation_and_continues(self):
        """한 영상 생성이 실패해도 실패 시도와 시간을 보존하고 다음 영상을 평가한다.

        :return: None.
        """
        path = ROOT / 'evaluation/video-rocket-calculus-fix-api-261101.json'
        service, requests = self.service(['invalid', 'invalid', boundaries(1)])
        with patch.object(evaluate_chapters, 'HttpTranslationService', return_value=service):
            report = evaluate_chapters.evaluate([path, path], 'http://translator:8080', date(2026, 11, 12))
        self.assertEqual([row['status'] for row in report['results']], ['failed', 'completed'])
        self.assertEqual(len(report['results'][0]['attempts']), 2)
        self.assertGreaterEqual(report['results'][0]['elapsed_seconds'], 0)

    def test_evaluation_cli_saves_report_and_signals_failure(self):
        """보고서를 새 파일로 저장하며 실패 행이 있으면 종료 코드 1을 반환한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            report = {'results': [{'input': 'saved.json', 'status': 'failed', 'attempts': []}]}
            with patch('sys.argv', ['evaluate_chapters', '--inputs', 'saved.json',
                                   '--evaluated-on', '2026-11-12', '--output', str(output)]), \
                    patch.object(evaluate_chapters, 'evaluate', return_value=report), \
                    patch('builtins.print'):
                self.assertEqual(evaluate_chapters.main(), 1)
            self.assertEqual(json.loads(output.read_text()), report)


if __name__ == '__main__':
    unittest.main()
