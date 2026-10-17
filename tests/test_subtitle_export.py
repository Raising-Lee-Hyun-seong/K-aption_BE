"""자막 파일의 시간·텍스트·부분 실패와 작업 다운로드 HTTP 계약을 검증한다."""
import io
import unittest

import av

from app.main import app
from app.schemas import TranscriptionResponse
from app.services.subtitle_export import render_subtitles
from tests import test_jobs


def sample_result() -> TranscriptionResponse:
    """정상 번역과 번역 실패가 섞인 자막 응답을 만든다.

    :return: 시각·영어·한국어 또는 실패를 포함하는 두 구간 응답.
    """
    return TranscriptionResponse(segments=[
        {'start': 3.6, 'end': 7.44, 'text': 'The velocity is constant.',
         'translation': '속도는 일정합니다.', 'translation_error': None},
        {'start': 8, 'end': 10, 'text': 'The net force is zero.',
         'translation': None, 'translation_error': 'translation_failed'},
    ])


class SubtitleExportTests(unittest.TestCase):
    """실제 자막 텍스트와 독립 FFmpeg 디코더로 변환 결과를 확인한다."""

    def test_srt_exact_timing_and_language_fallback(self):
        """SRT의 밀리초·구분자·한영 순서·실패 원문 대체를 정확히 검증한다.

        :return: None.
        """
        for language, first in (
            ('ko', '속도는 일정합니다.'), ('en', 'The velocity is constant.'),
            ('both', '속도는 일정합니다.\r\nThe velocity is constant.'),
        ):
            expected = ('1\r\n00:00:03,600 --> 00:00:07,440\r\n' + first
                        + '\r\n\r\n2\r\n00:00:08,000 --> 00:00:10,000\r\n'
                        + 'The net force is zero.\r\n\r\n')
            self.assertEqual(render_subtitles(sample_result(), 'srt', language), expected)

    def test_vtt_escapes_text_and_keeps_bilingual_order(self):
        """VTT는 원문의 태그·수식·빈 줄이 자막 구조나 마크업으로 해석되지 않게 한다.

        :return: None.
        """
        result = TranscriptionResponse(segments=[{
            'start': 0, 'end': 1, 'text': '<b>x</b> < 0 & y --> z\n\nextra',
            'translation': '값은 < 0 & 다음입니다.', 'translation_error': None,
        }])
        text = render_subtitles(result, 'vtt', 'both')
        self.assertEqual(text, 'WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n'
                         '값은 &lt; 0 &amp; 다음입니다.\n'
                         '&lt;b&gt;x&lt;/b&gt; &lt; 0 &amp; y --&gt; z extra\n\n')

    def test_rounding_hour_carry_and_submillisecond_interval(self):
        """시·분 경계 반올림과 1ms 미만 구간의 양수 길이를 보장한다.

        :return: None.
        """
        result = sample_result()
        result.segments = [result.segments[0].model_copy(update={'start': 3599.9995, 'end': 3600.0001})]
        self.assertIn('01:00:00,000 --> 01:00:00,001', render_subtitles(result))
        result.segments[0].start, result.segments[0].end = 0.0001, 0.0002
        self.assertIn('00:00:00.000 --> 00:00:00.001', render_subtitles(result, 'vtt'))

    def test_sorting_overlap_and_no_input_mutation(self):
        """입력을 수정하지 않고 시작순으로 정렬하며 겹친 구간을 임의로 재배치하지 않는다.

        :return: None.
        """
        result = sample_result()
        result.segments.reverse()
        result.segments[0].start = 5
        before = result.model_dump()
        text = render_subtitles(result, 'vtt')
        self.assertLess(text.index('00:00:03.600'), text.index('00:00:05.000'))
        self.assertIn('00:00:03.600 --> 00:00:07.440', text)
        self.assertEqual(result.model_dump(), before)

    def test_empty_and_blank_translation_fallback(self):
        """무음 문서와 빈 번역·오류가 있는 번역의 원문 대체를 확인한다.

        :return: None.
        """
        self.assertEqual(render_subtitles(TranscriptionResponse(segments=[])), '')
        self.assertEqual(render_subtitles(TranscriptionResponse(segments=[]), 'vtt'), 'WEBVTT\n\n')
        result = sample_result()
        result.segments = [result.segments[0]]
        for translation, error in ((' \n\t', None), ('무시할 번역', 'translation_failed')):
            result.segments[0].translation, result.segments[0].translation_error = translation, error
            text = render_subtitles(result, 'vtt', 'both')
            self.assertEqual(text.count('The velocity is constant.'), 1)

    def test_invalid_timeline_text_and_options_are_rejected(self):
        """형식·언어·역전 시각·비유한 시각·빈 원문을 유효한 파일로 내보내지 않는다.

        :return: None.
        """
        for file_format, language in (('ass', 'ko'), ('vtt', 'fr')):
            with self.assertRaises(ValueError):
                render_subtitles(sample_result(), file_format, language)
        for start, end in ((2, 1), (1, 1), (0, float('inf'))):
            result = sample_result()
            result.segments[0].start, result.segments[0].end = start, end
            with self.assertRaises(ValueError):
                render_subtitles(result)
        result = sample_result()
        result.segments[0].text = '\x00\n\t'
        with self.assertRaises(ValueError):
            render_subtitles(result)

    def test_independent_decoder_preserves_korean_and_math_symbols(self):
        """FFmpeg로 두 형식을 읽어 한국어·수식과 타임스탬프 위장 원문의 한 구간 처리를 확인한다.

        :return: None.
        """
        result = TranscriptionResponse(segments=[{
            'start': 0, 'end': 1, 'text': '00:00:10,000 --> 00:00:11,000\n\nextra',
            'translation': '값은 x < 0 & y > 1입니다.', 'translation_error': None,
        }])
        for file_format, decoder in (('srt', 'srt'), ('vtt', 'webvtt')):
            payload = render_subtitles(result, file_format, 'ko').encode('utf-8')
            with av.open(io.BytesIO(payload), format=decoder) as container:
                subtitles = [subtitle for packet in container.demux() for subtitle in packet.decode()]
            self.assertEqual(len(subtitles), 1)
            self.assertIn('값은 x < 0 & y > 1입니다.', subtitles[0].ass.decode())
            payload = render_subtitles(result, file_format, 'en').encode('utf-8')
            with av.open(io.BytesIO(payload), format=decoder) as container:
                subtitles = [subtitle for packet in container.demux() for subtitle in packet.decode()]
            self.assertEqual(len(subtitles), 1)


class SubtitleExportApiTests(test_jobs.JobTestCase):
    """작업 결과 다운로드의 응답·오류·저장 부수 효과를 확인한다."""

    request = test_jobs.JobApiTests.request

    async def test_download_modes_inline_headers_and_no_reprocessing(self):
        """완료 결과에서 언어·형식을 선택해 내려받고 재추론·추가 저장을 하지 않는다.

        :return: None.
        """
        self.translator.translate.side_effect = [{'translation': '첫 문장입니다.'}, ValueError('bad')]
        accepted = await self.submit()
        await self.finish()
        directory = self.root / accepted.job_id
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        stt_calls, translation_calls = self.stt.transcribe.call_count, self.translator.translate.call_count
        for file_format in ('srt', 'vtt'):
            for language in ('ko', 'en', 'both'):
                query = f'format={file_format}&language={language}'.encode()
                status, body, headers = await self.request('GET', f'/api/v1/jobs/{accepted.job_id}/subtitles',
                                                         query=query, decode_json=False)
                self.assertEqual(status, 200)
                self.assertIn(b'charset=utf-8', headers[b'content-type'])
                self.assertIn(f'-{language}.{file_format}'.encode(), headers[b'content-disposition'])
                self.assertTrue(headers[b'content-disposition'].startswith(b'attachment;'))
                self.assertEqual(headers[b'cache-control'], b'no-store')
                self.assertIn('Second.', body.decode())
        status, _, headers = await self.request('GET', f'/api/v1/jobs/{accepted.job_id}/subtitles',
                                               query=b'format=vtt&download=false', decode_json=False)
        self.assertEqual(status, 200)
        self.assertEqual(headers[b'content-type'], b'text/vtt; charset=utf-8')
        self.assertTrue(headers[b'content-disposition'].startswith(b'inline;'))
        self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})
        self.assertEqual((stt_calls, translation_calls),
                         (self.stt.transcribe.call_count, self.translator.translate.call_count))

    async def test_export_error_contract_and_expiry(self):
        """옵션·없는 작업·실패·손상·만료가 기존 작업 결과와 같은 상태로 구분된다.

        :return: None.
        """
        base = '/api/v1/jobs/' + '0' * 32 + '/subtitles'
        for query in (b'format=ass', b'language=fr', b'download=invalid'):
            status, _, _ = await self.request('GET', base, query=query)
            self.assertEqual(status, 422)
        status, _, _ = await self.request('GET', base)
        self.assertEqual(status, 404)
        self.stt.transcribe.side_effect = RuntimeError('private')
        accepted = await self.submit()
        await self.finish()
        path = f'/api/v1/jobs/{accepted.job_id}/subtitles'
        status, _, _ = await self.request('GET', path)
        self.assertEqual(status, 409)
        self.stt.transcribe.side_effect = None
        accepted = await self.submit()
        await self.finish()
        path = f'/api/v1/jobs/{accepted.job_id}/subtitles'
        (self.root / accepted.job_id / 'result.json').write_text('{broken')
        status, body, _ = await self.request('GET', path)
        self.assertEqual(status, 500)
        self.assertNotIn('private', str(body))
        self.clock.return_value += 86401
        status, _, _ = await self.request('GET', path)
        self.assertEqual(status, 404)

    async def test_empty_result_and_invalid_export_timeline(self):
        """무음 SRT·VTT는 200으로 제공하고 손상된 시간 관계는 500으로 거부한다.

        :return: None.
        """
        self.stt.transcribe.return_value = {'segments': []}
        accepted = await self.submit()
        await self.finish()
        path = f'/api/v1/jobs/{accepted.job_id}/subtitles'
        for query, expected in ((b'format=srt', b''), (b'format=vtt', b'WEBVTT\n\n')):
            status, body, _ = await self.request('GET', path, query=query, decode_json=False)
            self.assertEqual((status, body), (200, expected))
        result = sample_result()
        result.segments[0].end = result.segments[0].start
        (self.root / accepted.job_id / 'result.json').write_text(result.model_dump_json())
        status, _, _ = await self.request('GET', path)
        self.assertEqual(status, 500)

    async def test_openapi_documents_binary_text_formats(self):
        """다운로드 옵션과 실제 자막 미디어 유형이 OpenAPI에 포함된다.

        :return: None.
        """
        operation = app.openapi()['paths']['/api/v1/jobs/{job_id}/subtitles']['get']
        self.assertEqual(set(operation['responses']['200']['content']), {'application/x-subrip', 'text/vtt'})
        self.assertTrue({'format', 'language', 'download'} <= {p['name'] for p in operation['parameters']})
