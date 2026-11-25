"""모델 호출 없이 용어집 자동 생성의 원문 결합과 출력 파싱·필터를 검증한다."""
import unittest

from scripts.build_glossary import chunks, merge_terms, parse_terms, transcript_text

TRANSCRIPT = 'The lac operon is regulated. The repressor binds lactose. We use a for loop and break. PCR amplifies DNA.'


class BuildGlossaryTests(unittest.TestCase):
    """모델 출력이 지저분해도 원문에 나온 용어만 한 번씩 남기는지 검증한다."""

    def test_transcript_joins_segments_and_truncates(self):
        """구간 텍스트를 공백으로 잇고 최대 글자 수로 자른다.

        :return: None.
        """
        response = {'segments': [{'text': ' Hello '}, {'text': 'world.'}]}
        self.assertEqual(transcript_text(response), 'Hello world.')
        self.assertEqual(transcript_text(response, max_chars=5), 'Hello')

    def test_parse_keeps_terms_found_in_transcript(self):
        """코드 블록 속 JSON을 읽고, 원문에 없는 용어·중복·빈 번역어를 뺀다.

        :return: None.
        """
        output = ('Here you go:\n```json\n[{"en": "lac operon", "ko": "lac 오페론"}, {"en": "repressor", "ko": "억제인자"},'
                  '{"en": "Repressor", "ko": "리프레서"}, {"en": "ribosome", "ko": "리보솜"}, {"en": "lactose", "ko": ""},'
                  '{"en": "break", "ko": "break"}, {"en": "PCR", "ko": "PCR"}, {"en": "loop", "ko": "loop"}]\n```')
        self.assertEqual(parse_terms(output, TRANSCRIPT), [
            {'en': 'lac operon', 'ko': 'lac 오페론'}, {'en': 'repressor', 'ko': '억제인자'},
            {'en': 'break', 'ko': 'break'}, {'en': 'PCR', 'ko': 'PCR'}, {'en': 'loop', 'ko': 'loop'}])

    def test_parse_drops_single_letters_and_foreign_script(self):
        """g·a 같은 한 글자 용어와 한자가 섞인 번역어('较단한 사슬')를 버린다.

        :return: None.
        """
        output = '[{"en": "a", "ko": "에이"}, {"en": "repressor", "ko": "较억제"}, {"en": "lactose", "ko": "락토스"}]'
        self.assertEqual(parse_terms(output, TRANSCRIPT), [{'en': 'lactose', 'ko': '락토스'}])

    def test_parse_rejects_broken_output_and_respects_limit(self):
        """JSON이 없거나 깨지면 빈 목록, 개수 제한을 넘으면 앞에서부터 자른다.

        :return: None.
        """
        self.assertEqual(parse_terms('no json here', TRANSCRIPT), [])
        self.assertEqual(parse_terms('[{"en": "lac operon",', TRANSCRIPT), [])
        output = '[{"en": "repressor", "ko": "억제인자"}, {"en": "lactose", "ko": "락토스"}]'
        self.assertEqual(len(parse_terms(output, TRANSCRIPT, limit=1)), 1)

    def test_chunks_respect_size_and_merge_prefers_first(self):
        """원문을 문장 경계로 나누고, 조각별 결과를 합칠 때 먼저 나온 번역어를 남긴다.

        :return: None.
        """
        parts = chunks('A b c. D e f. G h i.', size=8)
        self.assertEqual(parts, ['A b c.', 'D e f.', 'G h i.'])
        merged = merge_terms([[{'en': 'lactose', 'ko': '락토스'}], [{'en': 'Lactose', 'ko': '젖당'},
                                                                   {'en': 'glucose', 'ko': '포도당'}]], 10)
        self.assertEqual(merged, [{'en': 'lactose', 'ko': '락토스'}, {'en': 'glucose', 'ko': '포도당'}])


if __name__ == '__main__':
    unittest.main()
