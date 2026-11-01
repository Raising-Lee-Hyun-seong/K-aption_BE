"""물리적 일과 계산을 뜻하는 work out의 문맥별 용어 선택을 검증한다."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from app.services.translation_http import HttpTranslationService
from app.services.translation_qwen import render_chatml


ROOT = Path(__file__).resolve().parents[1]


class GlossaryContextTests(unittest.TestCase):
    """프롬프트와 필수 용어 검증이 같은 문맥 구분을 사용하는지 확인한다."""

    def service(self):
        """모델을 로드하지 않고 실제 Qwen 프롬프트와 검증 서비스를 준비한다.

        :return: Qwen 용어 설정과 채팅 템플릿 대역을 갖춘 번역 서비스.
        """
        service = HttpTranslationService.__new__(HttpTranslationService)
        service.model_family = 'qwen3'
        service._configure_text(ROOT / 'templates/korean.json')
        service._configure_qwen()
        service.tokenizer = SimpleNamespace(
            encode=Mock(return_value=[1]),
            apply_chat_template=Mock(side_effect=render_chatml))
        return service

    def test_work_out_does_not_require_physics_work(self):
        """인접·분리된 계산 구동사는 물리적 일 용어나 불필요한 재시도를 요구하지 않는다.

        :return: None.
        """
        service = self.service()
        for source in ('I want to work out a better formula for this.',
                       'We will work this out by the end of this unit.',
                       'We will work it out.',
                       'We will work the formula out.',
                       'We will work this better formula out.',
                       'We will WORK THAT OUT.'):
            with self.subTest(source=source):
                self.assertNotIn('work', [entry['en'] for entry in service._matching_terms(source)])
                prompt = service._build_prompt(source, '')
                self.assertNotIn('work=일', prompt)
                self.assertIsNone(service._rejection_reason('더 나은 공식을 구하겠습니다.', source, ''))
        service._generate_output = Mock(return_value={
            'text': '더 나은 공식을 구하겠습니다.', 'tokens': 12,
            'finish_reason': 'stop', 'stop_reason': None})
        result = service.translate('I want to work out a better formula for this.', '', 128)
        self.assertEqual(result['translation'], '더 나은 공식을 구하겠습니다.')
        self.assertEqual(service._generate_output.call_count, 1)

    def test_physics_work_is_still_required(self):
        """물리적 일과 같은 문장에 남은 일은 계산 구동사와 함께 있어도 검증한다.

        :return: None.
        """
        service = self.service()
        for source in ('The work done is positive.',
                       'Work done by the force is positive.',
                       'We work out the work done.',
                       'The work is positive, and we can work it out.',
                       'The work that is carried out is positive.',
                       'The work the engine puts out is positive.',
                       'We calculate the work the force does out of this system.',
                       'The work out of the engine is positive.'):
            with self.subTest(source=source):
                self.assertIn('work', [entry['en'] for entry in service._matching_terms(source)])
                output = '힘은 양수입니다.' if 'force' in source else '양수입니다.'
                self.assertEqual(service._rejection_reason(output, source, ''), 'missing_term')
                self.assertIsNone(service._rejection_reason('힘이 한 일은 양수입니다.', source, ''))

    def test_other_terms_and_longer_compounds_remain_unchanged(self):
        """구동사 제외는 다른 용어와 긴 복합 용어의 우선 선택에 영향을 주지 않는다.

        :return: None.
        """
        service = self.service()
        glossary = [{'en': 'work', 'ko': '일'},
                    {'en': 'force', 'ko': '힘'},
                    {'en': 'net force', 'ko': '알짜힘'}]
        self.assertEqual(service._matching_terms('We work out the net force.', glossary),
                         [glossary[2]])
        self.assertEqual(service._matching_terms('We work out the net force and the force.', glossary),
                         [glossary[2], glossary[1]])
        self.assertEqual(service._matching_terms('The network is outside.', glossary), [])
        custom = [{'en': 'work function', 'ko': '일 함수'}, {'en': 'work', 'ko': '일'}]
        self.assertEqual(service._matching_terms('The work function is known.', custom), [custom[0]])


if __name__ == '__main__':
    unittest.main()
