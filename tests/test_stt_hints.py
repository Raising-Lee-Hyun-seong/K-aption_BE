"""선택적 STT 용어 힌트 전달과 기본 전사·시간 범위 계약을 검증한다."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from transcribe import TranscriptionService


class SttHintTests(unittest.TestCase):
    """실제 모델 대신 전사 대역으로 힌트와 오디오 검증 동작을 검사한다."""

    def test_optional_hints_reach_model_without_changing_the_raw_text(self):
        """용어 힌트를 엔진에 전달하되 숫자·함수 원문과 시각을 수정하지 않는다.

        :return: None.
        """
        for prompt, words in ((None, None), ('A physics lecture.', 'prime, integral')):
            with self.subTest(prompt=prompt):
                service = TranscriptionService(initial_prompt=prompt, hotwords=words)
                model = Mock()
                model.transcribe.return_value = (iter([SimpleNamespace(
                    start=0.1, end=0.9, text=' 2 vR prime of t. ')]),
                    SimpleNamespace(language='en', language_probability=1.0))
                service._model = model
                with patch('transcribe.validate_media_duration'), patch(
                        'faster_whisper.audio.decode_audio', return_value=np.zeros(16000)):
                    result = service.transcribe(Path('fixture.wav'))
                self.assertEqual(model.transcribe.call_args.kwargs['initial_prompt'], prompt)
                self.assertEqual(model.transcribe.call_args.kwargs['hotwords'], words)
                self.assertEqual(model.transcribe.call_args.kwargs['beam_size'], 5)
                self.assertTrue(model.transcribe.call_args.kwargs['vad_filter'])
                self.assertEqual(result['segments'], [{'start': 0.1, 'end': 0.9,
                                                      'text': '2 vR prime of t.'}])
