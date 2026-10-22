"""기존 Whisper 특징과의 수치 동등성·전체 음량 정규화·FFT 임시 배열 상한을 검증한다."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from faster_whisper.feature_extractor import FeatureExtractor

from app.services.stt_features import MemoryBoundedFeatureExtractor
from transcribe import TranscriptionService


class SttFeatureTests(unittest.TestCase):
    """모델 다운로드 없이 실제 upstream 특징 추출 결과와 비교한다."""

    def test_matches_upstream_across_boundaries_and_global_volume_changes(self):
        """조용한 구간·큰 음량·경계 프레임에서도 원래 전체 음성 특징과 일치한다.

        :return: None.
        """
        random = np.random.default_rng(42)
        for size, padding in ((1, 160), (16000, 0), (16001, 160), (16000 * 35, 160)):
            with self.subTest(size=size, padding=padding):
                audio = random.normal(0, 1e-6, size).astype(np.float32)
                audio[size // 2:] *= 100000
                original = audio.copy()
                base = FeatureExtractor()
                actual = MemoryBoundedFeatureExtractor(base, block_frames=37)(audio, padding=padding)
                expected = FeatureExtractor()(audio, padding=padding)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(audio, original)
                self.assertEqual(actual.dtype, np.float32)

    def test_fft_batch_size_is_bounded_without_losing_frames(self):
        """긴 입력도 FFT 배치 상한을 지키며 마지막 부분 프레임까지 유지한다.

        :return: None.
        """
        audio = np.zeros(16000 * 65 + 13, dtype=np.float32)
        extractor = MemoryBoundedFeatureExtractor(FeatureExtractor(), block_frames=500)
        with patch('app.services.stt_features.np.fft.rfft', wraps=np.fft.rfft) as fft:
            result = extractor(audio)
        self.assertGreater(fft.call_count, 10)
        self.assertTrue(all(call.args[0].shape[0] <= 500 for call in fft.call_args_list))
        self.assertEqual(result.shape, FeatureExtractor()(audio).shape)
        np.testing.assert_array_equal(result, FeatureExtractor()(audio))

    def test_chunk_settings_and_invalid_inputs(self):
        """창 설정을 기존 전사기에 전달하고 잘못된 블록 크기·다채널 입력을 거부한다.

        :return: None.
        """
        base = FeatureExtractor(feature_size=128)
        extractor = MemoryBoundedFeatureExtractor(base)
        result = extractor(np.zeros(16000, dtype=np.float32), chunk_length=20)
        self.assertEqual(extractor.n_samples, 320000)
        self.assertEqual(extractor.nb_max_frames, 2000)
        self.assertEqual(result.shape[0], 128)
        for value in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                MemoryBoundedFeatureExtractor(base, value)
        with self.assertRaises(ValueError):
            extractor(np.zeros((2, 16000), dtype=np.float32))

    def test_lazy_model_is_wrapped_once_and_reused(self):
        """지연 로드한 모델에만 특징 추출기를 적용하고 중복 모델을 생성하지 않는다.

        :return: None.
        """
        model = SimpleNamespace(feature_extractor=FeatureExtractor())
        service = TranscriptionService()
        with patch('faster_whisper.WhisperModel', return_value=model) as load:
            self.assertIs(service._get_model(), model)
            self.assertIs(service._get_model(), model)
        self.assertIsInstance(model.feature_extractor, MemoryBoundedFeatureExtractor)
        load.assert_called_once_with('small', device='cpu', compute_type='int8')
