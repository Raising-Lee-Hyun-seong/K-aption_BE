"""Whisper의 전역 log-Mel 정규화를 유지하면서 FFT 임시 배열을 작은 블록으로 제한한다."""

import numpy as np

from timing import measure


class MemoryBoundedFeatureExtractor:
    """기존 특징 설정을 위임하고 동일한 프레임·Mel 필터를 블록 단위로 계산한다."""

    def __init__(self, extractor, block_frames: int = 3000):
        """기존 faster-whisper 특징 추출기와 FFT 블록 상한을 보관한다.

        :param extractor: n_fft·hop_length·mel_filters 등 기존 Whisper 특징 설정 객체.
        :param block_frames: 한 FFT에 전달할 최대 프레임 수. 양의 정수. 기본값은 30초 분량이다.
        :return: None. 오디오 분할·모델 로딩·전사 문맥 변경은 하지 않는다.
        :raises ValueError: 블록 상한이 양의 정수가 아닌 경우.
        """
        if type(block_frames) is not int or block_frames <= 0:
            raise ValueError('FFT block size must be a positive integer')
        self.extractor, self.block_frames = extractor, block_frames

    def __getattr__(self, name: str):
        """Whisper 전사기가 읽는 특징 설정을 원래 추출기에서 제공한다.

        :param name: sampling_rate·nb_max_frames·time_per_frame 등 속성 이름.
        :return: 원래 특징 추출기의 속성값.
        :raises AttributeError: 원래 추출기에 해당 속성이 없는 경우.
        """
        return getattr(self.extractor, name)

    def __call__(self, waveform: np.ndarray, padding=160, chunk_length=None) -> np.ndarray:
        """단일 채널 전체 음성의 특징을 계산하되 큰 STFT·FFT 배열 생성을 피한다.

        :param waveform: 16kHz 등 원래 sampling_rate의 1차원 음성 배열. 읽기만 한다.
        :param padding: 원래 Whisper와 동일하게 오른쪽에 추가할 0 샘플 수. 기본 160.
        :param chunk_length: Whisper의 창 길이 초. None이면 기존 설정을 유지한다.
        :return: 원래 프레임 수·전역 정규화를 유지한 float32 log-Mel 특징 배열.
        :raises ValueError: 단일 채널 1차원 입력이 아니거나 패딩 설정이 유효하지 않은 경우.

        최종 특징·패딩 오디오는 음성 길이에 비례하지만 FFT 임시 배열은 block_frames로 제한한다.
        chunk_length를 지정하면 원래 추출기의 n_samples·nb_max_frames 설정도 갱신한다.
        """
        if waveform.ndim != 1:
            raise ValueError('Whisper features require mono audio')
        if chunk_length is not None:
            self.extractor.n_samples = chunk_length * self.sampling_rate
            self.extractor.nb_max_frames = self.n_samples // self.hop_length
        with measure('stt_features', audio_seconds=waveform.size / self.sampling_rate,
                     block_frames=self.block_frames) as metrics:
            waveform = waveform.astype(np.float32, copy=False)
            if padding:
                waveform = np.pad(waveform, (0, padding))
            padded = np.pad(waveform, (self.n_fft // 2, self.n_fft // 2), mode='reflect')
            del waveform
            # 원래 STFT의 마지막 프레임 하나를 제외하는 규칙을 그대로 적용한다.
            frames = (padded.size - self.n_fft) // self.hop_length
            features = np.empty((self.mel_filters.shape[0], frames), dtype=np.float32)
            window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)
            for start in range(0, frames, self.block_frames):
                end = min(start + self.block_frames, frames)
                samples = padded[start * self.hop_length:(end - 1) * self.hop_length + self.n_fft]
                windows = np.lib.stride_tricks.sliding_window_view(samples, self.n_fft)[::self.hop_length]
                spectrum = np.fft.rfft(windows * window, n=self.n_fft, axis=-1).T.astype(np.complex64)
                magnitudes = np.abs(spectrum) ** 2
                features[:, start:end] = self.mel_filters @ magnitudes
            # 블록별 최대값을 쓰면 음량이 다른 구간의 특징이 바뀌므로 전체 최대값으로 정규화한다.
            np.clip(features, a_min=1e-10, a_max=None, out=features)
            np.log10(features, out=features)
            np.maximum(features, features.max() - 8.0, out=features)
            features += 4.0
            features /= 4.0
            metrics['feature_frames'] = frames
            return features
