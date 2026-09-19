"""MP4의 음성을 faster-whisper로 인식해 UTF-8 텍스트로 저장합니다."""

import argparse
from pathlib import Path
from time import perf_counter


def main():
    total_start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="입력 MP4 경로")
    parser.add_argument("-o", "--output", type=Path, help="출력 TXT 경로")
    parser.add_argument("--model", default="small", help="모델 이름 (기본: small)")
    parser.add_argument("--language", default=None, help="언어 코드: ko, en 등 (기본: 자동 감지)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compute-type", default=None, help="기본: CPU는 int8, CUDA는 float16")
    parser.add_argument("--timestamps", action="store_true", help="텍스트에 구간별 시작/종료 시간 포함")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else video.with_suffix(".txt")
    if not video.is_file():
        parser.error(f"입력 파일을 찾을 수 없습니다: {video}")
    if output == video:
        parser.error("출력 경로는 입력 영상과 달라야 합니다.")
    if output.exists():
        parser.error(f"출력 파일이 이미 있습니다. -o로 다른 경로를 지정하세요: {output}")

    from faster_whisper import WhisperModel

    compute_type = args.compute_type or ("int8" if args.device == "cpu" else "float16")
    print(f"모델 로딩: {args.model} / {args.device} / {compute_type}", flush=True)
    model_start = perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    model_seconds = perf_counter() - model_start

    # PyAV가 MP4의 오디오를 직접 읽으므로 WAV로 미리 변환할 필요가 없습니다.
    transcription_start = perf_counter()
    segments, info = model.transcribe(
        str(video),
        language=args.language,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
    )
    print(f"인식 언어: {info.language} (확률: {info.language_probability:.1%})", flush=True)

    # segments는 지연 실행되는 generator이므로 순회해야 실제 인식이 진행됩니다.
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        timed_text = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}"
        print(timed_text, flush=True)
        lines.append(timed_text if args.timestamps else text)

    # 오디오 디코딩, VAD, 언어 감지와 generator 순회를 모두 포함합니다.
    transcription_seconds = perf_counter() - transcription_start

    # 인식 완료 후 저장하여 도중 실패 시 불완전한 결과 파일을 만들지 않습니다.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"저장 완료: {output} ({len(lines)}개 구간)")
    if not lines:
        print("인식된 대사가 없습니다.")

    total_seconds = perf_counter() - total_start
    print("\n소요 시간")
    print(f"  모델 로딩: {model_seconds:.2f}초 (최초 다운로드 시 다운로드 시간 포함)")
    print(f"  음성 인식: {transcription_seconds:.2f}초 (전처리 및 구간 출력 포함)")
    print(f"  전체 실행: {total_seconds:.2f}초 ({total_seconds / 60:.2f}분, 파일 저장 포함)")


if __name__ == "__main__":
    main()
