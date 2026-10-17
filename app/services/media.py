"""업로드 파일의 저장 크기와 미디어 재생 길이를 검사한다."""
from pathlib import Path
from typing import BinaryIO

MAX_UPLOAD_BYTES = 1024 ** 3
MAX_MEDIA_SECONDS = 90 * 60
UPLOAD_TOO_LARGE = "The uploaded file exceeds the 1 GiB limit"
MEDIA_TOO_LONG = "The uploaded media exceeds the 90 minute limit"


class UploadTooLargeError(ValueError):
    """업로드 파일의 실제 바이트 수가 저장 상한을 초과했음을 나타낸다."""


def copy_upload(source: BinaryIO, destination: BinaryIO) -> int:
    """파일을 작은 청크로 복사하고 크기 상한을 초과하면 저장을 중단한다.

    :param source: 현재 위치부터 읽을 업로드 바이너리 스트림.
    :param destination: 업로드 데이터를 기록할 임시 파일 스트림.
    :return: 성공적으로 복사한 총 바이트 수.
    :raises UploadTooLargeError: 입력이 1GiB를 초과할 때. 이미 기록된 데이터는 호출자가 삭제한다.
    :raises OSError: 파일 읽기 또는 저장에 실패할 때.
    """
    total = 0
    while chunk := source.read(min(1024 * 1024, MAX_UPLOAD_BYTES - total + 1)):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError(UPLOAD_TOO_LARGE)
        destination.write(chunk)
    return total


def validate_media_duration(media_path: Path) -> None:
    """미디어의 알려진 컨테이너·스트림 길이를 확인해 90분 초과 파일을 거부한다.

    파일은 검사 후 닫는다. 길이 메타데이터가 없으면 디코딩 단계의 실제
    오디오 길이 검사로 넘기며, 메타데이터만으로 허용 여부를 확정하지 않는다.

    :param media_path: 길이를 검사할 영상 또는 음성 파일 경로.
    :return: None.
    :raises ValueError: 미디어를 열 수 없거나 알려진 재생 길이가 90분을 초과할 때.
    """
    import av
    from av.error import FFmpegError

    try:
        with av.open(str(media_path), metadata_errors="ignore") as container:
            durations = []
            if container.duration is not None:
                durations.append(container.duration / av.time_base)
            for stream in container.streams:
                if stream.type in {"audio", "video"} and stream.duration is not None:
                    if stream.time_base is not None:
                        durations.append(stream.duration * stream.time_base)
            if any(duration > MAX_MEDIA_SECONDS for duration in durations):
                raise ValueError(MEDIA_TOO_LONG)
    except (FFmpegError, OSError) as exc:
        raise ValueError("The uploaded media does not contain decodable audio") from exc
