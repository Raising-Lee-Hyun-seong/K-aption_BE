"""STT의 짧은 영어 조각을 문장과 시간 경계에 따라 자막 번역 단위로 묶는다."""
import re

SENTENCE_END = re.compile(r"[.!?][\"'’”\)\]]*$")
MAX_GAP_SECONDS = 2.0


def group_segments(
    segments: list[dict], max_duration: float = 18.0, max_chars: int = 350
) -> list[dict]:
    """순서를 유지하며 STT 조각을 묶고 첫·마지막 시각과 원본 인덱스를 반환한다.

    문장 끝의 마침표·물음표·느낌표, 두 구간 사이 2초를 초과하는 공백,
    합친 길이·시간 상한에서 그룹을 닫는다. 빈 원문은 제외한다. 원본 구간을
    분할하지 않으므로 단일 구간이 상한을 넘으면 시간과 원문을 그대로 보존한다.
    상한이 양수가 아니면 ValueError를 발생시킨다.

    :param segments: 시작·종료 시각과 text를 담은 STT 구간 목록.
    :param max_duration: 여러 구간을 합칠 때의 최대 시간(초). 긴 단일 구간은 보존한다.
    :param max_chars: 합친 원문에 허용할 최대 문자 수. 연결 공백을 포함한다.
    :return: start·end·text·source_indices를 담은 자막 묶음 목록.
    :raises ValueError: 최대 시간 또는 최대 문자 수가 양수가 아닐 때.
    """
    if max_duration <= 0 or max_chars <= 0:
        raise ValueError("Subtitle grouping limits must be positive")

    grouped = []
    pending = None
    for index, segment in enumerate(segments):
        text = " ".join(segment["text"].split())
        if not text:
            continue
        if pending is not None:
            gap = segment["start"] - pending["end"]
            duration = segment["end"] - pending["start"]
            chars = len(pending["text"]) + 1 + len(text)
            if gap > MAX_GAP_SECONDS or duration > max_duration or chars > max_chars:
                grouped.append(pending)
                pending = None

        if pending is None:
            pending = {
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
                "source_indices": [index],
            }
        else:
            pending["end"] = segment["end"]
            pending["text"] += " " + text
            pending["source_indices"].append(index)

        if SENTENCE_END.search(text):
            grouped.append(pending)
            pending = None

    if pending is not None:
        grouped.append(pending)
    return grouped
