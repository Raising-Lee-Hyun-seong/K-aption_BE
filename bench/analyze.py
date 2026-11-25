"""벤치마크 결과(bench/out)를 클립별 속도·STT WER·번역 실패·토큰·예상 비용 표로 집계한다."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.estimate_cost import parse_events  # noqa: E402
from scripts.evaluate_lecture_quality import normalize_words, word_error_counts  # noqa: E402

CUE_TIME = re.compile(r'(\d+):(\d\d):(\d\d)[.,](\d+)\s*-->\s*(\d+):(\d\d):(\d\d)[.,](\d+)')


def read_text(path: Path) -> str:
    """PowerShell 5의 UTF-16 출력과 UTF-8 파일을 모두 읽는다.

    :param path: 읽을 파일 경로. 없으면 빈 문자열.
    :return: 디코딩한 문자열.
    """
    if not path.exists():
        return ''
    data = path.read_bytes()
    if data[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return data.decode('utf-16', errors='replace')
    return data.decode('utf-8-sig', errors='replace')


def parse_vtt(text: str) -> list:
    """WebVTT를 시작·종료(초)와 태그·화자 표기를 지운 텍스트 큐 목록으로 바꾼다.

    :param text: VTT 파일 내용.
    :return: start·end·text 사전 목록. [MUSIC] 같은 효과음 표기와 'PROFESSOR:' 화자명은 제거한다.
    """
    cues, current = [], None
    for line in text.splitlines():
        match = CUE_TIME.search(line)
        if match:
            g = [int(v) for v in match.groups()]
            current = {'start': g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 10 ** len(match.group(4)),
                       'end': g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 10 ** len(match.group(8)),
                       'text': ''}
            cues.append(current)
        elif current is not None and line.strip():
            cleaned = re.sub(r'<[^>]+>|\[[^\]]*\]|\([A-Z ]+\)', ' ', line)
            cleaned = re.sub(r'^\s*[A-Z][A-Z .\'-]{1,30}:', ' ', cleaned)
            current['text'] += ' ' + cleaned.strip()
    return [c for c in cues if c['text'].strip()]


def request_events(events: list) -> list:
    """시간 로그를 HTTP 요청 순서대로 묶는다.

    :param events: parse_events 결과.
    :return: 요청 ID 등장 순서대로 정렬한 (request_id, 이벤트 목록) 목록. 전사 요청만 남긴다.
    """
    order, grouped = [], {}
    for event in events:
        rid = event.get('request_id', '-')
        if rid == '-':
            continue
        if rid not in grouped:
            grouped[rid] = []
            order.append(rid)
        grouped[rid].append(event)
    return [(rid, grouped[rid]) for rid in order
            if any(e.get('event') == 'stt_inference' for e in grouped[rid])]


def summarize_clip(name: str, response: dict, events: list, vtt_text: str, client_seconds: float) -> dict:
    """한 클립의 응답·서버 로그·기준 자막으로 속도·정확도·실패·토큰 지표를 계산한다.

    :return: 클립 지표 사전. 기준 자막이 없으면 WER은 None.
    """
    segments = response.get('segments', [])
    phase = {}
    for e in events:
        if e.get('event') in ('stt', 'translation_batch', 'sentence_grouping', 'http_request'):
            phase[e['event']] = float(e['elapsed_seconds'])
    audio = next((float(e['audio_seconds']) for e in events
                  if e.get('event') == 'stt_inference' and 'audio_seconds' in e), None)
    attempts = [e for e in events if e.get('event') == 'translation_attempt']
    reasons = Counter(e.get('rejection_reason') for e in attempts if e.get('rejection_reason') not in (None, 'None'))
    prompt = sum(int(e.get('prompt_tokens', 0)) for e in attempts)
    output = sum(int(e.get('raw_output_tokens', 0)) for e in attempts)
    wer = None
    if vtt_text:
        cues = parse_vtt(vtt_text)
        if cues:
            first = cues[0]['start']
            hyp = ' '.join(s['text'] for s in segments if s['end'] > first - 0.5)
            wer = word_error_counts(normalize_words(' '.join(c['text'] for c in cues)), normalize_words(hyp))
    failed = sum(1 for s in segments if s.get('translation_error') or not s.get('translation'))
    minutes = audio / 60 if audio else None
    return {
        'name': name, 'audio_seconds': audio, 'client_seconds': client_seconds,
        'stt_seconds': phase.get('stt'), 'translation_seconds': phase.get('translation_batch'),
        'rtf_total': round(client_seconds / audio, 3) if audio else None,
        'rtf_stt': round(phase['stt'] / audio, 3) if audio and 'stt' in phase else None,
        'rtf_translation': round(phase['translation_batch'] / audio, 3) if audio and 'translation_batch' in phase else None,
        'segments': len(segments), 'failed_segments': failed, 'attempts': len(attempts),
        'rejection_reasons': dict(reasons),
        'prompt_tokens': prompt, 'output_tokens': output,
        'prompt_tokens_per_min': round(prompt / minutes) if minutes else None,
        'output_tokens_per_min': round(output / minutes) if minutes else None,
        'wer': round(wer['word_error_rate'], 4) if wer and wer['word_error_rate'] is not None else None,
        'wer_detail': wer,
    }


def main() -> None:
    """bench/out을 읽어 summary.json·summary.md를 쓴다.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'out')
    args = parser.parse_args()
    out = args.out
    rows = list(csv.DictReader(read_text(out / 'results.csv').splitlines()))
    log_text = read_text(out / 'backend_logs.txt')
    per_request = request_events(parse_events(log_text.splitlines()))
    clips = []
    for index, row in enumerate(rows):
        events = per_request[index][1] if index < len(per_request) else []
        if row['run'] == 'warmup' or row['http_status'] != '200':
            continue
        response = json.loads((out / f"{row['name']}.{row['run']}.json").read_text(encoding='utf-8'))
        vtt = read_text(out / f"{row['name']}.vtt")
        clips.append(summarize_clip(row['name'], response, events, vtt, float(row['client_seconds'])))
    (out / 'summary.json').write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding='utf-8')
    head = '| 클립 | 길이(초) | 전체 RTF | STT RTF | 번역 RTF | WER | 실패/구간 | 시도 | 입력·출력 토큰/분 |\n|---|---|---|---|---|---|---|---|---|\n'
    lines = [f"| {c['name']} | {c['audio_seconds'] and round(c['audio_seconds'])} | {c['rtf_total']} | {c['rtf_stt']} | "
             f"{c['rtf_translation']} | {c['wer']} | {c['failed_segments']}/{c['segments']} | {c['attempts']} | "
             f"{c['prompt_tokens_per_min']} / {c['output_tokens_per_min']} |" for c in clips]
    (out / 'summary.md').write_text(head + '\n'.join(lines) + '\n', encoding='utf-8')
    print(head + '\n'.join(lines))


if __name__ == '__main__':
    main()
