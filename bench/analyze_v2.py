"""2차 실험(bench/out2) 단계별 결과를 1차 결과(bench/out)와 비교해 summary_v2.md로 정리한다.

비교 항목: 처리 속도(RTF), 실패 구간, 형식 오류(새 검증 규칙 기준), 용어 준수율.
"""

import json
from pathlib import Path
import re
import sys

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))

from analyze import read_text, request_events, summarize_clip  # noqa: E402
from scripts.estimate_cost import parse_events  # noqa: E402
from app.services.translation_validation import preservation_problem  # noqa: E402
import csv  # noqa: E402


def term_adherence(segments: list, terms: list) -> tuple:
    """원문에 용어가 나온 (용어, 구간) 쌍 중 번역에 지정 번역어가 들어간 수를 센다.

    :param segments: 전사 응답의 segments.
    :param terms: {"en", "ko"} 목록.
    :return: (지정 번역어가 들어간 수, 전체 수).
    """
    ok = total = 0
    for segment in segments:
        if not segment.get('translation'):
            continue
        for term in terms:
            if re.search(r'(?<![A-Za-z])' + re.escape(term['en']) + r'(?![A-Za-z])', segment['text'], re.I):
                total += 1
                ok += term['ko'].replace(' ', '') in segment['translation'].replace(' ', '')
    return ok, total


def format_errors(segments: list) -> int:
    """새 검증 규칙(허용 문자·한영 혼합·종결어미 중복)에 걸리는 반환 구간 수를 센다.

    :return: 형식 오류 구간 수.
    """
    return sum(1 for s in segments if s.get('translation')
               and preservation_problem(s['text'], s['translation']) in
               ('unexpected_script', 'mixed_script_word', 'duplicated_ending'))


def phase_rows(phase_dir: Path) -> list:
    """한 단계 폴더의 클립별 지표를 계산한다.

    :return: summarize_clip 결과에 segments를 붙인 목록.
    """
    rows = list(csv.DictReader(read_text(phase_dir / 'results.csv').splitlines()))
    per_request = request_events(parse_events(read_text(phase_dir / 'backend_logs.txt').splitlines()))
    clips = []
    for index, row in enumerate(rows):
        if row['run'] == 'warmup' or row['http_status'] != '200':
            continue
        response = json.loads((phase_dir / f"{row['name']}.{row['run']}.json").read_text(encoding='utf-8'))
        events = per_request[index][1] if index < len(per_request) else []
        clip = summarize_clip(row['name'], response, events, read_text(phase_dir / f"{row['name']}.vtt"),
                              float(row['client_seconds']))
        clip['segments_data'] = response['segments']
        clips.append(clip)
    return clips


def glossary_for(phase: str, clip: str) -> list:
    """단계·클립에 맞는 용어 목록(기본 8개 제외)을 고른다.

    :return: {"en", "ko"} 목록. 해당 용어집이 없으면 빈 목록.
    """
    glossary = BENCH / 'glossary'
    if phase.startswith('C3_auto_'):
        path = glossary / f'auto_{clip}.json'
    elif clip.startswith('bio_'):
        path = glossary / 'bio_terms.json'
    elif clip.startswith('cs_'):
        path = glossary / 'cs_terms.json'
    else:
        return json.loads((ROOT / 'data/physics_glossary.json').read_text(encoding='utf-8'))
    return json.loads(path.read_text(encoding='utf-8'))[8:] if path.exists() else []


def main() -> None:
    """모든 단계를 읽어 표를 만들고 1차 결과의 같은 클립과 나란히 적는다.

    :return: None.
    """
    baseline = {}
    first = BENCH / 'out'
    if (first / 'results.csv').exists():
        for clip in phase_rows(first):
            baseline[clip['name']] = clip
    lines = ['| 단계 | 클립 | RTF(1차→이번) | 실패 | 형식 오류 | 용어 준수율(1차→이번) |',
             '|---|---|---|---|---|---|']
    for phase_dir in sorted(p for p in (BENCH / 'out2').iterdir() if p.is_dir()):
        if not (phase_dir / 'results.csv').exists():
            continue
        for clip in phase_rows(phase_dir):
            terms = glossary_for(phase_dir.name, clip['name'])
            ok, total = term_adherence(clip['segments_data'], terms)
            base = baseline.get(clip['name'])
            if base:
                bok, btotal = term_adherence(base['segments_data'], terms)
                adherence = f"{bok}/{btotal} → {ok}/{total}"
                rtf = f"{base['rtf_total']} → {clip['rtf_total']}"
            else:
                adherence, rtf = f"{ok}/{total}", f"{clip['rtf_total']}"
            lines.append(f"| {phase_dir.name} | {clip['name']} | {rtf} | {clip['failed_segments']}/{clip['segments']} | "
                         f"{format_errors(clip['segments_data'])} | {adherence} |")
    text = '\n'.join(lines) + '\n'
    (BENCH / 'out2' / 'summary_v2.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
