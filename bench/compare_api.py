"""같은 클립의 Qwen 번역(bench/out)과 외부 API 번역(bench/out3/<모델>)을 구간별로 나란히 비교한다.

사용법: python bench/compare_api.py --model-dir bench/out3/openai_gpt-5-mini
결과: <model-dir>/compare_<클립>.md (구간별 표) 와 화면 요약
"""
import argparse
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def load(path: Path) -> list:
    """응답 JSON에서 구간 목록을 읽는다(UTF-8/UTF-16 모두 허용).

    :param path: /api/v1/transcriptions 응답 파일.
    :return: 구간 dict 목록.
    """
    raw = path.read_bytes()
    text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8-sig')
    return json.loads(text)['segments']


def cell(text) -> str:
    """마크다운 표 칸에 넣을 수 있게 줄바꿈과 | 를 바꾼다.

    :param text: 원문 또는 번역문(None이면 실패).
    :return: 표 칸 문자열.
    """
    return '(실패 · 원문 유지)' if not text else str(text).replace('|', '/').replace('\n', ' ')


def main() -> None:
    """모델 폴더의 클립마다 Qwen 결과와 짝지어 비교표를 만든다.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    for api_file in sorted(model_dir.glob('*.1.json')):
        clip = api_file.name[:-len('.1.json')]
        qwen_file = BENCH / 'out' / f'{clip}.1.json'
        if not qwen_file.exists():
            print(f'{clip}: Qwen 결과 없음, 건너뜀')
            continue
        qwen, api = load(qwen_file), load(api_file)
        rows = ['| # | 시간 | 영어(STT) | Qwen3-4B | ' + model_dir.name + ' |', '|---|---|---|---|---|']
        api_by_start = {round(s['start'], 1): s for s in api}
        q_fail = a_fail = 0
        for i, seg in enumerate(qwen):
            other = api_by_start.get(round(seg['start'], 1), {})
            q_fail += not seg.get('translation')
            a_fail += not other.get('translation')
            rows.append(f"| {i} | {seg['start']:.0f}s | {cell(seg['text'])} | {cell(seg.get('translation'))} | "
                        f"{cell(other.get('translation')) if other else '(구간 없음)'} |")
        out = model_dir / f'compare_{clip}.md'
        out.write_text(f'# {clip}: Qwen3-4B vs {model_dir.name}\n\n' + '\n'.join(rows) + '\n', encoding='utf-8')
        print(f'{clip}: 구간 {len(qwen)}개, 실패 Qwen {q_fail} / API {a_fail} -> {out}')


if __name__ == '__main__':
    main()
