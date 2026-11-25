"""서버 시간 로그의 실제 토큰 수로 요청별 영상 1분당 토큰과 외부 API 예상 비용을 계산한다."""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIELD = re.compile(r'(\w+)=(\S+)')


def parse_events(lines) -> list:
    """`event=` 로 시작하는 시간 로그 줄을 키·값 사전 목록으로 바꾼다.

    :param lines: Uvicorn·Docker 로그 문자열 반복자. 시간 로그가 아닌 줄은 무시한다.
    :return: event·request_id 등을 담은 사전 목록. 값은 모두 문자열이다.
    """
    events = []
    for line in lines:
        start = line.find('event=')
        if start < 0:
            continue
        events.append(dict(FIELD.findall(line[start:])))
    return events


def summarize_requests(events: list) -> list:
    """요청 ID별로 영상 길이와 모든 번역 시도(재시도 포함)의 토큰 합을 모은다.

    :param events: parse_events 결과 또는 같은 형식의 사전 목록.
    :return: request_id·audio_minutes·attempts·prompt_tokens·output_tokens 사전 목록.
             영상 길이(stt_inference)가 없는 요청은 1분당 값을 계산할 수 없어 제외한다.
    """
    requests = {}
    for event in events:
        row = requests.setdefault(event.get('request_id', '-'), {
            'audio_minutes': None, 'attempts': 0, 'prompt_tokens': 0, 'output_tokens': 0})
        if event.get('event') == 'stt_inference' and 'audio_seconds' in event:
            row['audio_minutes'] = float(event['audio_seconds']) / 60
        elif event.get('event') == 'translation_attempt' and 'prompt_tokens' in event:
            row['attempts'] += 1
            row['prompt_tokens'] += int(event['prompt_tokens'])
            row['output_tokens'] += int(event.get('raw_output_tokens', 0))
    return [{'request_id': key, **row} for key, row in requests.items()
            if row['audio_minutes'] and row['attempts']]


def estimate(summary: dict, prices: dict) -> dict:
    """한 요청의 1분당 토큰과 가격표의 모델별 강의 1시간 예상 비용(USD)을 계산한다.

    :param summary: summarize_requests 결과의 한 요소.
    :param prices: data/api_prices.json 형식의 가격표.
    :return: 1분당 입력·출력 토큰과 번역·STT 모델별 1시간 비용 사전.

    토큰 수는 Qwen 토크나이저 기준이다. 외부 모델은 토크나이저가 달라 근사값이다.
    """
    minutes = summary['audio_minutes']
    per_min_in = summary['prompt_tokens'] / minutes
    per_min_out = summary['output_tokens'] / minutes
    translation = {
        name: round(60 * (per_min_in * p['input_per_million']
                          + per_min_out * p['output_per_million']) / 1_000_000, 4)
        for name, p in prices['translation'].items()}
    stt = {name: round(60 * p['per_minute'], 4) for name, p in prices['stt'].items()}
    return {'request_id': summary['request_id'], 'audio_minutes': round(minutes, 2),
            'attempts': summary['attempts'],
            'input_tokens_per_minute': round(per_min_in), 'output_tokens_per_minute': round(per_min_out),
            'translation_usd_per_hour': translation, 'stt_usd_per_hour': stt}


def main() -> None:
    """로그 파일을 읽어 요청별 추정 결과를 JSON으로 출력한다.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='docker compose logs 등으로 저장한 서버 로그 파일')
    parser.add_argument('--prices', type=Path, default=ROOT / 'data/api_prices.json')
    args = parser.parse_args()
    prices = json.loads(args.prices.read_text(encoding='utf-8'))
    with args.log.open(encoding='utf-8', errors='replace') as lines:
        summaries = summarize_requests(parse_events(lines))
    report = {'prices_checked_on': prices['checked_on'],
              'requests': [estimate(row, prices) for row in summaries]}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
