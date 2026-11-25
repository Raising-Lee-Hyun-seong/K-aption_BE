"""강의 STT 원문에서 전공 용어집(영어→한국어)을 LLM으로 자동 생성한다.

VideoLingo처럼 번역 전에 강의 단위 용어집을 먼저 만들어, 한 강의 안에서 같은 용어가
같은 번역어로 유지되게 하는 것이 목적이다. OpenAI 호환 `/v1/chat/completions`를 쓰므로
로컬 llama.cpp 번역 서버(translator:8080)나 외부 API에 모두 연결할 수 있다.

사용 예 (backend 컨테이너 안에서 로컬 Qwen 사용):
    python build_glossary.py --input result.json --base-url http://translator:8080 > terms.json
"""

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request

BASE_TERMS = [
    {"en": "rocket", "ko": "로켓"}, {"en": "magnitude", "ko": "크기"}, {"en": "constant", "ko": "상수"},
    {"en": "sled", "ko": "썰매"}, {"en": "integrated", "ko": "적분"}, {"en": "integrating", "ko": "적분"},
    {"en": "primes", "ko": "프라임"}, {"en": "prime", "ko": "프라임"},
]
PROMPT = (
    "You are building a terminology glossary for Korean subtitles of a university lecture.\n"
    "From the transcript below, list up to {limit} technical terms (nouns or noun phrases, "
    "including programming keywords) whose Korean translation must stay consistent.\n"
    "Use the standard Korean term used in Korean university textbooks. Keep code keywords "
    "(for, if, break, print) and abbreviations (DNA, PCR, cAMP) in English.\n"
    "Return only a JSON array like [{{\"en\": \"lactose\", \"ko\": \"락토스\"}}]. No explanations.\n\n"
    "Transcript:\n{transcript}"
)


def transcript_text(response: dict, max_chars: int = 20000) -> str:
    """전사 응답의 영어 구간을 이어 붙여 프롬프트용 원문을 만든다.

    :param response: `/api/v1/transcriptions` 응답 사전(segments[].text).
    :param max_chars: 프롬프트에 넣을 최대 글자 수. 긴 강의는 앞부분만 쓴다.
    :return: 공백으로 이은 영어 원문.
    """
    text = ' '.join(segment['text'].strip() for segment in response.get('segments', []))
    return text[:max_chars]


def parse_terms(output: str, transcript: str, limit: int = 40) -> list:
    """모델 출력에서 JSON 배열을 찾아 원문에 실제로 나온 용어만 남긴다.

    :param output: 모델이 생성한 텍스트. 코드 블록이나 앞뒤 설명이 붙어 있어도 된다.
    :param transcript: 용어가 실제로 등장했는지 확인할 영어 원문.
    :param limit: 남길 최대 용어 수.
    :return: {"en", "ko"} 사전 목록. 중복·원문에 없는 용어·빈 번역어는 뺀다.
    """
    match = re.search(r'\[.*\]', output, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    seen, terms = set(), []
    for item in items:
        if not isinstance(item, dict):
            continue
        en, ko = str(item.get('en', '')).strip(), str(item.get('ko', '')).strip()
        if not en or not ko or en.lower() in seen:
            continue
        if not re.search(r'(?<![A-Za-z])' + re.escape(en) + r'(?![A-Za-z])', transcript, re.IGNORECASE):
            continue
        if not re.search(r'[가-힣]', ko) and ko.lower() != en.lower():
            continue
        # 한 글자 영어(g·a·t 같은 염기 이름)는 문장 전체를 오염시키고, 한자·가나 등이 섞인 번역어는 버린다.
        if len(en) < 2 or re.search(r'[^\s\w\'′.\-+()/가-힣]|[\u3040-\u30ff\u3400-\u9fff]', ko):
            continue
        seen.add(en.lower())
        terms.append({'en': en, 'ko': ko})
    return terms[:limit]


def chunks(text: str, size: int = 1500) -> list:
    """문장 경계를 살려 원문을 요청 하나에 들어갈 크기로 나눈다.

    로컬 번역 서버는 컨텍스트가 2048토큰이라, 원문 약 1,500자(약 400토큰)씩 나눠 요청한다.

    :param text: 영어 원문.
    :param size: 조각 하나의 최대 글자 수.
    :return: 원문 조각 목록.
    """
    parts, current = [], ''
    for sentence in re.split(r'(?<=[.?!])\s+', text):
        if current and len(current) + len(sentence) + 1 > size:
            parts.append(current)
            current = ''
        current = (current + ' ' + sentence).strip()
    if current:
        parts.append(current)
    return parts


def merge_terms(groups: list, limit: int) -> list:
    """조각별 용어 목록을 합치며 먼저 나온 번역어를 우선한다(강의 안 일관성).

    :param groups: 조각별 parse_terms 결과 목록.
    :param limit: 남길 최대 용어 수.
    :return: 중복 없는 {"en", "ko"} 목록.
    """
    seen, merged = set(), []
    for group in groups:
        for term in group:
            if term['en'].lower() not in seen:
                seen.add(term['en'].lower())
                merged.append(term)
    return merged[:limit]


def request_terms(base_url: str, transcript: str, limit: int, model: str, api_key: str = '') -> str:
    """OpenAI 호환 채팅 API에 용어집 생성을 한 번 요청한다.

    :return: 모델이 생성한 텍스트.
    """
    body = json.dumps({'model': model, 'temperature': 0, 'max_tokens': 600,
                       'messages': [{'role': 'user', 'content': PROMPT.format(limit=limit, transcript=transcript)}]})
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(base_url.rstrip('/') + '/v1/chat/completions', body.encode('utf-8'), headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())['choices'][0]['message']['content']


def main() -> None:
    """전사 결과를 읽어 용어집 JSON(기본 용어 8개 + 생성 용어)을 표준 출력에 쓴다.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='전사 응답 JSON 파일')
    parser.add_argument('--base-url', default='http://translator:8080')
    parser.add_argument('--model', default='qwen3-4b-instruct-2507')
    parser.add_argument('--api-key', default='')
    parser.add_argument('--limit', type=int, default=40)
    parser.add_argument('--chunk-chars', type=int, default=1500)
    args = parser.parse_args()
    transcript = transcript_text(json.loads(args.input.read_text(encoding='utf-8')))
    groups = []
    for part in chunks(transcript, args.chunk_chars):
        try:
            groups.append(parse_terms(request_terms(args.base_url, part, 15, args.model, args.api_key), part, 15))
        except OSError as exc:
            print(f'chunk failed: {exc}', file=sys.stderr)
    terms = merge_terms(groups, args.limit)
    sys.stdout.write(json.dumps(BASE_TERMS + terms, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
