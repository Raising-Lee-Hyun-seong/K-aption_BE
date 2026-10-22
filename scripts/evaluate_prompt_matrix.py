"""보존된 평가 자료의 과거 프롬프트·진단을 재현하는 모델 실행 없는 도구다."""
import argparse
from datetime import date, datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.translation import TranslationService
from app.services.translation_validation import protected_symbols
from scripts.evaluate_candidates import (
    build_prompt, inspect_output, load_cases,
    save_report, sha256_file,
)

VARIANTS = ('koreanlm-current', 'koreanlm-simple', 'qwen-base', 'qwen-glossary',
            'qwen-glossary-v2', 'qwen-relations-v2', 'qwen-relations-v3')
GLOSSARY_ADDITIONS = ROOT / 'evaluation/physics_glossary_additions_v2.json'
GLOSSARY_ADDITIONS_V3 = ROOT / 'evaluation/physics_glossary_additions_v3.json'
QWEN_INSTRUCTION = (
    'Translate the following English text into Korean. Preserve negation, numbers '
    'and mathematical relations. Output only the translation.'
)
QWEN_PRESERVATION = (
    ' Keep mathematical symbols, operations, inequalities and corrections unchanged. '
    'Preserve the subject and whether a sentence is a statement or question. '
    'Do not add facts, units, explanations or instructions.'
)
QWEN_RELATIONS = (
    ' Translate definitions as complete grammatical Korean sentences, preserving '
    'the defined quantity and its defining relation. In mathematical speech, '
    'preserve the full equality, including factors and division, even when '
    'a parenthetical comment interrupts it. Do not infer missing relations '
    'or repair uncertain source text.'
)
QWEN_MATH = (
    ' In mathematical speech, preserve division expressed by "over". '
    'Descriptions of factor placement refer to position in the expression, '
    'not an increase in value. Prime denotes a mathematical prime mark, '
    'not a prime number. Keep variable labels rather than translating them as words.'
)


class HistoricalPromptService(TranslationService):
    """과거 프롬프트 기록의 순수 텍스트 재현만 제공하며 모델을 실행하지 않는다."""

    def _build_prompt(self, text: str, context: str, retry: bool = False) -> str:
        """영어 원문·앞 영어 문맥·필수 용어와 수식 보존 지시로 프롬프트를 구성한다.

        :param text: 검사하거나 번역할 입력 텍스트.
        :param context: 앞 영어 원문 등의 참고 문맥. 프롬프트에는 마지막 256자만 사용한다.
        :param retry: 참이면 한국어 완결 문장으로 답하도록 재시도 지시를 추가한다.
        :return: 번역 지시와 영어 입력이 결합된 모델 프롬프트 문자열.
        """
        instruction = "다음 영어 원문을 한국어로 번역하세요. 설명 없이 번역문만 답하세요."
        terms = self._matching_terms(text)
        if terms:
            instruction += " 용어: " + "; ".join(
                f'{entry["en"]}={entry["ko"]}' for entry in terms) + "."
        if context:
            instruction += f" 이전 영어 문맥: {context[-256:]} (참고만 하세요.)"
        instruction += " 숫자와 수식 기호를 보존하고 원문에 없는 단위를 추가하지 마세요."
        symbols = protected_symbols(text)
        if symbols:
            instruction += " 그대로 쓸 기호: " + ", ".join(symbols) + "."
        if retry:
            instruction += " 반드시 한글이 포함된 완전한 한국어 문장 하나만 답하세요."
        return self.template["prompt_input"].format(
            instruction=instruction,
            input=text,
        )



def prompt_service() -> TranslationService:
    """모델 로딩 없이 현재 API의 프롬프트·용어 매칭 로직만 준비한다.

    :return: 템플릿·용어집만 설정된 실험 전용 TranslationService 객체.
    """
    service = HistoricalPromptService.__new__(HistoricalPromptService)
    service.template = json.loads((ROOT / 'templates/korean.json').read_text())
    service.glossary = json.loads((ROOT / 'data/physics_glossary.json').read_text())
    return service


def variant_prompt(
    variant: str, tokenizer, service: TranslationService, text: str,
    context: str = '', retry: bool = False, context_format: str = 'inline',
) -> str:
    """기준 번역을 사용하지 않고 지정된 실험 조건의 번역 프롬프트를 구성한다.

    :param variant: VARIANTS에 정의된 기준 또는 보완 실험 조건.
    :param tokenizer: Qwen 채팅 템플릿을 제공하는 모델 토크나이저.
    :param service: 현재 템플릿·용어집·원문 매칭 로직을 가진 실험 객체.
    :param text: 번역할 영어 원문. 기준 번역은 전달하지 않는다.
    :param context: Qwen 조건에서만 사용할 앞 영어 문맥. 마지막 256자를 참고로 제공한다.
    :param retry: Qwen 재시도에서 완결된 한국어 출력 지시를 추가할지 여부.
    :param context_format: inline이면 지시에 문맥을 붙이고 system이면 별도 시스템 메시지로 분리한다.
    :return: 모델에 그대로 전달할 프롬프트 문자열.
    :raises ValueError: 알려지지 않은 실험 조건일 때.
    """
    if context_format not in ('inline', 'system'):
        raise ValueError(f'Unknown context format: {context_format}')
    if variant == 'koreanlm-current':
        return service._build_prompt(text, '')
    if variant == 'koreanlm-simple':
        return service.template['prompt_input'].format(
            instruction='다음 영어 원문을 한국어로 번역하세요. 번역문만 출력하세요.', input=text)
    if variant == 'qwen-base':
        return build_prompt(tokenizer, {'original_repository': 'Qwen/Qwen3-4B-Instruct-2507'}, text)
    if variant in ('qwen-glossary', 'qwen-glossary-v2', 'qwen-relations-v2', 'qwen-relations-v3'):
        instruction = QWEN_INSTRUCTION + QWEN_PRESERVATION
        matcher = service
        if variant.endswith(('-v2', '-v3')):
            matcher = TranslationService.__new__(TranslationService)
            additions = GLOSSARY_ADDITIONS_V3 if variant.endswith('-v3') else GLOSSARY_ADDITIONS
            matcher.glossary = service.glossary + json.loads(additions.read_text())
        if variant.startswith('qwen-relations-'):
            instruction += QWEN_RELATIONS
        if variant == 'qwen-relations-v3':
            instruction += QWEN_MATH
            symbols = protected_symbols(text)
            if symbols:
                instruction += ' Keep these source symbols: ' + ', '.join(symbols) + '.'
        terms = matcher._matching_terms(text)
        if terms:
            instruction += ' Terminology: ' + '; '.join(
                f'{entry["en"]}={entry["ko"]}' for entry in terms) + '.'
        messages = []
        if context and context_format == 'inline':
            instruction += ('\nPrevious English context (reference only; do not translate): '
                            + context[-256:])
        elif context:
            messages.append({'role': 'system', 'content': (
                "Translate only the current English source in the user's message. "
                'The previous English context below is background only. Never translate, '
                'quote, summarize or repeat that context. Never output context labels. '
                'Output only the Korean translation of the current source. '
                '\nPrevious English context:\n' + context[-256:])})
        if retry:
            instruction += '\nReturn only complete Korean translation sentences.'
        messages.append({'role': 'user', 'content': instruction + '\n\n' + text})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True)
    raise ValueError(f'Unknown variant: {variant}')


def verify_koreanlm(directory: Path) -> dict:
    """기존 KoreanLM의 설정·가중치 인덱스와 파일을 확인하고 재현용 해시를 기록한다.

    :param directory: 기존 KoreanLM 4bit 디렉터리.
    :return: 원본 모델 출처와 로컬 파일별 크기·SHA256 사전.
    :raises FileNotFoundError: 필수 파일이나 인덱스가 참조한 가중치가 없을 때.
    :raises ValueError: 4bit가 아니거나 가중치가 비어 있거나 모델 경로를 벗어날 때.

    원본 가중치의 공개 해시와 대조한 검증은 아니며 로컬 실행 파일의 동일성만 기록한다.
    """
    config = json.loads((directory / 'config.json').read_text())
    if config.get('quantization', {}).get('bits') != 4:
        raise ValueError('KoreanLM must use the existing 4bit configuration')
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    if not names:
        raise ValueError('Empty KoreanLM weight index')
    metadata = ['config.json', 'model.safetensors.index.json', 'source.json',
                'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                'generation_config.json']
    files = []
    for name in names + metadata:
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Model file must remain inside the model directory')
        if path.stat().st_size == 0:
            raise ValueError(f'Empty model file: {name}')
        files.append({'name': name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return {'original_source': json.loads((directory / 'source.json').read_text()),
            'files': files, 'verification_scope': 'local fingerprints, not upstream LFS verification'}


def diagnose_output(service: TranslationService, source: str, output: str, finish_reason: str) -> dict:
    """원시 출력을 바꾸지 않고 형식·보존 검사와 현재 용어 검사를 진단 정보로 기록한다.

    :param service: 원문 용어 매칭과 현재 검증 로직을 제공하는 실험 객체.
    :param source: 검사 기준인 영어 원문.
    :param output: 정리하지 않은 원시 생성 텍스트.
    :param finish_reason: 보존된 평가 출력의 생성 종료 사유.
    :return: 기본 진단과 원시 텍스트에 적용한 API 검증 사유의 사전. 의미 판정이 아니다.
    """
    return {'validation_problems': inspect_output(source, output, finish_reason),
            'raw_validator_rejection_reason': service._rejection_reason(output, source, '')}
