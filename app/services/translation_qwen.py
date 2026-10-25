"""Qwen3 Instruct에 맞는 문맥 분리·수학 번역 지시와 공통 번역 검증를 제공한다."""

import json
from pathlib import Path
import re

from app.services.translation import TranslationService
from app.services.translation_inputs import protect_math, restore_math, select_context
from app.services.translation_normalization import has_math_operator_words, normalize_math_operators
from app.services.translation_validation import protected_symbols, spoken_math_relations

ROOT = Path(__file__).resolve().parents[2]
FORMAL_STYLE = 'Translate into polite Korean. Use 존댓말 endings such as 합니다 and 입니다. '
INSTRUCTION = (
    'Translate the following English text into Korean. Preserve negation, numbers '
    'and mathematical relations. Output only the translation.'
    ' Keep mathematical symbols, operations, inequalities and corrections unchanged. '
    'Preserve the subject and whether a sentence is a statement or question. '
    'Do not add facts, units, explanations or instructions.'
    ' Translate definitions as complete grammatical Korean sentences, preserving '
    'the defined quantity and its defining relation. In mathematical speech, '
    'preserve the full equality, including factors and division, even when '
    'a parenthetical comment interrupts it. Do not infer missing relations '
    'or repair uncertain source text.'
    ' In mathematical speech, preserve division expressed by "over". '
    'Descriptions of factor placement refer to position in the expression, '
    'not an increase in value. Prime denotes a mathematical prime mark, '
    'not a prime number. Keep variable labels rather than translating them as words.'
)
CONTEXT_INSTRUCTION = (
    "Translate only the current English source in the user's message. "
    'The previous English context below is background only. Never translate, '
    'quote, summarize or repeat that context. Never output context labels. '
    'Output only the Korean translation of the current source. '
)


def informal_ending(translation: str) -> bool:
    """문장 끝의 뚜렷한 평서형 반말과 일부 구어체 반말을 검사한다.

    :param translation: 정리된 한국어 번역 후보.
    :return: 명백한 반말 종결형을 발견하면 True. 일반적인 문법·존대 분류기는 아니다.

    합니다·입니다·합시다와 같은 존댓말은 허용한다. 명사구·수식만으로 끝난
    표현을 자동으로 고치거나 모든 문장의 존댓말 완전성을 보장하지 않는다.
    """
    for match in re.finditer(r'([가-힣]+)(?=[.!?](?:\s|$)|$)', translation):
        ending = match.group(1)
        plain_endings = ('한다', '된다', '하다', '이다', '있다', '없다', '같다', '아니다', '싶다',
                         '했다', '겠다', '었다', '았다', '는다', '온다', '준다',
                         '크다', '작다', '그렇다', '멈춘다', '진다')
        if not ending.endswith(('니다', '시다')) and ending.endswith(plain_endings):
            return True
        if ending.endswith(('했어', '있어', '없어', '뭐야', '거야', '해봐', '해라', '하자', '해보자')):
            return True
    return False


def render_chatml(messages: list, tokenize: bool = False,
                  add_generation_prompt: bool = True) -> str:
    """Qwen3 Instruct 2507의 도구 없는 시스템·사용자 메시지 템플릿을 구성한다.

    :param messages: role이 system 또는 user이고 content가 문자열인 메시지 목록.
    :param tokenize: 문자열 구성만 지원하므로 False여야 한다.
    :param add_generation_prompt: 마지막 assistant 시작 표식 추가 여부. 기본 True.
    :return: 공식 비추론 채팅 템플릿과 같은 ChatML 문자열.
    :raises ValueError: 지원하지 않는 역할·토큰화 옵션·메시지 내용인 경우.
    """
    if tokenize or any(message['role'] not in ('system', 'user')
                       or not isinstance(message['content'], str) for message in messages):
        raise ValueError('Only untokenized system/user messages are supported')
    prompt = ''.join(f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n'
                     for message in messages)
    return prompt + ('<|im_start|>assistant\n' if add_generation_prompt else '')


class QwenPromptMixin:
    """HTTP 서비스에서 Qwen 번역 지시·문맥 분리·대상 검증을 공유한다."""

    def _configure_qwen(self) -> None:
        """번역용 추가 용어와 로켓·썰매 대상 보존 검사를 준비한다.

        :return: None. 공개 용어 JSON을 읽고 서비스 용어 목록을 설정한다.
        :raises OSError: 추가 용어 파일을 읽을 수 없는 경우.
        :raises ValueError: 용어 JSON 형식이 잘못된 경우.
        """
        additions = json.loads((ROOT / 'data/qwen_translation_terms.json').read_text(encoding='utf-8'))
        self.prompt_glossary = self.glossary + additions
        self.glossary = self.glossary + [entry for entry in additions if entry['en'] in ('rocket', 'sled')]

    def _build_prompt(self, text: str, context: str, retry: bool = False) -> str:
        """기존 고정 지시를 유지하며 의존 문맥과 수식 보호 입력을 나눠 구성한다.

        :param text: 현재 번역할 영어 원문.
        :param context: 앞 영어 원문. 의존 단서가 있을 때 마지막 256자만 시스템에서 참고한다.
        :param retry: 참이면 완결된 한국어 번역 문장만 출력하도록 재지시한다.
        :return: 고정 지시·선택적 문맥과 모델용 수식 보호 입력을 둔 메시지 문자열.
        """
        source, replacements = protect_math(text)
        context = select_context(text, context)
        instruction = (
            'Translate the following English text into Korean. Preserve negation, numbers '
            'and mathematical relations. Output only the translation.')
        if replacements:
            instruction += (' Copy each [MATH_*] placeholder exactly once, in order. '
                            'Use a complete Korean sentence around it.')
        symbols = protected_symbols(source)
        if symbols:
            instruction += ' Keep these source symbols: ' + ', '.join(symbols) + '.'
            instruction += (
                ' Render spoken arithmetic operators as symbols: plus as +, minus as -, '
                'equals as =, and over as /. Do not leave English operator words. '
                'Use Korean prose around the expression rather than a formula alone.')
        terms = self._matching_terms(source, self.prompt_glossary)
        if terms:
            instruction += ' Terminology: ' + '; '.join(
                f'{entry["en"]}={entry["ko"]}' for entry in terms) + '.'
        relations = spoken_math_relations(source)
        if relations:
            examples = []
            for operator, left, right in relations:
                if operator == 'of':
                    examples.append(f'{left}({right}) (function value, not multiplication)')
                elif left:
                    examples.append(f'{left}/({right})')
                else:
                    examples.append(f'division by {right} (do not infer the numerator)')
            instruction += ' Preserve these source relations: ' + '; '.join(examples) + '.'
        system = FORMAL_STYLE + INSTRUCTION
        if context:
            system += ' ' + CONTEXT_INSTRUCTION + '\nPrevious English context:\n' + context[-256:]
        messages = [{'role': 'system', 'content': system}]
        if retry:
            instruction += '\nReturn only complete polite Korean translation sentences ending in 합니다/입니다.'
            if symbols:
                instruction += (' Start with Korean words and then give the expression; '
                                'a formula alone is not a complete Korean translation.')
        messages.append({'role': 'user', 'content': (instruction.strip() + '\n\n' + source).lstrip()})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _restore_translation(self, translation: str, text: str) -> tuple:
        """수식 표식을 복원하고 원문으로 확인한 연산어만 정규화해 기존 검증에 전달한다.

        :param translation: 일반 출력 표식을 정리한 번역 후보.
        :param text: API에 보존된 보호 전 영어 원문.
        :return: 복원·정규화한 후보와 표식 실패 사유 또는 None. 검증 수락 여부는 별도로 판정한다.
        """
        restored, problem = restore_math(translation, text)
        if problem:
            return restored, problem
        return normalize_math_operators(restored, text), None

    def _rejection_reason(self, translation: str, text: str, context: str):
        """공통 검증과 함께 반말 종결형·번역되지 않은 영어 수학 연산어를 검사한다.

        :param translation: 정리된 한국어 번역 후보.
        :param text: 현재 영어 원문.
        :param context: 앞 영어 원문. 공통 검증에만 전달한다.
        :return: 첫 실패 사유 코드 또는 None. 일반적인 의미 정확도 판정은 아니다.
        """
        reason = super()._rejection_reason(translation, text, context)
        if reason:
            return reason
        if informal_ending(translation):
            return 'informal_speech'
        if protected_symbols(text):
            if has_math_operator_words(translation, text):
                return 'untranslated_math_operator'
            for operator in ('over', 'plus', 'minus', 'equals'):
                pattern = r'\b' + operator + r'\b'
                if re.search(pattern, text, re.IGNORECASE) and re.search(pattern, translation, re.IGNORECASE):
                    return 'untranslated_math_operator'
        return None
