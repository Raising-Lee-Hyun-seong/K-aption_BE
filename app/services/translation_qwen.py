"""Qwen3 Instruct에 맞는 문맥 분리·수학 번역 지시와 MLX 번역 서비스를 제공한다."""

import json
from pathlib import Path
import re

from app.services.translation import TranslationService
from app.services.translation_validation import protected_symbols

ROOT = Path(__file__).resolve().parents[2]
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
    """MLX와 llama.cpp에서 같은 Qwen 번역 지시·문맥 분리·대상 검증을 공유한다."""

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
        """현재 영어 원문과 참고 문맥을 분리해 Qwen 채팅 프롬프트를 만든다.

        :param text: 현재 번역할 영어 원문.
        :param context: 앞 영어 원문. 마지막 256자만 시스템 메시지에서 참고한다.
        :param retry: 참이면 완결된 한국어 번역 문장만 출력하도록 재지시한다.
        :return: Qwen 채팅 프롬프트. 기존 KoreanLM 선택 시에는 기존 프롬프트.
        """
        if getattr(self, 'model_family', 'koreanlm') != 'qwen3':
            return super()._build_prompt(text, context, retry)
        instruction = INSTRUCTION
        symbols = protected_symbols(text)
        if symbols:
            instruction += ' Keep these source symbols: ' + ', '.join(symbols) + '.'
            instruction += (
                ' Render spoken arithmetic operators as symbols: plus as +, minus as -, '
                'equals as =, and over as /. Do not leave English operator words. '
                'Use Korean prose around the expression rather than a formula alone.')
        terms = self._matching_terms(text, self.prompt_glossary)
        if terms:
            instruction += ' Terminology: ' + '; '.join(
                f'{entry["en"]}={entry["ko"]}' for entry in terms) + '.'
        messages = []
        if context:
            messages.append({'role': 'system', 'content': (
                "Translate only the current English source in the user's message. "
                'The previous English context below is background only. Never translate, '
                'quote, summarize or repeat that context. Never output context labels. '
                'Output only the Korean translation of the current source. '
                '\nPrevious English context:\n' + context[-256:])})
        if retry:
            instruction += '\nReturn only complete Korean translation sentences.'
            if symbols:
                instruction += (' Start with Korean words and then give the expression; '
                                'a formula alone is not a complete Korean translation.')
        messages.append({'role': 'user', 'content': instruction + '\n\n' + text})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _rejection_reason(self, translation: str, text: str, context: str):
        """공통 검증과 함께 수학 문맥의 번역되지 않은 영어 연산어를 검사한다.

        :param translation: 정리된 한국어 번역 후보.
        :param text: 현재 영어 원문.
        :param context: 앞 영어 원문. 공통 검증에만 전달한다.
        :return: 첫 실패 사유 코드 또는 None. 일반적인 의미 정확도 판정은 아니다.
        """
        reason = super()._rejection_reason(translation, text, context)
        if reason or getattr(self, 'model_family', 'koreanlm') != 'qwen3':
            return reason
        if protected_symbols(text):
            for operator in ('over', 'plus', 'minus', 'equals'):
                pattern = r'\b' + operator + r'\b'
                if re.search(pattern, text, re.IGNORECASE) and re.search(pattern, translation, re.IGNORECASE):
                    return 'untranslated_math_operator'
        return None


class QwenTranslationService(QwenPromptMixin, TranslationService):
    """Qwen3 Instruct MLX 4bit를 로드하고 공통 번역 검증·재시도를 적용한다."""

    def __init__(self, model_path: Path, template_path: Path, device: str = 'auto'):
        """Qwen3 구조를 확인한 뒤 MLX 모델과 번역 용어를 준비한다.

        :param model_path: 고정 Qwen3 Instruct 2507 MLX 4bit 디렉터리.
        :param template_path: 기존 공통 설정 초기화용 템플릿 경로.
        :param device: auto는 MLX 기본 장치, cpu는 CPU 강제 사용.
        :return: None. 모델을 메모리에 로드한다.
        :raises OSError: 모델 설정·가중치·공통 파일을 읽을 수 없는 경우.
        :raises ValueError: 모델 구조·장치·양자화 설정이 지원 범위와 다른 경우.
        """
        config = json.loads((model_path / 'config.json').read_text(encoding='utf-8'))
        if config.get('model_type') != 'qwen3':
            raise ValueError('Expected a Qwen3 Instruct MLX model')
        self.model_family = 'qwen3'
        super().__init__(model_path, template_path, device)
        self._configure_qwen()
