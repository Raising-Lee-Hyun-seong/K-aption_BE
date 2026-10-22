"""모델 실행 없이 과거 Qwen 평가 프롬프트와 출력 진단을 재현한다."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.translation import TranslationService
from scripts.evaluate_lecture_quality import evaluate_groups, full_groups, summarize
from scripts.evaluate_prompt_matrix import (
    GLOSSARY_ADDITIONS, GLOSSARY_ADDITIONS_V3, prompt_service, variant_prompt,
)


class EvaluationQwenService(TranslationService):
    """API를 바꾸지 않고 기존 출력 검사·재시도에 Qwen 전용 프롬프트를 연결한다."""

    def __init__(self, tokenizer, context_format: str = 'system', variant: str = 'qwen-relations-v2'):
        """모델 없이 보존된 평가 프롬프트의 재현 설정을 준비한다.

        :param tokenizer: 과거 채팅 템플릿을 재현할 텍스트 도우미.
        :param context_format: 참고 문맥 형식. inline 또는 system.
        :param variant: 보존된 과거 프롬프트 변형 이름.
        :return: None. 모델이나 추론 엔진을 로드하지 않는다.
        """
        self.tokenizer = tokenizer
        self.context_format, self.variant = context_format, variant
        self.glossary = prompt_service().glossary
        self._lock = Lock()
        self.raw_attempts = []

    def _build_prompt(self, text: str, context: str, retry: bool = False) -> str:
        """Qwen 보완 조건에 앞 영어 문맥과 재시도 지시를 연결한다.

        :param text: 현재 번역할 영어 원문.
        :param context: 앞 영어 원문. 마지막 256자만 참고로 제공한다.
        :param retry: 문맥 없는 재시도에서 완결된 한국어 문장 지시를 추가할지 여부.
        :return: Qwen 채팅 템플릿으로 구성한 프롬프트 문자열.
        """
        return variant_prompt(self.variant, self.tokenizer, self, text, context, retry,
                              self.context_format)

    def _generate_output(self, prompt: str, max_tokens: int) -> dict:
        """기존 스트림 조기 종료를 수행하고 정리 전 출력과 프롬프트를 평가 근거로 보존한다.

        :param prompt: 모델에 전달할 프롬프트 문자열.
        :param max_tokens: 한 시도에서 생성할 최대 토큰 수.
        :return: 기존 생성기의 text·tokens·finish_reason·stop_reason 사전.
        """
        result = super()._generate_output(prompt, max_tokens)
        self.raw_attempts.append({'prompt': prompt, **result})
        return result
