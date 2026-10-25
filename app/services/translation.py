"""모델을 로드하지 않고 번역 출력 검증·직렬 처리·재시도·시간 기록을 제공한다."""
import json
import re
from pathlib import Path
from threading import Lock
from time import perf_counter

from timing import measure
from app.services.translation_validation import (
    generation_problem, preservation_problem, protected_symbols,
)

OUTPUT_MARKERS = ("참고 문맥:", "번역할 대상:", "입력:", "출력:")
HANGUL_PATTERN = re.compile(r"[가-힣]")
GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "data/physics_glossary.json"


class TranslationTransportError(ValueError):
    """제한된 재시도 이후 번역 서버 연결·응답 처리가 실패했음을 나타낸다."""


class TranslationService:
    """번역 전송 구현과 분리된 출력 검증·직렬 처리·재시도 계약을 제공한다."""
    def __init__(self, template_path: Path):
        """용어집·프롬프트 설정과 직렬 추론 잠금을 준비한다.

        :param template_path: 공통 텍스트 설정 JSON 경로.
        :return: None. 모델이나 GPU 라이브러리는 로드하지 않는다.
        :raises OSError: 텍스트 설정 또는 용어집 읽기 실패 시.
        """
        self._configure_text(template_path)

    def _configure_text(self, template_path: Path) -> None:
        """공통 텍스트 설정·용어집과 직렬 번역 잠금을 준비한다.

        :param template_path: 공통 텍스트 설정 JSON 경로.
        :return: None.
        :raises OSError: 설정 또는 용어집 읽기 실패 시.
        """
        self.template = json.loads(template_path.read_text(encoding='utf-8'))
        self.glossary = json.loads(GLOSSARY_PATH.read_text(encoding='utf-8'))
        self._lock = Lock()

    def _matching_terms(self, text: str, glossary=None) -> list:
        """단어 경계로 용어를 찾고 겹치는 위치에서는 긴 복합 용어를 우선한다.

        :param text: 검사하거나 번역할 입력 텍스트.
        :param glossary: 선택적 용어 목록. None이면 서비스의 필수 검증 용어집을 사용한다.
        :return: 원문과 매칭된 영어·한국어 용어 항목 목록.
        """
        matched, occupied = [], []
        for entry in sorted(self.glossary if glossary is None else glossary,
                            key=lambda entry: -len(entry["en"])):
            spans = [match.span() for match in re.finditer(
                r"\b" + re.escape(entry["en"]) + r"\b", text, re.IGNORECASE)
                if not any(match.start() < end and match.end() > start
                           for start, end in occupied)]
            if spans:
                matched.append(entry)
                occupied.extend(spans)
        return matched

    def _build_prompt(self, text: str, context: str, retry: bool = False) -> str:
        """구체적인 번역 서비스가 제공해야 하는 프롬프트 계약을 정의한다.

        :param text: 현재 영어 원문.
        :param context: 앞 영어 문맥.
        :param retry: 문맥 없이 재시도하는지 여부.
        :return: 직렬화된 프롬프트 문자열.
        :raises NotImplementedError: 구체적인 프롬프트 구현 없이 호출한 경우.
        """
        raise NotImplementedError('Use the Qwen HTTP translation service')

    @staticmethod
    def _clean_translation(translation: str) -> str:
        """생성 텍스트의 프롬프트 표식과 일부 설명·따옴표를 제거해 번역문을 추출한다.

        :param translation: 정리하거나 검증할 한국어 번역 후보 문자열.
        :return: 표식·불필요한 줄·바깥 따옴표를 정리한 번역 문자열.
        """
        cleaned = translation.strip()
        for marker in ("### Response:", "### 응답:", "출력:", "번역할 대상:"):
            if marker in cleaned:
                cleaned = cleaned.rsplit(marker, 1)[-1].strip()

        lines = [
            line.strip()
            for line in cleaned.splitlines()
            if line.strip()
            and not line.strip().startswith(("참고 문맥:", "입력:", "###"))
        ]
        cleaned = " ".join(lines).strip()
        if "->" in cleaned:
            left, right = cleaned.split("->", 1)
            cleaned = (right if HANGUL_PATTERN.search(right) else left).strip()

        quoted = re.findall(r'["“]([^"”]+)["”]', cleaned)
        if ("번역하면" in cleaned or cleaned.startswith("입력된")) and quoted:
            korean_quotes = [text for text in quoted if HANGUL_PATTERN.search(text)]
            if korean_quotes:
                cleaned = korean_quotes[-1]
        return cleaned.strip().strip('"“”')

    @staticmethod
    def _is_valid_translation(translation: str) -> bool:
        """생성 결과에 한글이 포함되고 금지된 프롬프트 표식이 없는지 검사한다.

        :param translation: 정리하거나 검증할 한국어 번역 후보 문자열.
        :return: 한글을 포함하고 금지된 출력 표식이 없으면 True.
        """
        return bool(HANGUL_PATTERN.search(translation)) and not any(
            marker in translation for marker in OUTPUT_MARKERS
        )

    def _is_acceptable(self, translation: str, text: str, context: str) -> bool:
        """출력 형식·용어·수식 보존과 반복 검사로 번역 수락 여부를 반환한다.

        :param translation: 정리하거나 검증할 한국어 번역 후보 문자열.
        :param text: 검사하거나 번역할 입력 텍스트.
        :param context: 앞 영어 원문 등의 참고 문맥. 프롬프트에는 마지막 256자만 사용한다.
        :return: 모든 구현된 검증을 통과하면 True. 의미 정확도를 보장하지 않는다.
        """
        return self._rejection_reason(translation, text, context) is None

    def _rejection_reason(self, translation: str, text: str, context: str):
        """형식·문맥 복사·용어·숫자·수식 검증의 첫 실패 사유를 반환한다.

        :param translation: 정리하거나 검증할 한국어 번역 후보 문자열.
        :param text: 검사하거나 번역할 입력 텍스트.
        :param context: 앞 영어 원문 등의 참고 문맥. 프롬프트에는 마지막 256자만 사용한다.
        :return: 처음 발견한 거부 사유 코드. 구현된 검증을 통과하면 None.
        """
        compact = re.sub(r"\s+", "", translation)
        if not self._is_valid_translation(translation):
            return "invalid_format"
        if context and compact == re.sub(r"\s+", "", context):
            return "context_copy"
        problem = generation_problem(translation) or preservation_problem(text, translation)
        if problem:
            return problem
        if not all(re.sub(r"\s+", "", entry["ko"]) in compact
                   for entry in self._matching_terms(text)):
            return "missing_term"
        return None

    def _stream_output(self, prompt: str, max_tokens: int):
        """전송 구현이 제공해야 하는 생성 스트림 계약을 정의한다.

        :param prompt: 직렬화된 번역 프롬프트.
        :param max_tokens: 최대 생성 토큰 수.
        :return: 생성기 구현을 요구한다.
        :raises NotImplementedError: 구체적인 전송 서비스 없이 호출한 경우.
        """
        raise NotImplementedError('Use the HTTP translation service')

    def _restore_translation(self, translation: str, text: str) -> tuple:
        """구체 서비스의 입력 보호 복원 지점을 제공하며 기본 구현은 후보를 유지한다.

        :param translation: 일반 출력 표식을 정리한 번역 후보.
        :param text: 보호 전 현재 영어 원문.
        :return: 번역 후보와 선택적 복원 실패 사유 코드. 기본값은 (후보, None).
        """
        return translation, None

    def _generate_output(self, prompt: str, max_tokens: int) -> dict:
        """토큰 스트림을 수집하고 반복·지시문 유출을 발견하면 생성을 조기 종료한다.

        :param prompt: 토큰화하거나 모델에 전달할 프롬프트 문자열.
        :param max_tokens: 한 번 생성할 최대 출력 토큰 수.
        :return: text·tokens·finish_reason·stop_reason을 담은 생성 결과 사전.
        """
        stream = self._stream_output(prompt=prompt, max_tokens=max_tokens)
        output, token_count, finish_reason, stop_reason = "", 0, None, None
        try:
            for response in stream:
                output += response.text
                token_count = response.generation_tokens
                finish_reason = response.finish_reason
                stop_reason = generation_problem(output)
                if stop_reason:
                    break
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        return {"text": output, "tokens": token_count, "finish_reason": finish_reason,
                "stop_reason": stop_reason}

    def translate(self, text: str, context: str, max_tokens: int) -> dict:
        """입력 보호를 복원한 번역을 검증·재시도하고 대기·생성·전체 시간을 기록한다.

        :param text: 검사하거나 번역할 입력 텍스트.
        :param context: 앞 영어 원문 등의 참고 문맥. 프롬프트에는 마지막 256자만 사용한다.
        :param max_tokens: 한 번 생성할 최대 출력 토큰 수.
        :return: translation·prompt_tokens·generated_tokens·elapsed_seconds(초) 사전.
        :raises ValueError: 입출력 합계가 2048토큰을 넘거나 두 번 모두 출력 검증에 실패할 때.
        """
        start = perf_counter()
        with measure("translation_total", attempts=0) as metrics:
            with measure("translation_queue"):
                self._lock.acquire()
            try:
                translation = ""
                prompt_tokens = 0
                for attempt, retry in enumerate((False, True), 1):
                    metrics["attempts"] = attempt
                    with measure("translation_attempt", attempt=attempt) as attempt_metrics:
                        prompt = self._build_prompt(text, context if not retry else "", retry)
                        try:
                            prompt_tokens = len(self.tokenizer.encode(prompt))
                        except TranslationTransportError:
                            attempt_metrics['rejection_reason'] = 'translator_unavailable'
                            attempt_metrics['accepted'] = False
                            if retry:
                                raise
                            continue
                        attempt_metrics["prompt_tokens"] = prompt_tokens
                        attempt_metrics["max_tokens"] = max_tokens
                        if prompt_tokens + max_tokens > 2_048:
                            raise ValueError("Prompt and output exceed the 2048-token model limit")
                        with measure("translation_generate", attempt=attempt) as generation_metrics:
                            try:
                                generated = self._generate_output(prompt, max_tokens)
                            except TranslationTransportError:
                                attempt_metrics['rejection_reason'] = 'translator_unavailable'
                                attempt_metrics['accepted'] = False
                                if retry:
                                    raise
                                continue
                            generation_metrics["raw_output_tokens"] = generated["tokens"]
                            generation_metrics["finish_reason"] = generated["finish_reason"]
                            generation_metrics["early_stop_reason"] = generated["stop_reason"]
                        translation = self._clean_translation(generated["text"])
                        translation, restoration_problem = self._restore_translation(translation, text)
                        reason = generated["stop_reason"] or restoration_problem
                        if not reason and generated["finish_reason"] == "length":
                            reason = "token_limit"
                        reason = reason or self._rejection_reason(translation, text, context)
                        accepted = reason is None
                        attempt_metrics["raw_output_tokens"] = generated["tokens"]
                        attempt_metrics["finish_reason"] = generated["finish_reason"]
                        attempt_metrics["rejection_reason"] = reason
                        attempt_metrics["accepted"] = accepted
                    if accepted:
                        break
                else:
                    raise ValueError("Model did not produce a valid Korean translation")
                return {
                    "translation": translation,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated["tokens"],
                    "elapsed_seconds": perf_counter() - start,
                }
            finally:
                self._lock.release()
