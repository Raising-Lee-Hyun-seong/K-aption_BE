"""KoreanLM MLX 4bit 모델로 문맥을 반영한 한국어 번역을 생성하고 출력을 정리·검증한다."""
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


class TranslationService:
    """4bit 번역 모델을 재사용하며 문맥을 반영한 번역을 직렬 생성·검증한다."""
    def __init__(self, model_path: Path, template_path: Path):
        """4bit 모델 설정을 검증하고 모델·토크나이저·샘플러·프롬프트 템플릿을 로드한다.

        :param model_path: config.json과 가중치가 있는 MLX 4bit 모델 디렉터리.
        :param template_path: 번역 프롬프트 템플릿 JSON 경로.
        :return: None.
        :raises FileNotFoundError: 모델 설정 파일이 없을 때.
        :raises ValueError: 모델 양자화 설정이 4bit가 아닐 때.
        """
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"4bit model not found at {model_path}. Run `make model` first."
            )

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("quantization", {}).get("bits") != 4:
            raise ValueError(f"Expected a 4bit MLX model at {model_path}")

        with measure("translation_model_load"):
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler

            self.model, self.tokenizer = load(str(model_path))
            self.sampler = make_sampler(temp=0)
        self.template = json.loads(template_path.read_text(encoding="utf-8"))
        self.glossary = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
        self._lock = Lock()

    def _matching_terms(self, text: str) -> list:
        """단어 경계로 용어를 찾고 겹치는 위치에서는 긴 복합 용어를 우선한다.

        :param text: 검사하거나 번역할 입력 텍스트.
        :return: 원문과 매칭된 영어·한국어 용어 항목 목록.
        """
        matched, occupied = [], []
        for entry in sorted(self.glossary, key=lambda entry: -len(entry["en"])):
            spans = [match.span() for match in re.finditer(
                r"\b" + re.escape(entry["en"]) + r"\b", text, re.IGNORECASE)
                if not any(match.start() < end and match.end() > start
                           for start, end in occupied)]
            if spans:
                matched.append(entry)
                occupied.extend(spans)
        return matched

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

    def _generate_output(self, prompt: str, max_tokens: int) -> dict:
        """토큰 스트림을 수집하고 반복·지시문 유출을 발견하면 생성을 조기 종료한다.

        :param prompt: 토큰화하거나 모델에 전달할 프롬프트 문자열.
        :param max_tokens: 한 번 생성할 최대 출력 토큰 수.
        :return: text·tokens·finish_reason·stop_reason을 담은 생성 결과 사전.
        """
        from mlx_lm import stream_generate

        stream = stream_generate(self.model, self.tokenizer, prompt=prompt,
                                 max_tokens=max_tokens, sampler=self.sampler)
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
        """번역을 검증·재시도하고 대기·생성·전체 시간을 기록해 결과와 통계를 반환한다.

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
                        prompt_tokens = len(self.tokenizer.encode(prompt))
                        attempt_metrics["prompt_tokens"] = prompt_tokens
                        attempt_metrics["max_tokens"] = max_tokens
                        if prompt_tokens + max_tokens > 2_048:
                            raise ValueError("Prompt and output exceed the 2048-token model limit")
                        with measure("translation_generate", attempt=attempt) as generation_metrics:
                            generated = self._generate_output(prompt, max_tokens)
                            generation_metrics["raw_output_tokens"] = generated["tokens"]
                            generation_metrics["finish_reason"] = generated["finish_reason"]
                            generation_metrics["early_stop_reason"] = generated["stop_reason"]
                        translation = self._clean_translation(generated["text"])
                        reason = generated["stop_reason"]
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
                    raise ValueError("KoreanLM did not produce a valid Korean translation")
                return {
                    "translation": translation,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated["tokens"],
                    "elapsed_seconds": perf_counter() - start,
                }
            finally:
                self._lock.release()
