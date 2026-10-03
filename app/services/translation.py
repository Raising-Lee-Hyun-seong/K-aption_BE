"""KoreanLM MLX 4bit 모델로 문맥을 반영한 한국어 번역을 생성하고 출력을 정리·검증한다."""
import json
import re
from pathlib import Path
from threading import Lock
from time import perf_counter


OUTPUT_MARKERS = ("참고 문맥:", "번역할 대상:", "입력:", "출력:")
HANGUL_PATTERN = re.compile(r"[가-힣]")


class TranslationService:
    """4bit 번역 모델을 재사용하며 문맥을 반영한 번역을 직렬 생성·검증한다."""
    def __init__(self, model_path: Path, template_path: Path):
        """4bit 모델 설정을 검증하고 모델·토크나이저·샘플러·프롬프트 템플릿을 로드한다."""
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"4bit model not found at {model_path}. Run `make model` first."
            )

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("quantization", {}).get("bits") != 4:
            raise ValueError(f"Expected a 4bit MLX model at {model_path}")

        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tokenizer = load(str(model_path))
        self.sampler = make_sampler(temp=0)
        self.template = json.loads(template_path.read_text(encoding="utf-8"))
        self._lock = Lock()

    def _build_prompt(self, text: str, context: str, retry: bool = False) -> str:
        """영어 문장과 이전 번역 문맥으로 번역 프롬프트를 구성하고 재시도 지시를 추가한다."""
        instruction = (
            "영어 강의 문장 한 개를 자연스러운 한국어로 번역하세요. "
            "번역된 한국어 문장만 출력하세요. 원문, 문맥, 설명, 따옴표, "
            "'참고 문맥', '번역할 대상', '입력', '출력' 같은 표식을 출력하지 마세요."
        )
        if context:
            instruction += (
                f" 앞 문장의 한국어 번역은 의미 파악에만 사용하세요: {context}"
            )
        if retry:
            instruction += " 반드시 한글이 포함된 완전한 한국어 문장 하나만 답하세요."
        return self.template["prompt_input"].format(
            instruction=instruction,
            input=text,
        )

    @staticmethod
    def _clean_translation(translation: str) -> str:
        """생성 텍스트의 프롬프트 표식과 일부 설명·따옴표를 제거해 번역문을 추출한다."""
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
            cleaned = cleaned.split("->", 1)[0].strip()

        quoted = re.findall(r'["“]([^"”]+)["”]', cleaned)
        if ("번역하면" in cleaned or cleaned.startswith("입력된")) and quoted:
            korean_quotes = [text for text in quoted if HANGUL_PATTERN.search(text)]
            if korean_quotes:
                cleaned = korean_quotes[-1]
        return cleaned.strip().strip('"“”')

    @staticmethod
    def _is_valid_translation(translation: str) -> bool:
        """생성 결과에 한글이 포함되고 금지된 프롬프트 표식이 없는지 검사한다."""
        return bool(HANGUL_PATTERN.search(translation)) and not any(
            marker in translation for marker in OUTPUT_MARKERS
        )

    def translate(self, text: str, context: str, max_tokens: int) -> dict:
        """토큰 한도를 검사해 번역을 생성하고 실패 시 문맥 없이 재시도한 뒤 결과·통계를 반환한다."""
        from mlx_lm import generate

        start = perf_counter()
        with self._lock:
            translation = ""
            prompt_tokens = 0
            for retry in (False, True):
                prompt = self._build_prompt(text, context if not retry else "", retry)
                prompt_tokens = len(self.tokenizer.encode(prompt))
                if prompt_tokens + max_tokens > 2_048:
                    raise ValueError("Prompt and output exceed the 2048-token model limit")
                generated = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=self.sampler,
                    verbose=False,
                )
                translation = self._clean_translation(generated)
                if self._is_valid_translation(translation):
                    break
            else:
                raise ValueError("KoreanLM did not produce a Korean-only translation")
        elapsed_seconds = perf_counter() - start
        return {
            "translation": translation,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(self.tokenizer.encode(translation)),
            "elapsed_seconds": elapsed_seconds,
        }
