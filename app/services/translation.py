import json
from pathlib import Path
from threading import Lock
from time import perf_counter


class TranslationService:
    def __init__(self, model_path: Path, template_path: Path):
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

    def translate(self, text: str, context: str, max_tokens: int) -> dict:
        from mlx_lm import generate

        target = text
        if context:
            target = f"참고 문맥: {context}\n번역할 대상: {text}"
        prompt = self.template["prompt_input"].format(
            instruction=(
                "영어 강의의 대상 문장만 한국어로 번역하세요. "
                "참고 문맥은 번역하지 말고 설명 없이 번역문만 출력하세요."
            ),
            input=target,
        )
        prompt_tokens = len(self.tokenizer.encode(prompt))
        if prompt_tokens + max_tokens > 2_048:
            raise ValueError("Prompt and output exceed the 2048-token model limit")

        start = perf_counter()
        with self._lock:
            translation = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=self.sampler,
                verbose=False,
            ).strip()
        elapsed_seconds = perf_counter() - start
        return {
            "translation": translation,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(self.tokenizer.encode(translation)),
            "elapsed_seconds": elapsed_seconds,
        }
