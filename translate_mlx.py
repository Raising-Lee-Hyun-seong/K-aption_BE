"""Translate one English segment with local 4bit KoreanLM on Apple Silicon."""
import argparse
import json
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', help='English segment (without timestamps)')
    parser.add_argument('--context', default='', help='Adjacent English segments')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/koreanlm-4bit')
    parser.add_argument('--max-tokens', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 1024:
        parser.error('--max-tokens must be between 1 and 1024')
    if not (args.model / 'config.json').is_file():
        parser.error('Local model missing. Prepare and quantize KoreanLM first.')
    config = json.loads((args.model / 'config.json').read_text())
    if config.get('quantization', {}).get('bits') != 4:
        parser.error('Expected a 4bit MLX model')
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    start = perf_counter()
    model, tokenizer = load(str(args.model))
    print(f'Model loaded: {perf_counter() - start:.2f}s', flush=True)
    template = json.loads((ROOT / 'templates/korean.json').read_text())
    target = args.text
    if args.context:
        target = f'참고 문맥: {args.context}\n번역할 대상: {args.text}'
    prompt = template['prompt_input'].format(
        instruction='영어 강의의 대상 문장만 한국어로 번역하세요. 참고 문맥은 번역하지 말고 설명 없이 번역문만 출력하세요.',
        input=target)
    if len(tokenizer.encode(prompt)) + args.max_tokens > 2048:
        parser.error('Prompt and output exceed the 2048-token model limit')
    generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
             sampler=make_sampler(temp=0), verbose=True)


if __name__ == '__main__':
    main()
