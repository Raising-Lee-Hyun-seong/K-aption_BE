"""고정 리비전의 KoreanLM을 다운로드하고 MLX 변환에 사용할 FP16 safetensors를 준비한다."""
import gc
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'quantumaikr/KoreanLM'
REVISION = 'f4351abcdd6a933afbaffad0badf60c273e71920'
DEST = ROOT / 'models/koreanlm-fp16'
CACHE = ROOT / '.cache/huggingface'


def download(name):
    """고정된 모델 리비전의 파일을 Hugging Face 캐시에 다운로드하고 경로를 반환한다.

    :param name: 고정 모델 리비전에서 다운로드할 파일 이름.
    :return: 다운로드한 파일의 로컬 캐시 경로 문자열.
    """
    return hf_hub_download(MODEL, name, revision=REVISION, cache_dir=str(CACHE))


def main():
    """원본 가중치를 FP16 safetensors로 변환하고 토크나이저·설정·가중치 인덱스를 저장한다.

    :return: None.

    고정 모델을 캐시에 내려받고 models/koreanlm-fp16에 FP16 가중치·토크나이저·설정을 저장한다.
    """
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'tokenizer.model', 'tokenizer_config.json',
                 'special_tokens_map.json', 'generation_config.json']:
        shutil.copyfile(download(name), DEST / name)
    index = json.loads(Path(download('pytorch_model.bin.index.json')).read_text())
    shards = sorted(set(index['weight_map'].values()))
    weight_map = {}
    for number, name in enumerate(shards, 1):
        target = DEST / f'model-{number:05d}-of-{len(shards):05d}.safetensors'
        if target.exists():
            print(f'Already converted: {target.name}', flush=True)
        else:
            print(f'Downloading: {name}', flush=True)
            path = download(name)
            print(f'Converting: {name}', flush=True)
            # mmap avoids loading the entire original FP32 shard into RAM.
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            converted = {k: v.to(torch.float16).contiguous() if v.is_floating_point()
                         else v.contiguous() for k, v in state.items()}
            temp = target.with_suffix('.partial')
            save_file(converted, str(temp), metadata={'format': 'pt'})
            temp.replace(target)
            del converted, state
            gc.collect()
        weight_map.update({k: target.name for k, v in index['weight_map'].items() if v == name})
    config = json.loads((DEST / 'config.json').read_text())
    config['torch_dtype'] = 'float16'
    tokenizer = AutoTokenizer.from_pretrained(str(DEST))
    # Original config says BOS=0/EOS=1, but tokenizer has BOS=1/EOS=2.
    config['bos_token_id'] = tokenizer.bos_token_id
    config['eos_token_id'] = tokenizer.eos_token_id
    generation = json.loads((DEST / 'generation_config.json').read_text())
    generation.update(bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    (DEST / 'generation_config.json').write_text(json.dumps(generation, indent=2))
    (DEST / 'config.json').write_text(json.dumps(config, indent=2))
    (DEST / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {}, 'weight_map': weight_map}, indent=2))
    (DEST / 'source.json').write_text(json.dumps({'model': MODEL, 'revision': REVISION}, indent=2))
    print(f'Prepared: {DEST}', flush=True)


if __name__ == '__main__':
    main()
