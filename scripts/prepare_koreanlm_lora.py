"""공식 PEFT LoRA를 검증·다운로드하고 전체 모델 없이 MLX 형식으로 변환한다."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.request import urlopen

import torch
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'quantumaikr/KoreanLM-LoRA'
REVISION = '1447f2ae33e32cb8af610758b90adda0c05670a2'
HASHES = {
    'adapter_model.bin': 'f7f1707df4e86b215ad34347ea3dc9a508c7b46d6f15abf296cd556245aa98fb',
    'adapter_config.json': '6fa154dd2066678e519479b1f82c1530981a45ecbe7b73a94b7903a182322e2d',
}
KEY_PATTERN = re.compile(
    r'^base_model\.model\.model\.layers\.(\d+)\.self_attn\.'
    r'(q_proj|v_proj)\.lora_([AB])(?:\.default)?\.weight$'
)


def sha256_file(path: Path) -> str:
    """파일을 일정 크기로 읽어 SHA256을 계산한다.

    :param path: 해시를 계산하거나 요청을 보낼 대상 경로.
    :return: 파일 내용의 SHA256 16진수 문자열.
    """
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(source: Path) -> dict:
    """설정·가중치가 고정된 공식 파일의 체크섬과 일치하는지 확인한다.

    :param source: 공식 PEFT 설정과 가중치가 있는 디렉터리.
    :return: 검증된 파일 이름과 SHA256의 매핑.
    :raises ValueError: 공식 체크섬과 일치하지 않을 때.
    :raises FileNotFoundError: 필수 원본 파일이 없을 때.
    """
    actual = {}
    for filename, expected in HASHES.items():
        actual[filename] = sha256_file(source / filename)
        if actual[filename] != expected:
            raise ValueError(f'Checksum mismatch: {filename}')
    return actual


def download_sources(source: Path) -> None:
    """인증 정보를 읽지 않고 고정 리비전의 공개 파일을 받아 체크섬 확인 후 저장한다.

    :param source: 공식 PEFT 설정과 가중치가 있는 디렉터리.
    :return: None.
    :raises ValueError: 기존 파일 또는 다운로드 파일의 체크섬이 다를 때.
    :raises urllib.error.URLError: 공개 다운로드 요청에 실패할 때.

    고정 리비전의 파일을 다운로드·저장하며 인증 파일을 읽지 않는다. 정상 원본은 재사용한다.
    """
    source.mkdir(parents=True, exist_ok=True)
    for filename, expected in HASHES.items():
        target = source / filename
        if target.exists():
            if sha256_file(target) != expected:
                raise ValueError(f'Checksum mismatch: {filename}')
            continue
        partial = target.with_suffix(target.suffix + '.partial')
        try:
            url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}'
            with urlopen(url, timeout=60) as response, partial.open('wb') as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            if sha256_file(partial) != expected:
                raise ValueError(f'Checksum mismatch: {filename}')
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def convert_tensors(state: dict, adapter: dict, base: dict) -> tuple:
    """지원 설정·전체 층·키·모양·유한 값을 검사하고 MLX 전치 텐서와 설정을 반환한다.

    :param state: 검증할 PEFT 이름과 PyTorch 텐서의 매핑.
    :param adapter: PEFT 어댑터 설정 사전.
    :param base: 기반 Llama 모델의 구조 설정 사전.
    :return: MLX 이름·전치 텐서 사전과 MLX 어댑터 설정 사전의 튜플.
    :raises ValueError: 미지원 설정 또는 키·층·모양·자료형·유한값·완전성 검사에 실패할 때.
    """
    if base.get('model_type') != 'llama' or adapter.get('peft_type') != 'LORA':
        raise ValueError('Only Llama PEFT LoRA is supported')
    if (adapter.get('bias', 'none') != 'none' or adapter.get('fan_in_fan_out', False)
            or any(adapter.get(key) for key in ('modules_to_save', 'rank_pattern',
                       'alpha_pattern', 'use_rslora', 'use_dora'))):
        raise ValueError('Unsupported LoRA configuration')
    if sorted(adapter.get('target_modules', [])) != ['q_proj', 'v_proj']:
        raise ValueError('Expected only q_proj and v_proj')
    rank, alpha = adapter['r'], adapter['lora_alpha']
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError('LoRA rank must be a positive integer')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('LoRA alpha must be positive and finite')
    hidden, layers, heads = (base[key] for key in
                             ('hidden_size', 'num_hidden_layers', 'num_attention_heads'))
    if min(hidden, layers, heads) <= 0 or hidden % heads:
        raise ValueError('Invalid base model dimensions')
    head_dim = base.get('head_dim') or hidden // heads
    output_dims = {'q_proj': heads * head_dim,
                   'v_proj': (base.get('num_key_value_heads') or heads) * head_dim}
    converted = {}
    for key, tensor in state.items():
        match = KEY_PATTERN.fullmatch(key)
        if not match:
            raise ValueError(f'Unexpected adapter key: {key}')
        layer, projection, side = int(match[1]), match[2], match[3]
        name = f'model.layers.{layer}.self_attn.{projection}.lora_{side.lower()}'
        if layer >= layers or name in converted:
            raise ValueError(f'Out-of-range or duplicate adapter key: {key}')
        shape = (rank, hidden) if side == 'A' else (output_dims[projection], rank)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f'Unexpected tensor shape: {key}; expected {shape}')
        if tensor.dtype not in (torch.float16, torch.float32) or not torch.isfinite(tensor).all():
            raise ValueError(f'Unsupported dtype or nonfinite tensor: {key}')
        converted[name] = tensor.detach().cpu().T.contiguous().numpy()
    expected = {f'model.layers.{layer}.self_attn.{projection}.lora_{side}'
                for layer in range(layers) for projection in output_dims for side in ('a', 'b')}
    if set(converted) != expected:
        raise ValueError(f'Incomplete adapter: {len(expected - set(converted))} missing tensors')
    config = {'fine_tune_type': 'lora', 'num_layers': layers,
              'lora_parameters': {'rank': rank, 'scale': alpha / rank, 'dropout': 0.0,
                                  'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}
    return converted, config


def verify_numerics(state: dict, converted: dict, scale: float, mlx_scale: float) -> dict:
    """전체 모델 없이 모든 투영의 PEFT 선형 계산과 저장된 MLX 행렬 계산을 CPU에서 비교한다.

    :param state: 검증할 PEFT 이름과 PyTorch 텐서의 매핑.
    :param converted: MLX 이름으로 매핑하고 전치한 NumPy 텐서 사전.
    :param scale: PEFT 참조 계산에 사용할 alpha/r 배율.
    :param mlx_scale: 변환된 MLX 계산에 사용할 배율.
    :return: 확인한 투영 쌍 수·최대 절대 차이·허용 오차·계산 환경 사전.
    :raises AssertionError: 변환 계산이 PEFT 참조값의 허용 오차를 벗어날 때.
    """
    generator = torch.Generator().manual_seed(20261005)
    checked, maximum = 0, 0.0
    for key, a in state.items():
        match = KEY_PATTERN.fullmatch(key)
        if match[3] != 'A':
            continue
        b_key = key.replace('lora_A', 'lora_B')
        b = state[b_key].float()
        prefix = f'model.layers.{int(match[1])}.self_attn.{match[2]}'
        a_mlx = torch.from_numpy(converted[prefix + '.lora_a'].copy()).float()
        b_mlx = torch.from_numpy(converted[prefix + '.lora_b'].copy()).float()
        x = torch.randn((2, a.shape[1]), generator=generator)
        reference = torch.nn.functional.linear(torch.nn.functional.linear(x, a.float()), b) * scale
        result = ((x @ a_mlx) @ b_mlx) * mlx_scale
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
        maximum = max(maximum, float((result - reference).abs().max()))
        checked += 1
    return {'projection_pairs_checked': checked, 'max_abs_difference': maximum,
            'rtol': 1e-5, 'atol': 1e-6, 'backend': 'CPU PyTorch; actual MLX runtime not executed'}


def adapter_signal(state: dict) -> dict:
    """각 A·B 행렬의 영값 여부를 집계하고 모든 B가 0이면 델타가 항상 0임을 표시한다.

    :param state: 검증할 PEFT 이름과 PyTorch 텐서의 매핑.
    :return: A/B별 텐서·영값·비영 원소 수와 모든 B가 0인지 표시하는 사전.
    """
    counts = {}
    for side in ('A', 'B'):
        tensors = [tensor for key, tensor in state.items() if KEY_PATTERN.fullmatch(key)[3] == side]
        counts[side] = {'tensor_count': len(tensors),
                        'zero_tensor_count': sum(not bool(torch.count_nonzero(tensor)) for tensor in tensors),
                        'nonzero_elements': sum(int(torch.count_nonzero(tensor)) for tensor in tensors)}
    all_b_zero = counts['B']['tensor_count'] > 0 and counts['B']['zero_tensor_count'] == counts['B']['tensor_count']
    return {'matrices': counts, 'all_b_zero': all_b_zero,
            'delta_identically_zero': all_b_zero,
            'interpretation': ('모든 B가 0이므로 scale * B @ A = 0이다. 이 어댑터는 학습된 변화를 추가하지 않는다.'
                               if all_b_zero else '0이 아닌 B가 있다. 품질 개선 여부는 별도 검증이 필요하다.')}


def verify_mlx_numerics(state: dict, path: Path, scale: float, mlx_scale: float) -> dict:
    """전체 모델 없이 실제 MLX 파일 로딩·CPU 행렬 계산을 PEFT 계산과 비교한다.

    :param state: 검증할 PEFT 이름과 PyTorch 텐서의 매핑.
    :param path: 다시 읽어 검증할 MLX safetensors 파일 경로.
    :param scale: PEFT 참조 계산에 사용할 alpha/r 배율.
    :param mlx_scale: 변환된 MLX 계산에 사용할 배율.
    :return: MLX CPU에서 확인한 투영 쌍 수·최대 절대 차이·허용 오차 사전.
    :raises AssertionError: MLX 계산이 PEFT 참조값의 허용 오차를 벗어날 때.
    """
    import mlx.core as mx
    import numpy as np

    previous_device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        arrays = mx.load(str(path), format='safetensors', stream=mx.cpu)
        generator = torch.Generator().manual_seed(20261005)
        checked, maximum = 0, 0.0
        for key, a in state.items():
            match = KEY_PATTERN.fullmatch(key)
            if match[3] != 'A':
                continue
            b = state[key.replace('lora_A', 'lora_B')].float()
            prefix = f'model.layers.{int(match[1])}.self_attn.{match[2]}'
            x = torch.randn((2, a.shape[1]), generator=generator)
            reference = torch.nn.functional.linear(torch.nn.functional.linear(x, a.float()), b) * scale
            result = (mx.array(x.numpy()) @ arrays[prefix + '.lora_a'].astype(mx.float32)
                      @ arrays[prefix + '.lora_b'].astype(mx.float32)) * mlx_scale
            mx.eval(result)
            actual = torch.from_numpy(np.array(result, copy=True))
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((actual - reference).abs().max()))
            checked += 1
        return {'projection_pairs_checked': checked, 'max_abs_difference': maximum,
                'rtol': 1e-5, 'atol': 1e-6, 'backend': 'MLX CPU vs PEFT-style PyTorch CPU'}
    finally:
        mx.set_default_device(previous_device)


def main() -> None:
    """필요할 경우 다운로드하고 검증·변환·저장 후 다시 읽어 수치 비교 보고서를 기록한다.

    :return: None.
    :raises ValueError: 입출력 디렉터리가 같거나 원본·설정 검증에 실패할 때.
    :raises AssertionError: 변환 수치 검증에 실패할 때.

    CLI 인자를 읽고 변환 가중치·설정·출처·평가 보고서를 저장한다. 전체 기반 모델은 로드하지 않는다.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--verify-mlx', action='store_true', help='모델 없이 MLX CPU 행렬 계산도 비교')
    parser.add_argument('--source', type=Path, default=ROOT / 'models/koreanlm-lora-peft')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/koreanlm-lora-mlx')
    parser.add_argument('--base-config', type=Path, default=ROOT / 'models/koreanlm-4bit/config.json')
    parser.add_argument('--report', type=Path, default=ROOT / 'evaluation/lora_conversion.json')
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Source and output directories must differ')
    if args.download:
        download_sources(args.source)
    hashes = verify_sources(args.source)
    adapter = json.loads((args.source / 'adapter_config.json').read_text())
    base = json.loads(args.base_config.read_text())
    state = torch.load(args.source / 'adapter_model.bin', map_location='cpu', weights_only=True)
    converted, config = convert_tensors(state, adapter, base)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = args.output / 'adapters.safetensors'
    partial = weights.with_suffix('.partial')
    try:
        save_file(converted, str(partial), metadata={'format': 'mlx'})
        numerical = verify_numerics(state, load_file(str(partial)),
                                   adapter['lora_alpha'] / adapter['r'],
                                   config['lora_parameters']['scale'])
        native = (verify_mlx_numerics(state, partial, adapter['lora_alpha'] / adapter['r'],
                                    config['lora_parameters']['scale'])
                  if args.verify_mlx else None)
        partial.replace(weights)
    finally:
        partial.unlink(missing_ok=True)
    (args.output / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = {'repository': REPOSITORY, 'revision': REVISION, 'source_sha256': hashes,
              'tensor_count': len(converted), 'mlx_config': config, 'numerical_check': numerical,
              'mlx_cpu_check': native,
              'adapter_signal': adapter_signal(state),
              'weights_bytes': weights.stat().st_size, 'output_sha256': sha256_file(weights),
              'configured_base_name': adapter['base_model_name_or_path'],
              'base_config': str(args.base_config.relative_to(ROOT)) if args.base_config.is_relative_to(ROOT) else str(args.base_config),
              'base_model_loaded': False, 'translation_quality_tested': False,
              'limitation': '구조·변환 수치만 검증했다. 기반 모델 리비전·전체 모델 추론 호환성은 미검증이다.'}
    (args.output / 'source.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
