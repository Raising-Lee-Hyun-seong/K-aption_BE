"""기존 MLX affine 4bit 가중치를 재양자화 없이 Q4_1 GGUF로 재포장하고 수치를 검증한다."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import gguf
from gguf.quants import dequantize
import numpy as np
from safetensors import safe_open
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS = {'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
               'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_output',
               'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up', 'mlp.down_proj': 'ffn_down'}


def sha256_file(path: Path) -> str:
    """파일을 작은 청크로 읽어 내용 체크섬을 계산한다.

    :param path: 체크섬을 계산할 공개 모델 파일 경로.
    :return: SHA256 16진수 문자열.
    """
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def permute_rows(rows: int, heads: int) -> np.ndarray:
    """HF·MLX의 Q/K RoPE 행을 llama.cpp 순서로 옮길 인덱스를 만든다.

    :param rows: Q 또는 K 출력 행 수. heads의 두 배로 나누어져야 한다.
    :param heads: Q는 attention head 수, K는 key/value head 수.
    :return: 재배열할 기존 행 인덱스의 일차원 배열.
    :raises ValueError: 행·헤드 수가 유효하지 않은 경우.
    """
    if heads < 1 or rows < 1 or rows % (heads * 2):
        raise ValueError('Q/K rows must be divisible by twice the head count')
    return np.arange(rows).reshape(heads, 2, rows // heads // 2).swapaxes(1, 2).reshape(-1)


def pack_q4_1(weight: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """MLX의 64개 단위 4bit 정수·배율·오프셋을 값 변경 없이 GGUF 32개 블록으로 옮긴다.

    :param weight: 행별 연속 8개 nibble을 저장한 2차원 uint32 배열.
    :param scales: 각 64개 원소의 float16 배율. 열 수는 weight 열 수의 1/8.
    :param biases: scales와 같은 모양·자료형의 float16 오프셋.
    :return: 행별 Q4_1 블록 바이트 배열. 32개 원소마다 20바이트를 사용한다.
    :raises ValueError: 자료형·모양이 MLX affine 4bit group size 64와 다를 때.
    """
    if (weight.ndim != 2 or weight.dtype != np.uint32 or weight.shape[1] % 8
            or scales.dtype != np.float16 or biases.dtype != np.float16
            or scales.shape != (weight.shape[0], weight.shape[1] // 8)
            or biases.shape != scales.shape or not np.isfinite(scales).all()
            or not np.isfinite(biases).all()):
        raise ValueError('Expected finite MLX uint32/float16 affine 4bit groups of 64')
    shifts = np.arange(8, dtype=np.uint32) * 4
    values = ((weight.reshape(-1, 4, 1) >> shifts) & 15).astype(np.uint8).reshape(-1, 32)
    blocks = np.empty((values.shape[0], 20), dtype=np.uint8)
    blocks[:, :2] = np.repeat(scales.reshape(-1), 2).astype('<f2').view(np.uint8).reshape(-1, 2)
    blocks[:, 2:4] = np.repeat(biases.reshape(-1), 2).astype('<f2').view(np.uint8).reshape(-1, 2)
    blocks[:, 4:] = values[:, :16] | (values[:, 16:] << 4)
    return blocks.reshape(weight.shape[0], -1)


def verify_q4_1(weight: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                packed: np.ndarray) -> None:
    """모든 행의 재포장 값이 공식 GGUF 역양자화와 정확히 일치하는지 작은 배치로 확인한다.

    :param weight: 재배열까지 반영한 기존 MLX uint32 가중치.
    :param scales: 기존 float16 배율.
    :param biases: 기존 float16 오프셋.
    :param packed: 검증할 Q4_1 바이트 배열.
    :return: None. 전체 모델을 FP16으로 풀어 저장하지 않는다.
    :raises ValueError: 어느 원소든 배율·오프셋·정수 값이 달라진 경우.
    """
    shifts = np.arange(8, dtype=np.uint32) * 4
    for start in range(0, weight.shape[0], 32):
        end = start + 32
        values = ((weight[start:end, :, None] >> shifts) & 15).reshape(-1, scales.shape[1], 64)
        expected = values.astype(np.float32) * scales[start:end, :, None].astype(np.float32)
        expected += biases[start:end, :, None].astype(np.float32)
        actual = dequantize(packed[start:end], gguf.GGMLQuantizationType.Q4_1)
        if not np.array_equal(expected.reshape(actual.shape), actual):
            raise ValueError('GGUF repacking changed a quantized weight value')


def tensor_layout(config: dict) -> list[tuple[str, str, tuple[int, ...], int]]:
    """현재 Llama 모델에서 허용한 모든 가중치와 GGUF 이름·모양·RoPE 헤드를 정의한다.

    :param config: 기존 MLX Llama 모델의 차원·층·헤드 설정.
    :return: 원본 접두어·GGUF 이름·논리적 모양·Q/K 헤드 수 튜플 목록. norm 헤드는 0.
    :raises ValueError: 모델 구조나 양자화 설정이 지원 범위와 다를 때.
    """
    quant = config.get('quantization', {})
    if (config.get('model_type') != 'llama' or quant.get('bits') != 4
            or quant.get('group_size') != 64 or quant.get('mode', 'affine') != 'affine'
            or config.get('rope_scaling') or config.get('rope_traditional', False)
            or config.get('tie_word_embeddings', False)):
        raise ValueError('Only untied Llama MLX affine 4bit/group64 without RoPE scaling is supported')
    hidden, middle, vocab = (config[key] for key in ('hidden_size', 'intermediate_size', 'vocab_size'))
    heads, kv = config['num_attention_heads'], config.get('num_key_value_heads', config['num_attention_heads'])
    dimensions = (hidden, middle, vocab, heads, kv, config['num_hidden_layers'])
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions)
            or hidden % heads or heads % kv or (hidden // heads) % 2
            or hidden % 64 or middle % 64):
        raise ValueError('Unsupported model dimensions')
    kv_rows = kv * (hidden // heads)
    result = [('model.embed_tokens', 'token_embd.weight', (vocab, hidden), 0),
              ('lm_head', 'output.weight', (vocab, hidden), 0),
              ('model.norm', 'output_norm.weight', (hidden,), 0)]
    for layer in range(config['num_hidden_layers']):
        prefix = f'model.layers.{layer}'
        for norm, target in (('input_layernorm', 'attn_norm'), ('post_attention_layernorm', 'ffn_norm')):
            result.append((f'{prefix}.{norm}', f'blk.{layer}.{target}.weight', (hidden,), 0))
        for source, target in PROJECTIONS.items():
            shape = ((middle, hidden) if source in ('mlp.gate_proj', 'mlp.up_proj') else
                     (hidden, middle) if source == 'mlp.down_proj' else
                     (kv_rows, hidden) if source in ('self_attn.k_proj', 'self_attn.v_proj') else (hidden, hidden))
            permutation_heads = heads if source == 'self_attn.q_proj' else kv if source == 'self_attn.k_proj' else 0
            result.append((f'{prefix}.{source}', f'blk.{layer}.{target}.weight', shape, permutation_heads))
    return result


def add_metadata(writer, config: dict, source: Path, source_hash: str) -> None:
    """원본 Llama 구조와 SentencePiece 토큰·점수·특수 ID를 GGUF에 기록한다.

    :param writer: 아직 헤더를 쓰지 않은 GGUFWriter.
    :param config: 기존 모델의 구조 설정.
    :param source: tokenizer.model과 tokenizer_config.json이 있는 모델 디렉터리.
    :param source_hash: 원본 safetensors SHA256.
    :return: None. 전달한 writer에 메타데이터를 추가한다.
    :raises ValueError: 어휘 수나 특수 토큰 설정이 현재 모델과 맞지 않는 경우.
    """
    writer.add_name('KoreanLM existing MLX 4bit repack')
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q4_1))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_context_length(config.get('max_position_embeddings', 2048))
    writer.add_embedding_length(config['hidden_size'])
    writer.add_block_count(config['num_hidden_layers'])
    writer.add_feed_forward_length(config['intermediate_size'])
    writer.add_head_count(config['num_attention_heads'])
    writer.add_head_count_kv(config.get('num_key_value_heads', config['num_attention_heads']))
    writer.add_layer_norm_rms_eps(config['rms_norm_eps'])
    writer.add_rope_dimension_count(config['hidden_size'] // config['num_attention_heads'])
    writer.add_rope_freq_base(config.get('rope_theta', 10000.0))
    writer.add_string('kaption.source_sha256', source_hash)
    processor = spm.SentencePieceProcessor(model_file=str(source / 'tokenizer.model'))
    if processor.vocab_size() != config['vocab_size']:
        raise ValueError('Tokenizer vocabulary differs from model vocabulary')
    tokens, scores, types = [], [], []
    for index in range(processor.vocab_size()):
        tokens.append(processor.id_to_piece(index).encode('utf-8'))
        scores.append(processor.get_score(index))
        kind = (gguf.TokenType.UNKNOWN if processor.is_unknown(index) else
                gguf.TokenType.CONTROL if processor.is_control(index) else
                gguf.TokenType.UNUSED if processor.is_unused(index) else
                gguf.TokenType.BYTE if processor.is_byte(index) else gguf.TokenType.NORMAL)
        types.append(int(kind))
    writer.add_tokenizer_model('llama')
    writer.add_tokenizer_pre('default')
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(processor.bos_id())
    writer.add_eos_token_id(processor.eos_id())
    writer.add_unk_token_id(processor.unk_id())
    writer.add_pad_token_id(processor.unk_id())
    tokenizer_config = json.loads((source / 'tokenizer_config.json').read_text())
    if (tokenizer_config.get('add_eos_token', False) or not tokenizer_config.get('add_bos_token', True)
            or any(int(key) > 2 for key in tokenizer_config.get('added_tokens_decoder', {}))):
        raise ValueError('Unsupported tokenizer special-token additions or BOS/EOS policy')
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)


def convert(source: Path, output: Path) -> dict:
    """단일 MLX 모델 파일을 행 단위로 검증·재포장해 새 GGUF와 변환 보고서를 저장한다.

    :param source: 기존 KoreanLM MLX 4bit 모델 디렉터리. 읽기만 한다.
    :param output: 새 GGUF 파일 경로. 기존 파일 덮어쓰기는 허용하지 않는다.
    :return: 입력·출력 체크섬과 전체 검증한 텐서·가중치 수를 담은 보고서.
    :raises FileExistsError: 출력 GGUF가 이미 있는 경우.
    :raises ValueError: 미지원 구조·누락·추가 텐서·모양·수치 불일치가 있는 경우.
    :raises OSError: 읽기·임시 기록·최종 저장에 실패한 경우.

    출력 부모와 임시 파일을 만들며 완료 후 원자적으로 GGUF를 배치한다.
    전체 FP16 중간 파일은 만들지 않고 결과 옆 .conversion.json에 보고서를 저장한다.
    """
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    config = json.loads((source / 'config.json').read_text())
    layout = tensor_layout(config)
    weights_file = source / 'model.safetensors'
    source_hash = sha256_file(weights_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gguf-', dir=output.parent) as temporary:
        temporary_path = Path(temporary) / 'model.gguf'
        writer = gguf.GGUFWriter(temporary_path, 'llama', use_temp_file=True)
        add_metadata(writer, config, source, source_hash)
        verified = 0
        try:
            with safe_open(weights_file, framework='np') as tensors:
                expected_keys = set()
                for prefix, _, shape, _ in layout:
                    expected_keys.update(prefix + suffix for suffix in (
                        ('.weight',) if len(shape) == 1 else ('.weight', '.scales', '.biases')))
                if set(tensors.keys()) != expected_keys:
                    raise ValueError('Missing or unexpected source tensors')
                for prefix, target, shape, heads in layout:
                    weight = tensors.get_tensor(prefix + '.weight')
                    if len(shape) == 1:
                        if (weight.shape != shape or weight.dtype not in (np.float16, np.float32)
                                or not np.isfinite(weight).all()):
                            raise ValueError(f'Invalid norm shape or values: {prefix}')
                        writer.add_tensor(target, weight.astype(np.float32))
                    else:
                        if weight.shape != (shape[0], shape[1] // 8):
                            raise ValueError(f'Invalid packed matrix shape: {prefix}')
                        scales, biases = (tensors.get_tensor(prefix + suffix) for suffix in ('.scales', '.biases'))
                        if heads:
                            order = permute_rows(shape[0], heads)
                            weight, scales, biases = weight[order], scales[order], biases[order]
                        packed = pack_q4_1(weight, scales, biases)
                        verify_q4_1(weight, scales, biases, packed)
                        writer.add_tensor(target, packed, raw_dtype=gguf.GGMLQuantizationType.Q4_1)
                        del scales, biases, packed
                    verified += int(np.prod(shape))
                    del weight
                    print(f'Verified {target}', flush=True)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=False)
        finally:
            writer.close()
            if writer.temp_file is not None:
                writer.temp_file.close()
        if sha256_file(weights_file) != source_hash:
            raise ValueError('Source weights changed during conversion')
        os.link(temporary_path, output)
    report = {'source_sha256': source_hash, 'gguf_sha256': sha256_file(output),
              'format': 'Q4_1', 'requantized': False, 'tensor_count': len(layout),
              'verified_weight_values': verified, 'bytes': output.stat().st_size}
    output.with_suffix('.conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    """지정한 기존 모델에서 새 GGUF를 만들고 전체 수치 검증 보고서를 출력한다.

    :return: None.
    :raises ValueError: 모델 구조 또는 변환 수치 검증이 실패한 경우.
    :raises OSError: 기존 모델 읽기 또는 변환 결과 저장에 실패한 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'models/koreanlm-4bit')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/koreanlm-gguf/model.gguf')
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
