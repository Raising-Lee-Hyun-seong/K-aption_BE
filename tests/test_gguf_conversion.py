"""모델 없이 MLX 4bit 정수 재포장·RoPE 행 순서·공식 GGUF 역양자화를 검증한다."""

import unittest

import gguf
from gguf.quants import dequantize
import numpy as np

from scripts.prepare_koreanlm_gguf import pack_q4_1, permute_rows, tensor_layout, verify_q4_1


class GgufConversionTests(unittest.TestCase):
    """재양자화 없이 정수·배율·오프셋과 Q/K 행 의미를 보존하는지 검사한다."""

    def fixture(self):
        """음수 오프셋·다양한 배율과 모든 nibble 값을 가진 두 그룹 대역을 만든다.

        :return: 기존 uint32 가중치·float16 배율·오프셋·원래 정수 값 튜플.
        """
        values = np.arange(4 * 128, dtype=np.uint32).reshape(4, 128) % 16
        shifts = np.arange(8, dtype=np.uint32) * 4
        words = np.bitwise_or.reduce(values.reshape(4, -1, 8) << shifts, axis=-1)
        scales = np.array([[.1, .3], [1, 2], [.01, .05], [.5, 4]], dtype=np.float16)
        biases = np.array([[-2, 1], [-3, 2], [-.1, -.5], [0, -10]], dtype=np.float16)
        return words, scales, biases, values

    def test_every_weight_matches_official_gguf_dequantization(self):
        """64개 배율을 두 개의 32개 블록으로 옮겨도 모든 원소 값이 정확히 보존된다.

        :return: None.
        """
        weight, scales, biases, values = self.fixture()
        packed = pack_q4_1(weight, scales, biases)
        self.assertEqual(packed.shape, (4, 80))
        actual = dequantize(packed, gguf.GGMLQuantizationType.Q4_1)
        expected = values.astype(np.float32) * np.repeat(scales.astype(np.float32), 64, axis=1)
        expected += np.repeat(biases.astype(np.float32), 64, axis=1)
        np.testing.assert_array_equal(actual, expected)
        verify_q4_1(weight, scales, biases, packed)

    def test_corrupted_integer_and_scale_are_detected(self):
        """파일 크기가 같아도 nibble 또는 배율 손상이 있으면 변환 검증에서 거부한다.

        :return: None.
        """
        weight, scales, biases, _ = self.fixture()
        original = pack_q4_1(weight, scales, biases)
        for column in (0, 4, 19):
            with self.subTest(column=column):
                corrupted = original.copy()
                corrupted[0, column] ^= 1
                with self.assertRaises(ValueError):
                    verify_q4_1(weight, scales, biases, corrupted)

    def test_unsupported_shapes_dtypes_and_nonfinite_scales_are_rejected(self):
        """잘못된 그룹·자료형·비유한 수를 재포장 전에 거부한다.

        :return: None.
        """
        weight, scales, biases, _ = self.fixture()
        for changed in ((weight.astype(np.uint64), scales, biases),
                        (weight[:, :7], scales, biases),
                        (weight, scales.astype(np.float32), biases),
                        (weight, scales[:, :1], biases),
                        (weight, scales * np.float16('inf'), biases)):
            with self.subTest(shape=changed[0].shape), self.assertRaises(ValueError):
                pack_q4_1(*changed)

    def test_qk_permutation_moves_rows_and_preserves_values(self):
        """공식 HF Llama 변환과 같은 Q/K RoPE 행 순서를 적용한다.

        :return: None.
        """
        order = permute_rows(8, 2)
        np.testing.assert_array_equal(order, [0, 2, 1, 3, 4, 6, 5, 7])
        matrix = np.arange(8 * 64).reshape(8, 64)
        np.testing.assert_array_equal(matrix[order][np.argsort(order)], matrix)
        with self.assertRaises(ValueError):
            permute_rows(7, 2)

    def test_layout_includes_all_gqa_and_mlp_dimensions(self):
        """GQA의 K/V·MLP 모양을 구분하고 필요한 모든 텐서를 한 번씩 정의한다.

        :return: None.
        """
        config = {'model_type': 'llama', 'quantization': {'bits': 4, 'group_size': 64},
                  'hidden_size': 128, 'intermediate_size': 256, 'vocab_size': 64,
                  'num_attention_heads': 4, 'num_key_value_heads': 2, 'num_hidden_layers': 2}
        layout = tensor_layout(config)
        self.assertEqual(len(layout), 21)
        self.assertEqual(len({row[1] for row in layout}), 21)
        rows = {row[1]: row for row in layout}
        self.assertEqual(rows['blk.0.attn_q.weight'][2:], ((128, 128), 4))
        self.assertEqual(rows['blk.0.attn_k.weight'][2:], ((64, 128), 2))
        self.assertEqual(rows['blk.0.ffn_down.weight'][2:], ((128, 256), 0))
        config['quantization']['bits'] = 8
        with self.assertRaises(ValueError):
            tensor_layout(config)
        config['quantization']['bits'] = 4
        config['num_attention_heads'] = 0
        with self.assertRaises(ValueError):
            tensor_layout(config)
