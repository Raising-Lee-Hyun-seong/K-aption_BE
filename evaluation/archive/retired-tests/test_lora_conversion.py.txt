"""모델 없이 LoRA 변환의 모양·키 완전성·체크섬·전치·배율 오류를 검증한다."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.numpy import load_file, save_file

from scripts.prepare_koreanlm_lora import adapter_signal, convert_tensors, verify_numerics, verify_sources


class LoraConversionTests(unittest.TestCase):
    """작은 비정방형 행렬과 서로 다른 Q/V 크기로 변환 오류가 거부되는지 검사한다."""
    def setUp(self):
        """GQA 구조의 작은 두 층과 서로 다른 값을 가진 LoRA 가중치를 구성한다.

        :return: None.
        """
        self.adapter = {'peft_type': 'LORA', 'r': 2, 'lora_alpha': 6,
                        'target_modules': ['q_proj', 'v_proj']}
        self.base = {'model_type': 'llama', 'hidden_size': 6, 'num_hidden_layers': 2,
                     'num_attention_heads': 2, 'num_key_value_heads': 1}
        generator = torch.Generator().manual_seed(11)
        self.state = {}
        for layer in range(2):
            for projection, size in (('q_proj', 6), ('v_proj', 3)):
                prefix = f'base_model.model.model.layers.{layer}.self_attn.{projection}'
                self.state[prefix + '.lora_A.weight'] = torch.randn((2, 6), generator=generator)
                self.state[prefix + '.lora_B.weight'] = torch.randn((size, 2), generator=generator)

    def test_saved_arrays_match_peft_with_non_square_tensors(self):
        """파일 저장·재로딩 후 모든 층의 PEFT 계산과 변환된 행렬의 계산이 일치한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        arrays, config = convert_tensors(self.state, self.adapter, self.base)
        self.assertEqual(arrays['model.layers.0.self_attn.q_proj.lora_a'].shape, (6, 2))
        self.assertEqual(arrays['model.layers.0.self_attn.v_proj.lora_b'].shape, (2, 3))
        self.assertEqual(config['lora_parameters']['scale'], 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapters.safetensors'
            save_file(arrays, str(path))
            result = verify_numerics(self.state, load_file(str(path)), 3, 3)
        self.assertEqual(result['projection_pairs_checked'], 4)

    def test_wrong_scale_and_projection_mapping_are_detected(self):
        """배율 착오와 다른 층 가중치의 잘못된 매핑은 수치 검증에서 실패한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        arrays, _ = convert_tensors(self.state, self.adapter, self.base)
        with self.assertRaises(AssertionError):
            verify_numerics(self.state, arrays, 3, 6)
        arrays['model.layers.0.self_attn.q_proj.lora_a'] = arrays['model.layers.1.self_attn.q_proj.lora_a']
        with self.assertRaises(AssertionError):
            verify_numerics(self.state, arrays, 3, 3)

    def test_missing_unknown_out_of_range_and_duplicate_keys(self):
        """누락·미지원·층 범위 초과·default 별칭 중복을 조용히 무시하지 않고 거부한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        key = next(iter(self.state))
        cases = []
        missing = dict(self.state)
        missing.pop(key)
        cases.append(missing)
        unknown = dict(self.state, unexpected=torch.zeros(1))
        cases.append(unknown)
        outside = dict(self.state)
        outside[key.replace('layers.0', 'layers.2')] = outside.pop(key)
        cases.append(outside)
        duplicate = dict(self.state)
        duplicate[key.replace('.weight', '.default.weight')] = duplicate[key]
        cases.append(duplicate)
        for state in cases:
            with self.subTest(keys=len(state)), self.assertRaises(ValueError):
                convert_tensors(state, self.adapter, self.base)

    def test_wrong_shape_nonfinite_and_dtype_are_rejected(self):
        """기반 모델과 다른 크기·손상 값·정수 가중치를 저장 전에 거부한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        key = next(iter(self.state))
        for tensor in (torch.zeros(6, 2), torch.full((2, 6), float('nan')),
                       torch.full((2, 6), float('inf')), torch.zeros(2, 6, dtype=torch.int64)):
            with self.subTest(dtype=tensor.dtype), self.assertRaises(ValueError):
                convert_tensors({**self.state, key: tensor}, self.adapter, self.base)

    def test_unsupported_peft_options_and_invalid_rank(self):
        """별도 변환이 필요한 옵션과 유효하지 않은 rank·alpha를 명시적으로 거부한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        for change in ({'bias': 'all'}, {'fan_in_fan_out': True}, {'use_rslora': True},
                       {'use_dora': True}, {'modules_to_save': ['lm_head']},
                       {'rank_pattern': {'q_proj': 1}}, {'alpha_pattern': {'q_proj': 1}},
                       {'r': 0}, {'r': True}, {'lora_alpha': float('nan')},
                       {'target_modules': ['q_proj']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                convert_tensors(self.state, {**self.adapter, **change}, self.base)

    def test_default_keys_work_without_modifying_input(self):
        """default가 들어간 PEFT 이름도 매핑하고 원래 설정과 텐서는 수정하지 않는다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        adapter_before = copy.deepcopy(self.adapter)
        state = {key.replace('.weight', '.default.weight'): tensor.clone()
                 for key, tensor in self.state.items()}
        arrays, _ = convert_tensors(state, self.adapter, self.base)
        self.assertEqual(len(arrays), 8)
        self.assertEqual(self.adapter, adapter_before)
        for key, tensor in self.state.items():
            torch.testing.assert_close(tensor, state[key.replace('.weight', '.default.weight')])

    def test_bad_source_checksum_is_rejected(self):
        """공식 체크섬과 다른 파일은 역직렬화 전에 거부한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'adapter_model.bin').write_bytes(b'wrong checkpoint')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                verify_sources(path)

    def test_zero_b_is_reported_as_no_effect_and_partial_zero_is_distinguished(self):
        """모든 B가 0인 무효 어댑터와 일부 B만 0인 어댑터를 구분한다.

        :return: None.
        :raises AssertionError: 검증 대상의 동작이 테스트 기대값과 다를 때.
        """
        self.assertFalse(adapter_signal(self.state)['all_b_zero'])
        zero = {key: (torch.zeros_like(tensor) if '.lora_B' in key else tensor)
                for key, tensor in self.state.items()}
        result = adapter_signal(zero)
        self.assertTrue(result['delta_identically_zero'])
        self.assertEqual(result['matrices']['B']['zero_tensor_count'], 4)
        self.assertEqual(result['matrices']['A']['zero_tensor_count'], 0)
        key = next(key for key in zero if '.lora_B' in key)
        zero[key] = self.state[key]
        self.assertFalse(adapter_signal(zero)['all_b_zero'])


if __name__ == '__main__':
    unittest.main()
