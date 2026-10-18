# KoreanLM CPU model distribution

Original model: https://huggingface.co/quantumaikr/KoreanLM

Original model revision: `f4351abcdd6a933afbaffad0badf60c273e71920`

Upstream project: https://github.com/quantumaikr/KoreanLM

The upstream model card declares Apache License 2.0. Its original model card
and the upstream project's license are included unchanged in this directory.
License source: https://github.com/quantumaikr/KoreanLM/blob/30ee9ef4e5d2fdba1cc0daaffc196023c3ab87d5/LICENSE

Modifications by the K-aption project (2026-10-18): original weights were
converted to FP16 and then MLX affine 4-bit/group64. Those quantized weights
were repacked into GGUF Q4_1 without further quantization. Q/K tensor rows
were reordered for llama.cpp RoPE layout. Tokenizer BOS/EOS identifiers were
aligned with the original tokenizer. This is a format conversion, not new
training. The GGUF includes the tokenizer; separate tokenizer files are not
required for CPU inference.

GGUF SHA256: `f4e6b85689909fa69289dd67d4c6f67ecc95fb8400eb95b77ff9be57f0eed70e`

Distribution tag: `koreanlm-q4_1-v1`. Three binary parts must be concatenated
in manifest order and verified using `docker/koreanlm-model.json`.

This model retains the original model's limitations. Format verification
does not establish translation accuracy or suitability for a particular use.
