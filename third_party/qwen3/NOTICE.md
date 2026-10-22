# Qwen3 translation model

Original model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

License: Apache-2.0. The upstream license is included unchanged as `LICENSE`.

All platforms use the Unsloth GGUF Q4_K_M conversion, revision
`a06e946bb6b655725eafa393f4a9745d460374c9`. URL, size and SHA256 are pinned
in `docker/qwen3-model.json`. K-aption has not modified the weights.
Conversion source: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF

The earlier ggml-org Q8_0 manifest is retained in `evaluation/qwen3-q8-model.json`
for comparison only. Previous MLX source metadata and historical test fixtures
are retained under `evaluation/archive/`; they are not runtime dependencies.

K-aption supplies translation prompts and validation. Passing validation does
not prove semantic accuracy. Quantization and CPU kernels can affect outputs.
