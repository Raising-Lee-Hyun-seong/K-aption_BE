# Qwen3 translation model

Original model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

License: Apache-2.0. The upstream license is included unchanged as `LICENSE`.

Mac uses the MLX Community 4-bit conversion, revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. File hashes are pinned in
`data/qwen3_mlx.json`. K-aption has not modified these model files.

CPU uses the ggml-org GGUF Q8_0 conversion, revision
`e6f794d44f9395d0184a966c27b5ae99ea356fcb`. The URL, size and SHA256 are pinned
in `docker/qwen3-model.json`. K-aption has not modified this model file.

Both are conversions of the same Qwen3 Instruct 2507 model. Different
quantization and runtimes may produce different outputs. K-aption supplies
translation prompts and validation; validation does not prove semantic accuracy.
