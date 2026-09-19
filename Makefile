.DEFAULT_GOAL := help

PYTHON ?= python3
VIDEO ?= MIT8_01F16_W06PS01-2_360p.mp4
OUTPUT ?= outputs/sample.en.txt
STT_MODEL ?= small
DEVICE ?= cpu
LANGUAGE ?= en
TEXT ?= The velocity is constant.
CONTEXT ?=
MAX_TOKENS ?= 128
MODEL_DIR ?= models/koreanlm-4bit

# Pass user text as environment data, not interpolated shell code.
export VIDEO OUTPUT STT_MODEL DEVICE LANGUAGE TEXT CONTEXT MAX_TOKENS MODEL_DIR

.PHONY: help setup-stt setup-mlx stt prepare quantize model translate

help:
	@printf '%s\n' \
	  'make setup-stt       STT 가상환경 및 패키지 설치' \
	  'make setup-mlx       MLX 가상환경 및 패키지 설치 (Apple Silicon)' \
	  'make stt             샘플 영상의 영어 STT 실행' \
	  'make model           원본 준비 + 4bit 변환 (최초 1회)' \
	  'make prepare         원본 다운로드 + FP16 중간 파일 생성' \
	  'make quantize        준비된 FP16 모델을 4bit로 변환' \
	  'make translate       기본 예시 문장 번역' \
	  'make translate TEXT="English sentence"' \
	  'make stt VIDEO="lecture.mp4" OUTPUT="outputs/lecture.txt"'

setup-stt:
	$(PYTHON) -m venv .venv-stt
	.venv-stt/bin/python -m pip install -r requirements-stt.txt

setup-mlx:
	$(PYTHON) -m venv .venv-mlx
	.venv-mlx/bin/python -m pip install -r requirements-mlx.txt

stt:
	@test -x .venv-stt/bin/python || { echo '먼저 make setup-stt를 실행하세요.'; exit 1; }
	.venv-stt/bin/python transcribe.py "$$VIDEO" --language "$$LANGUAGE" \
	  --model "$$STT_MODEL" --device "$$DEVICE" --timestamps --output "$$OUTPUT"

prepare:
	@test -x .venv-mlx/bin/python || { echo '먼저 make setup-mlx를 실행하세요.'; exit 1; }
	.venv-mlx/bin/python scripts/prepare_koreanlm.py

quantize:
	@test -x .venv-mlx/bin/mlx_lm.convert || { echo '먼저 make setup-mlx를 실행하세요.'; exit 1; }
	.venv-mlx/bin/mlx_lm.convert --hf-path models/koreanlm-fp16 \
	  --mlx-path "$$MODEL_DIR" --quantize --q-bits 4 --q-group-size 64 --dtype float16

# Recursive calls keep preparation before quantization even with make -j.
model:
	$(MAKE) prepare
	$(MAKE) quantize

translate:
	@test -x .venv-mlx/bin/python || { echo '먼저 make setup-mlx를 실행하세요.'; exit 1; }
	.venv-mlx/bin/python translate_mlx.py "$$TEXT" --context "$$CONTEXT" \
	  --model "$$MODEL_DIR" --max-tokens "$$MAX_TOKENS"
