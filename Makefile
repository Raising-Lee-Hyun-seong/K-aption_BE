.DEFAULT_GOAL := help

PYTHON ?= python3
MODEL_DIR ?= models/koreanlm-4bit
HOST ?= 0.0.0.0
PORT ?= 8000

export MODEL_DIR

.PHONY: help setup-api setup-mlx api prepare quantize model

help:
	@printf '%s\n' \
	  'make setup-api       FastAPI·STT·번역 실행 환경 설치' \
	  'make setup-mlx       MLX 가상환경 및 패키지 설치 (Apple Silicon)' \
	  'make api             FastAPI 서버 실행 (기본 포트: 8000)' \
	  'make model           원본 준비 + 4bit 변환 (최초 1회)' \
	  'make prepare         원본 다운로드 + FP16 중간 파일 생성' \
	  'make quantize        준비된 FP16 모델을 4bit로 변환'

setup-api:
	$(PYTHON) -m venv .venv-api
	.venv-api/bin/python -m pip install -r requirements-api.txt

setup-mlx:
	$(PYTHON) -m venv .venv-mlx
	.venv-mlx/bin/python -m pip install -r requirements-mlx.txt

api:
	@test -x .venv-api/bin/uvicorn || { echo '먼저 make setup-api를 실행하세요.'; exit 1; }
	KAPTION_MODEL_DIR="$(MODEL_DIR)" .venv-api/bin/uvicorn app.main:app \
	  --host "$(HOST)" --port "$(PORT)"

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
