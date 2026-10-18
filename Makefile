.DEFAULT_GOAL := help

PYTHON ?= python3
KAPTION_TRANSLATION_MODEL ?= qwen3
ifeq ($(KAPTION_TRANSLATION_MODEL),koreanlm)
MODEL_DIR ?= models/koreanlm-4bit
else
MODEL_DIR ?= models/qwen3-4b-instruct-2507-4bit
endif
KOREANLM_DIR ?= models/koreanlm-4bit
HOST ?= 0.0.0.0
PORT ?= 8000

export MODEL_DIR KAPTION_TRANSLATION_MODEL

.PHONY: help setup-api setup-mlx api prepare quantize model

help:
	@printf '%s\n' \
	  'make setup-api       FastAPI·STT·번역 실행 환경 설치' \
	  'make setup-mlx       MLX 가상환경 및 패키지 설치 (Apple Silicon)' \
	  'make api             FastAPI 서버 실행 (기본 포트: 8000)' \
	  'make model           Qwen3 MLX 4bit 다운로드·검증 (최초 1회)' \
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
	@set -e; \
	  model_parent=$$(dirname "$(KOREANLM_DIR)"); \
	  mkdir -p "$$model_parent"; \
	  quantize_tmp=$$(mktemp -d "$$model_parent/.koreanlm-quantize.XXXXXX"); \
	  trap 'rm -rf "$$quantize_tmp"' EXIT; \
	  .venv-mlx/bin/mlx_lm.convert --hf-path models/koreanlm-fp16 \
	    --mlx-path "$$quantize_tmp/model" --quantize --q-bits 4 --q-group-size 64 --dtype float16; \
	  mkdir -p "$(KOREANLM_DIR)"; \
	  cp -R "$$quantize_tmp/model/." "$(KOREANLM_DIR)/"

# 기본 Mac API는 고정 Qwen3 MLX 4bit 파일을 준비한다.
model:
	@test "$(KAPTION_TRANSLATION_MODEL)" = qwen3 || { echo '기존 KoreanLM 준비는 make prepare 후 make quantize를 실행하세요.'; exit 1; }
	@test -x .venv-api/bin/python || { echo '먼저 make setup-api를 실행하세요.'; exit 1; }
	.venv-api/bin/python scripts/prepare_qwen.py --output "$(MODEL_DIR)"
