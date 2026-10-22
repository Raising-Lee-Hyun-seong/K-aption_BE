.DEFAULT_GOAL := help
PYTHON ?= python3
COMPOSE = docker compose --env-file docker/cpu.defaults
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help setup-api api up down logs model
help:
	@printf '%s\n' 'make up        CPU backend·translator 시작' 'make down      종료 (모델·작업 볼륨 유지)' 'make logs      서비스 로그' 'make model     고정 Q4_K_M 다운로드·검증' 'make setup-api 개발용 Python 환경 설치' 'make api       이미 실행 중인 translator에 개발 backend 연결'
setup-api:
	$(PYTHON) -m venv .venv-api
	.venv-api/bin/python -m pip install -r requirements-api.txt
api:
	.venv-api/bin/uvicorn app.main:app --host "$(HOST)" --port "$(PORT)"
up:
	$(COMPOSE) up -d --build
down:
	$(COMPOSE) down
logs:
	$(COMPOSE) logs -f backend translator model-init
model:
	$(COMPOSE) run --rm --build model-init
