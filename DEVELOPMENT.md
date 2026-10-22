# K-aption BE 개발·운영 안내

## 실행 구조

모든 플랫폼은 Qwen3-4B-Instruct-2507 GGUF Q4_K_M을 llama.cpp Server CPU에서 실행합니다.
backend는 번역 가중치를 직접 로드하지 않습니다. Ollama·MLX·CUDA·Metal 런타임은 사용하지 않습니다.

```mermaid
flowchart LR
    Client[프론트엔드 / Swagger] --> Backend[FastAPI backend]
    Backend --> STT[faster-whisper Small CPU INT8]
    STT --> Text[문장 결합 / 문맥 / 용어집]
    Text --> HTTP[HTTP 번역 요청]
    HTTP --> Translator[llama.cpp Server CPU / Q4_K_M]
    Translator --> Check[출력 검증 / 재시도]
    Check --> Result[작업 상태 / SSE / SRT·VTT]
    Result --> Client
```

`model-init`은 최초 준비·캐시 SHA256 검증 후 종료합니다. backend는 translator health 통과 뒤 시작합니다.
모델·Whisper 캐시·작업 결과는 각각 `qwen-model`, `whisper-cache`, `jobs` 볼륨에 보존합니다.
translator는 internal 네트워크만 사용하고 호스트 포트를 공개하지 않습니다. backend만 localhost:8000을 공개합니다.
Dockerfile은 플랫폼을 강제하지 않습니다. 공식 CPU 이미지의 고정 OCI 인덱스에서 amd64·arm64 바이너리를 선택하고 backend Python 의존성도 해당 플랫폼에서 설치합니다.
x86 AVX2/F16C/FMA 빌드 옵션을 ARM64에 강제로 적용하지 않습니다.

## Windows와 Mac 실행

Docker Desktop을 켜고 프로젝트 폴더에서 아래 명령을 사용합니다. Windows는 WSL2 Linux 컨테이너를 선택합니다.

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f backend translator model-init
docker compose down
```

`docker compose up -d`는 이미지를 재빌드하지 않고 저장된 모델을 재사용합니다.
코드를 바꿨으면 `--build`를 붙입니다. 모델 검증만 다시 하려면 `docker compose run --rm model-init`을 실행합니다.
`make up`, `make down`, `make logs`, `make model`도 같은 작업을 실행합니다.
작업 도구와 CI에서는 비밀값 없는 `docker/cpu.defaults`를 `--env-file`로 지정해 프로젝트 `.env` 자동 읽기를 피합니다.

최초 모델 다운로드는 공개 HTTPS 배포만 사용하며 토큰·API 키가 필요하지 않습니다.
고정 명세는 `docker/qwen3-model.json`이고, 임시 파일 스트리밍·파일 크기·개별 및 전체 SHA256·배타 잠금·원자적 교체를 적용합니다.
기존 파일이 손상됐으면 자동 덮어쓰기하지 않고 실패합니다. `model-init` 로그를 확인해 손상 파일을 별도 백업한 뒤 다시 준비합니다.
전체 볼륨 삭제는 모델과 결과를 지우므로 복구의 기본 절차로 사용하지 않습니다.

## 모델과 CPU 설정

Q4_K_M 파일은 약 2.50GB이며 Q8_0 약 4.28GB보다 작습니다. Windows 16GB 장비에서 모델과 STT를 함께 운영할 메모리 여유를 확보하기 위해 변경했습니다.
원본 Qwen 모델 버전은 유지합니다. 양자화에 따른 번역 변화는 실제 비교 기록을 확인합니다.
모델 파일은 이미지에 포함하지 않습니다. 출처·라이선스는 `third_party/qwen3/NOTICE.md`와 `LICENSE`에 있고 model-init이 모델 볼륨에 복사합니다.
과거 Q8_0 명세는 비교용 `evaluation/qwen3-q8-model.json`, MLX 출처·은퇴한 테스트 입력은 `evaluation/archive/`에 보존합니다.

서버는 device none·GPU layers 0·KV/연산 오프로딩 비활성화·context 2048·parallel 1·cache RAM 0을 사용합니다.
메모리 매핑은 llama.cpp 기본 동작을 사용하고 메모리 잠금은 사용하지 않습니다.
`--no-repack`으로 CPU 가중치 재배치를 비활성화해 가중치를 회수 가능한 파일 매핑으로 유지합니다. 약 4.11GB ARM64 Docker VM의 5분 시험에서 기본 재배치는 OOM을 일으켰으며, 비활성화한 비교는 OOM·재시작 없이 끝났지만 번역 4구간이 실패했습니다. CPU 커널·파일 재접근에 따라 속도와 수치 결과가 달라질 수 있어 x86 성능 향상으로 해석하지 않습니다.
backend는 서버의 실제 채팅 템플릿과 토크나이저로 입력+출력 2048토큰 상한을 검사합니다.
참고 영어 문맥은 마지막 256자로 제한합니다. 생성은 temperature 0·top_k 1·top_p 1·repeat_penalty 1·seed 0입니다.

| 환경변수 | 기본값 | 의미 |
| --- | --- | --- |
| KAPTION_CPU_THREADS | 4 | translator 생성·프롬프트 평가 스레드 |
| KAPTION_TRANSLATOR_CPUS | 4 | translator CPU 코어 한도 |
| KAPTION_BACKEND_CPUS | 4 | backend CPU 코어 한도 |
| KAPTION_STT_THREADS | 4 | backend OpenMP 스레드 |
| KAPTION_TRANSLATOR_TIMEOUT | 120 | 번역 서버 HTTP 읽기·쓰기 제한 초, 연결은 최대 10초 |
| KAPTION_PORT | 8000 | localhost backend 공개 포트 |
| KAPTION_TRANSLATOR_URL | http://translator:8080 | backend 내부 서버 주소. Compose에서 고정 |
| KAPTION_MAX_ACTIVE_JOBS | 2 | 업로드·대기·실행을 합친 미완료 작업 한도 |
| KAPTION_JOB_TTL_SECONDS | 86400 | 완료 작업 보관 초 |

Compose 외부의 개발 backend는 기본 translator 주소가 http://localhost:8080입니다. CPU 서버가 먼저 실행돼 있어야 합니다.

```bash
make setup-api
make api
```

과거 `KAPTION_TRANSLATION_BACKEND`, `KAPTION_TRANSLATION_MODEL`, `KAPTION_TRANSLATION_DEVICE`, MLX 모델 경로·변환 명령은 사용하지 않습니다.
Qwen 모델·Q4 파일·CPU 실행은 기본 배포에서 고정되며 환경변수로 과거 엔진을 선택하지 않습니다.

## 문맥·검증·작업 처리

기본 물리 용어집 28개와 Qwen 추가 용어 8개를 프롬프트에 사용합니다. 기본 용어와 rocket·sled는 지정 번역 포함 여부도 검사합니다.
존댓말 지시·원문 문맥 분리·문장 및 수식 연결 결합·한글 형식·문맥 복사·반복·숫자·변수·영어 연산어 검증을 유지합니다.
검증 실패나 서버 장애는 최대 2회 시도하며 재시도에서는 이전 문맥을 제거합니다. 토큰 상한 종료·빈 응답·불완전 SSE 종료를 성공으로 처리하지 않습니다.
STT 추론과 번역은 공유 잠금으로 동시 실행을 막고 작업자는 하나입니다. 오디오 디코딩과 모델 준비는 추론 잠금 바깥에 있어 메모리 경쟁이 완전히 사라지는 것은 아닙니다.
Whisper는 지연 로드하고 재사용합니다. Qwen은 translator에서 한 번 로드돼 STT 중에도 메모리를 차지합니다.
긴 음성의 FFT 임시 배열이 backend 한도를 넘지 않도록 log-Mel 특징 계산을 최대 3000프레임씩 수행합니다. 입력 음성·VAD·전사 문맥은 나누지 않고, 패딩·프레임·Mel 필터와 전체 음성 최대값 정규화를 유지합니다. 기존 추출기와 수치 비교한 허용 오차는 `rtol=1e-5`, `atol=1e-6`입니다. 최종 특징과 디코딩 오디오는 길이에 비례하므로 모든 길이의 메모리가 고정되는 것은 아닙니다.
단일 backend 프로세스·Uvicorn worker 1을 유지해야 합니다. 같은 작업 저장소를 여러 worker가 공유하면 시작을 거부합니다.

## API 계약

Swagger는 `/swagger`, OpenAPI는 `/openapi.json`입니다.

| API | 용도 |
| --- | --- |
| GET /health | 원격 번역 준비·STT 로딩 여부 |
| GET /api/v1/capabilities | 확장자·업로드 크기·길이·언어 제한 |
| POST /api/v1/translations | 영어 원문·문맥 번역 |
| POST /api/v1/transcriptions | 동기 파일 처리 |
| POST /api/v1/jobs | 파일 작업 접수 |
| GET /api/v1/jobs/{job_id} | 진행 상태 |
| GET /api/v1/jobs/{job_id}/events | SSE 상태·heartbeat |
| POST /api/v1/jobs/{job_id}/cancel | 취소 요청 |
| GET /api/v1/jobs/{job_id}/result | 최종 구간 JSON |
| GET /api/v1/jobs/{job_id}/subtitles | SRT/VTT 다운로드 |

입력은 영어이며 MP4·MOV·MKV·WEBM·WAV·MP3·M4A, 최대 1GiB·90분입니다.
번역 검증 실패는 기존처럼 구간 `translation=null`, `translation_error=translation_failed`로 남고 다른 구간은 계속 처리합니다.
자막은 실패 구간에 영어를 사용합니다. 작업 취소는 진행 중 STT·현재 번역 호출이 끝나는 경계에서 처리합니다.
서버 health 실패는 `/health`의 준비 상태에 반영합니다. 컨테이너 재시작 시 진행 중 작업은 기존 복구 정책을 따릅니다.
용어·형식 검사 통과는 의미·전공 수식·맞춤법 정확도를 보장하지 않습니다.

## 메모리 설정과 측정

초기 제한은 backend 2,500,000,000 bytes + translator 4,000,000,000 bytes입니다. 모델 준비 서비스는 256MiB 제한으로 모델 로딩 전에 종료합니다.
컨테이너 한도 합계 6.5GB와 호스트 Docker 관련 RAM 7GB 목표를 구분합니다. 한도를 설정했다고 목표를 달성한 것은 아닙니다.
OOM·재시작·메모리 표본 누락을 정상 성공으로 표시하지 않습니다.

Windows 16GB 장비의 초기 시험 예로 `%UserProfile%\.wslconfig`에 다음을 설정할 수 있습니다.

```ini
[wsl2]
memory=6GB
processors=4
swap=0
```

설정은 모든 WSL2 배포판에 적용됩니다. 작업을 저장하고 Docker Desktop·WSL을 종료한 뒤 재시작해야 반영됩니다.
`wsl --shutdown`은 다른 WSL 배포판도 종료하므로 진행 작업을 먼저 정리합니다. [Microsoft 설정 문서](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
Mac은 Docker Desktop Resources의 VM 메모리를 시험 환경에 맞게 설정합니다. 기존 4.1GB VM 결과를 Windows 속도로 해석하지 않습니다.
위 VM 설정은 초기 시험값이며 호스트 7GB 달성을 보장하지 않습니다. 다른 컨테이너·WSL 배포판도 호스트 측정에 영향을 줍니다.

```bash
docker stats
docker compose exec translator curl --fail http://localhost:8080/health
docker compose logs translator model-init backend
```

`docker stats`의 working set·cgroup 사용량·개별 peak·동시 합계와 Docker Desktop/WSL2 호스트 사용량을 각각 기록합니다.
서로 다른 시점의 개별 peak를 더해 동시 사용량으로 보고하지 않습니다. 호스트는 Windows 작업 관리자에서 VmmemWSL과 Docker Desktop 관련 프로세스를 별도로 관찰합니다.
프로세스 working set 합계는 공유 메모리를 중복 계산할 수 있으므로 원시 값과 산정 방법을 함께 남깁니다.
호스트를 측정하지 않았으면 7GB 목표는 미검증입니다.

## 평가·성능 재현

기존 평가 JSON과 강의 자료는 보존하며 새 출력 경로를 사용합니다. 날짜는 실제 시험일을 명시합니다.

```bash
.venv-api/bin/python scripts/evaluate_api_translation.py --base-url http://127.0.0.1:8080 --manifest docker/qwen3-model.json --model models/qwen3-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --evaluated-on 2026-10-21 --output evaluation/new-q4-run.json
```

이 도구는 CPU 서버에 이미 로드된 모델 경로와 로컬 GGUF SHA256을 검사합니다. `--merge-math`는 운영 API와 같은 결합을 적용합니다.
위 명령은 별도로 실행한 로컬 CPU 비교 서버를 대상으로 합니다. 기본 Compose의 translator는 호스트에 공개하지 않으므로 localhost:8080으로 직접 접속할 수 없습니다. 운영 배포에 평가용 포트를 추가하지 않습니다.
`--holdout evaluation/holdout-linear-algebra-261021.json`은 기존 프롬프트 평가에 없던 강의 구간을 추가합니다. 기준 번역은 서버에 보내지 않습니다.
Q8 비교는 비교용 명세와 모델을 서버에 로드해 같은 입력·서버 바이너리·프롬프트·생성 설정으로 수행합니다. Q8은 기본 런타임에 로드하지 않습니다.
자동 검증 통과율에 의미 정확도 40%·전공 보존 30%·자연스러움 15%·누락 10%·속도 5%의 종합 점수를 임의 부여하지 않습니다. 수동 검토가 필요한 항목은 미확정으로 남깁니다.

Linux 또는 Mac의 실제 Docker 스택을 측정하려면 컨테이너 이름을 확인해 다음을 실행합니다.

```bash
.venv-api/bin/python scripts/benchmark_cpu.py --media sample-1min.mp4 --output evaluation/new-pipeline-run.json --container kaption_be-backend-1 --container kaption_be-translator-1 --backend-container kaption_be-backend-1
```

Docker Engine Unix 소켓에서 주기적으로 메모리·CPU 사용률·OOM·재시작 횟수와 해당 작업의 STT·번역 시간 로그를 수집합니다. 소켓 위치가 다르면 `--docker-socket`을 지정합니다.
파일 캐시·익명 메모리 bytes, 주요 페이지 폴트·파일 재접근·CPU throttling 누적 카운터도 저장합니다. 카운터 차이는 동일 컨테이너 수명 안에서 비교하며 재시작을 별도로 확인합니다.
Windows는 WSL 내부 Python에서 Docker Unix 소켓 접근이 가능한 환경으로 실행하거나 `docker stats`와 호스트 측정을 별도로 기록합니다.
1·5·30분의 실제 강의 파일로 시험하고 최초 모델 준비·STT·번역·전체 시간·메모리·OOM을 구분합니다. 반복 영상이나 합성 파일은 부하 시험임을 표시합니다.

Mac에서는 `ps -axo pid=,rss=,comm=`에서 `com.apple.Virtualization.VirtualMachine` PID를 찾고, 위 시험 명령에 `--macos-vm-pid PID`를 추가할 수 있습니다.
도구는 `lsof`에서 해당 VM이 Docker의 `Docker.raw`를 연 것을 확인한 뒤 Docker 프로세스와 VM의 현재 물리 footprint를 `footprint`로 수집합니다. 명령 인자·환경변수·파일 내용은 읽지 않습니다.
이 값은 RSS나 컨테이너 working set과 별도인 `host_macos`에 저장됩니다. 특권 helper처럼 조회할 수 없는 PID는 null·오류로 남기고 관측 합계에서 제외합니다. 누락·커널 귀속 한계·공유 메모리 중복·표본 사이 peak 때문에 7GB 목표를 자동 합격 처리하지 않습니다. Windows 측정도 대신하지 않습니다.

## 관련 테스트만 실행

```bash
.venv-api/bin/python scripts/run_changed_tests.py --files app/services/translation_http.py compose.yaml scripts/benchmark_cpu.py
```

전체 테스트 탐색은 하지 않습니다. 모델 실행 없는 CI 의존성은 `requirements-ci.txt`이며 GPU·torch를 설치하지 않습니다.
과거 모델 준비·MLX 생성 도구는 제거했고 순수한 과거 프롬프트·출력 진단 도우미만 남겼습니다.
실제 검증 완료·미검증 플랫폼과 측정 한계는 [CPU 통합 기록](evaluation/cpu-unification-261021.md)에 있습니다.
