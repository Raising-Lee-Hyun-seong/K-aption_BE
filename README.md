# K-aption BE

영어 영상·음성을 한국어 존댓말로 번역하고 SRT/VTT 자막을 만드는 CPU 전용 백엔드입니다.
Windows x86_64와 Apple Silicon Mac에서 **같은 명령·모델·API**를 사용합니다.

- 번역: **Qwen3-4B-Instruct-2507 · GGUF Q4_K_M · llama.cpp Server**
- 음성 인식: **faster-whisper Small · CPU INT8**
- FastAPI와 번역 서버는 별도 컨테이너이며 번역 서버는 Docker 내부에서만 접근합니다.
- Python·CUDA·API 키·Ollama·MLX를 따로 설치하지 않습니다.

## 1. 준비

Git과 Docker Desktop을 설치하고 Docker Desktop을 실행하세요.
Windows는 **WSL2 기반 Linux 컨테이너**, Mac은 **ARM64 네이티브 이미지**를 사용합니다.
기준 장비는 GPU 없는 Windows RAM 16GB입니다. 첫 설치에는 인터넷과 디스크 여유 10GB 이상을 권장합니다.

**컨테이너 한도는 backend 2.5GB + translator 4GB입니다. 호스트 Docker/WSL 메모리 7GB 이하 달성은 아직 검증 중입니다.**
Docker VM 메모리가 너무 작으면 처리 지연이나 OOM 종료가 생길 수 있습니다. Windows WSL2의 초기 시험 설정은 6GB이며 [메모리 설정·검증 범위](DEVELOPMENT.md#메모리-설정과-측정)를 확인하세요.

## 2. 처음 실행

PowerShell 또는 Mac 터미널에서 순서대로 실행합니다.

```bash
git clone https://github.com/Raising-Lee-Hyun-seong/K-aption_BE.git
cd K-aption_BE
docker compose up -d --build
docker compose logs -f backend translator model-init
```

처음에는 **약 2.50GB Q4_K_M 모델을 자동 다운로드하고 SHA256을 검사**합니다.
준비가 끝나면 backend 로그에 `Uvicorn running`이 나옵니다.
`Ctrl+C`로 로그 보기를 끝내도 서버는 계속 실행됩니다.
첫 음성 처리에서는 Whisper 모델도 다운로드합니다. 이후에는 볼륨 캐시를 재사용합니다.

브라우저에서 **http://localhost:8000/swagger** 를 엽니다.
상태 확인은 **http://localhost:8000/health** 입니다.

Docker에 **8 vCPU 이상**을 할당했다면 아래 명령으로 번역 CPU와 스레드를 8개로 실행할 수 있습니다.

```bash
docker compose --env-file docker/cpu8.defaults up -d --build
```

재시작·업데이트에도 같은 `--env-file docker/cpu8.defaults`를 사용하세요.
4 CPU로 돌아가려면 `--env-file docker/cpu.defaults`로 바꿉니다. 메모리·모델·STT 설정은 같습니다.
8 CPU의 실제 성능·번역 검증은 Mac ARM64 Docker 기준이며 Windows 실기기는 미검증입니다.
[수식 연산어 처리와 측정 결과](evaluation/math-operator-analysis-261025.md)를 참고하세요.

## 3. 영상 올리고 자막 받기

Swagger의 **Try it out → Execute**로 실행합니다.

1. `POST /api/v1/jobs`에서 영어 영상·음성 파일을 올립니다.
2. 응답의 `job_id`로 `GET /api/v1/jobs/{job_id}`를 조회합니다.
3. `status=completed`가 되면 아래 주소의 `작업_ID`를 바꿔 엽니다.

```text
http://localhost:8000/api/v1/jobs/작업_ID/subtitles?format=srt&language=ko
```

VTT는 `format=vtt`, 한영 자막은 `language=both`로 바꿉니다.
지원 형식은 MP4·MOV·MKV·WEBM·WAV·MP3·M4A이며 최대 1GiB·90분입니다.
번역 실패 구간은 영어로 남습니다. 숫자·수식·의미 오류가 있을 수 있어 결과 검토가 필요합니다.

`POST /api/v1/transcriptions`의 처리 시간은 `docker compose logs -f backend`에서 확인합니다.
요청이 끝나면 `event=http_request`·`path=/api/v1/transcriptions` 로그에 전체 소요 초(`elapsed_seconds`)와 분(`elapsed_minutes`)이 표시됩니다. STT뿐 아니라 번역·재시도까지 포함한 시간입니다. [단계별 시간 확인](DEVELOPMENT.md#요청-처리-시간-로그)도 참고하세요.

## 4. 종료·다시 실행·업데이트

```bash
# 종료: 모델·Whisper 캐시·작업 결과 유지
docker compose down

# 다시 실행
docker compose up -d

# 코드 업데이트 후 이미지 재빌드
git pull
docker compose up -d --build
```

`down -v`는 모델·캐시·작업 결과 볼륨까지 삭제합니다.
프로젝트 `.env`를 사용하지 않으려면 모든 명령에 `--env-file docker/cpu.defaults`를 붙일 수 있습니다.

## 문제가 생기면

| 증상 | 확인 |
| --- | --- |
| Docker 연결 실패 | Docker Desktop 실행 여부 확인 |
| 모델 준비 실패 | `docker compose logs model-init`에서 인터넷·공간·체크섬 오류 확인 |
| backend가 대기함 | `docker compose ps`와 `docker compose logs translator` 확인 |
| 느림·메모리 부족 | Docker VM 메모리·다른 작업·CPU 스레드 설정 확인 |
| 번역 서버 장애 | `/health`의 `translation_model_loaded` 및 translator 로그 확인 |

설정·개발 실행·측정 방법은 [DEVELOPMENT.md](DEVELOPMENT.md), 실제 검증과 한계는 [CPU 통합 기록](evaluation/cpu-unification-261021.md)에 있습니다.
