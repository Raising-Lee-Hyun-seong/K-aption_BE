# K-aption BE

영어 영상·음성을 한국어로 번역하고 SRT/VTT 자막을 만드는 백엔드입니다.
기본 번역 모델은 **Qwen3 4B Instruct 2507**입니다. Windows에서는 GPU 없이 Docker로 실행할 수 있습니다. 아래 명령은 **PowerShell** 기준입니다.

## 1. 준비

- Git과 Docker Desktop을 설치하고 **Docker Desktop을 실행**합니다.
- Docker는 **WSL2 기반 Linux 컨테이너**를 사용합니다.
- 대상 환경은 x86 Windows·RAM 16GB이며, Docker에 메모리 8GB 이상을 할당합니다. [메모리 설정 방법](DEVELOPMENT.md#windows-준비)
- 첫 설치에는 인터넷과 이미지·모델을 저장할 디스크 여유가 필요합니다.

**Python·CUDA·API 키는 따로 준비하지 않아도 됩니다. 모델도 자동으로 받습니다.**

## 2. 처음 실행

PowerShell에서 다음 명령을 순서대로 실행합니다.

```powershell
git clone https://github.com/Raising-Lee-Hyun-seong/K-aption_BE.git
cd K-aption_BE
docker compose --env-file docker/cpu.defaults -f compose.cpu.yaml up -d --build backend
docker compose --env-file docker/cpu.defaults -f compose.cpu.yaml logs -f backend
```

처음에는 이미지 빌드와 **약 4.28GB 모델 다운로드**로 시간이 걸립니다.
로그에 `Uvicorn running`이 나오면 준비 완료입니다. `Ctrl+C`로 로그 보기를 종료해도 서버는 계속 실행됩니다.

브라우저에서 **http://localhost:8000/swagger** 를 열어 사용합니다.
별도 영상 재생 화면은 포함하지 않으며, Swagger에서 파일 업로드와 API 실행을 할 수 있습니다.

## 3. 영상 올리고 자막 받기

Swagger에서 각 API의 **Try it out → Execute**로 실행합니다.

1. `POST /api/v1/jobs`에서 `language=en`으로 파일을 올립니다.
2. 응답의 `job_id`를 복사합니다.
3. `GET /api/v1/jobs/{job_id}`에 넣어 조회하고, `status`가 `completed`가 될 때까지 확인합니다.
4. 아래 주소의 `작업_ID`를 받은 값으로 바꿔 브라우저에서 열면 한국어 SRT가 다운로드됩니다.

```text
http://localhost:8000/api/v1/jobs/작업_ID/subtitles?format=srt&language=ko
```

- VTT가 필요하면 `format=vtt`, 한영 자막은 `language=both`로 바꿉니다.
- 입력은 영어 음성이며 MP4·MOV·MKV·WEBM·WAV·MP3·M4A를 지원합니다. 최대 1GiB·90분입니다.
- 첫 음성 처리에는 Whisper 모델도 자동 다운로드합니다. CPU에서는 처리 시간이 길 수 있으니 처음에는 짧은 파일로 확인하세요.
- 번역이 실패한 구간은 영어 원문으로 남습니다. 정상 응답도 오역이 있을 수 있으므로 중요한 수식·내용은 확인하세요. 결과는 기본 24시간 보관됩니다.

## 4. 종료·재실행·업데이트

아래 명령은 모두 `K-aption_BE` 폴더에서 실행합니다.

**종료**

```powershell
docker compose --env-file docker/cpu.defaults -f compose.cpu.yaml down
```

**다시 실행** — 저장된 모델을 재사용합니다.

```powershell
docker compose --env-file docker/cpu.defaults -f compose.cpu.yaml up -d backend
```

**최신 코드로 업데이트** — 기존 KoreanLM은 유지하고 새 Qwen 모델을 별도로 받습니다.

```powershell
git pull
docker compose --env-file docker/cpu.defaults -f compose.cpu.yaml up -d --build backend
```

일반 `down`은 모델과 결과를 유지합니다. **`down -v`는 모델·작업 결과·캐시까지 삭제**하므로 사용에 주의하세요.

## 실행이 안 될 때

| 증상 | 확인할 내용 |
| --- | --- |
| Docker 연결 오류 | Docker Desktop이 켜져 있는지 확인 |
| 다운로드 실패 | 인터넷·디스크 여유 확인 후 실행 명령 재시도 |
| 브라우저 접속 안 됨 | 로그에 `Uvicorn running`이 나왔는지 확인 |
| 메모리 부족·강제 종료 | Docker 메모리 설정 확인, 다른 무거운 프로그램 종료 |
| 모델 검증 실패 | [모델 복구 절차](DEVELOPMENT.md#최초-빌드와-실행) 참고 |

상태 확인 주소: http://localhost:8000/health

Mac에서는 `make model`로 Qwen 모델을 준비한 뒤 `make api`로 실행합니다. 기존 서버가 켜져 있다면 종료하고 다시 실행하세요.

Mac 실행·API 상세·환경 설정·성능 측정은 [개발 문서](DEVELOPMENT.md), 실제 검증 결과는 [CPU 검증 기록](evaluation/cpu-docker-smoke.md)에 있습니다.
Windows 실기기의 처리 속도와 긴 영상 성능은 아직 추가 검증이 필요합니다.
