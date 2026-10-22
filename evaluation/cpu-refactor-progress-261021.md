# CPU Docker 통합 중간 작업 기록

이 기록 이후의 정리·Q8/Q4 비교·amd64 검증과 최신 재개 지점은 [CPU 통합 검증 기록](cpu-unification-261021.md)에 있다.

작업일: 2026-10-21. **전체 리팩터링 완료 보고가 아니라 재개용 중간 기록이다.**
사용자는 문맥 여유가 부족해지면 진행 중인 단계까지만 마무리하고 중단하도록 요청했다.
현재 작업 트리는 미커밋이며 원격 push도 수행하지 않았다.

## 최신 요구사항과 범위

이전 Q8_0·llama-cpp-python 직접 실행 계획은 최신 요청으로 대체됐다.
Qwen3-4B-Instruct-2507은 유지하되 GGUF Q4_K_M·llama.cpp Server·HTTP 호출로 통합한다.
backend는 Whisper small CPU INT8·전처리·문맥·용어집·검증·재시도·작업 API·SSE·자막을 담당한다.
모델 서버는 내부 네트워크에서만 접근한다. Windows 16GB를 우선하며 Docker 관련 호스트 메모리 7GB 목표는 별도로 측정해야 한다.

분석 결과 기존 `translation.py`가 MLX 모델 로딩·생성·공통 검증을 함께 제공했고,
`translation_llamacpp.py`는 이를 상속했다. `translation_qwen.py`의 프롬프트·용어·존댓말 검증은 두 엔진이 공유했다.
기존 Compose는 linux/amd64 고정의 단일 컨테이너였다. STT와 번역은 각각 잠금만 있어 요청 간 동시 추론이 가능했다.

## 이번 단계에서 구현한 내용

- `translation.py`: MLX 로딩·생성 호출을 제거하고 공통 검증·재시도·로그를 전송 인터페이스와 분리했다.
- `translation_http.py`: 서버 `/apply-template`·`/tokenize`로 실제 토큰 상한을 확인하고 `/v1/chat/completions` SSE를 읽는다. 빈 응답·불완전 종료·HTTP 장애·시간 초과는 최대 2회 시도 안에서 처리한다. 조기 반복 감지 시 응답을 닫는다.
- `translation_qwen.py`: 로컬 MLX 서비스 클래스를 제거했다. 프롬프트·문맥 분리·존댓말·용어·숫자·수식 검증은 유지했다.
- `main.py`: OS·엔진 분기 대신 HTTP 서비스를 생성한다. STT CPU INT8을 강제하고 번역과 STT 추론 잠금을 공유한다. health는 원격 모델 상태를 조회한다.
- `compose.yaml`: 기본 공통 실행 파일. backend·translator를 분리하고 최초 1회 다운로드용 model-init 서비스를 추가했다. 기존 `compose.cpu.yaml`은 삭제했다.
- `Dockerfile.cpu`: Python backend와 다운로드 도구만 포함하며 MLX·torch·llama-cpp-python을 설치하지 않는다.
- `Dockerfile.translator`: 공식 CPU 이미지의 다중 아키텍처 OCI 인덱스를 digest로 고정했다. 직접 x86 빌드 플래그를 ARM에 전달하지 않는다.
- CPU 설정: device none, GPU layers 0, KV/연산 오프로딩 비활성화, 컨텍스트 2048, 슬롯 1, 생성 캐시 RAM 0, 스레드 환경변수 지원.
- 메모리 한도: backend 2,500,000,000 bytes + translator 4,000,000,000 bytes = 6.5GB. init은 모델 로딩 전에 완료하며 256MiB 제한이다. **한도 합계가 호스트 사용량 실측이나 7GB 달성 증명은 아니다.**
- Q4_K_M 공개 배포 고정 및 실제 다운로드·SHA256 검증 완료. 기존 Q8_0 명세는 `evaluation/qwen3-q8-model.json`에 복사 보존했다. 기존 모델·영상·평가 JSON은 삭제하지 않았다.
- HTTP 전송·생명주기 테스트를 추가하고 공통 검증 테스트에서 MLX 대역을 전송 대역으로 교체했다. 변경 테스트 선택기와 CI의 HTTP 의존성을 갱신했다.

모델 출처: [Unsloth 공개 GGUF 고정 리비전](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/a06e946bb6b655725eafa393f4a9745d460374c9).
파일 `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, 2,497,281,120 bytes,
SHA256 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`.

공식 CPU 이미지: `ghcr.io/ggml-org/llama.cpp:server@sha256:c7b03e45cff9c4e984acd39032a6f3a82c811075b72bf176e329f84b8598c135`.
Docker 인덱스에서 amd64·arm64 지원을 확인했고 ARM64 이미지 revision 라벨은 `339f3242c761f16c27e5ffcde1d35dcc31933a55`였다.
[공식 Docker 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md).

## 검증 현황

- **Windows 실기기: 미검증. amd64 빌드·실행: 이번 단계에서 미검증.**
- Mac Apple Silicon / Linux ARM64: 세 이미지 네이티브 빌드·Q4 모델 로딩·translator health·backend 시작 확인.
- 번역 API: HTTP 200, `The velocity is constant.` → `속도는 상수입니다.`. 입력 210토큰·생성 8토큰, 서비스 7.55초·HTTP 7.57초. 단일 예비 측정이다.
- 실제 backend 이미지에서 mlx·mlx_lm·llama_cpp·torch 패키지가 없음을 확인했다.
- translator 라이브러리 목록은 ARM CPU 변형들만 제공하며 CUDA·Metal·Vulkan 라이브러리가 없었다. 실행 로그도 GPU 미지원 빌드임을 표시했다.
- 테스트 VM은 ARM64·8 vCPU·총 메모리 4,106,604,544 bytes였다. Windows 16GB 환경을 대체하지 않는다.
- Docker stats 관측값: 모델 로딩 후 translator 2.948GiB·backend 81.91MiB. STT 로딩 후 한 시점에 translator 2.698GiB·backend 868.9MiB. **최대치나 호스트 전체 메모리가 아니다.**
- 60초 원본 음성으로 작업 API·SSE·STT→번역→SRT/VTT 실제 smoke를 진행했다. 최종 결과는 별도 JSON으로 저장한다. 측정 중 docker stats 명령이 15초를 초과해 샘플러가 종료됐으므로 메모리 표본에는 누락이 있다.

### 이번 단계 최종 결과

관련 테스트 156개를 통과했다. 전체 테스트를 탐색한 것이 아니라 HTTP 서비스·API 계약·작업·SSE·취소·자막·STT 입력·수식 및 번역 검증·모델 다운로드·변경 테스트 선택기에 한정했다. 이전 MLX 생성 대역은 공통 전송 대역으로 바꿨으며, 제거된 로컬 MLX 로더 전용 테스트 3개와 Qwen 로컬 엔진 테스트 4개는 같은 기능의 새 HTTP 검사로 대체했다.

[실제 ARM64 smoke JSON](cpu-split-arm64-smoke-261021.json): 60초 음성의 STT 11조각→4문장, 작업 completed·번역 반환 4/4·SSE status 이벤트 6회·SRT/VTT 모두 HTTP 200과 다운로드 헤더 확인. backend·translator의 OOMKilled는 모두 false였다. 번역문에는 원래 STT의 `rocket's lead` 오류와 불완전한 미분 관계가 남아 있어 의미 정확성을 주장하지 않는다.

| 구간 | 실측 초 |
| --- | ---: |
| Whisper 최초 모델 로딩 | 12.55 |
| STT 추론 | 8.93 |
| STT 서비스 전체 | 21.61 |
| 번역 4문장·재시도 포함 | 349.90 |
| 작업 처리 전체 | 371.53 |
| HTTP 접수·SSE·결과 조회 포함 smoke | 372.38 |

컨테이너 시작 이후 cgroup memory.peak는 backend 990,887,936 bytes, translator 3,792,146,432 bytes였다. **개별 최대값은 발생 시점이 달라 합산한 값을 동시 사용량으로 해석하지 않는다.** docker stats 샘플 30개 이후 수집이 중단됐으며, 호스트 오버헤드까지 포함한 실제 최대 RAM 7GB 목표는 미검증이다. VM 메모리 약 4.11GB에서 수행한 이 예비 측정은 매우 느렸고, 메모리 압박·서버 CPU 설정·Windows 실기기 성능을 후속 단계에서 조사해야 한다.

최종 backend 이미지 내부에서도 HTTP·수식 결합·STT 힌트 테스트 20개를 통과했다. 이미지 안의 main·공통 번역·Qwen 지시·HTTP 전송 파일 SHA256이 작업 트리와 일치하는지 확인했다. model-init 재실행 시 재다운로드 없이 캐시 SHA256 검증을 통과했다. 테스트 컨테이너·네트워크는 `down`으로 정리했고 모델·Whisper·작업 결과 볼륨과 호스트 모델 파일은 보존했다.

**중단 지점:** HTTP 분리와 ARM64 CPU 기본 실행 검증까지 마쳤다. 다음은 동일 서버 조건의 양자화 품질 비교와 남은 MLX/KoreanLM 도구·테스트 정리다. README·DEVELOPMENT·Makefile 및 Windows·7GB·장시간 검증은 아직 완료되지 않았다.

## 남은 단계

1. ARM64 smoke 결과를 정리하고 새 코드를 포함한 마지막 이미지 상태를 확인한다.
2. Q8_0과 Q4_K_M을 **같은 서버 버전·입력·프롬프트·생성 설정**으로 비교한다. 기존 저장 36구간과 별도 비학습 강의 구간을 사용한다. 과거 MLX·Python 엔진 결과만 비교해 양자화 영향을 단정하지 않는다.
3. 사용하지 않는 `translation_llamacpp.py`, MLX 준비·변환 스크립트, requirements-mlx, KoreanLM 호환 설정·테스트·Makefile을 호출 관계에 맞춰 제거한다. 남아 있는 과거 평가 실행 스크립트의 MLX 분기도 정리하되 결과·테스트 데이터는 보존한다.
4. `scripts/evaluate_api_translation.py`와 기존 benchmark 도구를 HTTP 서버 기준으로 교체한다. 현재 도구 일부는 예전 factory 인자와 로컬 모델을 참조하므로 아직 실행하면 안 된다.
5. 옛 로컬 엔진 전용 `tests/test_cpu_runtime.py`, `tests/test_llamacpp_translation.py`와 변환 관련 테스트는 아직 새 구성에 맞춰 정리하지 않았다. 이를 포함한 전체 테스트 통과를 주장하지 않는다.
6. amd64 CPU 빌드·실행, Mac ARM64 최종 회귀, 재시작·볼륨 재사용·장애·취소를 검증한다.
7. 1·5·30분 처리 성능과 메모리 증가·OOM을 측정하고, Windows Docker Desktop/WSL2의 실제 호스트 메모리를 별도로 기록한다. 현재 호스트 오버헤드와 7GB 목표는 미검증이다.
8. README·DEVELOPMENT·Makefile·환경변수 안내·아키텍처·모델 라이선스 고지를 갱신한다. **기존 README와 Makefile은 아직 이전 실행 환경 설명이므로 현재 변경의 안내로 사용하지 않는다.**
9. 관련 검사와 문서 검토 후에만 최종 결과를 보고한다. 커밋은 범위·전체 메시지를 제시하고 별도 승인받는다.

## 재개와 실행

실행 공통 명령은 `docker compose up -d --build`다. 작업 도구에서는 프로젝트 .env를 읽지 않도록 비밀값 없는 `--env-file docker/cpu.defaults`를 명시했다.
테스트는 사용자 기본 스택과 구분한 프로젝트명 `kaption-cpu-refactor`·포트 18002를 사용했다.

```bash
KAPTION_PORT=18002 docker compose --env-file docker/cpu.defaults -p kaption-cpu-refactor up -d --build
docker compose --env-file docker/cpu.defaults -p kaption-cpu-refactor logs -f backend translator
docker compose --env-file docker/cpu.defaults -p kaption-cpu-refactor down
```

`down -v`는 캐시·결과를 삭제하므로 사용하지 않는다. 테스트 종료 후 컨테이너는 중단하고 볼륨은 보존한다.
