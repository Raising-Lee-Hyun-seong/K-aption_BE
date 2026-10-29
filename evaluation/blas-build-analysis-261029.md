# OpenBLAS 준비 단계 결과 (2026-10-29)

2단계: 동일 소스의 BLAS OFF/ON 이미지 준비 완료. 성능 비교는 시작하지 않았다.

## 결과

| 항목 | OFF | ON |
| --- | --- | --- |
| 이미지 | `kaption-translator-blas:off-339f3242c` | `kaption-translator-blas:on-339f3242c` |
| 이미지 ID 앞부분 | `3a0804a18f61` | `a781cdd2c752` |
| 네이티브 플랫폼 | Linux ARM64 | Linux ARM64 |
| `GGML_BLAS` | OFF | ON |
| 장치 목록 | 없음 | `BLAS: OpenBLAS` |
| BLAS 백엔드 연결 | 모듈 없음 | `libggml-blas.so` → `libopenblas.so.0` |

서버·벤치 실행 파일과 CPU 커널 등 공통 바이너리 16개의 SHA256이 모두 같다.
각 이미지의 버전, 장치 목록, CMake 설정, 컴파일러, 패키지 목록, 바이너리 해시,
BLAS 연결 확인 명령 7개씩 총 14개가 종료 코드 0으로 완료됐다.
단위 테스트나 전체 테스트 모음은 실행하지 않았다.

## 비교 조건

- 소스: 운영 엔진과 같은 `339f3242c761f16c27e5ffcde1d35dcc31933a55`.
- 소스 아카이브 SHA256 검증 및 Ubuntu 24.04 기반 이미지 digest 고정.
- GNU 14.2.0, CMake 3.28.3, pthread OpenBLAS 0.3.26.
- OFF 빌드를 재사용해 ON을 구성했다. 두 런타임은 같은 패키지 레이어를 공유한다.
- `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`.
- CUDA·Metal·Vulkan·OpenCL 비활성화. 포함된 연산 백엔드는 CPU와 ON의 BLAS다.
- 두 이미지 모두 API에 필요 없는 내장 웹 UI는 제외했다.
- 아카이브에는 Git 메타데이터가 없어 버전 출력은 build 0/commit unknown이다.
  실제 출처는 검증된 아카이브 SHA256과 이미지 revision 라벨로 기록했다.
- APT 저장소 자체를 시점 고정하지 않았으므로 캐시 없는 미래 재빌드는 패키지 버전이
  달라질 수 있다. 이번 비교에는 기록된 완성 이미지 ID를 사용한다.

## 확인하지 않은 항목

모델 로딩, 번역 API 응답, 실제 그래프의 BLAS 연산 배정, 처리 시간, 정확도,
추론 시 메모리 사용은 아직 확인하지 않았다. 장치 목록의 BLAS 표시는 CPU 수학
라이브러리 등록을 뜻하며 GPU 사용이나 성능 향상을 증명하지 않는다.
AMD64 빌드와 Windows 실행도 미검증이다.

다음 단계에서는 Qwen3 Q4_K_M, CPU·스레드 8개, 번역 컨테이너 4GB,
Docker VM 약 6GB, 기존 `--no-repack`·오프로딩 차단·문맥·배치 조건을 유지한다.
새 OFF와 ON의 대표 6문장 비교부터 시작하고 운영 공식 이미지와 새 OFF의 차이도
별도로 확인한다. 실제 BLAS 연산이 사용되는지 확인하기 전에는 시간 비교를 해석하지 않는다.

## 운영 상태와 자료

기존 Compose·모델·프롬프트·서버 이미지를 변경하지 않았다.
빌드 후 백엔드와 번역 컨테이너는 모두 healthy, OOM false, 재시작 0이었다.
기존 평가 데이터와 미커밋 변경도 유지했다. Git 커밋은 수행하지 않았다.

- [빌드 설정 및 원시 확인 결과](blas-build-inventory-261029.json)
- [재현 명령과 주의할 비교 범위](../docker/README.blas-eval.md)
- [비교용 Dockerfile](../docker/Dockerfile.translator-blas-eval)

전체 빌드 로그는 로컬 `/private/tmp/kaption-blas-off-build-261029.log`와
`/private/tmp/kaption-blas-on-build-261029.log`에 있으며 SHA256을 inventory에 기록했다.
