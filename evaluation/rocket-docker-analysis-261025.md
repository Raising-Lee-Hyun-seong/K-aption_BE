# 프로젝트 로켓 영상 Docker 전체 재시험

2026-10-25. `MIT8_01F16_W06PS01-2_360p.mp4` 전체 **212.586667초(3분 32.59초), 21,729,088 bytes**를 현재 백엔드 이미지로 처리했다. 앞선 네이티브 시험과 영상 SHA256·프로젝트 실행 소스 해시가 일치한다.

## 실제 결과

| 항목 | 시간·결과 |
| --- | ---: |
| 업로드부터 완료 상태 조회까지 | **119.434초, 약 1분 59초** |
| 서버 파이프라인 | 118.764초 |
| STT 전체 | **28.550초** |
| STT 모델 로딩 / 추론 | 0.909 / 27.341초 |
| 번역 전체 | **90.209초** |
| 번역 서버 입력 처리 / 출력 생성 합계 | 64.143 / 25.922초 |
| 재시도 2회 추가 시간 | 10.213초 |
| 원시 STT / 번역 입력 | 39 / 17구간 |
| 번역 반환 / 실패 | **16 / 1구간** |
| 번역 시도 / 입력 토큰 / 출력 토큰 | 19 / 6,679 / 637 |
| 두 컨테이너 working set 합계의 표본 최대값 | **4,339,830,784 bytes, 약 4.34GB** |
| OOM / 시험 중 컨테이너 재시작 / 통계 수집 오류 | 없음 / 없음 / 없음 |

재시도 시간은 번역 시간에 포함되어 있으며 더해 합산하지 않는다. 번역 서버 입력·출력 시간은 시험 시작부터 작업 완료까지의 translator 로그에서 수집한 19개 task의 합이다. API 생성 시도 19회와 개수가 일치했다. 서버 입력 처리가 여전히 전체의 약 54%로 가장 큰 부분이다.

작업은 `completed`지만 번역 1개가 실패했으므로 **`pipeline_succeeded=false`를 보존**했다. SSE·결과 조회·SRT/VTT HTTP 200과 다운로드 헤더를 확인하고 실제 자막도 저장했다. 16개 반환을 의미 정확도 94.1%로 해석하지 않는다.

## 환경과 준비

- 멈춰 있던 Docker Desktop을 종료·재시작해 엔진을 복구했다. 설정은 **6144MiB**, Engine 실제 메모리 **6,210,043,904 bytes**, VM vCPU 8, Linux ARM64다. Windows amd64 검증이 아니다.
- 진행 중 작업이 없는 것을 확인한 뒤 `docker compose --env-file docker/cpu.defaults build backend`와 `up -d --no-deps backend`로 **현재 코드의 backend만 갱신**했다. 기존 translator와 모델·Whisper·작업 볼륨은 유지했다. 비밀 파일은 읽지 않았다.
- Qwen3-4B-Instruct-2507 Q4_K_M GGUF의 컨테이너 내 SHA256을 고정 명세와 대조했다. 기존 CPU 전용 translator는 device none·GPU layers 0·KV/연산 오프로딩 OFF·no-repack·context 2048·parallel 1·threads/threads-batch 4·cache RAM 0을 유지했다. backend 프롬프트 캐시는 true다.
- backend CPU 제한 4개·메모리 2,500,000,000 bytes, translator CPU 제한 4개·메모리 4,000,000,000 bytes다. 추가 컨테이너 swap 한도는 없다.
- 시작 health는 번역 모델 준비 완료·STT 미로드였다. 번역 서버 시작·모델 검증·이미지 빌드 시간은 처리 시간에서 제외하고 최초 STT 로딩은 포함했다. Docker backend에 관측용 래퍼를 삽입하지 않았다.
- 단일 영상 1회 실행이다. 메모리는 표본상 컨테이너 working set 합계로 VM·호스트 오버헤드와 짧은 피크는 포함하지 않는다. 호스트 RAM 16GB 전체 예산 충족 여부는 별도로 검증하지 않았다.

실제 명령:

```bash
.venv-api/bin/python scripts/benchmark_cpu.py \
  --base-url http://127.0.0.1:8000 \
  --media MIT8_01F16_W06PS01-2_360p.mp4 \
  --output evaluation/rocket-docker-full-261025.json \
  --container k-aption_be-backend-1 \
  --container k-aption_be-translator-1 \
  --backend-container k-aption_be-backend-1 \
  --deadline-seconds 600
```

실제 작업 API·SSE·자막을 실행한 시간이다. 동기 `/api/v1/transcriptions`와 공통 파이프라인을 쓰지만 해당 동기 엔드포인트 자체의 직접 측정은 아니다.

## 앞선 네이티브 결과와의 관계

| 항목 | Mac 네이티브 CPU | Mac ARM64 Docker 6GB |
| --- | ---: | ---: |
| 전체 | 120.406초 | 119.434초 |
| STT | 21.645초 | 28.550초 |
| 번역 | 98.318초 | 90.209초 |
| 원시 STT / 번역 입력 | 45 / 26 | 39 / 17 |
| 번역 반환 / 실패 | 23 / 3 | 16 / 1 |

이번 관측에서 전체 시간은 비슷했다. **코드·영상은 같아도 STT 출력과 묶인 입력이 달라 직접적인 Docker 오버헤드·속도 개선·정확도 개선 비교가 아니다.** 네이티브는 macOS Python 3.9·CTranslate2 4.8.2, Docker는 Linux Python 3.11·CTranslate2 4.8.3이며 CPU 제한·관측 방식도 다르다. 전체 환경과 패키지 버전을 보존했지만 어느 차이가 결과 변화를 만들었는지 분리 시험하지 않았다.

## 남은 품질 문제

- **실패: 인덱스 10, 116.48초의 `mr prime equals m of vr`.** 첫 시도는 `unexpected_script`, 재시도는 `missing_math_function`으로 거부됐다. 이전의 전 구간 `translator_unavailable` 장애는 이번 시험에서 발생하지 않았다.
- 이번 STT는 `rocket sled`, `to vr equal vr of t`처럼 네이티브와 다른 원문을 만들었다. 별도 물리 힌트를 켠 결과가 아니라 Docker의 기존 기본 STT 실행 결과다.
- 인덱스 13의 `v0 + u ln(mR / 2m0)`, 마지막 구간의 질량 감소·0보다 작음·초기 속도에서 뺌·썰매 정지 관계는 출력에 남았다.
- 인덱스 3은 반환됐지만 `dvr = u가 상수`라는 불완전한 관계가 남는다. 인덱스 12의 `u1` 등 STT 수식 불확실성도 해결된 것이 아니다. 모든 수식의 의미 정확성이 검증된 결과가 아니다.
- 일부 수식·설명은 최대 약 33초 구간으로 합쳐졌다. 문맥 보존에 도움이 될 수 있지만 실제 자막 가독성·분할 정책은 별도 평가가 필요하다.

입력 처리 비용 절감, 수식 구조 보존·실패 사유별 재시도 개선이라는 [앞선 분석](rocket-native-analysis-261025.md)의 우선순위는 유지된다. 네이티브에서 측정한 beam 1·물리 힌트·기각한 프롬프트 후보를 Docker에 적용하지 않았다.

Docker 메모리 6GB 설정과 현재 코드의 두 서비스를 실행 상태로 남겼다. 운영 코드·기본값 추가 수정·커밋·push는 하지 않았다. 전체 단위 테스트 모음을 실행하지 않고 요청한 전체 영상 재시험만 수행했다.

## 보존한 자료

- [전체 API·SSE·시간·Docker 메모리 표본](rocket-docker-full-261025.json)
- [이미지·CPU/메모리 한도·영상/모델/소스 해시·패키지 버전](rocket-docker-environment-261025.json)
- [단계별 집계·실패 사유·서버 입력/생성 원시 시간 로그](rocket-docker-analysis-261025.json)
- 실제 생성한 [SRT](rocket-docker-full-261025.srt) / [VTT](rocket-docker-full-261025.vtt)

영상 출처는 MIT OpenCourseWare의 [Rocket Sled 강의](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/resources/rocket-sled-integrate-the-rocket-equation/)와 [보존된 출처·불확실성](mit_rocket_reference.json)을 참조한다. 원본·파생 자막은 해당 출처의 CC BY-NC-SA 4.0 조건을 따른다.
