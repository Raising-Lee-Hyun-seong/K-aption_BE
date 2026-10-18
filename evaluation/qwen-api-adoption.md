# Qwen3 기본 API 적용 검증

검증일: 2026-10-18. 기존 KoreanLM의 심한 오역을 개선하기 위해 기본 번역 모델을 Qwen3-4B-Instruct-2507로 바꿨다. Mac은 기존 비교에서 검증한 MLX 4bit, Windows/Linux CPU는 ggml-org GGUF Q8_0을 사용한다. 서로 다른 양자화·엔진이므로 출력이 같다고 보장하지 않는다. OpenAI API는 사용하지 않는다.

## 실제 서비스에서 바뀐 동작

- `make model`이 고정 Qwen MLX 파일을 다운로드·검증하고 `make api`는 기본 Qwen 서비스를 선택한다. 기존 KoreanLM은 `make api KAPTION_TRANSLATION_MODEL=koreanlm`으로 실행한다. 기존 가중치는 삭제하지 않는다.
- Docker 첫 실행은 고정 공개 Qwen GGUF를 받는다. 파일명·크기·SHA256을 바꿔 기존 KoreanLM과 같은 볼륨에서 별도로 보관한다. API 키와 로그인은 필요 없다.
- Qwen의 공식 비추론 채팅 형식을 사용한다. 현재 원문은 사용자 메시지, 이전 영어 원문 마지막 256자는 참고 시스템 메시지로 분리한다. 재시도에서는 참고 문맥을 제거한다.
- 기존 28개 용어 검증과 추가 번역 지시를 공유한다. 로켓·썰매가 다른 대상으로 바뀌면 필수 용어 누락으로 재시도한다. 수학 구간의 영어 over·plus·minus·equals가 그대로 남으면 재시도한다. 기호 자체와 한글 연산 설명은 허용한다.
- 기존 2,048토큰 합계 제한·기본 128토큰·재시도 1회·부분 실패 API 계약은 유지한다. 실패 시 원문을 보존한다. 실행 중이던 서버에는 재시작 후 반영된다.

## 사용자 강의 36구간 재생

사용자가 보낸 사례와 같은 저장 STT를 그대로 사용하고 실제 API 팩토리의 서비스를 호출했다. 수식·영어 원문·문장 경계는 고치지 않았다. 원시 생성 시도와 최종 출력을 보존했다.

| 항목 | 기존 KoreanLM 기록 | Qwen Mac MLX 최종 | Qwen Mac CPU 최종 |
| --- | ---: | ---: | ---: |
| 반환 번역 | 22/36 | 35/36 | 35/36 |
| 번역 미제공 | 14 | 1 | 1 |
| 최종 조건 번역 시간 | 과거 기록, 동일 실행 아님 | 52.81초 | 155.70초 |
| 생성 시도 | 과거 기록 | 39 | 38 |

시간은 모델 로딩·STT·HTTP 업로드를 제외한 번역 호출 합계다. CPU는 Apple Silicon 네이티브 CPU 4스레드이며 Windows 성능이 아니다. 조건별 1회 측정이고 서로 속도 개선율로 비교하지 않는다. **반환 번역 수는 의미 정확도가 아니다.**

| 영어 원문 | 기존 문제 | 실제 Qwen 결과 예시 |
| --- | --- | --- |
| separation variables | 배추 속도·분리 변수 기술 | 변수 분리 기법 |
| shuffle the m onto the other side | m을 순회 | m을 다른 쪽으로 옮김 |
| dmR over mR | dmR 위에 mR | dmR / mR |
| initial mass | 질문으로 변형 | 초기 질량이라는 진술 유지 |
| velocity … sled … stop | 추적 계수로 변형 | 속도 감소와 썰매 정지 유지 |

남은 문제도 있다. Mac MLX의 stt-9는 수학 인자의 위치를 사람처럼 옮긴다. MLX stt-25는 첫 시도가 수식만 나오고 재시도에 영어 연산어가 남아 번역을 제공하지 않는다. CPU에서는 stt-25가 번역되지만 stt-23의 두 시도에 외국 문자가 섞여 거부된다. CPU stt-12의 '통합'은 앞 적분 문맥을 제대로 반영하지 못한 표현이다. `rocket's lead`, `2/to`, 끊긴 등식·함수 인자 등의 STT 불확실성도 남아 있으며 수식 묶음·괄호 범위를 자동으로 복원하지 않았다. 수락된 출력에도 어색한 표현과 의미 오류가 있어 사람 검토가 필요하다.

이 강의는 기존 개발에 사용한 자료다. 새로운 강의에 대한 정확도·전문가 검수 통과·완성된 번역 품질을 주장하지 않는다.

최종 실행: [Mac MLX](qwen-api-mlx-v3-261018.json), [Mac CPU](qwen-api-cpu-v3-261018.json). 개선 전·중간 후보도 별도 파일에 보존했다. 실행 자료의 `semantic_accuracy`는 수동 판정 점수로 오해하지 않도록 null이다.

## 재현

새 출력 경로를 지정하며 기존 보고서는 덮어쓰지 않는다. `make model`로 준비한 Mac에서는 다음과 같다.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_api_translation.py --backend mlx --model models/qwen3-4b-instruct-2507-4bit --evaluated-on 2026-10-18 --output evaluation/qwen-api-new-run.json
```

평가 도구는 실제 API와 같은 팩토리를 사용하지만 저장 STT 재생이며 실제 음성 인식·HTTP 전체 평가와 구분한다.

## 변경 범위 검사

Qwen 기본 선택·문맥 분리·용어 재시도·CPU ChatML·다운로드와 연결된 기존 검증에 한정해 97개 테스트와 모델 시작 실패 시 저장소 잠금 해제 테스트 1개를 통과했다. CPU Docker 이미지에서는 Qwen 연결 테스트 8개를 별도로 통과했다. 전체 테스트 모음은 실행하지 않았다.


## 실제 HTTP·Linux CPU 확인

Mac에서 사용자 서버와 다른 18001 포트·별도 작업 디렉터리로 `make api`를 실행했다. 실제 HTTP 번역 3건(변수 분리·dmR/mR·썰매 정지)은 모두 200으로 정상 번역했다. 생성 대역은 사용하지 않았다.

기존 1.60초 합성 영어 WAV를 실제 작업 API에 올려 202 접수→completed→결과 조회 및 한영 SRT/VTT 200을 확인했다. 작업 경과 시간은 단일 실행 약 3.00초였고, STT `The velocity is constant.`와 번역 `속도는 상수이다.`를 반환했다. 이 한 문장은 물리학 번역 품질의 독립 평가가 아니다. 임시 API 서버는 정상 종료했고 사용자 8000 포트는 건드리지 않았다.

최종 Linux x86 CPU 이미지에서는 이미 검증·다운로드한 GGUF를 읽기 전용으로 연결해 실제 Qwen 번역을 실행했다. `And then we have dmR over mR.` → `그리고 그 후에 우리는 dmR / mR을 가진다.`를 반환했다. 번역 115.45초·로딩 포함 118.38초·프로세스 최고 RSS 3,666.03MiB였다. Apple Silicon에서 x86을 에뮬레이션하고 Docker VM 메모리가 약 3.83GiB로 제한된 조건이므로, 이 시간을 Windows CPU 속도로 해석하지 않는다. 정상 종료와 네이티브 모델 해제도 확인했다. CPU 컨테이너 안에서는 실제 모델 추론, Mac에서 실행한 공통 다운로더에서는 공개 URL 전체 다운로드와 SHA256 검증을 확인했다. 최종 Linux 이미지에서도 같은 고정 Hugging Face URL에 인증 없이 범위 요청을 보내 실제 GGUF 헤더를 확인했다.

모델 파일 해시:

- MLX 4bit: `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`
- GGUF Q8_0: `ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1`
