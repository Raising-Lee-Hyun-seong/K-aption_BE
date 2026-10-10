# 번역 모델·프롬프트 네 조건 비교

평가일: 2026-10-10. Apple M4·16GiB. **Qwen3 용어집·보존 지시가 엄격한 의미 보존 19/22로 가장 나았고, KoreanLM 지시 간소화는 2/22로 악화됐다. 세 문장의 의미·구문 문제가 남아 API에 적용하지 않았다.**

## 같은 조건의 실행 결과

| 조건 | 의미 보존 | 부분 보존 | 의미 오류 | 22문장 1회차 | 2회차 | MLX 최고 할당량 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KoreanLM 현재 지시 | 11/22 | 1 | 10 | 73.92초 | 74.36초 | 4.44GB |
| KoreanLM 간소화 지시 | 2/22 | 4 | 16 | 76.02초 | 77.46초 | 4.26GB |
| Qwen3 기본 | 15/22 | 3 | 4 | 19.03초 | 19.19초 | 2.47GB |
| Qwen3 용어집·보존 지시 | 19/22 | 2 | 1 | 20.55초 | 20.44초 | 2.58GB |

입력은 기존 MIT 공식 기준 문장 10개와 합성 물리학 12개다. 각 조건을 별도 프로세스에서 하나씩 로드하고 예열 후 두 번 실행했다. 네 조건의 입력·구현 해시, 패키지 버전, 생성 설정과 입력 순서가 모두 일치함을 확인했다. 모든 조건에서 두 회차의 원시 출력이 22개 모두 같았다. 시간은 모델 로딩·출처 검증·예열을 제외한 생성 시간 합계다. MLX 할당량은 전체 시스템 RAM이나 다른 앱의 메모리가 아니다.

온도 0, 최대 256토큰, 입출력 합계 상한 2048토큰을 공통 적용했다. 문맥·재시도·출력 정리·규칙 기반 조기 중단을 껐다. KoreanLM 현재 지시 조건은 현재 API의 `_build_prompt(text, "")`를 그대로 사용하지만 API 전체 파이프라인을 실행한 것은 아니다. 기존 36구간의 176초·10/36과 이번 값을 직접 비교해 개선율이나 회귀를 계산할 수 없다.

의미 판정은 단일 Codex 평가자의 원문·출력 대조이며 독립 전문가 검수 전이다. 합력/알짜힘 등 정상 동의어는 허용했고 부분 보존은 엄격한 통과에 넣지 않았다. 자연스럽지 않은 표현이라도 의미가 보존되면 통과시켰다. 생략·잘림·잘못된 정의나 관계는 별도로 구분했다. 판정 근거는 각 조건의 검토 JSON에 남겼다.

## 개선된 점과 남은 오류

- KoreanLM 현재 지시의 합력·운동량·충격량 용어 안내는 간소화 조건보다 유효했다. 간소화 조건은 dt를 데이터 타입으로 바꾸고 운동량을 운동 에너지로 정의했으며 초기 질량 문장이 반복하다 256토큰 상한에서 잘렸다. 이 사례는 두 회차 모두 발생했다. 다른 조건은 모든 문장에서 정상 EOS로 종료했다.
- Qwen3 용어집·보존 지시는 변위, 충격량, 운동 에너지의 속력 구분, M r 및 0보다 작음·초기 속도에서 뺄셈 관계를 개선했다. 기본 조건과 비교한 엄격한 통과는 15개에서 19개로 늘었다.
- 속력 정의의 용어 구분은 개선됐지만 최종 출력이 `속력은 속도의 크다.`라서 크기라는 정의 문장을 완성하지 못했다. 이 문장은 부분 보존으로 판정했다.

| 남은 사례 | 최상 후보 원시 출력 | 판정 |
| --- | --- | --- |
| official:rocket-speed | 그리고 우리는 궤도의 속력을 최종적으로 얻고자 한다. | fail: 속력이라는 용어는 고쳤지만 rocket을 궤도로 바꿔 구하려는 대상을 잃었다. |
| official:constant-u | 그리고 우리는 dvr가 등가된다-- u는 상수이므로-- 이는 앞부분으로 옮겨지고, 그 다음에 dmr over mr가 있다. | partial: u가 상수이고 앞에 놓는다는 정보 및 dmr/mr 일부는 보존하지만 dvr가 등가된다는 문장이 불완전하고 전체 dvr = u dmr/mr 관계를 명확히 표현하지 않았다. |
| physics:speed | 속력은 속도의 크다. | partial: 속력·속도 용어 구분은 고쳤지만 magnitude를 크다가 되도록 옮겨 크기라는 정의를 완전한 문장으로 표현하지 못했다. |

## 자동 검사와 의미 판정의 차이

| 조건 | 기본 형식·기호 진단 문제 없음 | 현재 용어 검사까지 문제 없음 | 의미 보존 |
| --- | ---: | ---: | ---: |
| KoreanLM 현재 지시 | 17/22 | 17/22 | 11/22 |
| KoreanLM 간소화 지시 | 16/22 | 3/22 | 2/22 |
| Qwen3 기본 | 21/22 | 13/22 | 15/22 |
| Qwen3 용어집·보존 지시 | 22/22 | 22/22 | 19/22 |

**Qwen3 용어집 조건은 자동 검사 22/22를 통과했지만 엄격한 의미 보존은 19/22다.** rocket을 궤도로 바꿔도 속력이라는 단어가 있어 검사를 통과했다. 반대로 기본 Qwen3의 합력 같은 정상 동의어는 현재 용어 검사의 알짜힘 문자열 조건 때문에 거부될 수 있다. 이 값은 원시 텍스트에 대한 진단이며 API의 정리·조기 중단·재시도 후 성공률이 아니다.

## 재현과 저장 자료

[실험 계획과 판정 기준](prompt-matrix-plan.md), [측정 요약 JSON](prompt-matrix-comparison.json), [실행 도구](../scripts/evaluate_prompt_matrix.py), [테스트](../tests/test_prompt_matrix.py).

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant koreanlm-current
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant koreanlm-simple
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant qwen-base
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant qwen-glossary
```

가중치는 기존 로컬 파일을 재사용했다. KoreanLM은 로컬 파일과 출처 기록의 해시를 남긴 것이며 공개 원본 LFS 해시 대조를 했다고 주장하지 않는다. Qwen3는 고정 공개 파일 크기·LFS 해시를 다시 검증했다. 기준 Qwen3의 22개 출력은 기존 후보 실행과 전부 같아 이전 검토 판정도 동일하게 적용했다. 다운로드나 API 모델 교체는 하지 않았다.

- KoreanLM 현재 지시: [원시 실행](prompt-matrix-koreanlm-current.json), [문장별 판정](prompt-matrix-koreanlm-current-review.json)
- KoreanLM 간소화 지시: [원시 실행](prompt-matrix-koreanlm-simple.json), [문장별 판정](prompt-matrix-koreanlm-simple-review.json)
- Qwen3 기본: [원시 실행](prompt-matrix-qwen-base.json), [문장별 판정](prompt-matrix-qwen-base-review.json)
- Qwen3 용어집·보존 지시: [원시 실행](prompt-matrix-qwen-glossary.json), [문장별 판정](prompt-matrix-qwen-glossary-review.json)

## 다음 작업

Qwen3 용어집·보존 지시를 후속 후보로 선택한다. rocket 대상 보존, 속력의 크기 정의, u를 포함한 등식 연결의 세 사례를 먼저 보완한다. 예를 들어 rocket=로켓은 평가용 용어집의 별도 버전으로 실험하되 이번 네 조건의 원시 결과를 덮어쓰지 않는다. 복합 프롬프트 변경에서 어느 문구가 효과를 냈는지는 분리 실험 전까지 단정하지 않는다.

보완 결과를 같은 22문장에 확인한 뒤 이번 사용자 응답의 36구간으로 확대한다. 의미 판정을 위해 STT의 rocket's lead, separation variables, 2/to 및 미분식 파편도 별도로 표시해야 한다. 성공한 조합의 Qwen 전용 채팅 템플릿을 API에 연결하는 작업과 타임스탬프·실패 처리 회귀 검증은 그 다음이다. 단순히 KAPTION_MODEL_DIR만 Qwen으로 바꾸면 현재 KoreanLM 템플릿이 적용되므로 이번 실험과 같은 입력이 되지 않는다.

테스트 80개와 30개 Python 파일·210개 함수의 PyDoc 필드 검사를 통과했다. 이전 모델 복구 작업의 미커밋 변경은 그대로 유지했다. 이번 실험도 커밋하지 않았다.

## 문장별 비교

| 사례 | KoreanLM 현재 | KoreanLM 간소화 | Qwen3 기본 | Qwen3 용어집 |
| --- | --- | --- | --- | --- |
| official:rocket-speed | pass | partial | fail | fail |
| official:separation-integrate | fail | fail | pass | pass |
| official:multiply-dt | fail | fail | pass | pass |
| official:move-m | fail | fail | pass | pass |
| official:constant-u | fail | fail | partial | partial |
| official:integrable-equation | pass | fail | pass | pass |
| official:integration-limits | fail | fail | pass | pass |
| official:initial-mass | partial | fail | pass | pass |
| official:mass-less-negative | fail | fail | partial | pass |
| official:sled-stops | fail | fail | pass | pass |
| physics:net_force | fail | fail | pass | pass |
| physics:velocity | pass | pass | pass | pass |
| physics:mass | pass | fail | pass | pass |
| physics:acceleration | pass | fail | pass | pass |
| physics:speed | fail | fail | fail | partial |
| physics:displacement | fail | fail | fail | pass |
| physics:momentum | pass | fail | pass | pass |
| physics:impulse | pass | partial | fail | pass |
| physics:energy | pass | partial | partial | pass |
| physics:limits | pass | fail | pass | pass |
| physics:gravity | pass | pass | pass | pass |
| physics:friction | pass | partial | pass | pass |
