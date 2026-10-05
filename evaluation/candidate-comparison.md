# 로컬 번역 후보 실행 비교

평가일: 2026-10-05. Apple M4, 물리 메모리 16GiB. **같은 22문장에서 Qwen3의 엄격한 의미 보존은 15/22, TranslateGemma는 8/22였다. 두 모델 모두 주요 오역이 남아 API에 적용하지 않았다.** 의미 판정은 Codex 단일 평가자의 원문·출력 대조이며 독립 전문가 검수 전이다.

## 실행 결과

| 지표 | TranslateGemma 4B 4bit | Qwen3 4B Instruct 2507 4bit |
| --- | --- | --- |
| 모델 로딩 | 1.51초 | 0.71초 |
| 22문장 번역 1회차 | 18.53초 | 18.78초 |
| 22문장 번역 2회차 | 18.45초 | 18.92초 |
| MLX 프로세스 최고 메모리 | 2.36GB | 2.47GB |
| 정상 종료·생성 예외 없음 | 22/22 | 22/22 |
| 자동 검사 문제 없음 | 21/22 | 21/22 |
| 엄격한 의미 보존 | 8/22 | 15/22 |
| 부분 보존·불명확 | 5/22 | 3/22 |
| 핵심 의미 오류 | 9/22 | 4/22 |
| 두 회차 출력 일치 | 22/22 | 22/22 |

각 모델은 별도 프로세스에서 하나씩 로드했다. 평가에서 제외한 예열 1문장 뒤, MIT 공식 기준 문장 10개와 합성 물리학 문장 12개를 같은 순서로 두 번 실행했다. 최대 출력 256토큰, temperature=0, 문맥·용어집·재시도 없음이며 모델별 채팅 템플릿을 사용했다. 원시 출력을 정리·재생성·삭제하지 않았다. 토큰 상한 종료는 최종 실행에서 없었다.

최고 메모리는 로딩부터 생성까지 MLX가 기록한 프로세스 할당량이며 운영체제·다른 앱·전체 Python 메모리를 포함한 시스템 RAM 사용량이 아니다. 토큰 수·토크나이저가 다르므로 초당 토큰 속도만으로 모델 우열을 판정하지 않는다. 2회 반복은 시간 편차 확인을 위한 소규모 실행이다. 같은 출력이 나온 것이 품질 정확성을 입증하지 않는다.

기존 KoreanLM의 12문장·전체 강의 평가는 입력 묶음·문맥·용어집·토큰 상한·재시도 조건이 다르다. 이번 시간을 기존 176.13초와 직접 비교해 개선율을 계산하거나, 이번 의미 통과율을 전체 강의 정확도라고 말할 수 없다. 현재 기반 모델 가중치가 없어 같은 조건의 KoreanLM 재실행은 하지 않았다.

## 중요한 오역

| 입력 | TranslateGemma | Qwen3 | 판정 |
| --- | --- | --- | --- |
| Speed is the magnitude of velocity. | 속도는 속도의 크기입니다. | 속도는 속도의 크다. | 둘 다 속력·속도 구분과 정의를 잃음 |
| Impulse is the change in momentum. | 충동은 운동량의 변화입니다. | 운동량의 변화는 운동량이다. | 둘 다 충격량 정의 오류 |
| Momentum is the product of mass and velocity. | 동력은 질량과 속도의 산물입니다. | 운동량은 질량과 속도의 곱이다. | TranslateGemma의 동력은 오역, Qwen은 보존 |
| ...get the speed of the rocket. | 로켓의 속도를 알아내고 싶습니다. | 궤도의 속도를 최종적으로 얻고자 한다. | TranslateGemma는 속력 구분 미충족, Qwen은 로켓→궤도 오류 |
| ...integration limits... | 통합의 한계를 관리해야 합니다. | 적분 한계를 잘 관리해야 한다. | TranslateGemma는 적분 문맥을 잃음 |
| ...my sled will eventually come to a stop. | 결국 제 스키는 멈출 것입니다. | 내 슬레드는 결국 정지하게 된다. | TranslateGemma는 대상 변경, Qwen 음역은 의미 보존으로 허용 |

합력·알짜힘 같은 동의어는 의미가 보존되면 통과시켰다. 속력/속도, 질량/무게, 적분/일반적인 풀기는 물리학 기준에서 구분했다. 부분 보존은 엄격한 의미 통과에 포함하지 않는다. `M r` 누락은 두 모델 모두 자동 검사에 잡혔지만 위의 속력·충격량·동력 오류는 자동 검사를 통과했다. **자동 검사 21/22를 정확한 번역 21개로 해석하면 안 된다.**

## 실행 중 해결한 TranslateGemma 호환성

1. 기본 MLX EOS 목록은 [1]인데, 실제 모델은 `<end_of_turn>` ID 106으로 번역을 끝냈다. 이를 중단 조건에 추가하지 않으면 번역 뒤 턴 종료 표식이 반복돼 256토큰을 소모했다. 토크나이저 파일의 ID를 확인하고 실행 시 EOS를 [1, 106]으로 설정했다.
2. 배포 설정은 `text_config.rope_parameters.full_attention`에 linear factor=8을 명시하지만 MLX-LM 0.29.1의 Gemma 구현은 `rope_scaling`만 읽는다. 해당 값을 기존 필드로 대응해 실제 모델 로딩에 전달했다. 원본 파일·가중치를 바꾼 것이 아니며 최종 보고서에 덮어쓰기 설정을 기록했다.
3. Transformers 4.57.6의 로컬 토크나이저 감지 조건은 Gemma에도 Mistral 정규식 경고를 냈다. 실제 Gemma 사전의 공백 Split을 보존하도록 `fix_mistral_regex=False`를 명시했다. Mistral용 정규식으로 교체하지 않았다.

중단 조건 보완 전 11문장은 `candidate-translategemma-stop-diagnostic.json`, 위치 배율 대응 전 44개 출력은 `candidate-translategemma-rope-diagnostic.json`에 별도 보존했다. 이 자료를 최종 품질·시간 집계에 섞지 않았다. 최종 TranslateGemma 결과는 두 호환성 보완이 적용된 파일이다. Qwen은 기본 EOS [151645]를 사용하며 이 Gemma 대응이 필요 없다.

## 저장 파일과 재현

- 원시 출력·프롬프트·토큰·종료·시간: [TranslateGemma](candidate-translategemma.json), [Qwen](candidate-qwen.json)
- 입력·결과 해시와 22개 문장별 의미 근거: [TranslateGemma 검토](candidate-translategemma-review.json), [Qwen 검토](candidate-qwen-review.json)
- 측정 요약: [candidate-comparison.json](candidate-comparison.json)
- 실행 도구: [evaluate_candidates.py](../scripts/evaluate_candidates.py)

```bash
.venv-api/bin/python scripts/evaluate_candidates.py --candidate translategemma --download --prepare-only
.venv-api/bin/python scripts/evaluate_candidates.py --candidate qwen --download --prepare-only
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_candidates.py --candidate translategemma
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_candidates.py --candidate qwen
```

두 후보의 파일 크기·공개 LFS SHA256을 대조했으며 다운로드와 준비는 인증 파일 없이 공개 URL로 진행했다. 약 4.50GB가 models에 추가됐고 가중치는 Git 제외 대상이다. 기존 캐시는 삭제하지 않았다. API 모델·서버 설정은 바꾸지 않았고 커밋하지 않았다. 회귀 테스트 71개를 통과했다.

## 후속 판단

우선 Qwen3에 물리학 용어집과 수식 관계 보존 지시를 적용한 별도 22문장 실험을 권장한다. 합력 같은 정상 동의어를 허용하고, 속력·변위·충격량·로켓 대상 오류가 실제로 해결되는지 원문으로 재판정해야 한다. 단어를 넣었다는 이유만으로 정답으로 인정하지 않는다. 성공한 후보만 저장 STT 36묶음 평가로 확대한다. 현재 상태에서 API 교체는 진행하지 않는다.

## 문장별 판정

아래 판정의 자세한 원문·기준·근거는 모델별 검토 JSON에 보존했다. pass=의미 보존, partial=부분 보존·불명확, fail=핵심 오류다.

| 사례 | TranslateGemma | Qwen3 |
| --- | --- | --- |
| official:rocket-speed | partial | fail |
| official:separation-integrate | pass | pass |
| official:multiply-dt | fail | pass |
| official:move-m | pass | pass |
| official:constant-u | partial | partial |
| official:integrable-equation | fail | pass |
| official:integration-limits | fail | pass |
| official:initial-mass | fail | pass |
| official:mass-less-negative | partial | partial |
| official:sled-stops | fail | pass |
| physics:net_force | pass | pass |
| physics:velocity | pass | pass |
| physics:mass | pass | pass |
| physics:acceleration | pass | pass |
| physics:speed | fail | fail |
| physics:displacement | partial | fail |
| physics:momentum | fail | pass |
| physics:impulse | fail | fail |
| physics:energy | partial | partial |
| physics:limits | fail | pass |
| physics:gravity | pass | pass |
| physics:friction | pass | pass |
