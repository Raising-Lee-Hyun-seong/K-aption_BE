# M4·16GB 로컬 번역 모델 후보

조사일: 2026-10-05. **TranslateGemma 4B 4bit와 Qwen3 4B Instruct 2507 4bit를 실행 비교 후보로 선정했다.** 번역 전용 모델을 먼저, 일반 지시 모델을 다음으로 평가한다. 가중치 다운로드·전체 모델 실행·API 교체는 아직 하지 않았다.

## 비교 대상

| 항목 | TranslateGemma 4B | Qwen3 4B Instruct 2507 |
| --- | --- | --- |
| 선정 이유 | 번역을 위해 학습한 소형 모델 | 수식·과학 지시를 처리하는 비추론 지시 모델 비교군 |
| 원본 | google/translategemma-4b-it | Qwen/Qwen3-4B-Instruct-2507 |
| 평가 파일 | mlx-community/translategemma-4b-it-4bit | mlx-community/Qwen3-4B-Instruct-2507-4bit |
| 언어 확인 | 변환본 전용 템플릿에 en·ko가 있으며 실제 렌더링 성공 | Qwen3 공식 언어 목록에 영어·한국어가 있음 |
| 가중치 파일 합계 | 2,183,295,977바이트, 약 2.18GB / 2.03GiB | 2,263,022,417바이트, 약 2.26GB / 2.11GiB |
| 저장소 파일 총합 | 약 2.22GB | 약 2.28GB |
| 모델 종류 | gemma3 (텍스트 경로) | qwen3 |
| 설치된 MLX-LM | 0.29.1 변환본이며 로컬에 gemma3·gemma3_text 구현이 있음 | 로컬에 qwen3 구현이 있음 |
| 라이선스 | Gemma Terms of Use | Apache-2.0 |

용량은 공개 파일 메타데이터 합계이며 추론 메모리 사용량이 아니다. M4·16GB에서 단일 모델·짧은 입력으로 평가할 후보로 판단했지만, 실제 최고 메모리·속도·수치 호환성은 미검증이다. 두 모델을 동시에 로드하지 않는다. 준비된 4bit 파일을 직접 받아 원본 FP16 중간 파일 추가 생성을 피한다. 두 저장소 전체 파일의 합계는 약 4.50GB / 4.19GiB이며 중복 캐시·임시 파일은 별도다.

TranslateGemma는 원본 카드에 번역용 학습·입력 방식이 명시돼 있어 먼저 평가할 근거가 있다. Qwen의 일반 과학·수학 벤치마크가 물리학 번역 정확도를 입증하는 것은 아니다. 양자화된 두 파일은 원제작자가 배포한 원본이 아니라 MLX Community 변환본이며, 원본과 같은 품질을 보장하지 않는다. [Google 원본 카드](https://huggingface.co/google/translategemma-4b-it), [Qwen 원본 카드](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3 언어 목록](https://qwenlm.github.io/blog/qwen3/).

## 실제 사전 점검

- MLX-LM 0.29.1, MLX 0.29.3, Transformers 4.57.6 설치 상태를 확인했다.
- 공식 기준 문장 10개와 기존 합성 물리학 문장 12개를 모두 각 템플릿에 렌더링했다. 22개 모두 영어 원문이 한 번만 포함되고 한국어 대상 언어가 들어갔다. 토크나이저·가중치는 로드하지 않았다.
- TranslateGemma 전용 메시지는 user content 목록에 type=text, source_lang_code=en, target_lang_code=ko, text=원문을 넣는다. 커뮤니티 페이지의 일반 채팅 자동 예제 대신 [공식 입력 규칙](https://huggingface.co/google/translategemma-4b-it#usage)을 따른다. 기존 용어집·앞 영어 문맥을 원문 text에 붙이면 그것도 번역 대상이 되므로 첫 비교는 원문만 전달한다.
- Qwen3-4B-Instruct-2507은 비추론 전용 버전이다. 자체 채팅 템플릿으로 한국어 번역·수치·부정·수식 보존 지시를 구성한다. 기존 KoreanLM 템플릿은 사용하지 않는다.
- 리비전·파일별 크기·가중치 SHA256·템플릿 해시는 [후보 목록](model_candidates.json)과 [템플릿 점검](model_candidates_checks.json)에 고정했다. [원시 메타데이터](model_candidates_sources.json)도 보존했다.

원본 TranslateGemma 저장소는 수동 승인 게이트이며 커뮤니티 변환본은 공개 상태다. 공개 변환본에도 Gemma 조건이 적용된다. 이용·배포 시 [Gemma 약관](https://ai.google.dev/gemma/terms)을 따른다. 이번에는 메타데이터·템플릿만 확인했다.

## 제외한 후보

- HY-MT1.5-1.8B: 영어·한국어를 지원하는 약 1GB MLX 4bit 후보였지만, [공식 라이선스](https://huggingface.co/tencent/HY-MT1.5-1.8B/blob/dbad03788f49709801014c95d481a514c272ca52/License.txt)가 한국을 허용 지역에서 제외한다. 한국에서의 실행 비교 후보에서 제외했다. 커뮤니티 변환본의 공개 여부가 원본 조건을 바꾸지 않는다.
- NLLB-200-distilled-600M: [공식 카드](https://huggingface.co/facebook/nllb-200-distilled-600M)의 CC-BY-NC-4.0·연구 목적·프로덕션 배포 제외 조건 때문에 제품 적용 후보에서 제외했다. 작은 모델이라는 이유만으로 선정하지 않았다.

## 다음 실행 범위

1. 후보 리비전의 4bit 가중치와 필수 설정·토크나이저만 다운로드하고 명시된 해시·크기를 확인한다. 대용량 기존 KoreanLM 캐시는 지우지 않는다.
2. 별도 평가 실행으로 TranslateGemma, Qwen 순서로 한 모델씩 로드한다. API의 TranslationService를 교체하지 않고, 원시 응답을 그대로 저장한다. 각 모델의 템플릿·리비전·런타임 버전·프롬프트를 기록한다.
3. 같은 영어 입력 22개에 최대 출력 256토큰·temperature=0·문맥 없음·재시도 없음으로 1차 비교한다. 기존 128토큰·문맥·재시도 평가와 조건이 다르므로 기존 시간 대비 개선율로 해석하지 않는다. 256토큰 상한으로 끝난 출력은 미완성으로 별도 표시한다.
4. 용어·숫자·수식 검증 사유를 기록하되 출력 삭제나 재생성으로 실패를 숨기지 않는다. 22개 원문·기준 번역·후보 번역을 함께 검토해 주어·행위·부정·연산 관계·의미 없는 추가 내용을 판정한다. 동의어만 다르면 의미 오류로 판정하지 않는다. 기준 번역은 독립 전문가 검수 전이다.
5. 모델 로딩 시간, 각 번역 시간, 생성 토큰·종료 사유, MLX 최고 메모리와 실패 수를 함께 기록한다. 예열 1문장을 평가에서 제외하고 동일 입력을 반복 실행해 시간 편차를 확인한다. 모델별 공식 샘플링 권장은 추후 별도 실험으로 구분한다.
6. 의미 품질이 개선된 후보를 STT 36묶음으로 확대하고, 그 결과로 모델 적용과 문맥·용어집 기능을 결정한다. 메모리 부족 또는 로딩 비호환이면 그 상태를 보고하고 API 의존성을 일괄 변경하지 않는다.

## 변환 파일 출처

- [TranslateGemma MLX 변환본](https://huggingface.co/mlx-community/translategemma-4b-it-4bit/tree/5788ec08c047f3f2e17808101b8d9566ac930d58)
- [Qwen3 MLX 변환본](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/tree/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b)

## 실행 후 갱신

상기 내용은 실행 전 조사다. 두 후보의 22문장 반복 실행과 의미 대조를 완료했다. TranslateGemma의 턴 종료·위치 배율 호환성을 보완한 최종 결과는 [실행 비교 보고서](candidate-comparison.md)에 기록했다. 원시 조사 자료는 유지하고 후보 목록의 상태를 갱신했다. API 모델은 교체하지 않았다.
