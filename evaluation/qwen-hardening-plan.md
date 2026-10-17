# Qwen3 검증·수학 표현 보완 계획

보고 기준일: 2026-10-18. 기존 36구간 시스템 문맥 평가의 남은 실패를 바탕으로 하는 개발 실험이며 독립 검증이 아니다.

## 변경 범위

1. 공통 출력 검증에서 정상 수식 밑줄 별칭을 허용한다. 변수 누락·계수 변경·긴 식별자의 부분 일치는 계속 거부한다.
2. 관찰된 문맥 표식 유출을 조기 종료 대상으로 추가한다.
3. 원문에 없는 연속 키릴·가나·한자 문자 어구를 출력 검증에서 거부한다. 원문의 같은 어구, 단일 수식 문자·그리스 문자, 한글 바로 뒤 괄호 한자 병기는 허용한다. 일반적인 언어·의미 분류기가 아니며 나머지 오탐·누락 가능성을 명시한다.
4. 평가 전용 v3 조건에 sled, integrated/integrating, 수학 prime 용어를 추가하고 수식의 나눗셈·인자 위치·기호 보존을 지시한다. API 모델·프롬프트·용어집은 유지한다. 공통 검증 보완은 API에도 적용된다.

STT의 실제 단어·수식·문장 경계는 이번에 고치지 않는다. 원래 36구간을 유지해 입력 변경과 출력 검증·프롬프트 변경의 효과를 섞지 않는다.

## 사전 실행 조건

- 같은 36구간으로 공통 검증만 보완한 v2 기준과 v3 보완을 각각 1회 실행한다.
- 시스템 메시지의 앞 영어 256자, 최대 128토큰, 문맥 없는 재시도 1회 등 기존 조건을 유지한다.
- 기존 22문장은 v3로 2회 실행한다. 문맥·재시도·출력 정리 없이 256토큰을 사용한다.
- 기존 실행·판정 파일은 수정하지 않고 별도 출력 경로에 저장한다.
- 원문·결과를 전부 대조해 pass/fail/uncertain을 판정하고, 자동 수락·실패 표시·의미 오류를 분리한다. 언어 혼입 차단으로 실패 표시가 늘어도 의미 개선으로 집계하지 않는다.
- 이미 알려진 개발 세트에 맞춘 작업이므로 새 강의·독립 사람 검수 없이 API 모델 교체를 결정하지 않는다.

## 재현

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_qwen_lecture.py --variant qwen-relations-v2 --context-format system --evaluated-on 2026-10-18 --output evaluation/qwen-hardening-v2.json
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_qwen_lecture.py --variant qwen-relations-v3 --context-format system --evaluated-on 2026-10-18 --output evaluation/qwen-hardening-v3.json
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant qwen-relations-v3 --evaluated-on 2026-10-18 --output evaluation/qwen-hardening-cases.json
```
