# Qwen3 보완 실험 계획

보고 기준일: 2026-10-18. 기존 API는 KoreanLM을 유지한다. 로컬 Qwen3의 고정된 기존 가중치만 사용하며 다운로드·API 모델 교체는 하지 않는다.

## 사전 고정 조건

- 기존 공식 자막 10문장과 합성 물리학 12문장을 같은 순서로 두 번 실행한다.
- temperature=0, 최대 256토큰, 입출력 합계 2048토큰, 문맥·재시도·조기 중단·출력 정리 없음.
- `qwen-glossary`: 기존 용어집·보존 지시를 같은 실행 환경에서 재측정한다.
- `qwen-glossary-v2`: 기존 조건에 원문 매칭 용어 rocket=로켓, magnitude=크기, constant=상수를 추가한다.
- `qwen-relations-v2`: v2 용어와 함께 완결된 정의 문장, 삽입 설명으로 끊긴 등식의 인자·나눗셈 관계 보존을 일반적으로 지시한다. 불확실한 원문은 임의 복원하지 않는다.

새 용어는 평가 전용 JSON에 보관하며 API의 물리 용어집은 수정하지 않는다. 기준 번역·정답 수식·사례별 답변은 프롬프트에 넣지 않는다. 용어 추가와 관계 지시 추가를 분리해 관찰하되, 개별 용어의 인과 효과까지 단정하지 않는다. 원시 결과는 별도 경로에 기록하고 기존 결과를 덮어쓰지 않는다.

## 판정과 종료 기준

기존 `prompt-matrix-plan.md`의 pass/partial/fail 기준으로 모든 출력을 원문과 대조한다. 자동 형식·용어 진단과 의미 판정은 구분한다. 반복이 같으면 의미 판정을 중복 집계하지 않는다. 새 실패·부분 보존 사례와 기존 세 사례의 변경을 모두 기록한다.

이 실험은 이미 확인한 오류를 바탕으로 만든 개발 세트 평가다. 독립 검증이나 일반적인 정확도로 해석하지 않는다. 가장 나은 조건을 36개 저장 STT 구간으로 확대하며, 알려진 STT 오류와 문장 파편은 별도로 표시한다. 후보가 개선되지 않아도 결과를 보존하고 API 교체는 보류한다. 실제 API의 문맥·정리·검증·재시도 평가는 원시 생성과 분리해야 한다.

## 재현

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant qwen-glossary --evaluated-on 2026-10-18 --output evaluation/qwen-refinement-baseline.json
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant qwen-glossary-v2 --evaluated-on 2026-10-18 --output evaluation/qwen-refinement-terms.json
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_prompt_matrix.py --variant qwen-relations-v2 --evaluated-on 2026-10-18 --output evaluation/qwen-refinement-relations.json
```

출력 파일이 이미 있으면 덮어쓰지 않고 다른 `--output`을 지정한다.
