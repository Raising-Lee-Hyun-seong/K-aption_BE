# 과거 실행 자료

이 디렉터리는 현재 런타임이 사용하지 않는 출처 명세와 제거된 로컬 엔진 테스트의 입력·기대값 기록을 보존한다. `.py.txt`는 실행 대상이 아니다.

MLX 모델 준비·변환·직접 생성 도구는 제거했다. 과거 평가 결과 JSON·강의 기준 자료·용어 실험 자료와 로컬 모델 파일은 보존한다. 이전 실행 코드가 필요하면 Git 이력을 확인한다.

`scripts/evaluate_candidates.py`, `evaluate_prompt_matrix.py`, `evaluate_qwen_lecture.py`는 과거 평가의 순수 텍스트·체크섬 진단을 위한 도우미만 유지한다. 새 실제 번역 평가는 `scripts/evaluate_api_translation.py`로 CPU HTTP 서버를 호출한다.
