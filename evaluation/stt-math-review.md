# 원본 강의의 수학 STT 확인

검증일: 2026-10-18. `MIT8_01F16_W06PS01-2_360p.mp4` 원본을 프로젝트 내부 `TranscriptionService`와 Whisper small·CPU int8·beam 5·VAD로 다시 인식했다. 같은 모델을 재사용하며 기본 설정, 일반 수학 문맥 힌트, 용어 힌트를 비교했다. 정답 자막이나 완성된 수식은 힌트에 넣지 않았다.

| 조건 | 전사 시간(초) | 원시 구간 수 | 관찰 |
| --- | ---: | ---: | --- |
| 기본 | 25.26 | 45 | 기존의 `2 vR prime of t`, `rocket's lead`가 재현됨 |
| 수학 문맥 힌트 | 24.84 | 33 | `to vR prime ... vR of t`를 인식하지만 다른 단어 오류도 발생 |
| 용어 힌트 | 35.96 | 24 | `rocket sled`와 `to`를 인식하지만 문장부호·문장 경계가 줄어듦 |

원시 결과와 입력 SHA256은 [비교 JSON](stt-math-hints-261018.json)에 저장했다. 시간은 서비스가 반환한 추론 대기·전사 시간이며 모델 로딩·오디오 디코딩은 제외한다. 단일 강의에 대한 개발 실험으로 일반적인 정확도나 Windows 속도를 뜻하지 않는다. 문장 수 감소를 정확도 개선으로 해석하지 않는다.

## 판서 대조와 적용 결정

50초·88초·121초·166초 판서를 확인했다. 프라임·적분 구간·분수·로그 괄호가 보이지만 360p 해상도와 손에 가린 부분 때문에 함수 인자를 모두 확정할 수는 없다. 음성의 끊긴 `dvR equals u is a constant`만 번역하면 판서에 있는 전체 미분 관계가 전달되지 않는다. `m of vR`, `u1 over m`도 STT만으로 신뢰성 있게 복구되지 않았다.

기본 API에는 힌트를 켜지 않는다. 일부 용어 개선과 함께 다른 단어·문장 경계의 회귀가 발생했고, 개선을 일반화할 독립 검증 자료도 부족하다. 서비스 생성자의 `initial_prompt`, `hotwords`는 비교 실험을 위한 선택적 인자이며 기본값은 모두 None이다. 원문 숫자·변수나 타임스탬프를 자동 보정하지 않는다. 수식 복구에는 명확한 원본 대조 또는 별도 판서 인식 단계가 더 필요하다.

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false .venv-api/bin/python scripts/evaluate_stt_hints.py --media MIT8_01F16_W06PS01-2_360p.mp4 --evaluated-on 2026-10-18 --output evaluation/stt-hints-new-run.json
```

기존 결과를 덮어쓰지 않도록 새 출력 경로를 지정한다. MIT 영상·기준 자막의 출처와 라이선스는 [기준 자료](mit_rocket_reference.json)에 기록돼 있다.
