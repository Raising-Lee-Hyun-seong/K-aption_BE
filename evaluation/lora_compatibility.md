# KoreanLM LoRA 사전 점검

점검일: 2026-10-05. 공개 메타데이터·350바이트 설정과 HTTP HEAD만 조회했다. 가중치 다운로드, 모델 복원·변경, API 적용 및 커밋은 하지 않았다.

## 확인 결과

- 공식 `quantumaikr/KoreanLM-LoRA`는 공개·비게이트 저장소다. 고정할 리비전은 `1447f2ae33e32cb8af610758b90adda0c05670a2`다.
- `adapter_model.bin`은 8,434,381바이트(약 8.04MiB)다. 인증 없이 리디렉션을 따라간 HEAD 요청이 HTTP 200 및 같은 파일 크기를 반환했다. 파일 본문은 받지 않았다.
- 설정은 PEFT LoRA, rank 4, alpha 16, 대상 `q_proj`·`v_proj`, bias 없음, fan_in_fan_out=false다.
- 설정의 기반 모델명은 `decapoda-research/llama-7b-hf`다. 공식 추론 코드는 `quantumaikr/KoreanLM`을 명시적으로 로드한 뒤 이 어댑터를 적용한다. 모델명 차이만으로 비호환이라고 단정할 수 없으며, 학습 기반 리비전과 실제 가중치 호환성은 확인되지 않았다.
- 로컬 KoreanLM 설정은 Llama, hidden_size 4096, 32층·32헤드이며 MLX 4bit/group 64다. FP16·4bit의 실제 safetensors 가중치는 현재 없다. 모델 비교를 하려면 먼저 복원해야 한다.

## MLX 적용 경로

설치된 `mlx-lm==0.29.1`은 `load(..., adapter_path=...)`와 양자화된 선형층 위의 LoRA를 지원한다. 현재 API는 adapter_path를 사용하지 않는다.

PEFT 파일은 MLX에 직접 전달할 수 없다. MLX는 `adapters.safetensors`와 자체 `adapter_config.json`을 기대한다. 다음은 실제 텐서 확인 전의 변환 후보이며 호환성 검증 완료를 뜻하지 않는다.

| 항목 | 변환 후보 |
| --- | --- |
| 적용 층 | 32층의 self_attn.q_proj·self_attn.v_proj. 실제 체크포인트 키로 확인 필요 |
| A 텐서 | PEFT A를 전치해 MLX lora_a로 저장 |
| B 텐서 | PEFT B를 전치해 MLX lora_b로 저장 |
| 배율 | PEFT alpha/r = 16/4 = 4를 MLX scale로 설정 |
| 추론 dropout | 0.0 |
| 이름 | PEFT 접두사와 optional default를 확인하고 실제 MLX 모듈 이름으로 매핑 |

PEFT의 델타는 `(alpha/r) * B @ A`다. 설치된 MLX LoRALinear는 `scale * lora_b.T @ lora_a.T`를 사용하므로 위 전치·배율이 대응한다. 실제 A/B 모양, 누락·중복 키, 모든 층의 적용 여부 및 수치 일치는 가중치를 받은 뒤 별도로 검증해야 한다. MLX 로더가 strict=False로 어댑터를 읽기 때문에 오류 없이 로드됐다는 것만으로 전체 적용을 확인할 수 없다.

## 사전 점검 당시 제안한 실행 작업 (아래 후속 결과로 갱신)

1. 고정 리비전의 약 8MB 어댑터를 다운로드하고 제공된 SHA256을 확인한다. PyTorch 파일은 weights_only=True로 읽어 실제 키·모양을 검사한다.
2. MLX 형식 변환기를 만들고 비정방형 A/B의 수치 일치, 키 누락·중복·모양 오류를 검증한다.
3. 기존 약 25GiB 원본 캐시를 재사용해 4bit 기본 모델을 복원한다. 전체 FP16 중간 산출물 약 13GiB를 추가 저장하는 기존 make model 경로의 디스크 사용을 줄일 방법도 확인한다. 현재는 캐시를 삭제하지 않는다.
4. 동일 4bit 모델에 어댑터를 켜고 끄며 공식 기준 문장 10개를 비교한다. 수식·행위·부정·의미 보존과 시간·실패 수를 함께 판정한 뒤 API 적용을 결정한다.

양자화된 기본 모델 위에 별도 LoRA를 얹는 방식과 FP16에 먼저 병합한 뒤 양자화하는 방식은 수치가 같다고 보장할 수 없다. 초기 비교에서는 어댑터를 별도로 적용해 기본 모델·템플릿·검증 규칙을 고정한다. 품질 개선은 아직 측정하지 않았다.

## 근거

- [공식 모델 메타데이터](https://huggingface.co/api/models/quantumaikr/KoreanLM-LoRA?blobs=true)
- [고정 리비전의 어댑터 설정](https://huggingface.co/quantumaikr/KoreanLM-LoRA/blob/1447f2ae33e32cb8af610758b90adda0c05670a2/adapter_config.json)
- [공식 추론 코드](https://github.com/quantumaikr/KoreanLM/blob/main/generate.py)
- [PEFT LoRA 배율 문서](https://huggingface.co/docs/peft/main/package_reference/lora)
- 로컬 설치 소스: mlx_lm/utils.py, tuner/utils.py, tuner/lora.py, models/llama.py
- 원시 점검 자료: [lora_compatibility.json](lora_compatibility.json)

## 다운로드·변환 후 확인한 결과

같은 날 고정 리비전의 실제 파일을 다운로드해 체크섬·128개 텐서와 MLX 변환을 확인했다. **B 행렬 64개가 전부 0이므로 이 공개 어댑터의 델타는 항상 0이다.** 어댑터 누락이 기존 오역의 원인이라는 가설을 이 파일의 복원으로 해결할 수 없다. 전체 모델을 복원해 같은 어댑터의 품질 개선을 비교하는 작업은 우선순위에서 제외하고, 실제 학습된 어댑터 확보 가능성·모델 방향을 먼저 검토한다. 상세 검증과 한계는 [다운로드·변환 점검](lora_conversion.md)에 기록했다.
