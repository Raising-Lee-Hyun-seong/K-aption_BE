# K-aption BE

해외 대학 강의의 영어 음성을 인식하고, 강의 문맥과 전공 용어를 반영한 한국어 자막을 만드는 프로젝트입니다.

- 프로젝트 저장소: https://github.com/Raising-Lee-Hyun-seong/K-aption_BE
- 기반 코드: [quantumaikr/KoreanLM](https://github.com/quantumaikr/KoreanLM)
- 모델 식별자: `quantumaikr/KoreanLM` — 프로젝트 저장소 주소와 별개이므로 변경하지 않습니다.

## 현재 상태

| 항목 | 상태 |
| --- | --- |
| STT | 프로젝트 내부 `transcribe.py`에 구현. MP4 → 영어 TXT, 선택적으로 구간 시간 포함 |
| 샘플 영상 | `MIT8_01F16_W06PS01-2_360p.mp4` 사용 |
| 전공 용어집 | 아직 없음. 먼저 용어집 없는 번역을 확인한 후 강의에서 용어를 추출·검수할 예정 |
| KoreanLM | MLX 4bit 변환·추론 확인. 짧은 문장 평가에서 의미 오류 발견; 전체 품질 검증 필요 |
| API·자막·UI | FastAPI 통합, SRT/VTT 생성, 자막 재생은 후속 작업 |

샘플 영상은 현재 로컬 파일이며 Git에 추적되지 않습니다. 새로 clone한 환경에서는 파일을 프로젝트 루트에 별도로 준비해야 합니다.

## 빠른 실행: Makefile

프로젝트 루트에서 실행합니다. macOS/Linux의 `make`와 POSIX 셸 기준입니다. MLX는 Apple Silicon macOS 전용이며, Windows PowerShell에서는 아래 2절의 직접 실행 명령을 사용하세요. `make`가 없으면 운영체제의 개발 도구로 설치해야 합니다.

```bash
make                 # 사용 가능한 명령 보기
make setup-stt       # STT 환경 설치 (최초 1회)
make stt             # 샘플 영상 → outputs/sample.en.txt
make setup-mlx       # MLX 환경 설치 (최초 1회)
make model           # 원본 다운로드 → FP16 → 4bit (최초 1회)
make translate       # 기본 문장: The velocity is constant.
```

**현재 이 Mac에는 4bit 모델과 MLX 환경이 준비되어 있으므로 `make translate`부터 실행하면 됩니다.** 이미 모델이 있으면 `make model`을 다시 실행할 필요가 없습니다. 기존 출력 폴더가 있으면 양자화 도구가 중단하며, 기존 STT 출력 파일도 덮어쓰지 않습니다.

```bash
make translate TEXT='The net force acting on the object is zero.'
make translate TEXT='The velocity is constant.' CONTEXT='We are discussing uniform motion.' MAX_TOKENS=128
make stt VIDEO='lecture.mp4' OUTPUT='outputs/lecture.en.txt' STT_MODEL=small
```

설치할 Python 선택: `make setup-stt PYTHON=python3.12`. 세부 작업은 `make prepare`(원본·FP16 준비), `make quantize`(4bit 변환)로 나눌 수 있습니다. 다른 모델 출력 경로는 `MODEL_DIR=models/koreanlm-4bit-v2`로 지정하고 번역할 때도 같은 값을 사용합니다. 설치·다운로드는 실행 명령에 자동으로 포함하지 않았습니다.

## 1. 저장소 준비

```bash
git clone https://github.com/Raising-Lee-Hyun-seong/K-aption_BE.git
cd K-aption_BE
```

## 2. 샘플 영상으로 영어 STT 실행

모든 명령은 프로젝트 루트에서 실행합니다. Python 3.10~3.12와 샘플 영상을 준비하세요. STT 코드와 의존성은 이 저장소에 포함되어 있습니다. 가상환경을 활성화하지 않고 내부 Python을 직접 호출하므로 KoreanLM 환경과 혼동하지 않습니다.

### Windows (PowerShell)

Python 3.12 설치 시 Python Launcher를 함께 설치합니다. 아래는 PowerShell 명령이며, 실행 정책 변경이나 Activate.ps1 실행이 필요하지 않습니다.

```powershell
py -3.12 -m venv .venv-stt
.\.venv-stt\Scripts\python.exe -m pip install -r requirements-stt.txt
.\.venv-stt\Scripts\python.exe transcribe.py .\MIT8_01F16_W06PS01-2_360p.mp4 --language en --model small --device cpu --timestamps --output .\outputs\sample.en.txt
```

`py`가 없으면 Python 설치 및 Launcher를 확인하세요. Python 3.10/3.11을 설치했다면 `-3.12`를 해당 버전으로 바꿉니다.

### macOS (터미널: zsh/bash)

Apple Silicon에서는 ARM64 Python을 사용하세요. 이 STT 코드는 CPU로 실행하며 MPS 옵션은 제공하지 않습니다.

```bash
python3 -m venv .venv-stt
.venv-stt/bin/python -m pip install -r requirements-stt.txt
.venv-stt/bin/python transcribe.py \
  ./MIT8_01F16_W06PS01-2_360p.mp4 \
  --language en \
  --model small \
  --device cpu \
  --timestamps \
  --output ./outputs/sample.en.txt
```

### Linux (터미널: bash)

Python과 venv 지원이 필요합니다. Ubuntu/Debian에서 venv 생성이 실패하면 설치한 Python 버전에 맞는 `python3-venv` 패키지를 준비하세요.

```bash
python3 -m venv .venv-stt
.venv-stt/bin/python -m pip install -r requirements-stt.txt
.venv-stt/bin/python transcribe.py \
  ./MIT8_01F16_W06PS01-2_360p.mp4 \
  --language en \
  --model small \
  --device cpu \
  --timestamps \
  --output ./outputs/sample.en.txt
```

### 공통 동작 및 옵션

- `transcribe.py`: 음성 인식, 구간 시간 출력, TXT 저장, 처리 시간 기록.
- `requirements-stt.txt`: STT 전용 의존성. 기존 `requirements.txt`는 KoreanLM 관련 파일로 유지합니다.
- 기본값은 `small`, CPU `int8`, 언어 자동 감지입니다. 위 명령에서는 영어를 지정합니다.
- Mac에서는 CPU를 사용합니다. 준비된 NVIDIA CUDA 환경에서는 `--device cuda`로 실행할 수 있습니다.
- 전체 옵션: macOS/Linux는 `.venv-stt/bin/python transcribe.py --help`, Windows PowerShell은 `.\.venv-stt\Scripts\python.exe transcribe.py --help`
- 최초 실행 시 Whisper 모델을 다운로드합니다.
- 현재 코드는 PyAV로 MP4 음성을 직접 읽으므로 별도 WAV 추출 명령이 필요하지 않습니다.
- 결과는 `[시작s -> 종료s] 영어 문장` 형태의 TXT입니다. 아직 SRT/VTT가 아닙니다.
- 출력 파일이 이미 있으면 덮어쓰지 않고 종료합니다. 다시 실행할 때 다른 출력 파일명을 지정하세요.

## 3. KoreanLM 실행 환경 준비

KoreanLM은 영어 텍스트를 입력받는 언어모델입니다. 영상을 직접 입력하지 않습니다. 먼저 아래 짧은 문장 예제로 모델 로딩과 생성을 확인하고, 이후 STT 결과 중 한두 문장을 사용합니다.

### 장비와 메모리

KoreanLM은 모델 설정 기준 약 6.74B 파라미터의 Llama 모델이며, 입력과 출력의 합을 최대 2,048토큰 범위로 관리합니다.

### 현재 장비: MacBook Pro M4 · 통합 메모리 16GB

**FP16 방식은 메모리 여유가 부족하지만, 아래 MLX 4bit 방식은 실제 로딩·생성에 성공했습니다.** M4의 연산 지원과 모델을 담을 메모리는 별개의 조건입니다. 아래 표는 사용자 제공 사양과 모델 설정에 따른 이론적 비교이며, 실제 4bit 실행 수치는 검증 결과 절에 따로 기록합니다.

| 실행 방식 | 가중치만의 이론적 크기 | 16GB 장비 판단 |
| --- | --- | --- |
| FP32 CPU | 약 25.10GiB | 물리 메모리 초과. 실용적인 실행 경로로 부적합 |
| FP16 MPS | 약 12.55GiB | macOS·앱·추론 버퍼까지 고려하면 매우 빠듯함. 메모리 부족 또는 스왑으로 인한 지연 가능 |
| 8bit | 약 6.28GiB | 추가 메모리와 실행 엔진 호환성 검증 필요 |
| 4bit | 약 3.14GiB | 로딩·생성 확인 완료. 실제 MLX 최대 메모리 약 4.23~4.41GB. 번역 품질 미달 사례 있음 |

크기는 파라미터 수 × 비트 수로 계산한 값이며 실제 파일 크기나 최대 RAM 사용량이 아닙니다. 양자화 메타데이터, 일부 비양자화 계층, KV 캐시, 실행 버퍼가 추가됩니다. 이 모델의 FP16 KV 캐시는 배치 1·2,048토큰에서 약 1GiB가 추가될 수 있습니다. 통합 메모리는 CPU·GPU·운영체제가 공유합니다.

Apple Silicon용 [MLX LM](https://github.com/ml-explore/mlx-lm) 또는 Metal을 지원하는 [llama.cpp](https://github.com/ggml-org/llama.cpp)의 4bit 변환·추론 경로를 검증 후보로 삼습니다. 두 도구가 양자화를 지원한다는 사실이 KoreanLM 체크포인트·토크나이저 변환 성공을 보장하지는 않습니다. 변환 과정은 양자화된 모델을 실행할 때보다 더 많은 메모리가 필요할 수 있습니다.

[bitsandbytes 공식 지원 목록](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)에는 현재 Apple MPS가 포함되어 있지 않습니다. 기존 코드에 `load_in_4bit=True`만 추가하면 Mac에서 동작한다고 가정하지 않습니다.

검증 순서:

1. KoreanLM 가중치·토크나이저의 MLX 또는 GGUF 변환 호환성을 확인합니다.
2. 4bit 모델로 짧은 영어 문장을 생성하고 최대 메모리, 스왑 증가, 생성 속도를 측정합니다.
3. STT 프로세스 종료 후 번역을 실행해 두 모델이 동시에 메모리를 점유하지 않도록 합니다.
4. 5~10분 구간의 번역 품질과 용어 일관성을 평가합니다. 모델이 실행되는 것과 번역 품질 통과는 별도로 판정합니다.

원본 다운로드·형식 변환·4bit 양자화와 짧은 문장 3회 추론을 수행했습니다. 영상 전체 처리 시간과 5~10분 번역 품질은 아직 측정하지 않았습니다.

근거: [KoreanLM 모델 설정](https://huggingface.co/quantumaikr/KoreanLM/blob/main/config.json). 입력·출력 길이 제한과 파라미터 수 추정에 사용했습니다.

### MLX 4bit 실행 환경 (macOS Apple Silicon)

현재 실행 환경은 Python 3.9, MLX 0.29.3, mlx-lm 0.29.1입니다. 설치된 정확한 패키지는 `requirements-mlx.txt`에 기록합니다. STT 및 기존 학습 환경과 분리합니다.

```bash
python3 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx.txt
.venv-mlx/bin/python -c 'import mlx.core as mx; print(mx.array([1, 2]) + 1)'
```

이 경로는 Apple Silicon의 Metal GPU용입니다. Windows/Linux용 STT 실행 방법은 2절을 따르며, 이 MLX 실행 명령을 그대로 적용하지 않습니다.

## 4. KoreanLM 4bit 변환 및 번역

### 최초 준비

원본 KoreanLM은 PyTorch `.bin`으로 배포되어 현재 MLX 로더에 바로 넣을 수 없습니다. `scripts/prepare_koreanlm.py`가 원본 리비전을 고정해 다운로드하고 조각별로 FP16 safetensors로 변환합니다. torch는 이 형식 변환에 사용합니다.

```bash
.venv-mlx/bin/python scripts/prepare_koreanlm.py
.venv-mlx/bin/mlx_lm.convert \
  --hf-path ./models/koreanlm-fp16 \
  --mlx-path ./models/koreanlm-4bit \
  --quantize --q-bits 4 --q-group-size 64 --dtype float16
```

- 원본: `quantumaikr/KoreanLM`, 리비전 `f4351abcdd6a933afbaffad0badf60c273e71920`.
- 로컬 원본 캐시: `.cache/huggingface`, FP16 중간 파일: `models/koreanlm-fp16`, 최종 4bit: `models/koreanlm-4bit`.
- 원본·중간 파일·최종 파일을 모두 보관하므로 약 45GB 이상의 여유 공간을 확보합니다. 작업 중 추가 공간이 필요할 수 있습니다.
- 다운로드가 중단되면 준비 명령을 다시 실행합니다. 완성된 safetensors 조각은 건너뜁니다.
- MLX 변환은 대상 폴더가 이미 있으면 중단합니다. 재변환할 때는 새 `--mlx-path`를 지정하세요.
- 가중치와 가상환경은 Git에서 제외합니다. 원본 모델 업로드나 학습은 수행하지 않습니다.

### 한 문장 번역

```bash
.venv-mlx/bin/python translate_mlx.py \
  'The net force acting on the object is zero.' \
  --max-tokens 128
```

위 영어 문장은 동작 확인용 예시이며 샘플 영상에서 추출한 문장이 아닙니다. STT 결과의 시간 표기를 제외한 영어 문장으로 교체합니다. `--context '앞뒤 영어 문장'`으로 참고 문맥을 추가할 수 있습니다. 입력과 출력 합계가 2,048토큰을 넘으면 중단합니다.

모델은 **KoreanLM 기본 가중치의 4bit 버전**입니다. 별도 LoRA 어댑터를 적용하지 않습니다. 로딩·생성 시간과 MLX 최대 메모리를 콘솔에서 확인할 수 있습니다. MLX의 최대 메모리는 macOS 전체 메모리 사용량과 다릅니다.

### 기존 generate.py와의 차이

현재 `generate.py`는 원본 프로젝트의 Gradio·LoRA 추론 코드입니다. 다음 사항을 정리하기 전에는 첫 실행 경로로 사용하지 않습니다.

- `PeftModel.from_pretrained()`로 별도 `quantumaikr/KoreanLM-LoRA` 어댑터를 로딩합니다. 어댑터 접근 가능 여부와 기본 모델 호환성 확인이 필요합니다.
- `koreanlm.push_to_hub('KoreanLM-LoRA')`가 들어 있어 실행 도중 업로드를 시도합니다.
- Gradio 구형 API와 기본 공유 설정을 사용합니다.
- CPU에서도 마지막에 `.half()`를 호출합니다. 새 실행 경로는 `translate_mlx.py`를 사용합니다.

### 문제 해결

| 증상 | 확인할 내용 |
| --- | --- |
| 메모리 부족 / 프로세스 종료 | 짧은 입력으로도 모델 로딩이 실패하면 더 큰 메모리의 장비가 필요합니다. 생성 길이를 줄이는 것만으로 가중치 메모리는 줄지 않습니다. |
| Metal 초기화 실패 | Apple Silicon 네이티브 Python과 GPU 접근 권한을 확인합니다. |
| 모델 다운로드 실패 | 네트워크, 디스크 공간, 모델 저장소 접근 상태를 확인합니다. |
| 번역 대신 반복·설명이 나옴 | 기본 모델의 지시 수행 한계일 수 있습니다. 출력과 프롬프트를 기록하고 평가합니다. |
| 템플릿 파일을 찾지 못함 | 현재 디렉터리가 `K-aption_BE`인지 확인합니다. |

## 5. 이후 작업

1. 첨부 샘플로 영어 STT를 실행하고 원문과 시간을 검수합니다.
2. KoreanLM 최소 예제를 실행해 장비, 패키지 버전, 로딩·생성 시간과 번역 결과를 기록합니다.
3. 용어집 없는 번역을 기준선으로 보관합니다.
4. 실제 강의에서 전공 용어 20개를 선정하고 한국어 표준 번역어를 검수합니다.
5. 문장만 / 앞뒤 문맥 포함 / 문맥·용어집 포함 조건을 비교합니다.
6. 구간 ID·원문·번역·시간을 공통 데이터로 관리하고 API, SRT/VTT, 재생 기능을 연결합니다.

## MLX 4bit 검증 결과 (2026-09-19)

최종 모델 약 3.5GiB, group size 64, affine 4bit. 양자화 보조 데이터 포함 평균 4.5비트/가중치입니다. Python 3.9 / MLX 0.29.3 / mlx-lm 0.29.1에서 실행했습니다.

| 입력 | 실제 출력 | 생성 속도 | MLX 최대 메모리 | 판단 |
| --- | --- | --- | --- | --- |
| The net force acting on the object is zero. | 입력된 문장 … 물체의 중력이 없습니다. … | 18.06 토큰/초 | 4.230GB | 알짜힘을 중력으로 오역. 설명도 추가함 |
| The velocity is constant. | 속도는 일정합니다. | 19.89 토큰/초 | 4.226GB | 짧은 문장 번역 성공 |
| 첫 문장 + 알짜힘을 설명하는 영어 문맥 | 참고 문맥: 번역할 대상: 물체의 움직임은 네트 포스 또는 네트 포스의 합이 0인 경우입니다. | 21.82 토큰/초 | 4.407GB | 의미·용어·출력 형식 모두 추가 개선 필요 |

모델 로딩은 1.73~1.99초였습니다. 수치는 짧은 문장 3회의 관측값이며 전체 강의 성능을 보장하지 않습니다. 최대 메모리는 MLX가 보고한 값으로 OS 전체 RAM·스왑을 포함하지 않습니다. OS 스왑 증가량은 측정하지 않았습니다.

**실행 가능성은 확인했지만 번역 품질 기준은 통과하지 않았습니다.** 비양자화 모델과 동일 입력 비교를 하지 않았으므로 오류가 양자화 때문인지 원래 모델의 한계인지 구분할 수 없습니다. 다음 작업은 프롬프트·용어집 검증과 5~10분 평가입니다.

원시 로그는 로컬 `outputs/koreanlm-4bit-smoke.txt`, `outputs/koreanlm-4bit-velocity.txt`, `outputs/koreanlm-4bit-context.txt`에 저장했습니다. 로그는 Git에서 제외하며 위 표에 결과를 요약했습니다.

원본 config의 BOS=0/EOS=1과 토크나이저의 BOS=1/EOS=2가 달라, 변환본 config와 generation_config를 실제 토크나이저 기준으로 정정했습니다. 토크나이저 어휘와 모델 가중치를 새로 학습하거나 변경하지 않았습니다. 가중치에는 형식·정밀도 변환만 적용했습니다.

원본 캐시 약 25GiB와 FP16 중간 파일 약 13GiB를 보관 중입니다. 반복 추론에는 `models/koreanlm-4bit`만 사용합니다.

## 원본 및 참고 자료

- [KoreanLM 원본 코드](https://github.com/quantumaikr/KoreanLM)
- [KoreanLM 모델 카드](https://huggingface.co/quantumaikr/KoreanLM)
- [모델 설정](https://huggingface.co/quantumaikr/KoreanLM/blob/main/config.json)
- [원본 LoRA 어댑터 주소](https://huggingface.co/quantumaikr/KoreanLM-LoRA) — 이번 확인에서는 접근 여부를 검증하지 못함

기존 KoreanLM 기반 코드의 라이선스는 저장소의 `LICENSE`를 확인하세요. 원본 출처와 고지를 유지합니다. 강의 영상·생성 자막의 이용 조건은 코드와 별도로 확인하고 기록할 예정입니다.
