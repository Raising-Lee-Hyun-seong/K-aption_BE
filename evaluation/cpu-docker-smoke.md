# Windows CPU 이식 검증

기준일: 2026-10-18. 대상은 GPU 없는 Galaxy Book3급 x86 Windows·RAM 16GB다. Mac 기본 MLX 실행은 유지하고 Windows Docker는 동일 KoreanLM 가중치를 llama.cpp CPU에서 실행한다. 실제 Galaxy Book3의 속도·긴 강의 성능은 해당 장비에서 별도로 확인해야 한다.

## 동일 모델 GGUF 재포장

MLX affine 4bit·group size 64의 정수·float16 배율·오프셋을 Q4_1의 두 개 32원소 블록에 재배치했다. 정수·배율·오프셋을 다시 산출하지 않는다. Q/K는 공식 llama.cpp Llama 변환과 같은 RoPE 행 순서를 적용한다. norm은 원래 float16 값을 정확하게 표현하는 float32로 기록한다.

- 원본: `models/koreanlm-4bit/model.safetensors`
- 원본 SHA256: `62f67af9551e3488d2fb3d2c05b862f541ae04bcbb177e264d5a7778b5192a0d`
- 결과: `models/koreanlm-gguf/model.gguf`, 4,213,149,728바이트(약 3.92GiB)
- 결과 SHA256: `f4e6b85689909fa69289dd67d4c6f67ecc95fb8400eb95b77ff9be57f0eed70e`
- 291개 텐서·6,738,415,616개 가중치 값을 처리했다. 모든 양자화 행을 작은 배치로 공식 GGUF 역양자화 결과와 정확히 비교했다.
- 전체 FP16 중간 모델을 만들거나 새 모델을 다운로드하지 않았다. 원본은 변경하지 않았다.
- 정상 프롬프트의 256개 토큰 ID가 기존 Hugging Face 토크나이저와 모두 일치했다. BOS=1·EOS=2·어휘·점수·토큰 종류를 기존 SentencePiece에서 보존했다.

이는 저장 가중치의 보존 검증이다. 엔진의 계산 순서·정밀도 차이 때문에 생성 결과 전체가 동일하다는 보장은 아니며 번역 의미 정확도 검수도 아니다.

## 실제 Mac CPU 예비 실행

Apple Silicon Mac에서 GPU 계층·KQV·연산 오프로딩을 모두 비활성화하고 llama-cpp-python 0.3.19 CPU 4스레드로 기존 공통 번역 서비스를 실행했다.

| 항목 | 단일 실행 결과 |
| --- | --- |
| 모델 로딩 | 7.43초 |
| 입력 | `The velocity is constant.` |
| 출력 | `속도는 일정합니다.` |
| 프롬프트 토큰 | 256개 |
| 생성 토큰 | 13개 |
| 검증·재시도 정책을 포함한 번역 호출 | 6.17초 |
| 프로세스 최대 RSS | 5,009.05MiB, 약 4.89GiB |

이 수치는 Windows·Galaxy Book3의 속도가 아니며 STT·긴 강의·반복 측정·전체 컨테이너 메모리 사용량을 포함하지 않는다. 파일 캐시·장비 상태에 따른 차이도 통제하지 않았다. 한 문장 통과를 일반 번역 품질이나 16GB 장비 전체 처리 안정성으로 확대 해석하지 않는다.

## Docker 구성

- Python 3.11·Debian Bookworm·Linux x86_64, CPU 4개·컨테이너 RAM 상한 7GB·미완료 작업 2개·서버 worker 1개다. 대상 WSL 메모리는 8GB 이상을 별도로 확보한다.
- llama-cpp-python 0.3.19를 같은 Debian C 라이브러리에 맞춰 OpenBLAS CPU 소스 빌드한다. AVX2·F16C·FMA CPU를 대상으로 하며 CUDA·Metal·Vulkan을 끈다. 컴파일 도구는 빌드 단계에만 둔다.
- 공개 Linux CPU wheel은 이 Debian 이미지에서 `libc.musl-x86_64.so.1` 로드 오류를 재현해 사용하지 않았다.
- CPU 실행 이미지는 MLX·PyTorch·Transformers를 설치하지 않는다. 프롬프트·문맥·용어집·숫자·수식 검증·한 번 재시도·부분 실패 계약은 기존 서비스에서 공유한다.
- 초기 검증에서는 기존 MLX 모델을 읽기 전용으로 마운트해 GGUF를 준비했다. 현재 배포 구성은 호스트 모델 마운트 없이 고정 GitHub Release에서 GGUF를 자동으로 받는다. GGUF·작업·Whisper 캐시는 Docker 볼륨이다. `.dockerignore` 허용 목록으로 비밀 파일·영상·가상환경·모델을 빌드 컨텍스트에서 제외한다. Compose는 비밀값 없는 `docker/cpu.defaults`를 명시해 프로젝트 `.env`를 자동으로 읽지 않는다.
- CPU 측정 CLI는 기본 300초 외부 제한과 프로세스 최대 RSS·검증 수락 여부·시간 보고서를 제공한다. 결과 수락은 의미 정확도 판정이 아니다.

## 변경 범위 테스트

CPU 설정·수명주기·보고서 10개, llama.cpp 생성·UTF-8·EOS·길이·공통 재시도 7개, GGUF 재포장 수치·손상·RoPE·GQA 5개를 통과했다. 기존 가중치 초기화 경계 3개와 모델 시작 오류 시 저장소 잠금 해제 1개도 해당 사례만 확인했다. 전체 테스트는 실행하지 않았다.

동일한 CPU 전용 22개를 최종 Linux x86 이미지 안에서도 실행해 통과했다. 실행 이미지의 `pip check`도 통과했다.

## Linux x86 실제 모델·API 연결

검증 호스트는 Apple Silicon·Docker VM 약 4GB이며 Linux x86 컨테이너는 에뮬레이션이다. 아래 결과는 Windows 성능 수치가 아니다.

- 최종 Debian 소스 빌드 이미지에서 전체 모델 번역이 성공했다. 4스레드·3500MiB 컨테이너 제한에서 로딩 2.78초, 번역 105.83초, 최대 RSS 3471.74MiB였다. 출력은 같은 `속도는 일정합니다.`, 256개 프롬프트·13개 생성 토큰이다. 메모리 압박·에뮬레이션·캐시 상태가 포함된 단일 실행이다.
- README와 같은 Compose 명령으로 **Linux 안에서도 전체 모델을 재포장**했다. 원본·결과 SHA256과 291개 텐서·6,738,415,616개 값이 Mac 변환과 동일했다. GGUF를 Docker 작업 볼륨에 저장하고 서버가 그 파일을 사용했다.
- 격리된 `127.0.0.1:18000` 서버에서 약 1.60초의 합성 영어 WAV를 실제 작업 API에 업로드했다. POST 202 → 실제 Whisper small CPU·int8 인식 → 기존 문장 묶기 → CPU 번역 → completed → 결과·한영 SRT/VTT 200까지 확인했다. 모델·STT 대역을 사용하지 않았다.
- 결과는 `start=0.0`, `end=1.4`, 영어 `The velocity is constant.`, 한국어 `속도는 일정합니다.`, `translation_error=null`인 구간 1개였다. 완료 상태 SSE 스냅샷과 STT 후 health의 두 모델 준비 상태도 확인했다.
- 서버 로그의 단계 시간은 번역 모델 로딩 2.374초, 최초 Whisper 로딩·다운로드 13.739초, STT 4.467초, 번역 48.816초(1회 시도), 작업 처리 67.316초였다. 클라이언트의 상태 조회 간격을 포함한 작업 관찰 시간은 68.92초다. 독립 번역 측정과 메모리 제한·캐시·실행 조건이 다르므로 두 수치로 성능 개선율을 계산하지 않는다.
- 전체 흐름 중 해당 컨테이너의 cgroup 메모리 최고값은 3,895,681,024바이트(약 3.63GiB), `oom=0`, `oom_kill=0`이었다. 장시간·최대 입력의 메모리 적합성 검증은 아니다. 작업 완료 후 검증 서버를 정상 종료했고 모델·결과·Whisper 캐시 볼륨은 남겼다.

## 초기 MLX CPU 실험과 남은 확인

처음에는 기존 MLX 0.29.3 CPU wheel과 mlx-lm 0.29.1을 그대로 쓰는 경로를 검증했다. Linux x86에서 작은 4bit 행렬 곱·SiLU JIT와 임시 소형 Llama 2토큰 생성까지 성공했다. SiLU JIT는 실행 시 g++가 필요해 초기 이미지의 컴파일러 누락도 재현·수정했다.

하지만 Mac에서 실제 기존 모델의 CPU 로딩·한 문장 번역 명령은 219초 관찰 시점에도 반환하지 않아 종료했다. 당시 호출 스택에서 `QuantizedMatmul::eval_cpu` 연산을 확인했다. 그 결과를 Windows 속도로 해석하거나 정확한 성능 개선 배율을 계산하지 않는다. 최종 Docker는 위의 llama.cpp 경로를 사용한다.

검증 호스트의 Docker는 ARM64·약 4GB 메모리여서 x86 컨테이너는 에뮬레이션이다. CPU 실행·재포장·짧은 실제 STT·번역·API 흐름은 확인했지만 실제 Windows에서 긴 영상·메모리·취소·재생 동기화와 해당 CPU의 처리 속도를 추가 확인해야 한다. 기존 KoreanLM의 의미 오류와 짧은 물리 강의에 편중된 품질 평가 한계는 이번 실행 환경 이식으로 해결되지 않았다.

근거: [공식 Llama 변환과 Q/K 행 순서](https://github.com/ggml-org/llama.cpp/blob/b6000/convert_hf_to_gguf.py), [공식 GGUF Q4_1 구현](https://github.com/ggml-org/llama.cpp/blob/b6000/gguf-py/gguf/quants.py), [llama-cpp-python CPU 빌드](https://github.com/abetlen/llama-cpp-python).


## GitHub 모델 자동 배포

모델 릴리스: https://github.com/Raising-Lee-Hyun-seong/K-aption_BE/releases/tag/koreanlm-q4_1-v1

동일 GGUF를 1,610,612,736바이트 조각 2개와 991,924,256바이트 조각 1개로 나눠 공개했다. GitHub API가 반환한 세 자산의 실제 크기와 SHA256이 `docker/koreanlm-model.json`의 고정 값과 모두 일치했다. 원본 Apache 2.0 라이선스·모델 카드·변경 고지·기존 변환 검증 보고서도 함께 배포했다.

모델 자동 설치 테스트 7개를 로컬과 최종 Linux CPU Docker 이미지에서 통과했다. 조각 결합, 다운로드 없는 캐시 재사용, 크기 초과·누락·조각 손상·전체 해시 불일치 거부, 네트워크 실패 후 정리와 재시도, 기존 손상 파일 보존, 명세 검증을 확인했다. 변경 테스트 선택기 테스트 11개도 통과했다. 전체 테스트 모음은 실행하지 않았다.

호스트 모델을 마운트하지 않고 로그인 정보 없이 빈 `kaption-bootstrap-smoke-v1` 볼륨에서 최종 CPU 이미지의 기본 시작 명령을 실행했다. 공개 Release에서 4,213,149,728바이트 전체를 다운로드하고 조각별·전체 검증 후 모델 로딩과 `/health` 200 응답까지 확인했다. 응답은 `translation_model_loaded=true`, 첫 STT 요청 전이므로 `transcription_model_loaded=false`였다. 모델 로딩은 단일 실행 1.859초였으며 다운로드 시간이나 Windows 속도를 나타내지 않는다.

서버를 정상 종료한 다음 같은 볼륨을 연결한 새 컨테이너를 `--network none`으로 실행했다. 설치 스크립트가 체크섬을 다시 검증하고 `Model verified, using cached file`로 성공해 네트워크 없는 재사용을 확인했다. 검증용 컨테이너와 새 시험 볼륨만 정리했으며 기존 작업·모델·Whisper 캐시 볼륨은 유지했다. 이번 검증은 Apple Silicon의 Linux x86 에뮬레이션이며 Windows 실기기 검증은 아니다.
