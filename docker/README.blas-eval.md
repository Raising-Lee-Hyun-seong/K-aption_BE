# OpenBLAS 비교 이미지

운영 `Dockerfile.translator`, Compose, 모델, 프롬프트를 변경하지 않는 실험 전용 빌드다.
`Dockerfile.translator-blas-eval`의 두 타깃은 같은 소스·도구·CPU 커널을 사용하며
`GGML_BLAS=OFF/ON`만 다르다. 두 런타임 모두 같은 OpenBLAS 패키지를 설치한다.
OFF 이미지는 BLAS 백엔드를 빌드하지 않는다.

## 소스 준비

저장소 전체 대신 소스 아카이브 한 개만 있는 디렉터리를 빌드 컨텍스트로 사용한다.
환경 파일과 모델 파일을 컨텍스트에 넣지 않는다. 아래 명령은 저장소 루트에서 실행한다.

```sh
mkdir -p /private/tmp/kaption-blas-build-context-261029
curl -fL --retry 2 \
  https://codeload.github.com/ggml-org/llama.cpp/tar.gz/339f3242c761f16c27e5ffcde1d35dcc31933a55 \
  -o /private/tmp/kaption-blas-build-context-261029/kaption-llama-source-339f3242c.tar.gz
```

Dockerfile은 SHA256 `901a2a2b2eb389a08cae3fec3269914f5c03e8d39b1352db9a29cbd1886727f3`를
검증한 뒤 압축을 푼다. 기반 Ubuntu 이미지도 digest로 고정한다.
APT 패키지는 첫 빌드 당시 버전을 사용하므로 완성 이미지의 ID와
`/app/build-info/packages.txt`를 함께 보존한다. 나중에 캐시 없이 재빌드하면
패키지 버전이 달라질 수 있다.

## ARM64 네이티브 빌드

```sh
docker build --platform linux/arm64 --target blas-off \
  -f docker/Dockerfile.translator-blas-eval \
  -t kaption-translator-blas:off-339f3242c \
  /private/tmp/kaption-blas-build-context-261029

docker build --platform linux/arm64 --target blas-on \
  -f docker/Dockerfile.translator-blas-eval \
  -t kaption-translator-blas:on-339f3242c \
  /private/tmp/kaption-blas-build-context-261029
```

컴파일 작업 수 기본값은 2이며 ON 타깃은 OFF 빌드의 공통 산출물을 재사용한다.
AMD64 머신에서는 플랫폼을 `linux/amd64`로 바꾸고 별도 이미지 태그를 사용한다.
ARM에서 AMD64 에뮬레이션 성능을 Windows 성능으로 해석하지 않는다.
두 아키텍처 모두 `GGML_NATIVE=OFF`, `GGML_CPU_ALL_VARIANTS=ON`을 사용하며
아키텍처별 커널 선택은 llama.cpp에 맡긴다. GPU 백엔드는 비활성화한다.
API 비교에 필요 없는 내장 웹 UI는 두 이미지 모두 빌드하지 않는다.

## 모델을 읽지 않는 준비 확인

```sh
docker run --rm --network none --read-only --cpus 2 --memory 256m \
  -e OPENBLAS_NUM_THREADS=8 -e OMP_NUM_THREADS=8 \
  kaption-translator-blas:off-339f3242c --list-devices
docker run --rm --network none --read-only --cpus 2 --memory 256m \
  -e OPENBLAS_NUM_THREADS=8 -e OMP_NUM_THREADS=8 \
  kaption-translator-blas:on-339f3242c --list-devices
```

ON의 BLAS 장치는 CPU 연산 라이브러리이며 GPU를 의미하지 않는다.
소스 아카이브에는 `.git`이 없으므로 `--version`의 빌드 커밋은 `unknown`으로
나올 수 있다. 출처는 이미지 revision 라벨과 검증된 소스 SHA256으로 확인한다.
각 이미지의 `/app/build-info/CMakeCache.txt`에서 실제 설정을 확인할 수 있다.

준비 확인은 모델 로딩·번역 성공·실제 BLAS 연산 배정·성능 개선을 증명하지 않는다.
다음 단계에서는 같은 Q4_K_M 모델, CPU 8개, 스레드 8개, 메모리 4GB,
`--device none --n-gpu-layers 0 --no-kv-offload --no-op-offload --no-repack`,
기존 문맥·배치·검증 조건을 유지하고 대표 문장 비교부터 수행한다.
운영 공식 이미지와 새 OFF 이미지의 차이도 별도로 확인해야 한다.
