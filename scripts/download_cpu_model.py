"""고정 GitHub Release의 모델 조각을 받아 크기·SHA256을 검증하고 CPU 모델을 원자적으로 저장한다."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docker/koreanlm-model.json'


def file_digest(path: Path) -> str:
    """기존 모델 파일을 작은 청크로 읽어 SHA256을 계산한다.

    :param path: 검사할 모델 파일 경로.
    :return: SHA256 16진수 문자열.
    :raises OSError: 파일을 읽을 수 없는 경우.
    """
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(2 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    """다운로드 명세의 형식·파일명·크기·체크섬과 공개 HTTPS 주소를 검사한다.

    :param path: 저장소에서 관리하는 JSON 명세 경로.
    :return: 검증된 모델 배포 명세.
    :raises ValueError: 명세가 지원 형식과 다르거나 잘못된 경우.
    :raises OSError: 명세 파일을 읽을 수 없는 경우.
    """
    manifest = json.loads(path.read_text(encoding='utf-8'))
    entries = [manifest, *manifest['parts']]
    if (manifest['schema_version'] != 1 or not manifest['parts']
            or not manifest['release_url'].startswith('https://github.com/')
            or any(type(entry['bytes']) is not int or entry['bytes'] <= 0
                   or len(entry['sha256']) != 64
                   or any(char not in '0123456789abcdef' for char in entry['sha256'])
                   for entry in entries)
            or sum(part['bytes'] for part in manifest['parts']) != manifest['bytes']
            or len({part['name'] for part in manifest['parts']}) != len(manifest['parts'])
            or any(not name or name in ('.', '..')
                   or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-'
                          for char in name)
                   for name in [manifest['filename'], *(part['name'] for part in manifest['parts'])])):
        raise ValueError('Invalid CPU model release manifest')
    return manifest


@contextmanager
def model_lock(path: Path):
    """같은 저장 위치로의 다운로드를 Linux 컨테이너의 배타 잠금으로 직렬화한다.

    :param path: 모델과 같은 볼륨 안의 잠금 파일 경로.
    :return: 컨텍스트 관리자. 진입 시 제공하는 값은 None.
    :yield: 배타 잠금을 얻은 뒤 None을 한 번 제공한다.
    :raises OSError: 잠금 파일 생성·잠금 획득에 실패한 경우.
    """
    import fcntl

    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ensure_model(manifest: dict, target: Path) -> Path:
    """모델이 없으면 조각을 스트리밍 결합하고 개별·전체 검증 후 저장한다.

    :param manifest: load_manifest로 검증한 고정 배포 명세.
    :param target: Docker 영구 볼륨 내 모델 출력 경로.
    :return: 크기·SHA256이 일치하는 모델 경로.
    :raises ValueError: 기존 파일 또는 다운로드 크기·체크섬이 다른 경우.
    :raises OSError: 네트워크·디스크·잠금 작업에 실패한 경우.

    부모 디렉터리와 잠금 파일을 만든다. 기존 정상 모델은 다운로드 없이 재사용하고,
    기존 손상 모델은 보존한 채 실패한다. 실패한 다운로드 임시 파일은 삭제한다.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with model_lock(target.with_suffix(target.suffix + '.lock')):
        if target.exists():
            if target.stat().st_size != manifest['bytes'] or file_digest(target) != manifest['sha256']:
                raise ValueError(f'Existing model failed verification; remove or move it first: {target}')
            print(f'Model verified, using cached file: {target}', flush=True)
            return target
        print(f'Downloading CPU model ({manifest["bytes"] / 1024 ** 3:.2f} GiB); first run only.', flush=True)
        with tempfile.TemporaryDirectory(prefix='.model-download-', dir=target.parent) as directory:
            temporary = Path(directory) / 'model.partial'
            whole = hashlib.sha256()
            with temporary.open('wb') as output:
                for part in manifest['parts']:
                    print(f'Downloading {part["name"]}', flush=True)
                    request = Request(f'{manifest["release_url"]}/{part["name"]}',
                                      headers={'User-Agent': 'K-aption-model-installer/1'})
                    digest = hashlib.sha256()
                    size = 0
                    with urlopen(request, timeout=60) as response:
                        while chunk := response.read(2 * 1024 * 1024):
                            size += len(chunk)
                            if size > part['bytes']:
                                raise ValueError(f'Model part exceeded expected size: {part["name"]}')
                            digest.update(chunk)
                            whole.update(chunk)
                            output.write(chunk)
                    if size != part['bytes'] or digest.hexdigest() != part['sha256']:
                        raise ValueError(f'Model part verification failed: {part["name"]}')
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != manifest['bytes'] or whole.hexdigest() != manifest['sha256']:
                raise ValueError('Combined CPU model verification failed')
            os.replace(temporary, target)
        print(f'CPU model ready: {target}', flush=True)
        return target


def main() -> None:
    """명령행 설정을 읽고 검증된 CPU 모델을 준비하며 실패 시 서버 시작을 막는다.

    :return: None.
    :raises ValueError: 명세·다운로드·기존 모델 검증에 실패한 경우.
    :raises OSError: 파일·네트워크 작업에 실패한 경우.
    """
    parser = argparse.ArgumentParser(description='Download and verify the pinned CPU model release')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path,
                        default=Path(os.getenv('KAPTION_MODEL_DIR', '/data/models/koreanlm-q4_1.gguf')))
    args = parser.parse_args()
    ensure_model(load_manifest(args.manifest), args.output)


if __name__ == '__main__':
    main()
