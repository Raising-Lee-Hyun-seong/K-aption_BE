"""고정된 공개 Qwen3 MLX 4bit 파일을 검증하며 내려받아 Mac API 모델을 준비한다."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_cpu_model import ensure_model


def prepare(directory: Path, manifest_path: Path) -> None:
    """파일별 크기·SHA256을 고정한 MLX 모델을 원자적으로 다운로드한다.

    :param directory: 모델·토크나이저를 저장할 Mac 로컬 디렉터리.
    :param manifest_path: 공개 저장소 리비전과 파일별 검증값을 담은 JSON 경로.
    :return: None. 정상 파일은 재사용하고 새 파일은 다운로드 후 검증한다.
    :raises ValueError: 파일명·기존 파일·다운로드의 무결성이 잘못된 경우.
    :raises OSError: 네트워크·파일 작업에 실패한 경우.
    """
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    base = f'https://huggingface.co/{source["repository"]}/resolve/{source["revision"]}'
    for entry in source['files']:
        name = entry['name']
        if not name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('MLX model manifest must contain flat filenames')
        manifest = {'release_url': base, 'bytes': entry['bytes'],
                    'sha256': entry['sha256'], 'parts': [entry]}
        ensure_model(manifest, directory / name)


def main() -> None:
    """선택한 경로에 Qwen3 MLX 모델을 준비한다.

    :return: None.
    :raises ValueError: 모델 파일 무결성 검증에 실패한 경우.
    :raises OSError: 모델 파일 준비에 실패한 경우.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/qwen3-4b-instruct-2507-4bit')
    args = parser.parse_args()
    prepare(args.output, ROOT / 'data/qwen3_mlx.json')


if __name__ == '__main__':
    main()
