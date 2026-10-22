"""모델 자동 설치의 조각·전체 무결성, 캐시 재사용과 실패 시 파일 보존을 검증한다."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from scripts.download_cpu_model import ensure_model, load_manifest


class ModelDownloadTests(unittest.TestCase):
    """작은 바이너리 모델로 실제 파일 결합과 실패 처리를 검사한다."""

    def setUp(self):
        """분할 바이너리와 고정 체크섬 명세 및 임시 출력 위치를 준비한다.

        :return: None.
        """
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target = self.root / 'models/test.gguf'
        self.parts = [b'GGUF\x00\xff', b'weight\x00bytes']
        self.payload = b''.join(self.parts)
        self.manifest = {'schema_version': 1, 'filename': 'test.gguf',
                         'release_url': 'https://github.com/example/model/releases/download/v1',
                         'bytes': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest(),
                         'parts': [{'name': f'model.part{index}', 'bytes': len(part),
                                    'sha256': hashlib.sha256(part).hexdigest()}
                                   for index, part in enumerate(self.parts)]}

    def assert_no_partial_output(self):
        """실패한 설치가 최종 모델이나 부분 다운로드 파일을 남기지 않는지 확인한다.

        :return: None.
        """
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.glob('.model-download-*')), [])

    def test_download_combines_verified_parts_and_reuses_cache_offline(self):
        """조각을 순서대로 합치고 다음 실행에서는 네트워크 없이 검증된 모델을 쓴다.

        :return: None.
        """
        with patch('scripts.download_cpu_model.urlopen',
                   side_effect=[io.BytesIO(part) for part in self.parts]) as request:
            self.assertEqual(ensure_model(self.manifest, self.target), self.target)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(request.call_args_list[0].args[0].full_url.endswith('/model.part0'))
        with patch('scripts.download_cpu_model.urlopen') as request:
            ensure_model(self.manifest, self.target)
        request.assert_not_called()

    def test_bad_part_size_or_digest_never_installs_model(self):
        """짧거나 과도한 조각과 같은 크기의 손상 조각을 모두 거부한다.

        :return: None.
        """
        for bad in (b'x', b'x' * 100, b'x' * len(self.parts[0])):
            with self.subTest(part=bad), patch('scripts.download_cpu_model.urlopen',
                                             return_value=io.BytesIO(bad)):
                with self.assertRaises(ValueError):
                    ensure_model(self.manifest, self.target)
            self.assert_no_partial_output()

    def test_wrong_combined_digest_never_installs_model(self):
        """모든 조각이 맞더라도 전체 SHA256이 다르면 최종 저장을 막는다.

        :return: None.
        """
        self.manifest['sha256'] = '0' * 64
        with patch('scripts.download_cpu_model.urlopen',
                   side_effect=[io.BytesIO(part) for part in self.parts]):
            with self.assertRaisesRegex(ValueError, 'Combined'):
                ensure_model(self.manifest, self.target)
        self.assert_no_partial_output()

    def test_interrupted_download_cleans_up_and_can_retry(self):
        """중간 네트워크 실패의 임시 파일을 정리하고 다음 실행에서 재설치한다.

        :return: None.
        """
        with patch('scripts.download_cpu_model.urlopen',
                   side_effect=[io.BytesIO(self.parts[0]), URLError('offline')]):
            with self.assertRaises(URLError):
                ensure_model(self.manifest, self.target)
        self.assert_no_partial_output()
        with patch('scripts.download_cpu_model.urlopen',
                   side_effect=[io.BytesIO(part) for part in self.parts]):
            ensure_model(self.manifest, self.target)
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_corrupt_existing_model_is_preserved_without_download(self):
        """기존 모델 손상은 자동 덮어쓰기 없이 명시적 오류로 처리한다.

        :return: None.
        """
        self.target.parent.mkdir()
        self.target.write_bytes(b'x' * len(self.payload))
        with patch('scripts.download_cpu_model.urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'Existing model'):
                ensure_model(self.manifest, self.target)
        request.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b'x' * len(self.payload))

    def test_manifest_rejects_bad_sizes_hashes_names_and_urls(self):
        """잘못된 크기·해시·경로·주소가 있는 명세를 다운로드 전에 거부한다.

        :return: None.
        """
        invalid = []
        for key, value in (('bytes', 0), ('sha256', 'invalid'),
                           ('filename', '../test.gguf'), ('release_url', 'http://github.com/x')):
            manifest = deepcopy(self.manifest)
            manifest[key] = value
            invalid.append(manifest)
        for name in ('../part', 'part?query', 'part\\name'):
            manifest = deepcopy(self.manifest)
            manifest['parts'][0]['name'] = name
            invalid.append(manifest)
        path = self.root / 'manifest.json'
        for manifest in invalid:
            with self.subTest(manifest=manifest):
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_checked_in_manifest_pins_original_model_and_release(self):
        """실제 배포 명세가 기존 모델과 전체 크기·체크섬을 고정하는지 검사한다.

        :return: None.
        """
        root = Path(__file__).resolve().parents[1]
        manifest = load_manifest(root / 'evaluation/archive/koreanlm-model.json')
        self.assertEqual(manifest['model'], 'quantumaikr/KoreanLM')
        self.assertEqual(manifest['bytes'], 4213149728)
        self.assertEqual(manifest['sha256'], 'f4e6b85689909fa69289dd67d4c6f67ecc95fb8400eb95b77ff9be57f0eed70e')
        self.assertEqual(len(manifest['parts']), 3)
        self.assertTrue(all(part['bytes'] < 2 * 1024 ** 3 for part in manifest['parts']))
