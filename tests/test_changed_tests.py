"""변경 범위별 테스트 선택과 Git 비교 및 제한 실행을 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_changed_tests as runner


class ChangedTestsTests(unittest.TestCase):
    """전체 테스트로 확장하지 않고 변경 범위만 실행하는 계약을 검사한다."""

    def test_documentation_has_no_test_selection(self):
        """문서만 바뀌면 테스트를 수집하지 않는지 확인한다.

        :return: None.
        """
        self.assertEqual(runner.select_tests(['README.md', 'evaluation/report.md']), [])

    def test_feature_and_direct_test_changes_are_deduplicated(self):
        """자막 구현과 테스트를 함께 바꿔도 자막 테스트만 한 번 선택한다.

        :return: None.
        """
        self.assertEqual(runner.select_tests([
            './app/services/subtitle_export.py', 'tests/test_subtitle_export.py']),
            ['tests/test_subtitle_export.py'])

    def test_shared_api_changes_select_affected_contracts(self):
        """공용 API 변경은 작업·업로드·자막 계약을 포함하고 모델 평가는 제외한다.

        :return: None.
        """
        selected = runner.select_tests(['app/main.py'])
        self.assertIn('tests/test_jobs.py', selected)
        self.assertIn('tests/test_subtitle_export.py', selected)
        self.assertIn('tests/test_media_limits.py', selected)
        self.assertNotIn('tests/test_lora_conversion.py', selected)
        self.assertNotIn('tests/test_qwen_lecture.py', selected)

    def test_unmapped_code_fails_instead_of_silently_skipping(self):
        """새 코드나 의존성 파일에는 명시적인 테스트 연결을 요구한다.

        :return: None.
        """
        for path in ('app/services/new_feature.py', 'requirements-new.txt', 'tests/helpers.py'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'OWNERS'):
                runner.select_tests([path])

    def test_deleted_tests_are_not_imported(self):
        """삭제된 테스트 경로를 수집 대상으로 남기지 않는지 검사한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(runner.select_tests(['tests/test_removed.py'], Path(directory)), [])

    def test_git_diff_handles_spaces_deletions_and_initial_push(self):
        """실제 임시 저장소에서 최초 push와 공백 경로 및 삭제를 읽는다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', directory], check=True)
            subprocess.run(['git', '-C', directory, 'config', 'user.email', 'ci@example.invalid'], check=True)
            subprocess.run(['git', '-C', directory, 'config', 'user.name', 'CI Test'], check=True)
            path = root / 'space name.md'
            path.write_text('first', encoding='utf-8')
            subprocess.run(['git', '-C', directory, 'add', '.'], check=True)
            subprocess.run(['git', '-C', directory, 'commit', '-qm', 'fixture'], check=True)
            base = subprocess.check_output(['git', '-C', directory, 'rev-parse', 'HEAD'], text=True).strip()
            self.assertEqual(runner.changed_files('0' * 40, base, root), ['space name.md'])
            path.unlink()
            (root / 'new.md').write_text('next', encoding='utf-8')
            subprocess.run(['git', '-C', directory, 'add', '-A'], check=True)
            subprocess.run(['git', '-C', directory, 'commit', '-qm', 'next'], check=True)
            head = subprocess.check_output(['git', '-C', directory, 'rev-parse', 'HEAD'], text=True).strip()
            self.assertEqual(runner.changed_files(base, head, root), ['new.md', 'space name.md'])

    def test_invalid_revision_is_rejected_before_git(self):
        """옵션이나 축약 참조가 Git 인자로 전달되지 않는지 확인한다.

        :return: None.
        """
        with patch.object(runner.subprocess, 'run') as git:
            with self.assertRaises(ValueError):
                runner.changed_files('--output=/tmp/unwanted', 'f' * 40)
            git.assert_not_called()

    def test_pull_request_uses_common_ancestor(self):
        """PR 비교가 대상 브랜치에만 추가된 변경을 테스트 범위에 넣지 않는다.

        :return: None.
        """
        with patch.object(runner.subprocess, 'run') as git:
            git.return_value.stdout = b'app/services/subtitle_export.py\0'
            self.assertEqual(runner.changed_files('a' * 40, 'b' * 40, merge_base=True),
                             ['app/services/subtitle_export.py'])
            self.assertIn('--merge-base', git.call_args.args[0])

    def test_cli_list_does_not_import_selected_tests(self):
        """의존성이 없는 기본 Python에서도 테스트 선택만 조회할 수 있다.

        :return: None.
        """
        result = subprocess.run([
            'python3', str(runner.ROOT / 'scripts/run_changed_tests.py'), '--files',
            'app/services/subtitle_export.py', '--list'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'tests/test_subtitle_export.py\n')

    def test_runner_collects_only_the_requested_file_and_reports_failure(self):
        """다른 테스트의 import 오류를 피하고 선택한 테스트 실패를 종료 코드로 전달한다.

        :return: None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / 'tests'
            tests.mkdir()
            (tests / 'test_unrelated_fixture.py').write_text(
                '"""수집되지 않아야 하는 대역이다."""\nraise RuntimeError("unrelated")\n', encoding='utf-8')
            (tests / 'test_selected_fixture.py').write_text(
                '"""선택 실행 실패를 확인하는 대역이다."""\nimport unittest\n'
                'class Selected(unittest.TestCase):\n'
                '    """실패를 반환하는 대역이다."""\n'
                '    def test_failure(self):\n'
                '        """실패를 발생시킨다.\n\n        :return: None.\n        """\n'
                '        self.fail("selected failure")\n', encoding='utf-8')
            output = io.StringIO()
            with patch.object(runner, 'ROOT', root), redirect_stderr(output):
                self.assertEqual(runner.run_tests(['tests/test_selected_fixture.py']), 1)
            self.assertIn('selected failure', output.getvalue())
            self.assertNotIn('unrelated', output.getvalue())

    def test_no_selection_does_not_start_loader(self):
        """대상 없는 변경이 테스트 전체 탐색으로 바뀌지 않는지 검사한다.

        :return: None.
        """
        with patch.object(unittest.TestLoader, 'discover') as discover, redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_tests([]), 0)
            discover.assert_not_called()
