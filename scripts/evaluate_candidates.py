"""고정된 두 MLX 번역 후보를 검증·다운로드하고 같은 문장의 원시 출력·시간·메모리를 저장한다."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.translation_validation import generation_problem, preservation_problem


def sha256_file(path: Path) -> str:
    """파일을 분할해서 읽고 SHA256을 반환한다.

    :param path: 해시를 계산할 파일 경로.
    :return: 파일 내용의 SHA256 16진수 문자열.
    """
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: dict) -> str:
    """고정 메타데이터의 크기와 제공된 LFS 해시를 파일과 대조한다.

    :param path: 검증할 로컬 파일 경로.
    :param expected: bytes와 선택적 sha256이 있는 공개 파일 메타데이터.
    :return: 실제 파일의 SHA256 문자열.
    :raises ValueError: 크기 또는 제공된 SHA256이 다를 때.
    :raises FileNotFoundError: 파일이 없을 때.
    """
    if path.stat().st_size != expected['bytes']:
        raise ValueError(f'File size mismatch: {path.name}')
    actual = sha256_file(path)
    if expected.get('sha256') and actual != expected['sha256']:
        raise ValueError(f'Checksum mismatch: {path.name}')
    return actual


def load_cases() -> list:
    """공식 자막과 합성 물리학 입력을 출처·고유 식별자를 보존해 합친다.

    :return: 원문과 기준 번역·검토 항목을 담은 평가 사례 22개 목록.
    """
    official = json.loads((ROOT / 'evaluation/mit_rocket_cases.json').read_text())['cases']
    physics = json.loads((ROOT / 'evaluation/physics_cases.json').read_text())
    return [{**row, 'id': 'official:' + row['id'], 'suite': 'official'} for row in official] + [
        {**row, 'id': 'physics:' + row['id'], 'suite': 'physics'} for row in physics]


def build_prompt(tokenizer, candidate: dict, text: str) -> str:
    """모델별 채팅 템플릿으로 영어 원문의 한국어 번역 입력을 구성한다.

    :param tokenizer: apply_chat_template을 제공하는 로컬 토크나이저.
    :param candidate: 원본 모델 저장소가 명시된 후보 설정.
    :param text: 번역할 영어 원문. 앞 문맥과 용어집은 붙이지 않는다.
    :return: 생성 시작 표식까지 렌더링한 프롬프트 문자열.
    """
    if candidate['original_repository'].startswith('google/translategemma'):
        content = [{'type': 'text', 'source_lang_code': 'en', 'target_lang_code': 'ko', 'text': text}]
    else:
        content = ('Translate the following English text into Korean. Preserve negation, numbers '
                   'and mathematical relations. Output only the translation.\n\n' + text)
    return tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)


def inspect_output(source: str, output: str, finish_reason: str) -> list:
    """출력을 수정·거부하지 않고 형식·종료·숫자·수식 문제를 함께 기록한다.

    :param source: 보존 검사의 기준이 되는 영어 원문.
    :param output: 정리·재시도 전 모델의 원시 출력.
    :param finish_reason: MLX 종료 사유. stop 또는 length 등.
    :return: 발견한 문제 코드 목록. 빈 목록도 의미 정확도를 보장하지 않는다.
    """
    problems = []
    if finish_reason != 'stop':
        problems.append('token_limit' if finish_reason == 'length' else 'unknown_finish')
    if not re.search('[가-힣]', output):
        problems.append('no_hangul')
    for problem in (generation_problem(output), preservation_problem(source, output)):
        if problem:
            problems.append(problem)
    return problems


def configure_stops(tokenizer, candidate: dict) -> dict:
    """번역용 Gemma의 턴 종료 토큰을 실제 토큰 ID로 확인하고 중단 조건에 추가한다.

    :param tokenizer: MLX의 EOS 목록·토큰 사전·add_eos_token을 제공하는 토크나이저.
    :param candidate: 원본 모델 저장소가 명시된 후보 설정.
    :return: 기본·최종 EOS ID 목록과 추가한 종료 표식 사전.
    :raises ValueError: Gemma의 종료 표식이 사전에 없거나 예상 ID와 다를 때.

    런타임 토크나이저의 종료 조건만 수정하며 다운로드한 파일은 변경하지 않는다.
    """
    before = sorted(tokenizer.eos_token_ids)
    added = []
    if candidate['original_repository'].startswith('google/translategemma'):
        if tokenizer.get_vocab().get('<end_of_turn>') != 106:
            raise ValueError('Unexpected Gemma end-of-turn token')
        tokenizer.add_eos_token('<end_of_turn>')
        added.append('<end_of_turn>')
    return {'before': before, 'after': sorted(tokenizer.eos_token_ids), 'added': added}


def model_overrides(candidate: dict, config: dict) -> dict:
    """Gemma 배포 설정의 새 RoPE 필드를 설치된 MLX가 읽는 기존 필드로 대응한다.

    :param candidate: 원본 모델 저장소가 명시된 후보 설정.
    :param config: 다운로드한 모델의 원래 설정 사전. 직접 수정하지 않는다.
    :return: 필요한 경우 text_config를 포함하는 로더용 덮어쓰기 사전.
    :raises ValueError: 새 RoPE 방식이 미지원이거나 기존 필드와 충돌할 때.

    파일·가중치·배율을 바꾸지 않고 명시된 linear factor를 rope_scaling에 전달한다.
    """
    if not candidate['original_repository'].startswith('google/translategemma'):
        return {}
    text = copy.deepcopy(config['text_config'])
    full = text.get('rope_parameters', {}).get('full_attention')
    if not full:
        return {}
    if full.get('rope_type') != 'linear' or not isinstance(full.get('factor'), (int, float)) or full['factor'] <= 0:
        raise ValueError('Unsupported Gemma RoPE parameters')
    mapped = {'type': 'linear', 'factor': full['factor']}
    if text.get('rope_scaling'):
        old = text['rope_scaling']
        if (old.get('type', old.get('rope_type')) != 'linear' or old.get('factor') != full['factor']):
            raise ValueError('Conflicting Gemma RoPE parameters')
    text['rope_scaling'] = mapped
    return {'text_config': text}


def save_report(path: Path, report: dict) -> None:
    """진행 중 결과도 보존하도록 JSON을 임시 파일에 쓴 뒤 교체한다.

    :param path: 저장할 보고서 경로.
    :param report: 원시 출력·설정·시간 등을 담은 JSON 직렬화 가능한 사전.
    :return: None.

    대상 파일을 갱신하고 필요한 부모 디렉터리를 생성한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
