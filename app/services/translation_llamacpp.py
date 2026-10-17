"""동일 KoreanLM의 GGUF 가중치를 CPU에서 생성하고 기존 번역 검증·재시도를 공유한다."""

from pathlib import Path

from app.services.translation import TranslationService
from app.services.translation_validation import generation_problem
from timing import measure


class LlamaTokenizer:
    """기존 번역 서비스의 토큰 수 계산을 GGUF SentencePiece 토크나이저에 연결한다."""

    def __init__(self, model):
        """GGUF 모델의 토크나이저를 보관한다.

        :param model: tokenize 메서드를 제공하는 llama_cpp.Llama 객체.
        :return: None.
        """
        self.model = model

    def encode(self, text: str) -> list[int]:
        """기존 BOS 포함 정책으로 프롬프트를 토큰화한다.

        :param text: 공통 템플릿으로 만든 한국어 지시와 영어 원문.
        :return: BOS를 포함한 토큰 ID 목록.
        """
        return self.model.tokenize(text.encode('utf-8'), add_bos=True, special=True)


class LlamaCppTranslationService(TranslationService):
    """프롬프트·문맥·검증·재시도는 유지하고 토큰 생성만 llama.cpp CPU로 수행한다."""

    def __init__(self, model_path: Path, template_path: Path, threads: int = 4):
        """기존 모델을 재포장한 GGUF를 메모리 매핑하고 CPU 전용 추론을 준비한다.

        :param model_path: 준비 스크립트로 만든 기존 KoreanLM의 Q4_1 GGUF 파일 경로.
        :param template_path: 공통 번역 프롬프트 JSON 경로.
        :param threads: 생성·프롬프트 평가에 사용할 CPU 스레드 수. 양의 정수, 기본 4.
        :return: None.
        :raises FileNotFoundError: GGUF 파일이 없는 경우.
        :raises ValueError: 스레드 수·모델 로딩 또는 모델 구조가 유효하지 않은 경우.
        :raises OSError: GGUF·공통 템플릿·용어집 읽기에 실패한 경우.

        GPU 계층·KQV·연산 오프로딩을 비활성화하고 컨텍스트를 2048토큰으로 제한한다.
        """
        if threads < 1:
            raise ValueError('CPU thread count must be positive')
        if not model_path.is_file():
            raise FileNotFoundError('GGUF model is missing; run scripts/prepare_koreanlm_gguf.py first')
        from llama_cpp import Llama
        with measure('translation_model_load', backend='llama-cpp', device='cpu'):
            self.model = Llama(model_path=str(model_path), n_ctx=2048, n_gpu_layers=0,
                               offload_kqv=False, op_offload=False,
                               n_threads=threads, n_threads_batch=threads,
                               n_batch=256, n_ubatch=128, use_mmap=True, use_mlock=False,
                               seed=0, verbose=False)
        if self.model.metadata.get('general.architecture') != 'llama':
            self.model.close()
            raise ValueError('Expected a Llama architecture GGUF')
        self.tokenizer = LlamaTokenizer(self.model)
        try:
            self._configure_text(template_path)
        except (OSError, ValueError):
            self.model.close()
            raise

    def close(self) -> None:
        """모든 추론 종료 후 CPU 모델의 메모리 매핑과 네이티브 컨텍스트를 해제한다.

        :return: None.
        """
        self.model.close()

    def _generate_output(self, prompt: str, max_tokens: int) -> dict:
        """탐욕 생성으로 토큰을 읽고 기존 이상 탐지와 EOS·길이 종료 정보를 제공한다.

        :param prompt: 기존 템플릿·문맥·용어집으로 만든 프롬프트 문자열.
        :param max_tokens: 한 번 생성할 최대 토큰 수. 양의 정수.
        :return: 기존 MLX 서비스와 같은 text·tokens·finish_reason·stop_reason 사전.
        :raises ValueError: 최대 토큰 수가 양수가 아니거나 추론 입력이 잘못된 경우.

        요청마다 KV 상태를 초기화한다. 부분 UTF-8 토큰은 완성 전 이상 문자로 판정하지
        않으며 최종 불완전 바이트는 대체 문자로 남겨 공통 검증에서 거부한다.
        """
        if max_tokens < 1:
            raise ValueError('Maximum generated tokens must be positive')
        self.model.reset()
        stream = self.model.generate(self.tokenizer.encode(prompt), temp=0.0,
                                     top_k=1, repeat_penalty=1.0)
        ids, count, finish, reason, output = [], 0, None, None, ''
        try:
            for token in stream:
                count += 1
                if token == self.model.token_eos():
                    finish = 'stop'
                    break
                ids.append(token)
                output = self.model.detokenize(ids).decode('utf-8', errors='ignore')
                reason = generation_problem(output)
                if reason:
                    break
                if count >= max_tokens:
                    finish = 'length'
                    break
        finally:
            close = getattr(stream, 'close', None)
            if close:
                close()
        output = self.model.detokenize(ids).decode('utf-8', errors='replace')
        return {'text': output, 'tokens': count, 'finish_reason': finish,
                'stop_reason': reason or generation_problem(output)}
