"""학습·추론에 필요한 JSON 입출력, 프롬프트 구성 및 콜백 기반 스트리밍 도구를 제공한다."""
import os
import io
import json

import os.path as osp
from typing import Union

import gc
import traceback
from queue import Queue
from threading import Thread

import torch
import transformers


def _make_w_io_base(f, mode: str):
    """경로의 부모 디렉터리를 준비해 쓰기 파일을 열거나 기존 파일 객체를 반환한다."""
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    """읽기 파일 경로를 파일 객체로 변환하거나 기존 파일 객체를 반환한다."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """딕셔너리·리스트를 JSON으로 또는 문자열을 그대로 기록하고 파일을 닫는다."""
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """경로 또는 파일 객체에서 JSON을 읽고 파일을 닫은 뒤 파싱 결과를 반환한다."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict



class Prompter(object):
    """JSON 템플릿으로 모델 입력 프롬프트를 구성하고 생성 결과에서 응답을 추출한다."""
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        """선택한 프롬프트 템플릿을 읽고 상세 출력 여부를 설정한다."""
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "korean"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        """명령과 선택적 입력을 템플릿에 넣고 학습 정답이 있으면 뒤에 추가한다."""
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        """템플릿의 응답 구분자 뒤에 있는 생성 텍스트를 추출한다."""
        return output.split(self.template["response_split"])[1].strip()
    

class Stream(transformers.StoppingCriteria):
    """생성된 토큰을 콜백으로 전달하는 Transformers 중단 조건을 제공한다."""
    def __init__(self, callback_func=None):
        """토큰 생성 결과를 전달받을 콜백을 저장한다."""
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        """첫 번째 입력 시퀀스를 콜백에 전달하고 생성을 계속하도록 False를 반환한다."""
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """콜백 기반 함수를 별도 스레드에서 실행해 결과를 지연 이터레이터로 제공한다."""

    def __init__(self, func, kwargs={}, callback=None):
        """결과 큐와 종료 표식을 준비하고 콜백 함수를 실행할 작업 스레드를 시작한다."""
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            """생성 결과를 큐에 넣고 중지 요청이 있으면 실행 중단을 알린다."""
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            """대상 함수를 실행하고 오류 처리 후 종료 표식과 완료 콜백을 전달한다."""
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        """생성 결과를 순회할 이터레이터 자신을 반환한다."""
        return self

    def __next__(self):
        """큐에서 다음 결과를 기다려 반환하고 종료 표식을 받으면 순회를 끝낸다."""
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        """컨텍스트 관리 구문에서 사용할 이터레이터 자신을 반환한다."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료 시 콜백 작업을 중지하도록 표시한다."""
        self.stop_now = True