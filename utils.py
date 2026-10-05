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
    """경로의 부모 디렉터리를 준비해 쓰기 파일을 열거나 기존 파일 객체를 반환한다.

    :param f: 입출력 대상 파일 경로 또는 열린 파일 객체.
    :param mode: 파일을 열 때 사용할 입출력 모드.
    :return: 쓰기 가능한 기존 또는 새로 연 파일 객체.
    """
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    """읽기 파일 경로를 파일 객체로 변환하거나 기존 파일 객체를 반환한다.

    :param f: 입출력 대상 파일 경로 또는 열린 파일 객체.
    :param mode: 파일을 열 때 사용할 입출력 모드.
    :return: 읽기 가능한 기존 또는 새로 연 파일 객체.
    """
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """딕셔너리·리스트를 JSON으로 또는 문자열을 그대로 기록하고 파일을 닫는다.

    :param obj: 저장할 딕셔너리·리스트 또는 문자열.
    :param f: 입출력 대상 파일 경로 또는 열린 파일 객체.
    :param mode: 파일을 열 때 사용할 입출력 모드.
    :param indent: JSON 들여쓰기에 사용할 공백 수.
    :param default: JSON으로 직접 직렬화할 수 없는 객체의 변환 함수.
    :return: None.
    :raises ValueError: obj가 딕셔너리·리스트·문자열이 아닐 때.

    호출자가 전달한 열린 파일 객체도 저장 후 닫는다.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """경로 또는 파일 객체에서 JSON을 읽고 파일을 닫은 뒤 파싱 결과를 반환한다.

    :param f: 입출력 대상 파일 경로 또는 열린 파일 객체.
    :param mode: 파일을 열 때 사용할 입출력 모드.
    :return: JSON을 파싱한 Python 객체.
    :raises json.JSONDecodeError: 파일 내용이 유효한 JSON이 아닐 때.

    호출자가 전달한 열린 파일 객체도 읽은 후 닫는다.
    """
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict



class Prompter(object):
    """JSON 템플릿으로 모델 입력 프롬프트를 구성하고 생성 결과에서 응답을 추출한다."""
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        """선택한 프롬프트 템플릿을 읽고 상세 출력 여부를 설정한다.

        :param template_name: templates 디렉터리의 템플릿 이름. 빈 값이면 korean을 사용한다.
        :param verbose: 참이면 템플릿 정보와 구성한 프롬프트를 출력한다.
        :return: None.
        :raises ValueError: 선택한 프롬프트 템플릿 파일이 없을 때.
        """
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
        """명령과 선택적 입력을 템플릿에 넣고 학습 정답이 있으면 뒤에 추가한다.

        :param instruction: 모델이 수행할 작업 지시문.
        :param input: 프롬프트에 추가할 선택적 입력 원문.
        :param label: 학습 프롬프트 끝에 덧붙일 선택적 정답 문자열.
        :return: 지시문·선택적 입력·정답이 결합된 프롬프트 문자열.
        """
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
        """템플릿의 응답 구분자 뒤에 있는 생성 텍스트를 추출한다.

        :param output: 검사하거나 응답을 추출할 모델 생성 텍스트.
        :return: 응답 구분자 다음의 텍스트에서 앞뒤 공백을 제거한 문자열.
        :raises IndexError: 출력에 응답 구분자가 없을 때.
        """
        return output.split(self.template["response_split"])[1].strip()


class Stream(transformers.StoppingCriteria):
    """생성된 토큰을 콜백으로 전달하는 Transformers 중단 조건을 제공한다."""
    def __init__(self, callback_func=None):
        """토큰 생성 결과를 전달받을 콜백을 저장한다.

        :param callback_func: 생성된 첫 번째 시퀀스의 토큰을 전달받는 선택적 콜백.
        :return: None.
        """
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        """첫 번째 입력 시퀀스를 콜백에 전달하고 생성을 계속하도록 False를 반환한다.

        :param input_ids: 생성 중인 배치의 토큰 ID 텐서.
        :param scores: Transformers가 전달하는 토큰 점수. 이 구현에서는 사용하지 않는다.
        :return: 생성을 중단하지 않도록 항상 False.
        """
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """콜백 기반 함수를 별도 스레드에서 실행해 결과를 지연 이터레이터로 제공한다."""

    def __init__(self, func, kwargs={}, callback=None):
        """결과 큐와 종료 표식을 준비하고 콜백 함수를 실행할 작업 스레드를 시작한다.

        :param func: callback 인자를 받으며 별도 스레드에서 실행할 함수.
        :param kwargs: 대상 함수에 전달할 추가 키워드 인자 사전.
        :param callback: 작업 함수의 최종 반환값을 전달받는 선택적 완료 콜백.
        :return: None.
        """
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            """생성 결과를 큐에 넣고 중지 요청이 있으면 실행 중단을 알린다.

            :param val: 큐에 전달할 중간 생성 결과.
            :return: None.
            :raises ValueError: 컨텍스트 종료 등으로 작업 중지를 요청한 상태일 때.
            """
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            """대상 함수를 실행하고 오류 처리 후 종료 표식과 완료 콜백을 전달한다.

            :return: None.

            대상 함수의 ValueError는 무시하며 다른 예외는 traceback을 출력한 뒤 처리한다.
            """
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
        """생성 결과를 순회할 이터레이터 자신을 반환한다.

        :return: 순회할 Iteratorize 인스턴스 자신.
        """
        return self

    def __next__(self):
        """큐에서 다음 결과를 기다려 반환하고 종료 표식을 받으면 순회를 끝낸다.

        :return: 콜백이 큐에 넣은 다음 중간 결과.
        :raises StopIteration: 작업 스레드의 종료 표식을 받았을 때.
        """
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        """컨텍스트 관리 구문에서 사용할 이터레이터 자신을 반환한다.

        :return: 컨텍스트에서 사용할 Iteratorize 인스턴스 자신.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료 시 콜백 작업을 중지하도록 표시한다.

        :param exc_type: 컨텍스트 본문에서 발생한 예외 종류. 정상 종료면 None.
        :param exc_val: 컨텍스트 본문에서 발생한 예외 객체. 정상 종료면 None.
        :param exc_tb: 컨텍스트 본문 예외의 traceback. 정상 종료면 None.
        :return: None.
        """
        self.stop_now = True