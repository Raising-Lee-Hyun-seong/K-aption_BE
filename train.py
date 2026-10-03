"""KoreanLM 지도 학습을 위한 데이터 전처리, 배치 구성 및 모델 학습·저장을 수행한다."""
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다."
        "요청을 적절히 완료하는 응답을 작성합니다.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "아래는 작업을 설명하는 지침입니다."
        "요청을 적절히 완료하는 응답을 작성합니다.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    """학습에 사용할 기반 모델의 이름 또는 경로를 정의한다."""
    model_name_or_path: Optional[str] = field(default="quantumaikr/KoreanLM")


@dataclass
class DataArguments:
    """지도 학습 데이터 파일의 경로를 정의한다."""
    data_path: str = field(default='korean_data.json', metadata={
                           "help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """모델 저장 위치, 캐시 및 토큰 길이를 포함한 학습 옵션을 정의한다."""
    output_dir: str = field(
        default='./pretrained', metadata={"help": "Path to the pretrained model output."})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """특수 토큰을 추가하고 새 토큰의 임베딩을 기존 임베딩 평균으로 초기화한다."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-
                                                num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-
                                                  num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """문자열 목록을 토큰화하고 패딩을 제외한 토큰 길이를 반환한다."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """입력과 정답을 토큰화하고 입력 영역의 라벨을 마스킹해 정답만 학습하도록 구성한다."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """프롬프트와 정답을 토큰화한 지도 학습 데이터셋을 제공한다."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        """JSON 데이터를 읽어 프롬프트·정답 쌍을 구성하고 토큰과 라벨을 저장한다."""
        super(SupervisedDataset, self).__init__()
        logging.warning("데이터 로드 중...")
        list_data_dict = utils.jload(data_path)

        logging.warning("입력 서식 지정...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get(
                "input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("입력 토큰화하기... 시간이 좀 걸릴 수 있습니다...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        """학습 데이터셋의 예제 수를 반환한다."""
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """지정한 인덱스의 입력 토큰과 학습 라벨을 반환한다."""
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):

    """길이가 다른 학습 예제를 패딩해 하나의 배치로 구성한다."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """입력과 라벨을 패딩하고 어텐션 마스크가 포함된 배치를 반환한다."""
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Trainer에 전달할 학습 데이터셋과 배치 구성기를 생성한다."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    """실행 인자로 모델과 데이터를 준비하고 토크나이저를 Hub에 업로드한 뒤 학습 결과를 저장한다."""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    koreanlm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    print('tokenizer.pad_token_id', tokenizer.pad_token_id)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=koreanlm,
    )

    tokenizer.push_to_hub('open_llama_7b_hf')

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=koreanlm, tokenizer=tokenizer,
                      args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
