import json
import pandas as pd
from typing import Dict, Optional, Union
import transformers


class Question:
    def __init__(
        self,
        question: str,
        answer: str,
        options: Dict[str, str],
        answer_idx: str,
        prompt_prefix: Optional[str] = "",
        prompt_suffix: Optional[str] = "",
    ):
        self.question = question
        self.answer = answer
        self.options = options
        self.answer_idx = answer_idx
        self.prompt_prefix = prompt_prefix if prompt_prefix else ""  # handle None
        self.prompt_suffix = prompt_suffix if prompt_suffix else ""  # handle None

        self._full_question = None

    def _get_full_question(
        self, prompt_prefix: Optional[str] = None, prompt_suffix: Optional[str] = None
    ) -> str:
        if prompt_prefix:
            self.prompt_prefix = prompt_prefix
        if prompt_suffix:
            self.prompt_suffix = prompt_suffix

        formatted_answers = "\n".join([f"{k}: {v}" for k, v in self.options.items()])

        # note that we are putting in a newline between the question and the answers
        full_question = f"""{self.prompt_prefix}{self.question} 

{formatted_answers}
{self.prompt_suffix}"""

        self._full_question = full_question

    def query(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        model_kwargs: Optional[Dict[str, Union[str, int]]] = {},
        tokenizer_kwargs: Optional[Dict[str, Union[str, int]]] = {
            "return_tensors": "pt"
        },
        decode_outputs: bool = True,
    ):
        if not model_kwargs:
            model_kwargs = {}

        # if self.full_question is None:
        input_ids = self.encode(tokenizer, **tokenizer_kwargs).to(
            model.device, non_blocking=True
        )
        outputs = model.generate(input_ids, **model_kwargs)

        if decode_outputs:
            return tokenizer.decode(outputs[0])
        else:
            return outputs

    def encode(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        # tokenizer_kwargs: Optional[dict] = {"return_tensors": "pt"},
        **tokenizer_kwargs,
    ):
        return tokenizer.encode(self.full_question, **tokenizer_kwargs)

    def score(self, answer):
        if self.answer_idx in answer:
            return True
        else:
            return False

    @property
    def full_question(self):
        if self._full_question is None:
            self._get_full_question()
            return self._full_question
        else:
            return self._full_question

    @classmethod
    def from_series(self, series: Union[pd.Series, pd.DataFrame]):
        # TODO: add in keys of series as kwargs

        # if isinstance(element):

        return self(
            question=series["question"],
            answer=series["answer"],
            options=series["options"],
            answer_idx=series["answer_idx"],
            prompt_prefix=series["prompt_prefix"]
            if series.get("prompt_prefix", None)
            else "",
            prompt_suffix=series["prompt_suffix"]
            if series.get("prompt_suffix", None)
            else "",
        )

    @classmethod
    def from_dict(self, dict: Dict[str, Union[str, dict]]):
        return self(
            question=dict["question"],
            answer=dict["answer"],
            options=dict["options"],
            answer_idx=dict["answer_idx"],
            prompt_prefix=dict["prompt_prefix"]
            if dict.get("prompt_prefix", None)
            else "",
            prompt_suffix=dict["prompt_suffix"]
            if dict.get("prompt_suffix", None)
            else "",
        )
