import json
import pandas as pd
from typing import Dict, Optional, Union
import transformers


class Question:
    def __init__(
        self,
        question: str,
        reference: str,
        answer: str,
        options: Dict[str, str],
        answer_idx: str,
        prompt_prefix: Optional[str] = "",
        prompt_suffix: Optional[str] = "",
    ):
        self.question = question
        self.reference = reference
        self.answer = answer
        self.options = options
        self.answer_idx = answer_idx
        self.prompt_prefix = prompt_prefix if prompt_prefix else ""  # handle None
        self.prompt_suffix = prompt_suffix if prompt_suffix else ""  # handle None

        self._full_question = None

    def _get_full_question(
        self, prompt_prefix: Optional[str] = None,
        prompt_suffix: Optional[str] = None
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
            "return_tensors": "pt", "max_length":512,
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
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        self._get_full_question()
        return self._full_question
#         if self._full_question is None:
#             self._get_full_question()
#             return self._full_question
#         else:
#             return self._full_question

    @classmethod
    def from_series(self, series: Union[pd.Series, pd.DataFrame]):
        # TODO: add in keys of series as kwargs

        # if isinstance(element):

        return self(
            question=series["question"],
            reference=series["reference"],
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
            reference=dict["reference"],
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


class AnswerCollector:
    
    def __init__(self, rowvals=[], colnames=["answer"], **kwargs):
        self.answers = pd.DataFrame(rowvals, columns=colnames, **kwargs)
    
    @property
    def num_entry(self):
        return self.answers.shape[0]
    
    @property
    def num_collection(self):
        return len(self.answers.columns)
        
    @property
    def colnames(self):
        return self.answers.columns.values
    
    @classmethod
    def from_array(self, rowvals, **kwargs):
        return self(rowvals=rowvals, **kwargs)
    
    def add_entry(self, **kwargs):
        ''' Add a row to the answers dataframe (default values are nan).
        '''
        last_index = self.num_entry
        # self.answers.loc[last_index] = np.nan
        for k, v in kwargs.items():
            self.answers.at[last_index, k] = v
            
    def remove_entry(self, index):
        ''' Remove a row from the answer collection.
        '''
        self.answers = self.answers.drop(index, axis=0)
        
    def add_collection(self, collection, name=None):
        ''' Add a collection of values for a new column.
        '''
        self.answers[name] = collection
    
    def remove_collection(self, name):
        ''' Remove a column from the answer collection.
        '''
        self.answers = self.answers.drop(name, axis=1)
    
    def clean(self):
        pass
    
    def to_excel(self, filepath, sheet_name='Results', **kwargs):
        self.answers.to_excel(filepath, sheet_name=sheet_name, **kwargs)
        
    def to_csv(self, filepath, **kwargs):
        self.answers.to_csv(filepath, **kwargs)