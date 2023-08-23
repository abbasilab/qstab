from qstab.perturbations.base import Perturbation
from qstab.data.question import Question
import random


class RandomDrugPerturbation(Perturbation):
    def __init__(self, drug_list: list[str]):
        super().__init__()

        self.drug_list = drug_list

    def _apply(self, question: Question):
        num_opts = len(question.options)
        num_to_resample = random.randrange(1, num_opts)

        correct_answer = question.options.pop(question.answer_idx)

        keys = list(question.options.keys())
        random.shuffle(keys)

        for idx in range(num_to_resample):
            key = keys[idx]
            question.options[key] = random.choice(self.drug_list)

        question.options[question.answer_idx] = correct_answer

        options_sorted = {}
        for key in sorted(question.options.keys()):
            options_sorted[key] = question.options[key]

        question.options = options_sorted
