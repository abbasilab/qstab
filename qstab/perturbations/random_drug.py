from qstab.perturbations.base import Perturbation
from qstab.data.question import Question
import random
from typing import Optional


class RandomDrugPerturbation(Perturbation):
    def __init__(self, drug_list: list[str], num_perturbs: Optional[int] = None):
        super().__init__()

        self.drug_list = drug_list
        self.num_perturbs = num_perturbs

    def _apply(self, question: Question):
        num_opts = len(question.options)

        correct_answer = question.options.pop(question.answer_idx)

        keys = list(question.options.keys())
        random.shuffle(keys)

        if self.num_perturbs is None:
            num_to_resample = random.randrange(1, num_opts)
        elif self.num_perturbs > num_opts:
            raise ValueError(
                "Number of perturbations cannot be greater than the number of options."
            )
        else:
            num_to_resample = self.num_perturbs

        for idx in range(num_to_resample):
            key = keys[idx]
            question.options[key] = random.choice(self.drug_list)

        question.options[question.answer_idx] = correct_answer

        options_sorted = {}
        for key in sorted(question.options.keys()):
            options_sorted[key] = question.options[key]

        question.options = options_sorted
