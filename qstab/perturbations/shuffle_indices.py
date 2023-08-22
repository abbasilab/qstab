from qstab.perturbations.base import Perturbation
from qstab.data.question import Question
import random


class ShufflePerturbation(Perturbation):
    def __init__(self):
        super().__init__()

    def _apply(self, question: Question, ensure_shuffle: bool = True):
        options = question.options

        values = list(options.values())
        indices = list(range(len(values)))

        if ensure_shuffle:
            correct_index_numeric_map = {
                letter: idx for letter, idx in zip(options.keys(), indices)
            }
            correct_index_val = correct_index_numeric_map[question.answer_idx]

            # shuffle until nth item is not n
            while indices[correct_index_val] == correct_index_val:
                random.shuffle(indices)
        else:
            random.shuffle(indices)

        values_shuffled = [values[i] for i in indices]

        new_options = dict(zip(sorted(options.keys()), values_shuffled))
        question.options = new_options

        for key, val in new_options.items():
            if val == question.answer:
                question.answer_idx = key

        if question.full_question is not None:
            question._get_full_question()  # reset prompt if it has already been generated
