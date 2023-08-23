from qstab.perturbations.base import Perturbation
from qstab.data.question import Question
import random


class ScramblingPerturbation(Perturbation):
    def __init__(self):
        super().__init__()

        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.num_perturbs = 4

    def random_word_scrambling(self, word: str, with_replacement: bool = True):
        num_letters = len(word)

        if with_replacement:
            new_word = random.choices(self.alphabet, k=num_letters)
        else:
            new_word = random.sample(self.alphabet, k=num_letters)

        new_word = "".join(new_word)
        return new_word

    def _apply(self, question: Question):
        correct_answer = question.options.pop(question.answer_idx)
        keys = list(question.options.keys())

        for idx in range(self.num_perturbs):
            key = keys[idx]

            question.options[key] = self.random_word_scrambling(question.options[key])

        question.options[question.answer_idx] = correct_answer

        options_sorted = {}
        for key in sorted(question.options.keys()):
            options_sorted[key] = question.options[key]

        question.options = options_sorted
