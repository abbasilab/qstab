from abc import ABC, abstractmethod
from qstab.data import Question


class Perturbation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _apply(self, question: Question):
        """
        This method should modify the question in-place.

        Args:
            question (Question): An instance of Question to be modified.
        """

        pass

    def apply(self, question: Question):
        self._apply(question)

        if question.full_question is not None:
            # if question has not been modified, then wait, otherwise recompute
            # so that question is consistent with the perturbation
            question._get_full_question()
