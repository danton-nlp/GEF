from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict


class WordValidator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def is_valid_word(self, word, input_idx, beam_sequence, beam_scores):
        """
        TODO
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class BannedWords(WordValidator):
    def __init__(
        self, 
        banned_words=set(), 
        banned_words_by_input_idx: Dict[int, set] = {}
    ):
        self.banned_words = defaultdict(
            lambda: banned_words,
            banned_words_by_input_idx
        )

    def is_valid_word(self, word, input_idx, beam_sequence, beam_scores):
        return word not in self.banned_words[input_idx]


class OverlapValidator(WordValidator):
    def __init__(self, docs_to_summarize):
        self.docs_to_summarize = docs_to_summarize

    def is_valid_word(self, word, input_idx, beam_sequence, beam_scores):
        return word.lower() in self.docs_to_summarize[input_idx]
