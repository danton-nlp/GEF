from abc import ABC, abstractmethod


class WordValidator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def is_valid_word(
        self,
        word,
        input_idx,
        beam_sequence,
        beam_scores
    ):
        """
        TODO
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

class DictionaryValidator(WordValidator):
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def is_valid_word(
        self,
        word,
        input_idx,
        beam_sequence,
        beam_scores
    ):
        return word in self.dictionary


class OverlapValidator(WordValidator):
    def __init__(self, docs_to_summarize):
        self.docs_to_summarize = docs_to_summarize
    
    def is_valid_word(
        self,
        word,
        input_idx,
        beam_sequence,
        beam_scores
    ):
        return word.lower() in self.docs_to_summarize[input_idx]