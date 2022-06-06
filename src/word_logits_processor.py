from collections import defaultdict
import torch
from transformers import LogitsProcessor
from src.beam_validators import WordValidator


SPLIT_WORD_TOKENS = {" ", ".", ",", "_", "?", "!", "'"}


def should_backtrack(subword: str):
    return not subword.startswith(" ")


def is_subword_ending(subword: str):
    return subword.startswith(" ") or subword.endswith(" ")


class WordLogitsProcessor(LogitsProcessor):
    r"""
    [`WordLogitsProcessor`] enforcing constraints on words during beam search

    Args:
        tokenizer (`AutoTokenizer`):
            The model's tokenizer
        num_beams (`int`):
            Number of beams.
        word_validator (`WordValidator`):
            Responsible for checking whether the word is valid.
    """

    def __init__(self, tokenizer, num_beams, word_validator: WordValidator):
        self.tokenizer = tokenizer
        self.word_validator = word_validator
        self.num_beams = num_beams
        self.excluded_beams_by_input_idx = defaultdict(lambda: list())
        self.words_to_check_by_input_idx = defaultdict(lambda: 0)
        self.failed_sequences = set()

    def is_valid_beam(
        self,
        input_idx,  # input idx being processed
        sequence,  # sequence generated so far
        token_id,  # next token to be generated (argmax of beam_scores)
        beam_scores,  # probability of all tokens to be generated
    ):
        """
        Check whether beam is valid according to the passed validators.

        To enable validating on a word-level, this method backtracks
        to collect the predicted word when it detects that the predicted
        subword (token) is a word ending.
        """

        # Begin by checking whether tokens
        # to-be-generated indicate a phrase ending
        if "pegasus" in self.tokenizer.name_or_path:
            # For pegasus we need to include the previous token
            # since spaces are not included when decoding a single token:
            # https://github.com/huggingface/tokenizers/issues/826
            to_be_generated = self.tokenizer.decode(
                [sequence[-1], token_id], skip_special_tokens=True
            )
        else:
            to_be_generated = self.tokenizer.decode(token_id, skip_special_tokens=True)
        phrase_ending_idx = -1
        for j, char in enumerate(reversed(to_be_generated)):
            if char in SPLIT_WORD_TOKENS:
                phrase_ending_idx = j + 1
                break

        # if the predicted token indicates a phrase ending
        # backtrack to collect the phrase
        if phrase_ending_idx != -1:
            backtrack_phrase = ""
            candidate_gen = self.tokenizer.decode(
                list(sequence) + [token_id], skip_special_tokens=True
            )[:-phrase_ending_idx]
            prev_char_idx = len(candidate_gen) - 1

            while prev_char_idx >= 0:
                prev_char = candidate_gen[prev_char_idx]
                if prev_char not in SPLIT_WORD_TOKENS:
                    backtrack_phrase = prev_char + backtrack_phrase
                else:
                    # We encountered a split-word token
                    # stop backtracking UNLESS the backtracked word
                    # is an invalid phrase ending
                    if self.word_validator.is_maybe_invalid_phrase_ending(
                        prev_char + backtrack_phrase, input_idx
                    ):
                        backtrack_phrase = prev_char + backtrack_phrase
                    else:
                        break
                prev_char_idx -= 1
            self.words_to_check_by_input_idx[input_idx] += 1
            # Call validator to check whether the word is valid
            if not self.word_validator.is_valid_word(
                backtrack_phrase, input_idx, sequence, beam_scores
            ):
                return False
        return True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        blocked_beams_by_input_idx = defaultdict(lambda: 0)
        # for every beam (partially generated sentence)
        for beam_idx, (beam_input_ids, beam_scores) in enumerate(
            zip(input_ids, scores)
        ):
            top_k = beam_scores.topk(k=1)
            for prob, idx in zip(top_k[0], top_k[1]):
                input_idx = beam_idx // self.num_beams
                if not self.is_valid_beam(
                    input_idx, beam_input_ids, idx.item(), scores[beam_idx]
                ):
                    scores[beam_idx, :] = -float("inf")
                    self.excluded_beams_by_input_idx[input_idx].append(
                        (
                            beam_input_ids,
                            idx.item(),
                            prob.item(),
                        )
                    )
                    blocked_beams_by_input_idx[input_idx] += 1

        for input_idx, n_blocked in blocked_beams_by_input_idx.items():
            if n_blocked == self.num_beams:
                self.failed_sequences.add(input_idx)

        return scores
