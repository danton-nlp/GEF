from typing import List
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Constraint,
    LogitsProcessor,
    LogitsProcessorList,
)
from datasets import load_dataset
import argparse
import torch
from collections import defaultdict

from beam_validator import DictionaryValidator, WordValidator

SPLIT_WORD_TOKENS = {
    ' ',
    '.',
    ',',
    '_',
    '?',
    '!',
}

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
        self.excluded_beams_by_seq_idx = defaultdict(lambda: [])
        self.num_beams = num_beams
        self.words_to_check = defaultdict(lambda: 0)

    def is_valid_beam(
        self,
        doc_idx, # doc idx being summarized
        sequence,  # sequence generated so far
        token_id,  # next token to be generated (argmax of beam_scores)
        beam_scores # probability of all tokens to be generated
    ):
        """
            Check whether beam is valid according to the passed validators.

            To enable validating on a word-level, this method backtracks
            to collect the predicted word when it detects that the predicted
            subword (token) is a word ending.
        """
        # Token-level checks
        # if token_id in self.banned_token_ids:
        #     return False

        # Word-level checks
        current_subword = self.tokenizer.decode(token_id)
        backtrack_word = ""
        is_subword_ending = False
        for char in current_subword:
            if char in SPLIT_WORD_TOKENS:
                is_subword_ending = True
                break
            else:
                backtrack_word += char

        # if the predicted subword indicates a word ending
        # backtrack to collect the predicted word
        backtrack_done = False
        if is_subword_ending:
            prev_subword_idx = len(sequence) - 1
            while prev_subword_idx != 0 and not backtrack_done:
                prev_token_id = sequence[prev_subword_idx]
                prev_subword = self.tokenizer.decode(prev_token_id)
                prev_char_idx = len(prev_subword) - 1
                while prev_char_idx >= 0:
                    prev_char = prev_subword[prev_char_idx]
                    if prev_char not in SPLIT_WORD_TOKENS:
                        backtrack_word = prev_char + backtrack_word
                    else:
                        backtrack_done = True
                        break 
                    prev_char_idx -= 1
                prev_subword_idx -= 1
            self.words_to_check[backtrack_word] += 1
            # Call validator to check whether the word is valid
            if not self.word_validator.is_valid_word(
                doc_idx, 
                backtrack_word,
                sequence, 
                beam_scores
            ):
                return False
        return True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        
        # for every beam (partially generated sentence)
        for beam_idx, (beam_input_ids, beam_scores) in enumerate(
            zip(input_ids, scores)
        ):
            top_k = beam_scores.topk(k=1)
            for prob, idx in zip(top_k[0], top_k[1]):
                if not self.is_valid_beam(
                    beam_idx // self.num_beams,
                    beam_input_ids, 
                    idx.item(), 
                    scores[beam_idx]
                ):
                    scores[beam_idx, :] = -float("inf")
                    self.excluded_beams_by_seq_idx[beam_idx % self.num_beams].append((
                        len(beam_input_ids),
                        idx.item(),
                        prob.item(),
                    ))
                    
        return scores


def entropy(p_dist: torch.Tensor) -> float:
    """ "
    Calculates Shannon entropy for a probability distribution

    Args:
        p_dist: probability distribution (torch.Tensor)

    Returns:
        entropy (float)
    """
    # add epsilon because log(0) = nan
    p_dist = p_dist.view(-1) + 1e-12
    return -torch.mul(p_dist, p_dist.log()).sum(0).item()


def generate_summaries_with_constraints(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    docs_to_summarize: List[str],
    num_beams: int = 4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    input_token_ids = inputs.input_ids.to(device)

    factuality_enforcer = WordLogitsProcessor(
        tokenizer, 
        num_beams, 
        DictionaryValidator({
            "Edinburgh",
            "Wales",
            "prison",
            "charity",
            "homeless",
            "man",
            "a"
        })
    )
    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        # remove_invalid_values=True,
        logits_processor=LogitsProcessorList([factuality_enforcer])
        # max_length=0,
    )

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]

    # reshape model_output scores to (n_seqs x seq len x n_beams x vocab)
    model_beam_scores = (
        torch.stack(model_output.scores)
        .reshape(len(model_output.scores), len(generated_summaries), num_beams, -1)
        .permute(1, 0, 2, 3)
    )
    
    # Collect Beam Search Metadata
    beams_metadata = []
    if model_output.beam_indices is not None:
        for seq_idx in range(model_output.sequences.shape[0]):
            top_beam_indices = [x.item() for x in model_output.beam_indices[seq_idx]]
            seq_beams = {
                "beams": [list() for _ in range(num_beams)],
                "selected_beam_indices": top_beam_indices,
                "dropped_seqs": factuality_enforcer.excluded_beams_by_seq_idx[seq_idx]
            }
            beams_metadata.append(seq_beams)

            for idx, output_token_id in enumerate(model_output.sequences[seq_idx][1:]):
                # beam_idx = model_output.beam_indices[seq_idx][idx]
                for beam_idx in range(num_beams):
                    beam_probs = torch.exp(model_beam_scores[seq_idx][idx][beam_idx])
                    beam_top_alternatives = []
                    top_probs = torch.topk(beam_probs, k=num_beams)
                    for i, v in zip(top_probs.indices, top_probs.values):
                        beam_top_alternatives.append(
                            {
                                "token": tokenizer.decode(i),
                                "token_id": i.item(),
                                "probability": v.item(),
                            }
                        )
                    seq_beams["beams"][beam_idx].append(
                        {
                            "top_tokens": beam_top_alternatives,
                            "entropy": entropy(beam_probs),
                            # "token_id": output_token_id,
                            # "token": tokenizer.decode(output_token_id),
                            # "beam_token_prob": selected_beam_probs[output_token_id].item(),
                            # "beam_idx": beam_idx.item(),
                            # "token_in_input": output_token_id in input_set,
                        }
                    )

    return generated_summaries, beams_metadata


def load_model_and_tokenizer(path):
    return (
        AutoModelForSeq2SeqLM.from_pretrained(path),
        AutoTokenizer.from_pretrained(path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    # parser.add_argument("--sumtool_path", type=str, default="")
    # parser.add_argument("--data_subset", type=int, default=0)
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    xsum_test = load_dataset("xsum")["test"]

    summaries, metadata = generate_summaries_with_constraints(
        model, tokenizer, xsum_test["document"][0:2], num_beams=4
    )

    print(summaries)
