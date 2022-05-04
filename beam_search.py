from typing import List
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Constraint,
    LogitsProcessor,
    LogitsProcessorList,
    PhrasalConstraint,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import argparse
import torch
from collections import defaultdict

class FactualLogitsProcessor(LogitsProcessor):
    r"""
    [`FactualLogitsProcessor`] enforcing dynamic constraints on logits
    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, tokenizer, num_beams, banned_token_ids=set()):
        self.tokenizer = tokenizer
        self.banned_token_ids = banned_token_ids
        self.unfactual_tokens_by_seq_idx = defaultdict(lambda: [])
        self.num_beams = num_beams

    def is_factual_token(
        self,
        sequence,  # sequence generated so far
        token_id,  # next token to be generated
        token_probability,  # probability of token to be generated
    ):
        if token_id in self.banned_token_ids:
            return False
        return True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # for every beam (partially generated sentence)
        for beam_idx, (beam_input_ids, beam_scores) in enumerate(
            zip(input_ids, scores)
        ):
            # get the last token of this beam
            # last_word = self.tokenizer.decode(beam_input_ids[-1])
            top_k = beam_scores.topk(k=5)
            for prob, idx in zip(top_k[0], top_k[1]):
                if not self.is_factual_token(
                    beam_input_ids[-1], idx.item(), prob.item()
                ):
                    scores[beam_idx, idx] = -float("inf")
                    self.unfactual_tokens_by_seq_idx[beam_idx % self.num_beams].append((
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
    factuality_enforcer = FactualLogitsProcessor(tokenizer, num_beams, banned_token_ids={
        5295,  # ' Wales'
        9652,  # ' Edinburgh'
    })
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
                "dropped_seqs": factuality_enforcer.unfactual_tokens_by_seq_idx[seq_idx]
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
                    # print("appending at", beam_idx)
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
