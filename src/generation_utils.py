import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList
from src.word_logits_processor import WordLogitsProcessor


SUMMARY_FAILED_GENERATION = "<Failed generation: blocked all beams>"


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


def load_prior_model_and_tokenizer():
    return load_model_and_tokenizer("facebook/bart-large")


def load_model_and_tokenizer(
    path: str, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    return (
        AutoModelForSeq2SeqLM.from_pretrained(path).to(device),
        AutoTokenizer.from_pretrained(path),
    )


def load_bart_xsum_cmlm(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model = AutoModelForSeq2SeqLM.from_pretrained("model-checkpoints/entfa-cmlm").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-xsum", mask_token="###"
    )

    return model, tokenizer


def generate_summaries(
    model,
    tokenizer,
    docs_to_summarize,
    word_logits_processor: WordLogitsProcessor,
    num_beams=4,
    return_beam_metadata=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model.to(device)
    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    model_output = model.generate(
        inputs.input_ids.to(device),
        num_beams=num_beams,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        logits_processor=LogitsProcessorList(
            [] if word_logits_processor is None else [word_logits_processor]
        ),
    )
    generated_summaries = [
        (
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if (
                word_logits_processor is None
                or idx not in word_logits_processor.failed_sequences
            )
            else SUMMARY_FAILED_GENERATION
        )
        for idx, ids in enumerate(model_output.sequences)
    ]

    if not return_beam_metadata:
        return generated_summaries

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
                "score": model_output.sequences_scores[seq_idx].item(),
                "beams": [list() for _ in range(num_beams)],
                "selected_beam_indices": top_beam_indices,
                "dropped_seqs": word_logits_processor.excluded_beams_by_input_idx[
                    seq_idx
                ],
                "n_words_checked": word_logits_processor.words_to_check_by_input_idx[
                    seq_idx
                ],
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
