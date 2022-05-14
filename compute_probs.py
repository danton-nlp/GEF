import pprint
from typing import List, Tuple
from tqdm import tqdm

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from src.data_utils import load_xent
from src.generation_utils import load_model_and_tokenizer, load_xsum_with_mask_in_vocab
from src.prob_computation_utils import build_masked_inputs_and_targets


def compute_prior_and_posterior_probs(
    masked_inputs: List[str],
    targets: List[str],
    sources: List[str],
    entities: List[str],
    prior_model_and_tokenizer: Tuple[BartForConditionalGeneration, BartTokenizer],
    posterior_model_and_tokenizer: Tuple[BartForConditionalGeneration, BartTokenizer],
    verbose: bool = False,
) -> List[List[float]]:
    """
    Compute the joint prior and posterior probabilities of an masked entity, given
    a possible causal (left context only) or non-causal (left and right
    context). The prior probability is NOT conditioned on "source" (e.g.
    full article). The posterior probability is.

    Returns a tuple of 2 items:
    2. The joint prior probability of all tokens in the masked out entity.
    2. The joint posterior probability of all tokens in the masked out entity.
    """
    if len(masked_inputs) != len(targets):
        raise ValueError("number of inputs is not the same as the number of targets")

    entity_probs: List[List[float]] = []

    needed_data = zip(masked_inputs, targets, sources, entities)
    for masked_input, target, source, entity in needed_data:
        if verbose:
            print(f"{masked_input=}")
            print(f"{target=}")

        prior_input_tokenized = (
            prior_model_and_tokenizer[1]
            .encode(masked_input, return_tensors="pt")
            .squeeze(0)  # type: ignore
        )
        posterior_input_tokenized = (
            posterior_model_and_tokenizer[1]
            .encode(
                "<s>" + masked_input.replace("<mask>", "###") + "</s>" + source,
                add_special_tokens=False,
                truncation=True,
                return_tensors="pt",
            )
            .squeeze(0)  # type: ignore
        )

        with torch.no_grad():
            prior_entity_prob = compute_entitity_probability(
                input_tokenized=prior_input_tokenized,
                target_tokenized=(
                    prior_model_and_tokenizer[1]
                    .encode(target, return_tensors="pt")
                    .squeeze(0)  # type: ignore
                ),
                entity_tokenized=(
                    prior_model_and_tokenizer[1]
                    .encode(f" {entity}", return_tensors="pt", add_special_tokens=False)
                    .squeeze(0)  # type: ignore
                ),
                model=prior_model_and_tokenizer[0],
                tokenizer=prior_model_and_tokenizer[1],
                verbose=verbose,
            )

            posterior_entity_prob = compute_entitity_probability(
                input_tokenized=posterior_input_tokenized,
                target_tokenized=(
                    posterior_model_and_tokenizer[1]
                    .encode(target, return_tensors="pt")
                    .squeeze(0)  # type: ignore
                ),
                entity_tokenized=(
                    posterior_model_and_tokenizer[1]
                    .encode(f" {entity}", return_tensors="pt", add_special_tokens=False)
                    .squeeze(0)  # type: ignore
                ),
                model=posterior_model_and_tokenizer[0],
                tokenizer=posterior_model_and_tokenizer[1],  # type: ignore
                verbose=verbose,
            )
        entity_probs.append([prior_entity_prob, posterior_entity_prob])

    return entity_probs


def compute_entitity_probability(
    input_tokenized: torch.Tensor,
    target_tokenized: torch.Tensor,
    entity_tokenized: torch.Tensor,
    model: BartForConditionalGeneration,
    tokenizer=BartTokenizer,
    verbose: bool = False,
) -> float:
    def prefix_allowed_tokens_fn(_batch_id, input_ids):
        current_step = len(input_ids) - 1
        return target_tokenized[current_step].tolist()

    prediction = model.generate(
        input_tokenized.unsqueeze(0),
        num_beams=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=200,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )

    mask_input_idx = (input_tokenized == tokenizer.mask_token_id).nonzero().item()

    entity_token_prob_distribs = torch.vstack(
        prediction.scores[mask_input_idx : (mask_input_idx + len(entity_tokenized))]
    ).softmax(dim=1)

    # TODO: some tensor native operation here? close to `torch.gather`?
    probs_of_entity_tokens = torch.hstack(
        [
            entity_token_prob_distribs[i, token_id]
            for i, token_id in enumerate(entity_tokenized)
        ]
    )

    if verbose:
        all_probs_at_target_tokens = [
            [
                tokenizer.decode(target_tokenized[i]),
                distrib.softmax(dim=1).squeeze(0)[target_tokenized[i]].item(),
            ]
            for i, distrib in enumerate(prediction.scores)
        ]
        pprint.PrettyPrinter(indent=4).pprint(all_probs_at_target_tokens)

    return torch.prod(probs_of_entity_tokens).item()



if __name__ == "__main__":
    dataset = load_xent("test")

    example_entity_probs = {}
    for idx, example in enumerate(tqdm(dataset[3:10])):
        (
            inputs,
            targets,
            entities,
            sources,
            entity_labels,
        ) = build_masked_inputs_and_targets(example)

        entity_probs = compute_prior_and_posterior_probs(
            masked_inputs=inputs,
            targets=targets,
            sources=sources,
            entities=entities,
            prior_model_and_tokenizer=load_model_and_tokenizer("facebook/bart-large"),
            posterior_model_and_tokenizer=load_xsum_with_mask_in_vocab(),
        )

        pprint.PrettyPrinter(indent=4).pprint(
            list(zip(entities, entity_labels, entity_probs))
        )
