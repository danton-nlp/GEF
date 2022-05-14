import pprint
from functools import reduce
from operator import mul
from typing import Dict, List, Tuple, TypedDict

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from data_utils import load_xent


class XEntExample(TypedDict):
    source: str
    reference: str
    prediction: str
    entities: List[Dict]


# TODO: Delete me -- tech debt
def compute_prior_probs(
    masked_inputs: List[str],
    targets: List[str],
    entities: List[dict],
    prior_model_and_tokenizer: Tuple,
    verbose: bool = False,
) -> List[float]:
    """
    Compute the joint prior probability of an masked entity, given
    it's causal (left) context. As a prior probability, it's NOT conditioned
    on "source" (e.g. full article).

    Returns a tuple of 2 items:
    1. The probs of all tokens in the input (all non-masked tokens should have a
    prob close to 1, by problem framing)
    2. The joint probability of all tokens in the masked out entity.
    """
    if len(masked_inputs) != len(targets):
        raise ValueError("number of inputs is not the same as the number of targets")

    model, tokenizer = model_and_tokenizer

    def prefix_allowed_tokens_fn(_batch_id, input_ids):
        current_step = len(input_ids) - 1
        return target_tokenized[current_step].tolist()

    entity_probs = []

    for masked_input, target, entity in zip(masked_inputs, targets, entities):
        if verbose:
            print(f"{masked_input=}")
            print(f"{target=}")

        masked_input_tokenized = tokenizer.encode(
            masked_input,
            return_tensors="pt",
        ).squeeze(0)
        target_tokenized = tokenizer.encode(target, return_tensors="pt").squeeze(0)
        entity_tokenized = tokenizer.encode(
            f" {entity}", return_tensors="pt", add_special_tokens=False
        ).squeeze(0)

        if verbose:
            print(f"{masked_input_tokenized=}")
            print(f"{target_tokenized=}")

        prediction = model.generate(
            masked_input_tokenized.unsqueeze(0),
            num_beams=1,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=200,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        mask_input_idx = (
            (masked_input_tokenized == tokenizer.mask_token_id).nonzero().item()
        )

        entity_token_prob_distribs = torch.vstack(
            prediction.scores[mask_input_idx : (mask_input_idx + len(entity_tokenized))]
        ).softmax(dim=1)

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

        entity_probs.append(torch.prod(probs_of_entity_tokens))

    return entity_probs


def compute_prior_and_posterior_probs(
    masked_inputs: List[str],
    targets: List[str],
    sources: List[str],
    entities: List[dict],
    prior_model_and_tokenizer: Tuple[BartForConditionalGeneration, BartTokenizer],
    posterior_model_and_tokenizer: Tuple[BartForConditionalGeneration, BartTokenizer],
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
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

    entity_probs = []

    needed_data = zip(masked_inputs, targets, sources, entities)
    for masked_input, target, source, entity in needed_data:
        if verbose:
            print(f"{masked_input=}")
            print(f"{target=}")

        prior_input_tokenized = prior_model_and_tokenizer[1].encode(
            masked_input, return_tensors="pt"
        ).squeeze(0)
        posterior_input_tokenized = posterior_model_and_tokenizer[1].encode(
            masked_input,
            text_pair=source,
            return_tensors="pt",
        ).squeeze(0)

        with torch.no_grad():
            prior_entity_prob = compute_entitity_probability(
                input_tokenized=prior_input_tokenized,
                target_tokenized=prior_model_and_tokenizer[1].encode(target, return_tensors="pt").squeeze(0),
                entity_tokenized=prior_model_and_tokenizer[1].encode(
                    f" {entity}", return_tensors="pt", add_special_tokens=False
                ).squeeze(0),
                model=prior_model_and_tokenizer[0],
                tokenizer=prior_model_and_tokenizer[1],
                verbose=verbose,
            )

            posterior_entity_prob = compute_entitity_probability(
                input_tokenized=posterior_input_tokenized,
                target_tokenized=posterior_model_and_tokenizer[1].encode(target, return_tensors="pt").squeeze(0),
                entity_tokenized=posterior_model_and_tokenizer[1].encode(
                    f" {entity}", return_tensors="pt", add_special_tokens=False
                ).squeeze(0),
                model=posterior_model_and_tokenizer[0],
                tokenizer=posterior_model_and_tokenizer[1],
                verbose=verbose,
            )
        entity_probs.append([prior_entity_prob, posterior_entity_prob])

    return entity_probs


def compute_entitity_probability(
    input_tokenized: torch.Tensor,
    target_tokenized: torch.Tensor,
    entity_tokenized: torch.Tensor,
    model: BartForConditionalGeneration,
    tokenizer = BartTokenizer,
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


def build_causal_masked_inputs_and_targets(
    example: XEntExample,
) -> Tuple[List[str], List[str], List[str]]:
    """
    For a given example from the XEnt dataset, return a tuple of 2 lists:
    a list of features and targets respectively for a causal mask filling task.

    Example output:
    (
        ['Sydney has marked the first anniversary of the siege at the <mask>'],
        ['Sydney has marked the first anniversary of the siege at the Waverley']
    )

    """

    inputs, targets, entities = [], [], []
    prediction = example["prediction"]

    for entity in example["entities"]:
        inputs.append(prediction[0 : entity["start"]] + "<mask>")
        targets.append(prediction[: entity["end"]])
        entities.append(entity["ent"])

    return inputs, targets, entities


def build_masked_inputs_and_targets(
    example: XEntExample,
) -> Tuple[List[str], List[str], List[str]]:
    """
    For a given example from the XEnt dataset, return a tuple of 4 lists:
    - list of non-causal masked inputs
    - list of targets (summary with masked filled in)
    - list of masked entities
    - list of source articles for each summary

    Example output:
    (
        ['Sydney has marked the first anniversary of the siege at the <mask>'],
        ['Sydney has marked the first anniversary of the siege at the Waverley']
    )

    """

    inputs, targets, entities, sources = [], [], [], []
    prediction = example["prediction"]

    for entity in example["entities"]:
        masked_input = (
            prediction[0 : entity["start"]] + "<mask>" + prediction[entity["end"] :]
        )
        inputs.append(masked_input)
        targets.append(prediction)
        entities.append(entity["ent"])
        sources.append(example["source"])

    return inputs, targets, entities, sources


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    dataset = load_xent("test")

    for idx, example in enumerate(dataset):
        inputs, targets, entities = build_causal_masked_inputs_and_targets(example)
        prior_probs = compute_prior_probs(inputs, targets, entities, (model, tokenizer))
