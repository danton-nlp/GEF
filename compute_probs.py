import argparse
import pprint
from typing import List
from tqdm import tqdm

import torch
from transformers.tokenization_utils_base import BatchEncoding
from iterative_constraints import split_batches

from src.data_utils import load_xent
from src.generation_utils import load_model_and_tokenizer, load_bart_xsum_cmlm
from src.prob_computation_utils import build_masked_inputs_and_targets


def compute_probs_for_summary(
    masked_inputs: List[str],
    targets: List[str],
    sources: List[str],
    entities: List[str],
    prior_model_and_tokenizer,
    posterior_model_and_tokenizer,
    verbose: bool = False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 3,
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

    prior_entity_probs: List[float] = []
    posterior_entity_probs: List[float] = []

    needed_data = list(zip(masked_inputs, targets, sources, entities))
    inputs_batched = list(split_batches(needed_data, batch_size))

    for batch in inputs_batched:
        batch_masked_inputs = [x[0] for x in batch]
        batch_targets = [x[1] for x in batch]
        batch_sources = [x[2] for x in batch]
        batch_entities = [x[3] for x in batch]

        if verbose:
            print(f"{batch_masked_inputs=}")
            print(f"{batch_targets=}")

        prior_input_batch_tokenized = prior_model_and_tokenizer[1](
            batch_masked_inputs, return_tensors="pt", padding=True
        ).to(device)

        batch_formatted_posterior_inputs = [
            "<s>"
            + batch_masked_inputs[i].replace("<mask>", "###")
            + "</s>"
            + batch_sources[i]
            for i in range(len(batch_masked_inputs))
        ]

        posterior_input_batch_tokenized = posterior_model_and_tokenizer[1](
            batch_formatted_posterior_inputs,
            add_special_tokens=False,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        prior_target_tokenized = prior_model_and_tokenizer[1](
            batch_targets,
            return_tensors="pt",
            padding=True
        ).to(device)

        posterior_target_tokenized = posterior_model_and_tokenizer[1](
            batch_targets,
            return_tensors="pt",
            padding=True
        ).to(device)

        batch_entities_with_leading_space = [
            " " + entity
            for entity
            in batch_entities
        ]

        prior_entity_tokenized = prior_model_and_tokenizer[1](
            batch_entities_with_leading_space,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        posterior_entity_tokenized = posterior_model_and_tokenizer[1](
            batch_entities_with_leading_space,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        with torch.no_grad():
            batch_prior_entity_probs = compute_entitity_probability(
                input_tokenized=prior_input_batch_tokenized,
                target_tokenized=prior_target_tokenized,
                entity_tokenized=prior_entity_tokenized,
                model=prior_model_and_tokenizer[0],
                tokenizer=prior_model_and_tokenizer[1],
                verbose=verbose,
            )
            prior_entity_probs.extend(batch_prior_entity_probs)

            batch_posterior_entity_probs = compute_entitity_probability(
                input_tokenized=posterior_input_batch_tokenized,
                target_tokenized=posterior_target_tokenized,
                entity_tokenized=posterior_entity_tokenized,
                model=posterior_model_and_tokenizer[0],
                tokenizer=posterior_model_and_tokenizer[1],  # type: ignore
                verbose=verbose,
            )
            posterior_entity_probs.extend(batch_posterior_entity_probs)

    return list(zip(prior_entity_probs, posterior_entity_probs))


def compute_entitity_probability(
    input_tokenized: BatchEncoding,
    target_tokenized: BatchEncoding,
    entity_tokenized: BatchEncoding,
    model,
    tokenizer,
    verbose: bool = False,
) -> List[float]:

    target_input_ids = target_tokenized['input_ids']

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        current_step = len(input_ids) - 1
        return target_input_ids[batch_id, current_step].tolist()

    prediction = model.generate(
        input_tokenized['input_ids'],
        num_beams=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=200,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )

    mask_input_indices = (input_tokenized['input_ids'] == tokenizer.mask_token_id).nonzero()
    entity_token_lengths = entity_tokenized['attention_mask'].count_nonzero(dim=1)
    mask_input_indices_with_length = torch.hstack((
        mask_input_indices, entity_token_lengths.unsqueeze(1)
    ))

    probs_of_entity_tokens = []
    for idx_in_batch, starting_mask_token_pos, num_entity_tokens in mask_input_indices_with_length:
        entity_local_token_probs = []
        for entity_local_idx in range(num_entity_tokens):
            local_index_of_filled_in_token = starting_mask_token_pos + entity_local_idx
            vocab_index_for_entity_token = entity_tokenized['input_ids'][idx_in_batch][entity_local_idx]

            prob_of_entity_token = (
                prediction
                .scores[local_index_of_filled_in_token][idx_in_batch]
                .softmax(dim=0)
                [vocab_index_for_entity_token]
            )
            entity_local_token_probs.append(prob_of_entity_token)
        probs_of_entity_tokens.append(entity_local_token_probs)


    # entity_token_prob_distribs = torch.vstack(
    #     prediction.scores[mask_input_idx : (mask_input_idx + len(entity_tokenized))]
    # ).softmax(dim=1)

    # # TODO: some tensor native operation here? close to `torch.gather`?
    # probs_of_entity_tokens = torch.hstack(
    #     [
    #         entity_token_prob_distribs[i, token_id]
    #         for i, token_id in enumerate(entity_tokenized)
    #     ]
    # )

    # if verbose:
    #     all_probs_at_target_tokens = [
    #         [
    #             tokenizer.decode(target_tokenized[i]),
    #             distrib.softmax(dim=1).squeeze(0)[target_tokenized[i]].item(),
    #         ]
    #         for i, distrib in enumerate(prediction.scores)
    #     ]
    #     pprint.PrettyPrinter(indent=4).pprint(all_probs_at_target_tokens)

    return [
        torch.prod(torch.hstack(entity_probs)).item()
        for entity_probs
        in probs_of_entity_tokens
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'computes prior and posterior probabilities of specified split xent entities.'
    )
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None, help='debug: max number of examples to process')
    parser.add_argument(
        "--xent_split", type=str, default="test", choices=['test', 'train']
    )
    args = parser.parse_args()
    dataset = load_xent(args.xent_split)

    for idx, example in enumerate(tqdm(dataset[:args.max_examples])):
        (
            inputs,
            targets,
            entities,
            sources,
            entity_labels,
        ) = build_masked_inputs_and_targets(example)

        entity_probs = compute_probs_for_summary(
            masked_inputs=inputs,
            targets=targets,
            sources=sources,
            entities=entities,
            batch_size=args.batch_size,
            prior_model_and_tokenizer=load_model_and_tokenizer("facebook/bart-large"),
            posterior_model_and_tokenizer=load_bart_xsum_cmlm(),
        )

        pprint.PrettyPrinter(indent=4).pprint(
            list(zip(entities, entity_labels, entity_probs))
        )
