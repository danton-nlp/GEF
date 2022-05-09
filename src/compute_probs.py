from operator import mul
from functools import reduce
from typing import Dict, List, Tuple, TypedDict

from transformers import BartForConditionalGeneration, BartTokenizer

from data_utils import load_xent


class XEntExample(TypedDict):
    source: str
    reference: str
    prediction: str
    entities: List[Dict]


def compute_prior_probs(
    masked_inputs: List[str],
    targets: List[str],
    entities: List[dict],
    model_and_tokenizer: Tuple
) -> Tuple[List, float]:
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

    token_probs = []
    entity_probs = []

    for masked_input, target, entity in zip(masked_inputs, targets, entities):
        print(f'{masked_input=}')
        print(f'{target=}')

        masked_input_tokenized = tokenizer.encode(masked_input, return_tensors="pt")
        target_tokenized = tokenizer.encode(target, return_tensors="pt").squeeze(0)
        entity_tokenized = tokenizer.encode(
            f' {entity}',
            return_tensors='pt',
            add_special_tokens=False
        ).squeeze(0)

        prediction = model.generate(
            masked_input_tokenized,
            num_beams=1,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        target_probs_for_entity = []
        for target_token_id, token_score in zip(target_tokenized.tolist(), prediction.scores):
            target_token = tokenizer.decode(target_token_id, add_special_tokens=False)

            # TODO: refactor to an unsqueeze
            token_prob = token_score.softmax(dim=1)[0, target_token_id]
            print(f"{target_token_id} | '{target_token}' | {token_prob}")
            target_probs_for_entity.append([target_token_id, token_prob])

        token_probs.append(target_probs_for_entity)

        token_id_to_prob = dict(target_probs_for_entity)
        entity_probs.append(
            reduce(
                mul,
                [token_id_to_prob[entity_id].item() for entity_id in entity_tokenized.tolist()],
                1
            )
        )

    return token_probs, entity_probs


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
        entities.append(entity['ent'])


    return inputs, targets, entities


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    dataset = load_xent("test")

    for idx, example in enumerate(dataset):
        inputs, targets = build_causal_masked_inputs_and_targets(example)
        prior_probs = compute_prior_probs(inputs, targets, (model, tokenizer))
