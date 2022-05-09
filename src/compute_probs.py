from typing import Dict, List, Tuple, TypedDict

from transformers import BartForConditionalGeneration, BartTokenizer

from data_utils import load_xent


class XEntExample(TypedDict):
    source: str
    reference: str
    prediction: str
    entities: List[Dict]


def compute_prior_probs(
    masked_inputs: List[str], targets: List[str], model_and_tokenizer=Tuple
):
    if len(masked_inputs) != len(targets):
        raise ValueError("number of inputs is not the same as the number of targets")

    model, tokenizer = model_and_tokenizer

    def prefix_allowed_tokens_fn(_batch_id, input_ids):
        current_step = len(input_ids) - 1
        return target_tokenized[current_step].tolist()

    for masked_input, target in zip(masked_inputs, targets):
        print('MASKED INPUT:', masked_input)
        print(f'{target=}')
        masked_input_tokenized = tokenizer.encode(masked_input, return_tensors="pt")

        target_tokenized = tokenizer.encode(target, return_tensors="pt").squeeze(0)

        prediction = model.generate(
            masked_input_tokenized,
            num_beams=1,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        for target_token, token_score in zip(target_tokenized.tolist(), prediction.scores):
            target_parts = tokenizer.decode(target_token, add_special_tokens=False)

            # TODO: refactor to an unsqueeze
            token_score = token_score.softmax(dim=1)[0, target_token]
            print(f"{target_token} | '{target_parts}' | {token_score}")


def build_causal_masked_inputs_and_targets(
    example: XEntExample,
) -> Tuple[List[str], List[str]]:
    """
    For a given example from the XEnt dataset, return a tuple of 2 lists:
    a list of features and targets respectively for a causal mask filling task.

    Example output:
    (
        ['Sydney has marked the first anniversary of the siege at the <mask>'],
        ['Sydney has marked the first anniversary of the siege at the Waverley']
    )

    """

    inputs, targets = [], []
    prediction = example["prediction"]

    for entity in example["entities"]:
        inputs.append(prediction[0 : entity["start"]] + "<mask>")
        targets.append(prediction[: entity["end"]])

    return inputs, targets


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    dataset = load_xent("test")

    for idx, example in enumerate(dataset):
        inputs, targets = build_causal_masked_inputs_and_targets(example)
        prior_probs = compute_prior_probs(inputs, targets, (model, tokenizer))
