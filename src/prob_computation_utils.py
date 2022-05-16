from typing import List, Tuple, TypedDict
from src.data_utils import XEntExample
from src.entity_utils import MarkedEntity


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
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    For a given example from the XEnt dataset, return a tuple of 4 lists:
    - list of non-causal masked inputs
    - list of targets (summary with masked filled in)
    - list of masked entities
    - list of source articles for each summary
    - list of entity labels

    Example output:
    (
        ['Sydney has marked the first anniversary of the siege at the <mask>'],
        ['Sydney has marked the first anniversary of the siege at the Waverley']
    )

    """

    inputs, targets, entities, sources, labels = [], [], [], [], []
    prediction = example["prediction"]

    for entity in example["entities"]:
        masked_input = (
            prediction[0 : entity["start"]] + "<mask>" + prediction[entity["end"] :]
        )
        inputs.append(masked_input)
        targets.append(prediction)

        if entity['start'] != 0:
            entity_to_tokenize = " " + entity['ent']
        else:
            entity_to_tokenize = entity['ent']
        entities.append(entity_to_tokenize)

        sources.append(example["source"])
        labels.append(entity["label"])

    return inputs, targets, entities, sources, labels


class InferenceInput(TypedDict):
    source: str
    prediction: str
    entities: List[MarkedEntity]


def build_masked_inputs_and_targets_for_inference(
    inference_input: InferenceInput,
) -> Tuple[List[str], List[str], List[str], List[str]]:
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
    prediction = inference_input["prediction"]

    for entity in inference_input["entities"]:
        masked_input = (
            prediction[0 : entity["start"]] + "<mask>" + prediction[entity["end"] :]
        )
        inputs.append(masked_input)
        targets.append(prediction)
        entities.append(entity["ent"])
        sources.append(inference_input["source"])

    return inputs, targets, entities, sources
