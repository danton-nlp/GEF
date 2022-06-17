from typing import Dict, List, Tuple, TypedDict
import json
from datasets import load_dataset
import random
from sumtool.storage import get_summary_metrics, get_summaries

from src.generation_utils import SUMMARY_FAILED_GENERATION


class XSumDoc(TypedDict):
    document: str
    summary: str
    id: str


class XEntExample(TypedDict):
    source: str
    reference: str
    prediction: str
    entities: List[Dict]


def load_xsum_dict(split) -> Dict[str, XSumDoc]:
    return {x["id"]: x for x in load_dataset("xsum")[split]}


def load_xent(split: str) -> List[XEntExample]:
    if split not in ["test", "train"]:
        raise ValueError("must be test or train")
    with open(f"./data/xent/{split}.json", "r") as f:
        return json.load(f)


def persist_example_with_probs(
    output_filepath: str,
    output_dataset: List[dict],
    output_dataset_idx: int,
    example: XEntExample,
    entity_texts: List[str],
    entity_probs: List[Tuple[float]],
):
    xent_example_with_probs = example.copy()
    entity_probs_with_names = dict(zip(entity_texts, entity_probs))

    for entity in xent_example_with_probs["entities"]:
        prior, posterior = entity_probs_with_names[entity["ent"]]
        entity["prior_prob"] = prior
        entity["posterior_prob"] = posterior

    output_dataset[output_dataset_idx] = xent_example_with_probs
    with open(output_filepath, "w") as f:
        json.dump(output_dataset, f, indent=2)


DEBUG_IDS = {
    "34361828",
    "36456002",  # Lukaku
    "24403775",
    "32112735",  # One direction split
    "36203675",  # Dementia mobile game researchers
    "17996567",
    "36396523",  # Kaka
    "39368095",
    "37066389",  # "Omar Martinez"
    "37615223",
}


def load_debug_subset(xsum_test):
    return {
        sum_id: v["document"] for sum_id, v in xsum_test.items() if sum_id in DEBUG_IDS
    }


def load_shuffled_test_split(xsum_test, data_subset: str, N=100) -> Dict[str, str]:
    with open("data/xsum_shuffled_test_splits.json", "r") as f:
        shuffled_test_splits = json.load(f)

    summaries: List[Tuple[str, XSumDoc]] = [
        (sum_id, xsum_test[sum_id]) for sum_id in shuffled_test_splits[data_subset]
    ]

    return {k: v["document"] for (k, v) in summaries[:N]}


def split_batches(lst, size):
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def load_summaries_from_logs(path, max_iterations=5):
    with open(path, "r") as f:
        logs = json.load(f)

    sorted_keys = sorted([int(x) for x in logs["iterations"].keys()])

    sums_by_id = {}
    sum_ents_by_id = {}
    failed_sums_by_id = {}
    for iteration_idx in sorted_keys:
        summaries = logs["iterations"][str(iteration_idx)]["summaries"]
        for sum_id, data in summaries.items():
            # If summary generation failed, we don't update the summary dict
            # effectively falling back to the previously generated summary
            if data["summary"] != SUMMARY_FAILED_GENERATION:
                sums_by_id[sum_id] = data["summary"]
                sum_ents_by_id[sum_id] = data["labeled_entities"]
            else:
                # Keep track of iteration idx when summary generation failed
                failed_sums_by_id[sum_id] = iteration_idx
        if iteration_idx + 1 == max_iterations:
            break
    return (sums_by_id, sum_ents_by_id, failed_sums_by_id)


def get_gold_xsum_data():
    return (get_summaries("xsum", "gold"), get_summary_metrics("xsum", "gold"))
