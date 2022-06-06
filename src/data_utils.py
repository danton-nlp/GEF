from typing import Dict, List, Tuple, TypedDict
import json
from datasets import load_dataset
import random

from src.entity_utils import MarkedEntity


class XSumDoc(TypedDict):
    document: str
    summary: str
    id: str


class XEntExample(TypedDict):
    source: str
    reference: str
    prediction: str
    entities: List[MarkedEntity]


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
    "36456002",
    "24403775",
    "32112735",  # One direction split
    "36203675",  # Dementia mobile game researchers
    "17996567",
    "36396523",
    "39368095",
    "37066389",  # "Omar Martinez"
    "37615223",
}


def load_debug_subset(xsum_test):
    return {
        sum_id: v["document"] for sum_id, v in xsum_test.items() if sum_id in DEBUG_IDS
    }


def load_xent_test_set(xsum_test, gold_metadata, N=50, include_debug=False):
    """
    Construct test set from xent test.
    Optionally add debug_ids.
    """
    xent_test_summaries = {
        sum_id: x["document"]
        for sum_id, x in xsum_test.items()
        if "xent-test" in gold_metadata[sum_id]
    }
    rng_data_split = random.Random(42)
    # shuffle and get first N
    test_set = {
        k: v
        for (k, v) in rng_data_split.sample(
            list(xent_test_summaries.items()), len(xent_test_summaries)
        )[:N]
    }
    if include_debug:
        for sum_id in DEBUG_IDS:
            if sum_id not in test_set and "xent-train" not in gold_metadata[sum_id]:
                test_set[sum_id] = xsum_test[sum_id]["document"]

    return rng_data_split.sample(list(test_set.items()), len(test_set))


def load_extrinsic_test_set(
    xsum_test, baseline_metadata, gold_metadata, N, include_debug=False
):
    """
    Construct test set from baseline summaries with extrinsic entities.
    Exclude sums which are in xent train
    Optionally add debug_ids.
    """
    test_summaries = {
        sum_id: x["document"]
        for sum_id, x in xsum_test.items()
        if "xent-train" not in gold_metadata[sum_id]
        and len(
            [
                ent
                for ent in baseline_metadata[sum_id]["entities"]
                if not ent["in_source"]
            ]
        )
        > 0
    }
    print(f"Loading {N}/{len(test_summaries)} from extrinsic test set")
    rng_data_split = random.Random(42)
    # shuffle and get first N
    test_set = {
        k: v
        for (k, v) in rng_data_split.sample(
            list(test_summaries.items()), len(test_summaries)
        )[:N]
    }
    if include_debug:
        for sum_id in DEBUG_IDS:
            if sum_id not in test_set and "xent-train" not in gold_metadata[sum_id]:
                test_set[sum_id] = xsum_test[sum_id]["document"]

    return rng_data_split.sample(list(test_set.items()), len(test_set))


def split_batches(lst, size):
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
