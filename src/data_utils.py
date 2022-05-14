from typing import Dict, List, TypedDict
import json
from datasets import load_dataset
import random


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


def load_test_set(xsum_test, gold_metadata, N=50):
    """
    Construct test set from xent test.
    Also add debug_ids.
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

    for sum_id in DEBUG_IDS:
        if sum_id not in test_set and "xent-train" not in gold_metadata[sum_id]:
            test_set[sum_id] = xsum_test[sum_id]["document"]

    return rng_data_split.sample(list(test_set.items()), len(test_set))
