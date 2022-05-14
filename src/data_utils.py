from typing import Dict, List, TypedDict
import json
from datasets import load_dataset

XSumDoc = TypedDict("XsumDoc", {"document": str, "summary": str, "id": str})


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
