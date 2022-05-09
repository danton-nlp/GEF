from typing import Dict, TypedDict
from datasets import load_dataset

XSumDoc = TypedDict("XsumDoc", {
    "document": str,
    "summary": str,
    "id": str
})


def load_xsum_dict(split) -> Dict[str, XSumDoc]:
    return {x["id"]: x for x in load_dataset("xsum")[split]}
import json

# TODO: Likely define the types in the XEnt dataset.
def load_xent(split: str) -> dict:
    if split not in ['test', 'train']:
        raise ValueError('must be test or train')
    with open(f"./data/xent/{split}.json", "r") as f:
        return json.load(f)
