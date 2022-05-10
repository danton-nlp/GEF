from typing import Dict, TypedDict
from datasets import load_dataset

XSumDoc = TypedDict("XsumDoc", {
    "document": str,
    "summary": str,
    "id": str
})


def load_xsum_dict(split) -> Dict[str, XSumDoc]:
    return {x["id"]: x for x in load_dataset("xsum")[split]}
