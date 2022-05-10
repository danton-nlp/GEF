from sumtool.storage import store_summary_metrics
from src.data_utils import load_xsum_dict
import json


def load_EntFA(split: str):
    with open(f"./data/xent/{split}.json", "r") as f:
        return json.load(f)


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL = "gold"

if __name__ == "__main__":
    EntFA_train = load_EntFA("train")
    EntFA_test = load_EntFA("test")

    xsum_test = load_xsum_dict("test")
    xsum_by_source = {x["document"].replace("\n", " "): x for x in xsum_test.values()}
    xsum_by_gt = {x["summary"].replace("\n", " "): x for x in xsum_test.values()}

    summary_annotations = {}

    missing = 0
    for entfa_sum in EntFA_train + EntFA_test:
        if entfa_sum["source"] in xsum_by_source:
            xsum_doc = xsum_by_source[entfa_sum["source"]]
        elif entfa_sum["reference"] in xsum_by_gt:
            xsum_doc = xsum_by_gt[entfa_sum["reference"]]
        else:
            missing += 1
            continue
        xsum_id = xsum_doc["id"]
        summary_annotations[xsum_id] = {
            "xent": entfa_sum["entities"]
        }

    print(summary_annotations)
    print(f"Missing {missing}, total: {len(summary_annotations)}")
    store_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL, summary_annotations)
