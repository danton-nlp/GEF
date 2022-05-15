from sumtool.storage import get_summary_metrics, get_summaries
from src.data_utils import load_xsum_dict
import json


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL = "gold"

if __name__ == "__main__":
    xsum_test = load_xsum_dict("test")
    gold_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL)
    gold_summaries = get_summaries(SUMTOOL_DATASET, SUMTOOL_MODEL)

    persisted = []
    n_ents = 0
    n_sums = 0
    for sum_id, metadata in gold_metadata.items():
        if "our_annotations" in metadata:
            for gen_summary, entities in metadata["our_annotations"].items():
                sum_object = {
                    "source": xsum_test[sum_id]["document"],
                    "reference": gold_summaries[sum_id]["summary"],
                    "prediction": gen_summary,
                    "entities": entities,
                }
                n_sums +=1
                n_ents += len(entities)
                persisted.append(sum_object)
    with open(f"./data/xent-extended/test.json", "w") as f:
        json.dump(persisted, f, indent=2)

    print(f"Persisted {n_ents} entity annotations for {n_sums} summaries")
