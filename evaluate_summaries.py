from src.data_utils import load_test_set, load_xsum_dict
from src.detect_entities import detect_entities
from sumtool.storage import get_summary_metrics, get_summaries
from src.oracle import get_entity_annotations, oracle_label_entities
from src.entity_factuality import ANNOTATION_LABELS
from src.generation_utils import SUMMARY_FAILED_GENERATION
from src.metrics import rouge
import argparse
import json
import numpy as np


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL_GOLD = "gold"


def compute_metrics(sums_by_id, gold_sums, gold_metadata, xsum_test):
    summary_results = {}
    counters = {
        "factual": 0,
        "non_factual": 0,
        "unknown": 0,
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
    }

    summary_entities = {}
    for sum_id, summary in sums_by_id.items():
        summary_entities[sum_id] = detect_entities(
            summary, xsum_test[sum_id]["document"]
        )
    labeled_ents = oracle_label_entities(
        summary_entities,
        get_entity_annotations(summary_entities.keys(), gold_metadata),
    )

    for sum_id, summary in sums_by_id.items():
        source = xsum_test[sum_id]["document"]
        reference = gold_sums[sum_id]["summary"]
        rouge_scores = rouge([summary], [reference])
        non_factual = False
        has_unknown = False

        counters["rouge1"].append(rouge_scores["rouge1"]["f1"])
        counters["rouge2"].append(rouge_scores["rouge2"]["f1"])
        counters["rougeL"].append(rouge_scores["rougeL"]["f1"])

        if summary == SUMMARY_FAILED_GENERATION:
            non_factual = True

        for ent in labeled_ents[sum_id]:
            if ent["label"] in [
                ANNOTATION_LABELS["Non-factual"],
            ]:
                non_factual = True
            elif ent["label"] == "Unknown":
                has_unknown = True

        if non_factual:
            counters["non_factual"] += 1
        elif has_unknown:
            counters["unknown"] += 1
        else:
            counters["factual"] += 1

    metrics = {
        "factual": counters["factual"] / len(sums_by_id),
        "non_factual": counters["non_factual"] / len(sums_by_id),
        "unknown": counters["unknown"] / len(sums_by_id),
        "rouge1": np.mean(counters["rouge1"]),
        "rouge2": np.mean(counters["rouge2"]),
        "rougeL": np.mean(counters["rougeL"]),
    }

    return metrics, summary_results


def load_summaries_from_logs(path):
    with open(path, "r") as f:
        logs = json.load(f)

    sorted_keys = sorted([int(x) for x in logs["iterations"].keys()])

    sums_by_id = {}
    for idx in sorted_keys:
        summaries = logs["iterations"][str(idx)]["summaries"]
        for sum_id, data in summaries.items():
            sums_by_id[sum_id] = data["summary"]
    return sums_by_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    gold_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    gold_sums = get_summaries(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    xsum_test = load_xsum_dict("test")
    test_set_ids = {k for (k, v) in load_test_set(xsum_test, gold_metadata, 500)}

    print(f"Test results for {len(test_set_ids)} summaries")
    metrics = {}
    # Read from sumtool storage to get other summaries
    sumtool_sums = ["facebook-bart-large-xsum", "entity-filter-v2"]
    for sumtool_name in sumtool_sums:
        dataset = get_summaries(SUMTOOL_DATASET, sumtool_name)
        sums_by_id = {
            sum_id: x["summary"]
            for sum_id, x in dataset.items()
            if sum_id in test_set_ids
        }

        print(f"Model: {sumtool_name}")
        print(compute_metrics(sums_by_id, gold_sums, gold_metadata, xsum_test))

    print("Constrained oracle")
    sums_by_id = load_summaries_from_logs("results/test.json")
    print(compute_metrics(sums_by_id, gold_sums, gold_metadata, xsum_test))
    # Read logs to get our results
