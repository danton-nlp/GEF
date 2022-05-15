from src.annotation import prompt_annotation_flow
from src.data_utils import load_test_set, load_xsum_dict
from src.detect_entities import detect_entities
from sumtool.storage import get_summary_metrics, get_summaries
from src.entity_utils import count_entities, filter_entities
from src.oracle import get_entity_annotations, oracle_label_entities
from src.entity_factuality import ANNOTATION_LABELS
from src.generation_utils import SUMMARY_FAILED_GENERATION
from src.metrics import rouge
import argparse
import json
import numpy as np


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL_GOLD = "gold"


def get_labeled_entities(sums_by_id, gold_metadata, xsum_test):
    summary_entities = {}
    for sum_id, summary in sums_by_id.items():
        summary_entities[sum_id] = detect_entities(
            summary, xsum_test[sum_id]["document"]
        )
    labeled_ents = oracle_label_entities(
        summary_entities,
        get_entity_annotations(summary_entities.keys(), gold_metadata),
    )
    unknown_entities = filter_entities(
        lambda x: x["label"] == ANNOTATION_LABELS["Unknown"],
        labeled_ents,
    )

    if count_entities(unknown_entities) > 0:
        result = prompt_annotation_flow(
            unknown_entities,
            xsum_test,
            sums_by_id,
            gold_metadata,
        )
        if result is not False:
            labeled_ents = oracle_label_entities(
                summary_entities,
                get_entity_annotations(summary_entities.keys(), gold_metadata),
            )
    return labeled_ents


def compute_metrics(sums_by_id, gold_sums, gold_metadata, xsum_test):
    labeled_ents = get_labeled_entities(sums_by_id, gold_metadata, xsum_test)
    summary_results = {}
    counters = {
        "factual": 0,
        "non_factual": 0,
        "unknown": 0,
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
    }

    for sum_id, summary in sums_by_id.items():
        source = xsum_test[sum_id]["document"]
        reference = gold_sums[sum_id]["summary"]
        rouge_scores = rouge([summary], [reference])
        non_factual = False
        has_unknown = False

        # TODO: handle rogue score computation for failed gen
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

    MODEL_RESULTS = {
        "Constrained oracle": load_summaries_from_logs("results/test.json"),
    }
    for sumtool_name in ["facebook-bart-large-xsum", "entity-filter-v2"]:
        dataset = get_summaries(SUMTOOL_DATASET, sumtool_name)
        MODEL_RESULTS[sumtool_name] = {
            sum_id: x["summary"]
            for sum_id, x in dataset.items()
            if sum_id in test_set_ids
        }

    metrics = {}
    for label, sums_by_id in MODEL_RESULTS.items():
        print(f"Model: {label}")
        print(compute_metrics(sums_by_id, gold_sums, gold_metadata, xsum_test))
