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
import pprint


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL_GOLD = "gold"


def get_labeled_entities(sums_by_id, gold_metadata, xsum_test, should_annotate):
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

    if should_annotate and count_entities(unknown_entities) > 0:
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


def compute_metrics(sums_by_id, gold_sums, gold_metadata, xsum_test, should_annotate):
    labeled_ents = get_labeled_entities(
        sums_by_id, gold_metadata, xsum_test, should_annotate
    )
    summary_results = {}
    counters = {
        "factual": 0,
        "non_factual": 0,
        "unknown": 0,
        "entities": 0,
        "failed": 0,
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
    }
    for value in ANNOTATION_LABELS.values():
        counters[value] = 0

    for sum_id, summary in sums_by_id.items():
        source = xsum_test[sum_id]["document"]
        reference = gold_sums[sum_id]["summary"]

        non_factual = False
        has_unknown = False

        # TODO: handle rogue score computation for failed gen
        if summary != SUMMARY_FAILED_GENERATION:
            rouge_scores = rouge([summary], [reference])
            counters["rouge1"].append(rouge_scores["rouge1"]["f1"])
            counters["rouge2"].append(rouge_scores["rouge2"]["f1"])
            counters["rougeL"].append(rouge_scores["rougeL"]["f1"])

        if summary == SUMMARY_FAILED_GENERATION:
            non_factual = True
            counters["failed"] += 1

        for ent in labeled_ents[sum_id]:
            counters["entities"] += 1
            counters[ent["label"]] += 1
            if (
                ent["label"] == ANNOTATION_LABELS["Non-factual"]
                or ent["label"] == ANNOTATION_LABELS["Intrinsic"]
            ):
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
        "summaries": {
            "total": len(sums_by_id),
            "factual": counters["factual"] / len(sums_by_id),
            "non_factual": counters["non_factual"] / len(sums_by_id),
            "unknown": counters["unknown"] / len(sums_by_id),
            "failed": counters["failed"],
        },
        "entities": {
            "total": counters["entities"],
        },
        "rouge1": np.mean(counters["rouge1"]),
        "rouge2": np.mean(counters["rouge2"]),
        "rougeL": np.mean(counters["rougeL"]),
    }
    for value in ANNOTATION_LABELS.values():
        metrics["entities"][value] = counters[value]

    return metrics, summary_results


def load_summaries_from_logs(path, max_iterations=5):
    with open(path, "r") as f:
        logs = json.load(f)

    sorted_keys = sorted([int(x) for x in logs["iterations"].keys()])

    sums_by_id = {}
    for idx in sorted_keys:
        summaries = logs["iterations"][str(idx)]["summaries"]
        for sum_id, data in summaries.items():
            sums_by_id[sum_id] = data["summary"]
        if idx + 1 == max_iterations:
            break
    return sums_by_id


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", type=bool, default=False)
    parser.add_argument("--test_size", type=int, default=60)
    args = parser.parse_args()

    gold_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    gold_sums = get_summaries(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    xsum_test = load_xsum_dict("test")
    test_set_ids = {
        k for (k, v) in load_test_set(xsum_test, gold_metadata, args.test_size)
    }

    print(f"Test results for {len(test_set_ids)} summaries")

    MODEL_RESULTS = {
        # "Debug FBS w/ oracle, i=5": load_summaries_from_logs(
        #     "results/debug-oracle.json", max_iterations=5
        # ),
        # "Debug FBS w/ classifier, i=5": load_summaries_from_logs(
        #     "results/debug-classifier.json", max_iterations=5
        # ),
        "Test FBS w/ oracle, i=2": load_summaries_from_logs(
            "results/xent-test-oracle.json", max_iterations=2
        ),
        "Test FBS w/ oracle, i=5": load_summaries_from_logs(
            "results/xent-test-oracle.json", max_iterations=5
        ),
        "Test FBS w/ classifier, i=5": load_summaries_from_logs(
            "results/xent-test-classifier.json", max_iterations=5
        ),
        # "Test FBS w/ bad classifier, i=5": load_summaries_from_logs(
        #     "results/test-bad-classifier.json", max_iterations=5
        # ),
    }
    for sumtool_name in [
        "facebook-bart-large-xsum",  # Baseline
        "chen-corrector",  # Chen. et al replication project
        "entity-filter-v2",  # Nan. et al
    ]:
        dataset = get_summaries(SUMTOOL_DATASET, sumtool_name)
        MODEL_RESULTS[sumtool_name] = {
            sum_id: x["summary"] for sum_id, x in dataset.items()
        }

    metrics = {}
    for label, sums_by_id in MODEL_RESULTS.items():
        filtered_sums_by_id = {
            sum_id: x for sum_id, x in sums_by_id.items() if sum_id in test_set_ids
        }
        print(f"Model: {label}")
        pp.pprint(
            compute_metrics(
                filtered_sums_by_id, gold_sums, gold_metadata, xsum_test, args.annotate
            )[0]
        )
