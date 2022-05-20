from typing import List
from src.annotation import prompt_annotation_flow
from src.data_utils import load_extrinsic_test_set, load_xent_test_set, load_xsum_dict
from src.detect_entities import detect_entities
from sumtool.storage import get_summary_metrics, get_summaries
from src.entity_utils import MarkedEntity, count_entities, filter_entities
from src.oracle import get_entity_annotations, oracle_label_entities
from src.entity_factuality import ANNOTATION_LABELS
from src.generation_utils import SUMMARY_FAILED_GENERATION
from src.metrics import rouge
import argparse
import json
import numpy as np
import pprint
import pandas as pd
from termcolor import colored
from tqdm import tqdm


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL_GOLD = "gold"
SUMTOOL_MODEL_BASELINE = "facebook-bart-large-xsum"
pp = pprint.PrettyPrinter(indent=2)


def get_labeled_entities(
    sums_by_id,
    sum_ents_by_id,
    gold_metadata,
    xsum_test,
    should_annotate,
    entity_match_type,
):
    # Detect entities if they're not cached
    if len(sum_ents_by_id) == 0:
        for sum_id, summary in sums_by_id.items():
            sum_ents_by_id[sum_id] = detect_entities(
                summary, xsum_test[sum_id]["document"]
            )
    labeled_ents = oracle_label_entities(
        sum_ents_by_id,
        get_entity_annotations(sum_ents_by_id.keys(), gold_metadata),
        entity_match_type,
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
                sum_ents_by_id,
                get_entity_annotations(sum_ents_by_id.keys(), gold_metadata),
            )
    return labeled_ents


def mark_entities(summary, labeled_entities: List[MarkedEntity]):
    marked_summary = summary
    offset = 0
    for ent in sorted(labeled_entities, key=lambda x: x["start"]):
        position = ent["end"] + offset
        inserted_label = " [" + ent["label"] + "]"
        before = marked_summary[:position]
        after = marked_summary[position:]
        marked_summary = before + inserted_label + after
        offset += len(inserted_label)
    return marked_summary


def compute_metrics(
    sums_by_id,
    sum_ents_by_id,
    gold_sums,
    gold_metadata,
    xsum_test,
    should_annotate,
    entity_match_type,
    print_first_n,
    is_fbs,
    is_gold
):
    labeled_ents = get_labeled_entities(
        sums_by_id,
        sum_ents_by_id,
        gold_metadata,
        xsum_test,
        should_annotate,
        entity_match_type,
    )
    summary_results = {}
    counters = {
        "factual": 0,
        "non_factual": 0,
        "non_factual_intrinsic": 0,
        "non_factual_extrinsic": 0,
        "unknown": 0,
        "entities": 0,
        "skipped": 0,
        "failed": 0,
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "sum_with_extrinsic": 0,
    }
    for value in ANNOTATION_LABELS.values():
        counters[value] = 0

    print_counter = 0
    for sum_id, summary in tqdm(sorted(sums_by_id.items(), key=lambda x: x[0])):
        source = xsum_test[sum_id]["document"]
        reference = gold_sums[sum_id]["summary"]

        non_factual_extrinsic = False
        non_factual_intrinsic = False
        is_skipped = False
        has_unknown = False
        has_failed = False

        if summary == SUMMARY_FAILED_GENERATION:
            is_skipped = True
            has_failed = True
            counters["failed"] += 1
        else:
            rouge_scores = rouge([summary], [reference])
            counters["rouge1"].append(rouge_scores["rouge1"]["f1"])
            counters["rouge2"].append(rouge_scores["rouge2"]["f1"])
            counters["rougeL"].append(rouge_scores["rougeL"]["f1"])

        n_extrinsic = 0
        for ent in labeled_ents[sum_id]:
            if is_gold:
                if ent["in_source"]:
                    counters[ANNOTATION_LABELS["Factual"]] += 1
                else:
                    counters[ANNOTATION_LABELS["Non-hallucinated"]] += 1
                continue
            counters["entities"] += 1
            counters[ent["label"]] += 1

            # SKIP LOGIC for FBS
            if is_fbs:
                is_oracle = "predicted_label" not in ent
                if is_oracle and ent["label"] == ANNOTATION_LABELS["Non-factual"]:
                    is_skipped = True
                elif (
                    not is_oracle
                    and ent["predicted_label"] == ANNOTATION_LABELS["Non-factual"]
                ):
                    is_skipped = True

            if not is_skipped and ent["label"] == ANNOTATION_LABELS["Non-factual"]:
                non_factual_extrinsic = True
            elif ent["label"] == ANNOTATION_LABELS["Intrinsic"]:
                non_factual_intrinsic = True
            elif ent["label"] == "Unknown":
                has_unknown = True

            if ent["label"] in [
                ANNOTATION_LABELS["Factual"],
                ANNOTATION_LABELS["Non-factual"],
            ]:
                n_extrinsic += 1
        non_factual = not is_skipped and (non_factual_extrinsic or non_factual_intrinsic)
        summary_results[sum_id] = {
            "summary": summary,
            "is_non_factual": non_factual,
            "is_non_factual_extrinsic": non_factual_extrinsic,
            "is_non_factual_intrinsic": non_factual_intrinsic,
            "is_factual": not is_skipped and not non_factual and not has_unknown,
            "has_unknown": has_unknown,
            "is_skipped": is_skipped,
            "has_failed": has_failed,
            "n_extrinsic": n_extrinsic,
        }

        if print_counter < print_first_n:
            print(f"----XSUM ID {sum_id}----")
            print(mark_entities(summary, labeled_ents[sum_id]))
            print(
                f"Is factual? {colored(str(not non_factual), 'red' if non_factual else 'green')}"
            )
            print()
            print(f"GT summary: {xsum_test[sum_id]['summary']}")
            print("----")
            print()
            print_counter += 1

        if is_skipped:
            counters["skipped"] += 1
        elif non_factual:
            counters["non_factual"] += 1
        elif has_unknown:
            counters["unknown"] += 1
        else:
            counters["factual"] += 1

        if not is_skipped:
            if non_factual_extrinsic:
                counters["non_factual_extrinsic"] += 1
            if non_factual_intrinsic:
                counters["non_factual_intrinsic"] += 1

        if n_extrinsic > 0:
            counters["sum_with_extrinsic"] += 1

    total = len(sums_by_id)
    metrics = {
        "summaries": {
            "total": total,
            "factual": counters["factual"] / total,
            "non_factual": counters["non_factual"] / total,
            "non_factual_extrinsic": counters["non_factual_extrinsic"] / total,
            "non_factual_intrinsic": counters["non_factual_intrinsic"] / total,
            "skipped": counters["skipped"] / total,
            "failed": counters["failed"],
            "unknown": counters["unknown"] / total,
            "sum_with_extrinsic": counters["sum_with_extrinsic"] / total,
            "ents_per_sum": counters["entities"] / (total - counters["skipped"]),
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
    sum_ents_by_id = {}
    for idx in sorted_keys:
        summaries = logs["iterations"][str(idx)]["summaries"]
        for sum_id, data in summaries.items():
            sums_by_id[sum_id] = data["summary"]
            sum_ents_by_id[sum_id] = data["labeled_entities"]
        if idx + 1 == max_iterations:
            break
    return (sums_by_id, sum_ents_by_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", type=bool, default=True)
    parser.add_argument("--data_subset", type=str, default="test-extrinsic")
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--entity_label_match", type=str, default="contained")
    parser.add_argument("--print_first_n", type=int, default=0)
    parser.add_argument("--model_filter", type=str, default="")
    args = parser.parse_args()

    gold_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    baseline_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_BASELINE)
    gold_sums = get_summaries(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    xsum_test = load_xsum_dict("test")
    if args.data_subset == "full":
        test_set_ids = set(xsum_test.keys())
    elif args.data_subset == "full-extrinsic":
        test_set = load_extrinsic_test_set(
            xsum_test, baseline_metadata, gold_metadata, 10000
        )
        test_set_ids = {k for (k, v) in test_set}
    else:
        test_set = (
            load_extrinsic_test_set(
                xsum_test, baseline_metadata, gold_metadata, args.test_size
            )
            if args.data_subset == "test-extrinsic"
            else load_xent_test_set(xsum_test, gold_metadata, args.test_size)
        )
        test_set_ids = {k for (k, v) in test_set}

    print(f"Test results for {len(test_set_ids)} summaries")

    if args.data_subset == "full" or args.data_subset == "full-extrinsic":
        MODEL_RESULTS = {
            "fbs_classifier": load_summaries_from_logs(
                "results/full-classifier-knnv1.json", max_iterations=5
            ),
        }
    else:
        MODEL_RESULTS = {
            # "Test FBS w/ oracle, i=2": load_summaries_from_logs(
            #     f"results/{args.data_subset}-oracle.json", max_iterations=2
            # ),
            "fbs_oracle": load_summaries_from_logs(
                f"results/{args.data_subset}-oracle.json", max_iterations=5
            ),
            # "Test FBS w/ classifier v0, i=5": load_summaries_from_logs(
            #     f"results/{args.data_subset}-classifier-knnv0.json", max_iterations=5
            # ),
            "fbs_classifier": load_summaries_from_logs(
                f"results/{args.data_subset}-classifier-knnv1.json", max_iterations=5
            ),
            "fbs_classifier_i10": load_summaries_from_logs(
                f"results/{args.data_subset}-classifier-knnv1.json", max_iterations=10
            ),
            # "Test FBS w/ bad classifier, i=5": load_summaries_from_logs(
            #     "results/test-bad-classifier.json", max_iterations=5
            # ),
        }
    for sumtool_name, model_label in [
        ("facebook-bart-large-xsum", "baseline"),
        ("chen-corrector", "corrector"),  # Chen. et al replication project
        ("gold", "gold"),  # Chen. et al replication project
        # ("entity-filter-v2", "filtered"),  # Nan. et al
    ]:
        dataset = get_summaries(SUMTOOL_DATASET, sumtool_name)
        MODEL_RESULTS[model_label] = (
            {sum_id: x["summary"] for sum_id, x in dataset.items()},
            {},
        )

    aggregated_results = []
    summary_results = {}
    for model_label, (sums_by_id, sum_ents_by_id) in MODEL_RESULTS.items():
        if args.model_filter == "" or args.model_filter in model_label:
            filtered_sums_by_id = {
                sum_id: x for sum_id, x in sums_by_id.items() if sum_id in test_set_ids
            }
            filtered_ents_by_id = {
                sum_id: x for sum_id, x in sum_ents_by_id.items() if sum_id in test_set_ids
            }
            print(f"Model: {model_label}")
            agg_metrics, summaries = compute_metrics(
                filtered_sums_by_id,
                filtered_ents_by_id,
                gold_sums,
                gold_metadata,
                xsum_test,
                args.annotate and "gold" not in model_label.lower(),
                args.entity_label_match,
                args.print_first_n,
                is_fbs="fbs" in model_label.lower(),
                is_gold="gold" in model_label.lower()
            )
            pp.pprint(agg_metrics)
            aggregated_results.append(
                [
                    model_label,
                    agg_metrics["summaries"]["total"],
                    agg_metrics["summaries"]["factual"],
                    agg_metrics["summaries"]["non_factual"],
                    agg_metrics["summaries"]["non_factual_extrinsic"],
                    agg_metrics["summaries"]["non_factual_intrinsic"],
                    agg_metrics["summaries"]["unknown"],
                    agg_metrics["summaries"]["skipped"],
                    agg_metrics["summaries"]["failed"],
                    agg_metrics["summaries"]["ents_per_sum"],
                    agg_metrics["summaries"]["sum_with_extrinsic"],
                    agg_metrics["rouge1"],
                    agg_metrics["rouge2"],
                    agg_metrics["rougeL"],
                ]
            )
            for sum_id, sum_metrics in summaries.items():
                if sum_id not in summary_results:
                    summary_results[sum_id] = {}
                for metric, value in sum_metrics.items():
                    summary_results[sum_id][f"{model_label}_{metric}"] = value

    df_aggregated = pd.DataFrame(
        aggregated_results,
        columns=[
            "model",
            "total",
            "factual",
            "non_factual",
            "non_factual_extrinsic",
            "non_factual_intrinsic",
            "unknown",
            "skipped",
            "failed",
            "ents_per_sum",
            "sum_with_extrinsic",
            "rouge1",
            "rouge2",
            "rougeL",
        ],
    )

    df_aggregated.to_csv(
        f"results/{args.data_subset}-{args.test_size}.csv", index=False
    )
    df_summaries = pd.DataFrame.from_dict(summary_results, orient="index")
    df_summaries.to_json(f"results/{args.data_subset}-{args.test_size}-summaries.json")
