from collections import defaultdict
from typing import List
from src.annotation import prompt_annotation_flow
from src.detect_entities import detect_entities
from src.entity_utils import MarkedEntity, count_entities, filter_entities
from src.oracle import EntityMatchType, get_entity_annotations, oracle_label_entities
from src.entity_factuality import ANNOTATION_LABELS
from src.generation_utils import SUMMARY_FAILED_GENERATION
from src.metrics import rouge
import numpy as np
from tqdm import tqdm
from termcolor import colored


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
        inserted_label = f" [{ent['label']}]"
        before = marked_summary[:position]
        after = marked_summary[position:]
        marked_summary = before + inserted_label + after
        offset += len(inserted_label)
    return marked_summary


def evaluate_summary(
    summary: str,
    source: str,
    reference: str,
    labeled_entities: List[MarkedEntity],
):
    is_gold = summary == reference
    summary_eval = {
        "skipped": False,
        "failed": False,
        "count_entity_total": 0,
        "count_entity_extrinsic": 0,
        "count_entity_label": defaultdict(lambda: 0),
        "has_predicted_non_factual": False,
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
    }
    if summary == SUMMARY_FAILED_GENERATION:
        summary_eval["failed"] = True
        summary_eval["skipped"] = True
    else:
        rouge_scores = rouge([summary], [reference])
        summary_eval["rouge1"] = rouge_scores["rouge1"]["f1"]
        summary_eval["rouge2"] = rouge_scores["rouge2"]["f1"]
        summary_eval["rougeL"] = rouge_scores["rougeL"]["f1"]

        for ent in labeled_entities:
            summary_eval["count_entity_total"] += 1

            # Increment entity label count
            if is_gold:
                if ent["in_source"]:
                    summary_eval["count_entity_label"][
                        ANNOTATION_LABELS["Non-hallucinated"]
                    ] += 1
                else:
                    summary_eval["count_entity_label"][
                        ANNOTATION_LABELS["Factual"]
                    ] += 1
            else:
                summary_eval["count_entity_label"][ent["label"]] += 1

                # Detect non-factual predictions from FBS classifier
                if (
                    "predicted_label" in ent
                    and ent["predicted_label"] == ANNOTATION_LABELS["Non-factual"]
                ):
                    summary_eval["has_predicted_non_factual"] = True

    # Derive count of extrinsic entities from label counts
    for extrinsic_label in [
        ANNOTATION_LABELS["Factual"],
        ANNOTATION_LABELS["Non-factual"],
    ]:
        summary_eval["count_entity_extrinsic"] += summary_eval["count_entity_label"][
            extrinsic_label
        ]

    return summary_eval


def evaluate_factuality(
    sums_by_id,
    sum_ents_by_id,
    gold_sums,
    gold_metadata,
    xsum_test,
    should_annotate,
    entity_match_type: EntityMatchType,
    print_first_n,
    is_fbs,
    is_oracle,
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
    agg_results = {
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
        "count_entity_label": defaultdict(lambda: 0)
    }

    print_counter = 0
    for sum_id, summary in tqdm(sorted(sums_by_id.items(), key=lambda x: x[0])):
        summary_eval = evaluate_summary(
            summary,
            xsum_test[sum_id]["document"],
            gold_sums[sum_id]["summary"],
            labeled_ents[sum_id],
        )
        count_non_factual_extrinsic = summary_eval["count_entity_label"][
            ANNOTATION_LABELS["Non-factual"]
        ]
        count_non_factual_intrinsic = summary_eval["count_entity_label"][
            ANNOTATION_LABELS["Intrinsic"]
        ]

        # Init the possible eval states of a summary
        non_factual_extrinsic = False
        non_factual_intrinsic = False
        non_factual = False
        has_unknown = (
            summary_eval["count_entity_label"][ANNOTATION_LABELS["Unknown"]] > 0
        )
        count_extrinsic = 0

        # Skip logic
        is_skipped = False
        if summary_eval["failed"]:
            is_skipped = True
        elif is_fbs and is_oracle and count_non_factual_extrinsic:
            is_skipped = True
        elif is_fbs and summary_eval["has_predicted_non_factual"]:
            is_skipped = True

        # Update evaluation state if summary is not skipped
        if not is_skipped:
            non_factual_extrinsic = count_non_factual_extrinsic > 0
            non_factual_intrinsic = count_non_factual_intrinsic > 0
            non_factual = non_factual_extrinsic or non_factual_intrinsic
            count_extrinsic = summary_eval["count_entity_extrinsic"]

        summary_results[sum_id] = {
            "summary": summary,
            "is_non_factual": non_factual,
            "is_non_factual_extrinsic": non_factual_extrinsic,
            "is_non_factual_intrinsic": non_factual_intrinsic,
            "is_factual": not is_skipped and not non_factual and not has_unknown,
            "is_skipped": is_skipped,
            "has_unknown": has_unknown,
            "has_failed": summary_eval["failed"],
            "count_extrinsic": count_extrinsic,
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

        # Increment aggregated results
        if summary_eval["failed"]:
            agg_results["failed"] += 1

        # Only increment counts if summary is NOT skipped!
        if is_skipped:
            agg_results["skipped"] += 1
        else:
            agg_results["entities"] += summary_eval["count_entity_total"]
            for key, value in summary_eval["count_entity_label"].items():
                agg_results["count_entity_label"][key] += value

            if non_factual:
                agg_results["non_factual"] += 1
            elif has_unknown:
                agg_results["unknown"] += 1
            else:
                agg_results["factual"] += 1

            if non_factual_extrinsic:
                agg_results["non_factual_extrinsic"] += 1
            if non_factual_intrinsic:
                agg_results["non_factual_intrinsic"] += 1

            if count_extrinsic > 0:
                agg_results["sum_with_extrinsic"] += 1

            agg_results["rouge1"].append(summary_eval["rouge1"])
            agg_results["rouge2"].append(summary_eval["rouge2"])
            agg_results["rougeL"].append(summary_eval["rougeL"])

    total = len(sums_by_id)
    metrics = {
        "summaries": {
            "total": total,
            "factual": agg_results["factual"] / total,
            "non_factual": agg_results["non_factual"] / total,
            "non_factual_extrinsic": agg_results["non_factual_extrinsic"] / total,
            "non_factual_intrinsic": agg_results["non_factual_intrinsic"] / total,
            "skipped": agg_results["skipped"] / total,
            "failed": agg_results["failed"],
            "unknown": agg_results["unknown"] / total,
            "sum_with_extrinsic": agg_results["sum_with_extrinsic"] / total,
            "ents_per_sum": agg_results["entities"] / (total - agg_results["skipped"]),
        },
        "entities": {
            "total": agg_results["entities"],
        },
        "rouge1": np.mean(agg_results["rouge1"]),
        "rouge2": np.mean(agg_results["rouge2"]),
        "rougeL": np.mean(agg_results["rougeL"]),
    }
    for value in ANNOTATION_LABELS.values():
        metrics["entities"][value] = agg_results["count_entity_label"][value]

    return metrics, summary_results
