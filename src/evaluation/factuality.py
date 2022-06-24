from collections import defaultdict
from typing import DefaultDict, List, Optional, TypedDict
from src.annotation import prompt_annotation_flow
from src.detect_entities import detect_entities
from src.entity_utils import MarkedEntity, count_entities, filter_entities
from src.oracle import EntityMatchType, get_entity_annotations, oracle_label_entities
from src.entity_factuality import ANNOTATION_LABELS
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
    force_annotation_flow=False
):
    # Detect entities if they're not cached
    for sum_id, summary in sums_by_id.items():
        if sum_id not in sum_ents_by_id:
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
            force_annotation_flow
        )
        if result is not False:
            labeled_ents = oracle_label_entities(
                sum_ents_by_id,
                get_entity_annotations(sum_ents_by_id.keys(), gold_metadata),
                entity_match_type
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


class SummaryEval(TypedDict):
    skipped: bool
    failed: bool
    count_entity_total: int
    count_entity_extrinsic: int
    entity_extrinsic_factuality_ratio: Optional[float]
    count_entity_label: DefaultDict[str, int]
    count_entity_type: DefaultDict[str, DefaultDict[str, int]]
    has_predicted_non_factual: bool
    has_extrinsic_and_fully_factual: int
    rouge1: Optional[float]
    rouge2: Optional[float]
    rougeL: Optional[float]


def evaluate_summary(
    summary: str,
    source: str,
    reference: str,
    labeled_entities: List[MarkedEntity],
    compute_rouge: bool,
) -> SummaryEval:
    is_gold = summary == reference
    summary_eval: SummaryEval = {
        "skipped": False,
        "failed": False,
        "count_entity_total": 0,
        "count_entity_extrinsic": 0,
        "entity_extrinsic_factuality_ratio": None,
        "count_entity_label": defaultdict(lambda: 0),
        "count_entity_type": defaultdict(lambda: defaultdict(lambda: 0)),
        "has_predicted_non_factual": False,
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
        "has_extrinsic_and_fully_factual": 0,
    }
    if compute_rouge:
        rouge_scores = rouge([summary], [reference])
        summary_eval["rouge1"] = rouge_scores["rouge1"]["f1"]
        summary_eval["rouge2"] = rouge_scores["rouge2"]["f1"]
        summary_eval["rougeL"] = rouge_scores["rougeL"]["f1"]

    for ent in labeled_entities:
        summary_eval["count_entity_total"] += 1

        if is_gold:
            if ent["in_source"]:
                entity_label = ANNOTATION_LABELS["Non-hallucinated"]
            else:
                entity_label = ANNOTATION_LABELS["Factual"]
        else:
            entity_label = str(ent["label"])

            # Detect non-factual predictions from GEF classifier
            if (
                "predicted_label" in ent
                and ent["predicted_label"] == ANNOTATION_LABELS["Non-factual"]
            ):
                summary_eval["has_predicted_non_factual"] = True

        # Increment entity label count
        summary_eval["count_entity_label"][entity_label] += 1
        summary_eval["count_entity_type"][ent["type"]][entity_label] += 1
        summary_eval["count_entity_type"][ent["type"]]["total"] += 1

        if not ent["in_source"]:
            summary_eval["count_entity_extrinsic"] += 1

    # If there are any extrinsic hallucinations, then keep the ratio set to
    # None to flag that the ratio isn't helpful.
    #
    # Alternative could be that if there are no extrinsic hallucinations
    # that we set the ratio to 1, but we think that unfairly inflates the
    # the aggregate stat for models that include fewer extrinsic
    # hallucinations.
    count_entity_labels = summary_eval["count_entity_label"]
    count_total_extrinsic_hallucinations = (
        count_entity_labels[ANNOTATION_LABELS["Factual"]]
        + count_entity_labels[ANNOTATION_LABELS["Non-factual"]]
    )
    # Replace count entity extrinsic with labeled if there are no unknowns
    # We rely on human labels rather than ent["in_source"], i.e.
    # when "Ghana" is in source, but "Ghanian" is not, we still treat "Ghanian" as intrinsic
    if count_entity_labels[ANNOTATION_LABELS["Unknown"]] == 0:
        summary_eval["count_entity_extrinsic"] = count_total_extrinsic_hallucinations
    if count_total_extrinsic_hallucinations > 0:
        summary_eval["entity_extrinsic_factuality_ratio"] = (
            count_entity_labels[ANNOTATION_LABELS["Factual"]]
            / count_total_extrinsic_hallucinations
        )

    if (
        count_total_extrinsic_hallucinations > 0
        and summary_eval["entity_extrinsic_factuality_ratio"] == 1
    ):
        summary_eval["has_extrinsic_and_fully_factual"] = 1
    return summary_eval


def evaluate_factuality(
    sums_by_id,
    sum_ents_by_id,
    failed_sums_by_id,
    gold_sums,
    gold_metadata,
    xsum_test,
    should_annotate,
    entity_match_type: EntityMatchType,
    print_first_n,
    is_gef,
    is_oracle,
    compute_rouge=True,
    count_skips=False,
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
        "sum_with_extrinsic_factual": [],
        "extrinsic_factuality_ratios": [],
        "count_entity_label": defaultdict(lambda: 0),
        "count_entity_type": defaultdict(lambda: defaultdict(lambda: 0)),
        "count_entity_extrinsic": 0,
    }

    print_counter = 0
    for sum_id, summary in tqdm(sorted(sums_by_id.items(), key=lambda x: x[0])):
        summary_eval = evaluate_summary(
            summary,
            xsum_test[sum_id]["document"],
            gold_sums[sum_id]["summary"],
            labeled_ents[sum_id],
            compute_rouge,
        )
        if sum_id in failed_sums_by_id:
            summary_eval["failed"] = True
            summary_eval["skipped"] = True
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
        if count_skips:
            if summary_eval["failed"]:
                is_skipped = True
            elif is_gef and is_oracle and count_non_factual_extrinsic:
                is_skipped = True
            elif is_gef and summary_eval["has_predicted_non_factual"]:
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
            "count_extrinsic_factual": summary_eval["count_entity_label"][
                "Factual Hallucination"
            ],
            "extrinsic_factuality_ratio": summary_eval[
                "entity_extrinsic_factuality_ratio"
            ],
            "has_extrinsic_and_fully_factual": summary_eval[
                "has_extrinsic_and_fully_factual"
            ]
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

            for type, label_counts in summary_eval["count_entity_type"].items():
                for label, counts in label_counts.items():
                    agg_results["count_entity_type"][type][label] += counts

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

            if summary_eval["entity_extrinsic_factuality_ratio"]:
                agg_results["extrinsic_factuality_ratios"].append(
                    summary_eval["entity_extrinsic_factuality_ratio"]
                )
            agg_results["sum_with_extrinsic_factual"].append(
                summary_eval["has_extrinsic_and_fully_factual"]
            )

            agg_results["count_entity_extrinsic"] += summary_eval[
                "count_entity_extrinsic"
            ]

            if compute_rouge and not summary_eval["failed"]:
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
            "sum_with_extrinsic_factual": np.mean(
                agg_results["sum_with_extrinsic_factual"]
            ),
            "ents_per_sum": agg_results["entities"] / (total - agg_results["skipped"]),
        },
        "entities": {
            "total": agg_results["entities"],
            "count_extrinsic": agg_results["count_entity_extrinsic"],
            "extrinsic_factuality_ratio": {
                "mean": np.mean(agg_results["extrinsic_factuality_ratios"]),
                "median": np.median(agg_results["extrinsic_factuality_ratios"]),
                "stdev": np.std(agg_results["extrinsic_factuality_ratios"]),
            },
            "type": {},
        },
        "rouge1": np.mean(agg_results["rouge1"]),
        "rouge2": np.mean(agg_results["rouge2"]),
        "rougeL": np.mean(agg_results["rougeL"]),
    }
    for value in ANNOTATION_LABELS.values():
        metrics["entities"][value] = agg_results["count_entity_label"][value]

    for type, label_counts in agg_results["count_entity_type"].items():
        metrics["entities"]["type"][type] = dict(label_counts)

    return metrics, summary_results
