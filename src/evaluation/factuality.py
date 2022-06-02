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
    is_gold,
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
                counters["entities"] += 1
                continue

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

            # Only count entity if summary is not skipped
            if not is_skipped:
                counters["entities"] += 1
                counters[ent["label"]] += 1

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
        non_factual = not is_skipped and (
            non_factual_extrinsic or non_factual_intrinsic
        )
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
