from collections import defaultdict
from typing import Dict
from sumtool.storage import store_summary_metrics

from src.data_utils import XSumDoc
from src.entity_factuality import ANNOTATION_LABELS
from src.entity_utils import MarkedEntityLookup


def persist_updated_annotations(old_metadata, updated_annotations, summaries_by_id):
    updated_metadata = old_metadata.copy()
    for sum_id, new_annotations in updated_annotations.items():
        if sum_id in old_metadata:
            summary = summaries_by_id[sum_id]
            our_annotations = (
                updated_metadata[sum_id]["our_annotations"]
                if "our_annotations" in updated_metadata[sum_id]
                else {}
            )
            if summary not in our_annotations:
                our_annotations[summary] = []

            for annot in new_annotations:
                our_annotations[summary].append(annot)
            updated_metadata[sum_id]["our_annotations"] = our_annotations

    store_summary_metrics("xsum", "gold", updated_metadata)
    return updated_metadata


def annotate_entities(
    entity_lookup: MarkedEntityLookup,
    xsum_test: Dict[str, XSumDoc],
    generated_summaries: Dict[str, str],
) -> MarkedEntityLookup:
    updated_annotations = defaultdict(lambda: list())
    for sum_id, labeled_entities in entity_lookup.items():
        printed_sum = False
        for entity in labeled_entities:
            if entity["label"] == "Unknown":
                if not printed_sum:
                    print(f"----XSUM ID {sum_id}----")
                    print(f"{xsum_test[sum_id]['document']}")
                    print()
                    print(f"GT summary: {xsum_test[sum_id]['summary']}")
                    print("----")
                    print(f"Google search for article: https://www.google.com/search?q=bbc+{sum_id}")
                    print(f"Generated summary: {generated_summaries[sum_id]}")
                    printed_sum = True

                print(
                    f"What is the label of '{entity['ent']} (pos {entity['start']}:{entity['end']})? In source: {entity['in_source']}"
                )
                user_input = ""
                while user_input not in ["0", "1", "I", "U", "S", "E"]:
                    user_input = input(
                        "Non-factual (0), Factual (1), Intrinsic Non-factual (I), Extrinsic Non-factual (E), Unknown (U) or Skip & save annotations (S)\n"
                    )

                if user_input == "S":
                    return updated_annotations
                elif user_input == "I":
                    annotation = entity.copy()
                    annotation["label"] = ANNOTATION_LABELS["Intrinsic"]
                    updated_annotations[sum_id].append(annotation)
                elif user_input == "E":
                    annotation = entity.copy()
                    annotation["label"] = ANNOTATION_LABELS["Non-factual"]
                    updated_annotations[sum_id].append(annotation)
                elif user_input == "1":
                    annotation = entity.copy()
                    annotation["label"] = (
                        ANNOTATION_LABELS["Non-hallucinated"]
                        if entity["in_source"]
                        else ANNOTATION_LABELS["Factual"]
                    )
                    updated_annotations[sum_id].append(annotation)
                elif user_input == "0":
                    annotation = entity.copy()
                    annotation["label"] = (
                        ANNOTATION_LABELS["Intrinsic"]
                        if entity["in_source"]
                        else ANNOTATION_LABELS["Non-factual"]
                    )
                    updated_annotations[sum_id].append(annotation)
    return updated_annotations


def prompt_annotation_flow(
    unknown_entities: MarkedEntityLookup,
    xsum_test,
    sums_by_id,
    metadata,
    force_annotation_flow=False,
):
    n_unknown = sum([len(ents) for ents in unknown_entities.values()])
    valid_input = False
    response = ""
    while not valid_input and not force_annotation_flow:
        response = input(
            f"Would you like to annotate {n_unknown} unknown entities? (y/n)\n"
        ).strip()
        valid_input = response in ["y", "n"]
    if response == "y" or force_annotation_flow:
        updated_annotations = annotate_entities(unknown_entities, xsum_test, sums_by_id)
        summary_gold_metadata = persist_updated_annotations(
            metadata,
            updated_annotations,
            sums_by_id,
        )
        return updated_annotations, summary_gold_metadata
    else:
        return False
