from collections import defaultdict
import argparse
from typing import Dict
from src.data_utils import XSumDoc, load_debug_subset, load_test_set, load_xsum_dict
from src.detect_entities import detect_entities, is_entity_contained
from copy import deepcopy
from src.entity_utils import (
    MarkedEntityLookup,
    count_entities,
    filter_entities,
)
from src.entity_factuality import (
    EntityFactualityClassifier,
    ANNOTATION_LABELS,
)
from src.generation_utils import (
    SUMMARY_FAILED_GENERATION,
    generate_summaries,
    load_model_and_tokenizer,
)
from src.beam_validators import BannedPhrases
from src.word_logits_processor import WordLogitsProcessor
from sumtool.storage import get_summary_metrics, store_summary_metrics
from src.misc_utils import Timer, get_new_log_path
import json
import time


def oracle_label_entities(
    summary_entities: MarkedEntityLookup,
    annotations: MarkedEntityLookup,
) -> MarkedEntityLookup:
    labeled_entities: MarkedEntityLookup = {}
    for bbc_id, marked_entities in summary_entities.items():
        to_be_labeled = [x.copy() for x in marked_entities]
        for x in to_be_labeled:
            x["label"] = (
                "Unknown"
                if not x["in_source"]
                else ANNOTATION_LABELS["Non-hallucinated"]
            )
        for unlabeled_entity in to_be_labeled:
            for annotated_entity in annotations[bbc_id]:
                if is_entity_contained(
                    unlabeled_entity["ent"], annotated_entity["ent"]
                ):
                    unlabeled_entity["label"] = annotated_entity["label"]
        labeled_entities[bbc_id] = to_be_labeled
    return labeled_entities


def prompt_labeling(
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
                    print(f"Generated summary: {generated_summaries[sum_id]}")
                    printed_sum = True

                print(
                    f"What is the label of '{entity['ent']} (pos {entity['start']}:{entity['end']})?"
                )
                user_input = ""
                while user_input not in ["0", "1", "I", "U", "S"]:
                    user_input = input(
                        "Non-factual (0), Factual (1), Intrinsic (I), Unknown (U) or Skip & save annotations (S)\n"
                    )

                if user_input == "S":
                    return updated_annotations
                elif user_input == "1":
                    annotation = entity.copy()
                    annotation["label"] = ANNOTATION_LABELS["Factual"]
                    updated_annotations[sum_id].append(annotation)
                elif user_input == "I":
                    annotation = entity.copy()
                    annotation["label"] = ANNOTATION_LABELS["Intrinsic"]
                    updated_annotations[sum_id].append(annotation)
                elif user_input == "0":
                    annotation = entity.copy()
                    annotation["label"] = ANNOTATION_LABELS["Non-factual"]
                    updated_annotations[sum_id].append(annotation)
    return updated_annotations


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL_GOLD = "gold"


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

    store_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD, updated_metadata)
    return updated_metadata


def get_entity_annotations(metadata, sum_id):
    annots = []
    for key in ["xent-train", "xent-test", "our_annotations"]:
        if key in metadata[sum_id]:
            for ents in metadata[sum_id][key].values():
                annots += ents
    return annots


def persist_iteration(
    logging_path,
    parser_args,
    iteration_log,
    iteration_idx,
    tokenizer,
    id_to_idx,
    gen_summaries_by_id,
    oracle_labeled_entities,
    generation_metadata,
    banned_phrases_by_sum_id,
    iteration_stats,
):
    if iteration_idx not in iteration_log:
        iteration_log[iteration_idx] = {"summaries": {}}
    iteration_log[iteration_idx]["stats"] = iteration_stats
    for sum_id in gen_summaries_by_id.keys():
        iteration_log[iteration_idx]["summaries"][sum_id] = {
            "banned_phrases": list(banned_phrases_by_sum_id[sum_id]),
            "summary": gen_summaries_by_id[sum_id],
            "generation_metadata": {
                "score": generation_metadata[id_to_idx[sum_id]]["score"],
                "dropped_seqs": [
                    tokenizer.decode(dropped_seq[0])
                    for dropped_seq in generation_metadata[id_to_idx[sum_id]][
                        "dropped_seqs"
                    ]
                ],
                "n_words_checked": generation_metadata[id_to_idx[sum_id]][
                    "n_words_checked"
                ],
            },
            "labeled_entities": oracle_labeled_entities[sum_id],
        }
    with open(logging_path, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "args": vars(parser_args),
                "iterations": iteration_log,
            },
            f,
            indent=2,
        )


def compute_stats(results_by_sum_id):
    summary_stats = {
        "completed": 0,
        "total": len(results_by_sum_id),
        "factual": 0,
        "non_factual": 0,
        "failed": 0,
        "unknown": 0,
    }
    entity_stats = {
        "label": {
            ANNOTATION_LABELS["Non-factual"]: 0,
            ANNOTATION_LABELS["Factual"]: 0,
            ANNOTATION_LABELS["Unknown"]: 0,
            ANNOTATION_LABELS["Non-hallucinated"]: 0,
            ANNOTATION_LABELS["Intrinsic"]: 0,
        },
        "type": {},
    }
    for results in results_by_sum_id.values():
        if results["completed"]:
            summary_stats["completed"] += 1
        ents_non_factual = len(results[ANNOTATION_LABELS["Non-factual"]])
        ents_unknown = len(results[ANNOTATION_LABELS["Unknown"]])

        if results["failed"]:
            summary_stats["failed"] += 1
        elif ents_non_factual > 0:
            summary_stats["non_factual"] += 1
        elif ents_unknown > 0:
            summary_stats["unknown"] += 1
        else:
            summary_stats["factual"] += 1

        for label in ANNOTATION_LABELS.values():
            label_results = results[label]
            entity_stats["label"][label] += len(label_results)
            for ent in label_results:
                if ent["type"] not in entity_stats["type"]:
                    entity_stats["type"][ent["type"]] = {
                        "total": 0,
                        ANNOTATION_LABELS["Non-factual"]: 0,
                        ANNOTATION_LABELS["Factual"]: 0,
                        ANNOTATION_LABELS["Unknown"]: 0,
                        ANNOTATION_LABELS["Non-hallucinated"]: 0,
                        ANNOTATION_LABELS["Intrinsic"]: 0,
                    }
                entity_stats["type"][ent["type"]]["total"] += 1
                entity_stats["type"][ent["type"]][label] += 1

    return {
        "summary": summary_stats,
        "entity": entity_stats,
    }


def print_results(label, results_by_sum_id, type="summary"):
    if type == "summary":
        stats = compute_stats(results_by_sum_id)["summary"]
        print(
            f"""
        [{label}]
        - Factual: {stats['factual']} ({stats['factual']/stats['total']:.2%})
        - Non-factual: {stats['non_factual']} ({stats['non_factual']/stats['total']:.2%})
        - Failed: {stats['failed']} ({stats['failed']/stats['total']:.2%})
        - Unknown: {stats['unknown']} ({stats['unknown']/stats['total']:.2%})
        """
        )
    elif type == "entity":
        stats = compute_stats(results_by_sum_id)["entity"]
        print(
            f"""
        [{label}]
        - Non-factual: {stats[ANNOTATION_LABELS['Non-factual']]}
        - Factual: {stats[ANNOTATION_LABELS['Factual']]}
        - Non-hallucinated: {stats[ANNOTATION_LABELS['Non-hallucinated']]}
        - Unknown: {stats[ANNOTATION_LABELS['Unknown']]}
        """
        )


def split_batches(lst, size):
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickled_classifier", type=str, default="")
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--annotate", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--data_subset", type=str, default="debug", help="debug|xent|full"
    )
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    iteration_log = {}
    logging_path = get_new_log_path("logs-iterative") + ".json"
    summary_gold_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    xsum_test = load_xsum_dict("test")
    num_beams = 4

    if args.pickled_classifier != "":
        clf_factuality = EntityFactualityClassifier(args.pickled_classifier)
    else:
        clf_factuality = None

    xent_test_summaries = {
        sum_id: x["document"]
        for sum_id, x in xsum_test.items()
        if "xent-test" in summary_gold_metadata[sum_id]
    }

    if args.data_subset == "debug":
        docs_to_summarize = load_debug_subset(xsum_test)
    elif args.data_subset == "xent-test":
        docs_to_summarize = xent_test_summaries
    elif "xent-test-" in args.data_subset:
        docs_to_summarize = {
            k: v
            for k, v in list(xent_test_summaries.items())[
                : int(args.data_subset.split("-")[-1])
            ]
        }
    elif args.data_subset in ["test-anton", "test-daniel", "test"]:
        test_set = load_test_set(xsum_test, summary_gold_metadata)
        if "anton" in args.data_subset:
            print("Loaded Anton's test subset")
            test_set = test_set[:25]
        elif "daniel" in args.data_subset:
            print("Loaded Daniel's test subset")
            test_set = test_set[25:50]
        else:
            print("Loaded full test set")
        docs_to_summarize = {k: v for (k, v) in test_set}
    elif args.data_subset == "full":
        docs_to_summarize = {sum_id: x["document"] for sum_id, x in xsum_test.items()}
    else:
        raise argparse.ArgumentError(args.data_subset, message="Invalid argument")

    # initialize with no constraints
    banned_phrases_by_sum_id = defaultdict(lambda: set())

    should_prompt_labeling = args.annotate

    # ...until convergence / max iterations
    n_iterations = 0
    results_by_sum_id = {}
    for sum_id in docs_to_summarize.keys():
        results_by_sum_id[sum_id] = {
            "summary": "",
            "completed": False,
            "failed": False,
            ANNOTATION_LABELS["Factual"]: [],
            ANNOTATION_LABELS["Non-factual"]: [],
            ANNOTATION_LABELS["Unknown"]: [],
            ANNOTATION_LABELS["Non-hallucinated"]: [],
            ANNOTATION_LABELS["Intrinsic"]: [],
        }

    while n_iterations < args.max_iterations:
        prev_banned_phrases_by_sum_id = deepcopy(banned_phrases_by_sum_id)
        new_constraints = 0
        incomplete_docs = [
            (sum_id, summary)
            for sum_id, summary in docs_to_summarize.items()
            if not results_by_sum_id[sum_id]["completed"]
        ]
        batches = list(split_batches(incomplete_docs, args.batch_size))
        with Timer(f"Iteration {n_iterations}, {len(incomplete_docs)} docs"):
            for batch_idx, batch_sums in enumerate(batches):
                print(f"Batch {batch_idx+1}/{len(batches)}")
                # Generate summaries
                id_to_idx = {}
                model_input = []
                banned_phrases_by_input_idx = {}
                for sum_id, summary in batch_sums:
                    input_idx = len(model_input)
                    id_to_idx[sum_id] = input_idx
                    model_input.append(summary)
                    banned_phrases_by_input_idx[input_idx] = banned_phrases_by_sum_id[
                        sum_id
                    ]

                factuality_enforcer = WordLogitsProcessor(
                    tokenizer,
                    num_beams,
                    BannedPhrases(
                        banned_phrases_by_input_idx=banned_phrases_by_input_idx
                    ),
                )
                with Timer(f"Generating {len(model_input)} summaries"):
                    gen_summaries, generation_metadata = generate_summaries(
                        model,
                        tokenizer,
                        model_input,
                        factuality_enforcer,
                        num_beams=num_beams,
                        return_beam_metadata=True,
                    )
                gen_summaries_by_id = {
                    bbc_id: gen_summaries[input_idx]
                    for bbc_id, input_idx in id_to_idx.items()
                }
                for sum_id, summary in gen_summaries_by_id.items():
                    results_by_sum_id[sum_id]["summary"] = summary

                #  Detect & classify entities
                with Timer("Detecting & classifying entities"):
                    summary_entities = {}
                    for bbc_id, input_idx in id_to_idx.items():
                        summary_entities[bbc_id] = detect_entities(
                            gen_summaries[input_idx], model_input[input_idx]
                        )

                    # Set predicted labels from classifier
                    if clf_factuality is not None:
                        summary_entities = clf_factuality.classify_entities(
                            summary_entities, gen_summaries_by_id
                        )

                    # Set labels from oracle
                    oracle_labeled_entities = oracle_label_entities(
                        summary_entities,
                        {
                            sum_id: get_entity_annotations(
                                summary_gold_metadata, sum_id
                            )
                            for sum_id in summary_entities.keys()
                        },
                    )

                # Update results based on oracle labels & banned words based on predictions
                for sum_id, labeled_entities in oracle_labeled_entities.items():
                    banned_phrases = banned_phrases_by_sum_id[sum_id]
                    no_constraints = True
                    # reset results
                    for label in ANNOTATION_LABELS.values():
                        results_by_sum_id[sum_id][label] = []
                    for ent in labeled_entities:
                        # Use classifier label if classifier is loaded, otherwise use the oracle
                        label = (
                            ent["predicted_label"]
                            if clf_factuality is not None
                            else ent["label"]
                        )
                        results_by_sum_id[sum_id][label].append(ent)
                        if label == ANNOTATION_LABELS["Non-factual"]:
                            no_constraints = False
                            if ent["ent"] not in banned_phrases:
                                new_constraints += 1
                                banned_phrases.add(ent["ent"])

                    if no_constraints:
                        results_by_sum_id[sum_id]["completed"] = True

                    if gen_summaries_by_id[sum_id] == SUMMARY_FAILED_GENERATION:
                        results_by_sum_id[sum_id]["failed"] = True

                if args.verbose:
                    for sum_id, summary in gen_summaries_by_id.items():
                        print(f"[{sum_id}]: {summary}")
                        print(
                            f"- constraints: {list(prev_banned_phrases_by_sum_id[sum_id])}"
                        )
                        print(
                            f"- non-factual: {[ent['ent'] for ent in results_by_sum_id[sum_id][ANNOTATION_LABELS['Non-factual']]]}"
                        )
                        print(
                            f"- factual: {[ent['ent'] for ent in results_by_sum_id[sum_id][ANNOTATION_LABELS['Factual']]]}"
                        )
                        print(
                            f"- non-hallucinated: {[ent['ent'] for ent in results_by_sum_id[sum_id][ANNOTATION_LABELS['Non-hallucinated']]]}"
                        )
                        print(
                            f"- unknown: {[ent['ent'] for ent in results_by_sum_id[sum_id][ANNOTATION_LABELS['Unknown']]]}"
                        )

                # Manual annotation
                unknown_entities = filter_entities(
                    lambda x: x["label"] == ANNOTATION_LABELS["Unknown"],
                    oracle_labeled_entities,
                )
                if should_prompt_labeling and count_entities(unknown_entities) > 0:
                    if (
                        input("Would you like to label unknown entities? (y/n)\n")
                        == "y"
                    ):
                        updated_annotations = prompt_labeling(
                            unknown_entities, xsum_test, gen_summaries_by_id
                        )
                        summary_gold_metadata = persist_updated_annotations(
                            summary_gold_metadata,
                            updated_annotations,
                            gen_summaries_by_id,
                        )
                        if args.verbose:
                            print("Updated annotations:", dict(updated_annotations))
                        for sum_id, annotations in updated_annotations.items():
                            for annot in annotations:
                                if annot["label"] == ANNOTATION_LABELS["Non-factual"]:
                                    new_constraints += 1
                                    banned_phrases_by_sum_id[sum_id].add(annot["ent"])
                                    results_by_sum_id[sum_id]["completed"] = False
                    else:
                        should_prompt_labeling = False

                persist_iteration(
                    logging_path,
                    args,
                    iteration_log,
                    n_iterations,
                    tokenizer,
                    id_to_idx,
                    gen_summaries_by_id,
                    oracle_labeled_entities,
                    generation_metadata,
                    prev_banned_phrases_by_sum_id,
                    compute_stats(results_by_sum_id),
                )

                print_results(
                    "Batch Summary Stats",
                    {
                        k: v
                        for k, v in results_by_sum_id.items()
                        if k in gen_summaries_by_id
                    },
                )

            iter_total = len(docs_to_summarize)
            print_results("Iteration Summary Stats", results_by_sum_id)

            # break if no new constriants
            if new_constraints == 0:
                print("No new constraints found, done...")
                break

            n_iterations += 1

            if n_iterations + 1 > args.max_iterations:
                print("Reached max iterations!")
                break
            else:
                print(f"Added {new_constraints} constraints!")

        print()
        print()
