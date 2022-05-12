from collections import defaultdict
import argparse
from typing import Dict, List
from src.data_utils import XSumDoc, load_xsum_dict
from src.detect_entities import detect_entities, is_entity_contained
from src.entity_utils import (
    LabeledEntity,
    LabeledEntityLookup,
    MarkedEntityLookup,
    count_entities,
    filter_entities,
)
from src.generation_utils import SUMMARY_FAILED_GENERATION, generate_summaries, load_model_and_tokenizer
from src.beam_validators import BannedPhrases
from src.word_logits_processor import WordLogitsProcessor
from sumtool.storage import get_summary_metrics, store_summary_metrics
from src.misc_utils import Timer, get_new_log_path
import json
import time


ANNOTATION_LABELS = {
    "Non-factual": "Non-factual Hallucination",
    "Factual": "Factual Hallucination",
    "Non-hallucinated": "Non-hallucinated",
    "Unknown": "Unknown",
}


def oracle_classify_entities(
    summary_entities: MarkedEntityLookup,
    annotations: LabeledEntityLookup,
) -> LabeledEntityLookup:
    labeled_entities: LabeledEntityLookup = {}
    for bbc_id, marked_entities in summary_entities.items():
        to_be_labeled: List[LabeledEntity] = [x.copy() for x in marked_entities]
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
    entity_lookup: LabeledEntityLookup,
    xsum_test: Dict[str, XSumDoc],
    generated_summaries: Dict[str, str],
) -> LabeledEntityLookup:
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
                while user_input not in ["0", "1", "U", "S"]:
                    user_input = input(
                        "Non-factual (0), Factual (1), Unknown (U) or Skip & save annotations (S)\n"
                    )

                if user_input == "S":
                    return updated_annotations
                elif user_input == "1":
                    annotation = entity.copy()
                    annotation["label"] = ANNOTATION_LABELS["Factual"]
                    updated_annotations[sum_id].append(annotation)
                elif user_input == "0":
                    annotation = entity.copy()
                    annotation["label"] = ANNOTATION_LABELS["Non-factual"]
                    updated_annotations[sum_id].append(annotation)
    return updated_annotations


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL_GOLD = "gold"


def persist_updated_annotations(old_metadata, updated_annotations):
    updated_metadata = old_metadata.copy()
    for sum_id, new_annotations in updated_annotations.items():
        if sum_id in old_metadata:
            old_annotations = (
                updated_metadata[sum_id]["our_annotations"]
                if "our_annotations" in updated_metadata[sum_id]
                else []
            )
            updated_metadata[sum_id]["our_annotations"] = (
                old_annotations + new_annotations
            )

    store_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD, updated_metadata)
    return updated_metadata


def get_entity_annotations(metadata, sum_id):
    annots = []
    for key in ["xent", "our_annotations"]:
        if key in metadata[sum_id]:
            annots += metadata[sum_id][key]
    return annots


def persist_iteration(
    logging_path,
    parser_args,
    iteration_log,
    id_to_idx,
    gen_summaries_by_id,
    all_labeled_entities,
    generation_metadata,
    banned_phrases_by_sum_id,
):
    iteration_log.append(
        {
            "summaries": {
                sum_id: {
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
                    "labeled_entities": all_labeled_entities[sum_id],
                }
                for sum_id in gen_summaries_by_id.keys()
            }
        }
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--annotate", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument(
        "--data_subset", type=str, default="debug", help="debug|x-ent|all"
    )
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    iteration_log = []
    logging_path = get_new_log_path("logs-iterative") + ".json"
    summary_gold_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)
    xsum_test = load_xsum_dict("test")
    num_beams = 4

    if args.data_subset == "debug":
        docs_to_summarize = {
            x["id"]: x["document"]
            for x in [
                xsum_test["34361828"],
                xsum_test["36456002"],
                xsum_test["24403775"],
                xsum_test["32112735"],  # One direction split
                xsum_test["36203675"],  # Dementia mobile game researchers
                xsum_test["17996567"],
                xsum_test["36396523"],
                xsum_test["39368095"],
            ]
        }
    elif args.data_subset == "x-ent":
        docs_to_summarize = {
            sum_id: x["document"]
            for sum_id, x in xsum_test.items()
            if "x-ent" in summary_gold_metadata[sum_id]
        }
    else:
        docs_to_summarize = {sum_id: x["document"] for sum_id, x in xsum_test.items()}

    # initialize with no constraints
    banned_phrases_by_sum_id = defaultdict(lambda: set())

    should_prompt_labeling = args.annotate

    incomplete_sum_ids = set(docs_to_summarize.keys())
    unknown_ents_by_sum_id = defaultdict(lambda: set())
    factual_ents_by_sum_id = defaultdict(lambda: set())
    non_hallucinated_ents_by_sum_id = defaultdict(lambda: set())

    # ...until convergence / max iterations
    n_iterations = 0
    while n_iterations < args.max_iterations:
        print(f"-- Iteration {n_iterations} --")
        # Generate summaries
        id_to_idx = {}
        model_input = []
        banned_phrases_by_input_idx = {}
        for sum_id, summary in docs_to_summarize.items():
            if sum_id in incomplete_sum_ids:
                input_idx = len(model_input)
                id_to_idx[sum_id] = input_idx
                model_input.append(summary)
                banned_phrases_by_input_idx[input_idx] = banned_phrases_by_sum_id[
                    sum_id
                ]

        factuality_enforcer = WordLogitsProcessor(
            tokenizer,
            num_beams,
            BannedPhrases(banned_phrases_by_input_idx=banned_phrases_by_input_idx),
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
            bbc_id: gen_summaries[input_idx] for bbc_id, input_idx in id_to_idx.items()
        }

        #  Detect & classify entities
        with Timer("Detecting & classifying entities"):
            summary_entities = {}
            for bbc_id, input_idx in id_to_idx.items():
                summary_entities[bbc_id] = detect_entities(
                    gen_summaries[input_idx], model_input[input_idx]
                )

            # 4) See if NER is factual according to model (oracle / classification)
            all_labeled_entities = oracle_classify_entities(
                summary_entities,
                {
                    sum_id: get_entity_annotations(summary_gold_metadata, sum_id)
                    for sum_id in summary_entities.keys()
                },
            )

        # Process entity labels
        new_constraints = 0
        incomplete_sum_ids = set()
        for sum_id, labeled_entities in all_labeled_entities.items():
            banned_phrases = banned_phrases_by_sum_id[sum_id]
            # clear
            unknown_ents_by_sum_id[sum_id] = set()
            factual_ents_by_sum_id[sum_id] = set()
            non_hallucinated_ents_by_sum_id[sum_id] = set()

            for ent in labeled_entities:
                if (
                    ent["label"] == ANNOTATION_LABELS["Non-factual"]
                    and ent["ent"] not in banned_phrases
                ):
                    banned_phrases.add(ent["ent"])
                    new_constraints += 1
                    incomplete_sum_ids.add(sum_id)
                elif ent["label"] == "Unknown":
                    unknown_ents_by_sum_id[sum_id].add(ent["ent"])
                elif ent["label"] == ANNOTATION_LABELS["Factual"]:
                    factual_ents_by_sum_id[sum_id].add(ent["ent"])
                elif ent["label"] == ANNOTATION_LABELS["Non-hallucinated"]:
                    non_hallucinated_ents_by_sum_id[sum_id].add(ent["ent"])
        print(
            f"""
[Summary Entity Stats]
- Non-factual: {new_constraints}
- Factual: {sum([len(x) for x in factual_ents_by_sum_id.values()])}
- Non-hallucinated: {sum([len(x) for x in non_hallucinated_ents_by_sum_id.values()])}
- Unknown: {sum([len(x) for x in unknown_ents_by_sum_id.values()])}
"""
        )
        if args.verbose:
            for sum_id, summary in gen_summaries_by_id.items():
                print(f"[{sum_id}]: {summary}")
                print(
                    f"- non-factual (constraints): {list(banned_phrases_by_sum_id[sum_id])}"
                )
                print(f"- factual: {list(factual_ents_by_sum_id[sum_id])}")
                print(
                    f"- non-hallucinated: {list(non_hallucinated_ents_by_sum_id[sum_id])}"
                )
                print(f"- unknown: {list(unknown_ents_by_sum_id[sum_id])}")

        # Manual annotation
        unknown_entities = filter_entities(
            lambda x: x["label"] == ANNOTATION_LABELS["Unknown"], all_labeled_entities
        )
        if should_prompt_labeling and count_entities(unknown_entities) > 0:
            if input("Would you like to label unknown entities? (y/n)\n") == "y":
                updated_annotations = prompt_labeling(
                    unknown_entities, xsum_test, gen_summaries_by_id
                )
                summary_gold_metadata = persist_updated_annotations(
                    summary_gold_metadata, updated_annotations
                )
                if args.verbose:
                    print("Updated annotations:", dict(updated_annotations))
                for sum_id, annotations in updated_annotations.items():
                    for annot in annotations:
                        if annot["label"] == ANNOTATION_LABELS["Non-factual"]:
                            new_constraints += 1
                            banned_phrases_by_sum_id[sum_id].add(annot["ent"])
            else:
                should_prompt_labeling = False

        persist_iteration(
            logging_path,
            args,
            iteration_log,
            id_to_idx,
            gen_summaries_by_id,
            all_labeled_entities,
            generation_metadata,
            banned_phrases_by_sum_id,
        )

        n_failed = len([
            1
            for summary in gen_summaries_by_id.values()
            if summary == SUMMARY_FAILED_GENERATION
        ])
        n_incomplete = len(incomplete_sum_ids)
        n_total = len(docs_to_summarize)
        n_converged = n_total - n_incomplete - n_failed
        print(
            f"""
[Summary Stats]
- Factual: {n_converged} ({n_converged/n_total:.2%})
- Non-factual: {n_incomplete} ({n_incomplete/n_total:.2%})
- Failed: {n_failed} ({n_failed/n_total:.2%})
- Unknown: {sum([1 for x in unknown_ents_by_sum_id.values() if len(x) > 0])}
"""
        )
        # break if no new constriants
        if new_constraints == 0:
            print("No new constraints found, done...")
            break
        else:
            print(f"Added {new_constraints} constraints!")
        n_iterations += 1

        print()
        print()

    # Persist results
    pass
    # Evaluate?
    pass
