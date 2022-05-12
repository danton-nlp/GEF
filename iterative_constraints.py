from collections import defaultdict
import argparse
from typing import Dict, List
from src.data_utils import XSumDoc, load_xsum_dict
from src.detect_entities import detect_entities, is_entity_contained
from src.entity_utils import LabeledEntity, LabeledEntityLookup, MarkedEntityLookup, count_entities, filter_entities
from src.generation_utils import generate_summaries, load_model_and_tokenizer
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
    gen_summaries_by_id,
    all_labeled_entities,
    generation_metadata,
):
    iteration_log.append(
        {
            sum_id: {
                "summary": gen_summaries_by_id[sum_id],
                "generation_metadata": {
                    "score": generation_metadata[id_to_idx[sum_id]]["score"],
                    "dropped_seqs": [
                        tokenizer.decode(dropped_seq[0])
                        for dropped_seq in generation_metadata[id_to_idx[sum_id]]["dropped_seqs"]
                    ],
                    "n_words_checked": generation_metadata[id_to_idx[sum_id]]["n_words_checked"],
                },
                "labeled_entities": all_labeled_entities[sum_id],
            }
            for sum_id in gen_summaries_by_id.keys()
        }
    )
    with open(logging_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "args": vars(parser_args),
            "iterations": iteration_log
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--annotate", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    # parser.add_argument("--data_subset", type=int, default=0)
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    iteration_log = []
    logging_path = get_new_log_path("logs-iterative") + ".json"

    xsum_test = load_xsum_dict("test")
    num_beams = 4

    # 1) Select subset of data to work on
    docs_to_summarize = {
        x["id"]: x["document"]
        for x in [
            xsum_test["36396523"],
            xsum_test["39368095"],
        ]
    }

    summary_gold_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_GOLD)

    # initialize with no constraints
    banned_phrases_by_input_idx = defaultdict(lambda: set())

    should_prompt_labeling = args.annotate

    # ...until convergence / max iterations
    n_iterations = 0
    while n_iterations < args.max_iterations:
        print(f"-- Iteration {n_iterations} --")
        # Generate summaries
        factuality_enforcer = WordLogitsProcessor(
            tokenizer,
            num_beams,
            BannedPhrases(
                banned_phrases_by_input_idx=dict(banned_phrases_by_input_idx)
            ),
        )
        id_to_idx = {}
        model_input = []
        for idx, (sum_id, summary) in enumerate(docs_to_summarize.items()):
            id_to_idx[sum_id] = idx
            model_input.append(summary)
        print("Generating summaries...")
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
        with Timer("Detecting & classifying entities..."):
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
        unknown_ents = defaultdict(lambda: set())
        factual_ents = defaultdict(lambda: set())
        non_hallucinated_ents = defaultdict(lambda: set())
        for sum_id, labeled_entities in all_labeled_entities.items():
            input_idx = id_to_idx[sum_id]
            banned_phrases = banned_phrases_by_input_idx[input_idx]
            for ent in labeled_entities:
                if (
                    ent["label"] == ANNOTATION_LABELS["Non-factual"]
                    and ent["ent"] not in banned_phrases
                ):
                    banned_phrases.add(ent["ent"])
                    new_constraints += 1
                elif ent["label"] == "Unknown":
                    unknown_ents[sum_id].add(ent["ent"])
                elif ent["label"] == ANNOTATION_LABELS["Factual"]:
                    factual_ents[sum_id].add(ent["ent"])
                elif ent["label"] == ANNOTATION_LABELS["Non-hallucinated"]:
                    non_hallucinated_ents[sum_id].add(ent["ent"])

        print(
            f"""
[Summary Entity Stats]
- Non-factual: {new_constraints}
- Factual: {len(factual_ents)}
- Non-hallucinated: {len(non_hallucinated_ents)}
- Unknown: {len(unknown_ents)}
"""
        )
        if args.verbose:
            for sum_id, summary in gen_summaries_by_id.items():
                print(f"[{sum_id}]: {summary}")
                print(
                    f"- non-factual (constraints): {banned_phrases_by_input_idx[id_to_idx[sum_id]]}"
                )
                print(f"- factual: {factual_ents[sum_id]}")
                print(f"- non-hallucinated: {non_hallucinated_ents[sum_id]}")
                print(f"- unknown: {unknown_ents[sum_id]}")

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
                print(dict(updated_annotations))
                for sum_id, annotations in updated_annotations.items():
                    for annot in annotations:
                        if annot["label"] == ANNOTATION_LABELS["Non-factual"]:
                            new_constraints += 1
                            banned_phrases_by_input_idx[id_to_idx[sum_id]].add(
                                annot["ent"]
                            )
            else:
                should_prompt_labeling = False

        persist_iteration(
            logging_path,
            args,
            iteration_log,
            gen_summaries_by_id,
            all_labeled_entities,
            generation_metadata
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
