import argparse
from typing import Union
from src.oracle import EntityMatchType
from src.evaluation.factuality import evaluate_factuality
from src.generation_utils import SUMMARY_FAILED_GENERATION
from src.data_utils import (
    get_gold_xsum_data,
    load_shuffled_test_split,
    load_xsum_dict,
)
import json
import pprint
import editdistance
import os
from tqdm import tqdm


pp = pprint.PrettyPrinter(indent=2)


def collect_iteration_stats(
    logs_path,
    xsum_test,
    test_set_ids: Union[set, None],
    entity_match_type: EntityMatchType = "strict_all",
    should_annotate=False,
):
    is_oracle = "oracle" in logs_path
    gold_sums, gold_metadata = get_gold_xsum_data()

    with open(logs_path, "r") as f:
        logs = json.load(f)

    sorted_keys = sorted([int(x) for x in logs["iterations"].keys()])

    iteration_stats = []

    failed_sums_by_id = {}
    sums_by_id = {}
    sum_ents_by_id = {}
    baseline_eval_sums_by_id = {}
    for iteration_idx in sorted_keys:
        iteration_data = logs["iterations"][str(iteration_idx)]
        summaries = iteration_data["summaries"]
        current_iteration_stats = {
            "iteration": iteration_idx,
            "summary": {},
            "summary_generated": len(summaries),
        }
        updated_sums_by_id = {}
        for sum_id, data in summaries.items():
            if test_set_ids is None or sum_id in test_set_ids:
                if data["summary"] == SUMMARY_FAILED_GENERATION:
                    failed_sums_by_id[sum_id] = iteration_idx
                else:
                    updated_sums_by_id[sum_id] = data["summary"]
                    sums_by_id[sum_id] = data["summary"]
                    sum_ents_by_id[sum_id] = data["labeled_entities"]
                    current_iteration_stats["summary"][sum_id] = {
                        "summary": data["summary"]
                    }

        print()
        print(f"Iteration {iteration_idx}")
        print(f"Summaries generated: {current_iteration_stats['summary_generated']}")
        stats_factuality, eval_sums_by_id = evaluate_factuality(
            sums_by_id,
            sum_ents_by_id,
            failed_sums_by_id,
            gold_sums,
            gold_metadata,
            xsum_test,
            should_annotate,
            entity_match_type=entity_match_type,
            print_first_n=0,
            is_fbs=True,
            is_oracle=is_oracle,
            compute_rouge=False,
        )
        current_iteration_stats["factuality_summary"] = stats_factuality["summaries"]
        current_iteration_stats["factuality_entities"] = stats_factuality["entities"]

        # print(iteration_data["stats"]["entity"])
        # print(iteration_data["stats"]["summary"])
        # print(stats_factuality)
        print("Updated sums:", len(updated_sums_by_id))
        if len(updated_sums_by_id) == 4:
            pprint.pprint(updated_sums_by_id)

        # compute edit stats
        if iteration_idx == 0:
            baseline_eval_sums_by_id = eval_sums_by_id
        if iteration_idx > 0:
            print("Computing edit stats")
            for sum_id, sum in tqdm(updated_sums_by_id.items()):
                if sum != SUMMARY_FAILED_GENERATION:
                    original_sum = baseline_eval_sums_by_id[sum_id]["summary"]
                    # rouge_scores = rouge([sum], [original_sum])
                    current_iteration_stats["summary"][sum_id]["edit_stats"] = {
                        "edit_distance_token": editdistance.eval(
                            original_sum.lower().split(" "), sum.lower().split(" ")
                        ),
                        # "rouge1": rouge_scores["rouge1"]["f1"],
                        # "rouge2": rouge_scores["rouge2"]["f1"],
                        # "rougeL": rouge_scores["rougeL"]["f1"],
                    }

        iteration_stats.append(current_iteration_stats)
    return iteration_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", type=bool, default=False)
    parser.add_argument("--data_subset", type=str, default="bart-test-extrinsic")
    parser.add_argument("--test_size", type=int, default=100)
    args = parser.parse_args()

    xsum_test = load_xsum_dict("test")
    test_set_ids = (
        set(
            load_shuffled_test_split(xsum_test, args.data_subset, args.test_size).keys()
        )
        if args.data_subset
        not in ["bart-debug", "pegasus-debug", "bart-full", "pegasus-full"]
        else None
    )

    for model in ["oracle", "classifier-knnv1"]:
        if os.path.exists(f"results/fbs-logs/{args.data_subset}-{model}.json"):
            iteration_stats = collect_iteration_stats(
                f"results/fbs-logs/{args.data_subset}-{model}.json",
                xsum_test,
                test_set_ids,
                should_annotate=args.annotate,
            )
            test_size = f"-{args.test_size}" if test_set_ids is not None else ""
            with open(
                f"results/iteration-changes/{args.data_subset}{test_size}-{model}.json",
                "w",
            ) as f:
                json.dump(iteration_stats, f, indent=2, sort_keys=True)
        else:
            print(f"Logs for {model} does not exist!")
