from collections import defaultdict
from src.data_utils import (
    get_gold_xsum_data,
    load_shuffled_test_split,
    load_xsum_dict,
)
from src.evaluation.factuality import get_labeled_entities
from sumtool.storage import get_summary_metrics, get_summaries
import argparse
import pprint
from evaluate_summaries import load_model_results_for_subset
from tqdm import tqdm


pp = pprint.PrettyPrinter(indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--entity_label_match", type=str, default="strict_all")
    parser.add_argument(
        "--data_subsets", type=str, default="bart-test-extrinsic,pegasus-test-extrinsic"
    )
    parser.add_argument("--print_first_n", type=int, default=0)
    parser.add_argument("--model_filter", type=str, default="")
    parser.add_argument("--count_skips", type=bool, default=False)
    args = parser.parse_args()

    baseline_metadata = get_summary_metrics("xsum", "facebook-bart-large-xsum")
    gold_sums, gold_metadata = get_gold_xsum_data()
    xsum_test = load_xsum_dict("test")

    for data_subset in args.data_subsets.split(","):
        print(f"Data subset: {data_subset}")
        test_set_ids = set(
            load_shuffled_test_split(xsum_test, data_subset, args.test_size).keys()
        )
        print(f"Annotating {len(test_set_ids)} summaries")
        MODEL_RESULTS = load_model_results_for_subset(data_subset)

        bart_models = [
            ("facebook-bart-large-xsum", "baseline-bart"),
            ("meng-rl", "meng-rl"),  # Hallucinated, but factual! Paper
            ("pinocchio-fallback-ours", "pinocchio"),  # King et. al paper
            ("chen-corrector", "corrector"),  # Chen. et al replication project
            # ("entity-filter-v2", "filtered"),  # Nan. et al
            ("gold", "gold"),
        ]
        pegasus_models = [
            ("google-pegasus-xsum", "baseline-pegasus"),
            ("gold", "gold"),
        ]

        for (sumtool_name, model_label) in (
            bart_models if "bart" in data_subset else pegasus_models
        ):
            dataset = get_summaries("xsum", sumtool_name)
            MODEL_RESULTS[model_label] = (
                {sum_id: x["summary"] for sum_id, x in dataset.items()},
                {},
                {},
            )

        aggregated_results = []
        summary_results = {}
        merged_sums_by_id = defaultdict(lambda: {})
        for model_label, (
            sums_by_id,
            sum_ents_by_id,
            failed_sums_by_id,
        ) in MODEL_RESULTS.items():
            filtered_sums_by_id = {
                sum_id: x
                for sum_id, x in sums_by_id.items()
                if sum_id in test_set_ids
            }
            filtered_ents_by_id = {
                sum_id: x
                for sum_id, x in sum_ents_by_id.items()
                if sum_id in test_set_ids
            }
            if args.model_filter == "" or args.model_filter in model_label:
                for sum_id, summary in filtered_sums_by_id.items():
                    merged_sums_by_id[sum_id][model_label] = summary

        for sum_id, sums_by_model in tqdm(list(merged_sums_by_id.items())):
            for model_label, summary in sums_by_model.items():
                should_annotate =  "gold" not in model_label
                labeled_ents = get_labeled_entities(
                    {sum_id: summary},
                    {},
                    gold_metadata,
                    xsum_test,
                    should_annotate=should_annotate,
                    entity_match_type=args.entity_label_match,
                    force_annotation_flow=should_annotate
                )
