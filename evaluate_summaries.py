from src.data_utils import (
    get_gold_data,
    load_extrinsic_test_set,
    load_xent_test_set,
    load_xsum_dict,
    load_summaries_from_logs,
)
from src.evaluation.factuality import evaluate_factuality
from sumtool.storage import get_summary_metrics, get_summaries
import argparse
import pprint
import pandas as pd


SUMTOOL_DATASET = "xsum"
SUMTOOL_MODEL_BASELINE = "facebook-bart-large-xsum"
pp = pprint.PrettyPrinter(indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", type=bool, default=False)
    parser.add_argument("--data_subset", type=str, default="test-extrinsic")
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--entity_label_match", type=str, default="contained")
    parser.add_argument("--print_first_n", type=int, default=0)
    parser.add_argument("--model_filter", type=str, default="")
    args = parser.parse_args()

    baseline_metadata = get_summary_metrics(SUMTOOL_DATASET, SUMTOOL_MODEL_BASELINE)
    gold_sums, gold_metadata = get_gold_data()
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
                sum_id: x
                for sum_id, x in sum_ents_by_id.items()
                if sum_id in test_set_ids
            }
            print(f"Model: {model_label}")
            agg_metrics, summaries = evaluate_factuality(
                filtered_sums_by_id,
                filtered_ents_by_id,
                gold_sums,
                gold_metadata,
                xsum_test,
                args.annotate and "gold" not in model_label.lower(),
                args.entity_label_match,
                args.print_first_n,
                is_fbs="fbs" in model_label.lower(),
                is_oracle="oracle" in model_label.lower(),
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
