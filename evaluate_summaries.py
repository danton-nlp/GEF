from src.data_utils import (
    get_gold_xsum_data,
    load_shuffled_test_split,
    load_xsum_dict,
    load_summaries_from_logs,
)
from src.detect_entities import detect_entities
from src.evaluation.factuality import evaluate_factuality
from sumtool.storage import get_summary_metrics, get_summaries
import argparse
import pprint
import pandas as pd
import json


pp = pprint.PrettyPrinter(indent=2)


def load_model_results_for_subset(data_subset: str, beam_suffix: str = ""):
    if "fully-annotated" in data_subset:
        return {
            "gef_classifier": load_summaries_from_logs(
                "results/gef-logs/bart-full-classifier-knnv1.json",
                max_iterations=100,
            ),
            # "gef_pegasus_classifier": load_summaries_from_logs(
            #     f"results/gef-logs/pegasus-full-classifier-knnv1.json", max_iterations=5
            # ),
        }
    else:
        results = {}
        if "bart" in data_subset:
            data_prefix = "bart-full"
        else:
            data_prefix = data_subset
        for (name, loc, max_iterations) in [
            (
                "gef_oracle",
                f"results/gef-logs/{data_subset}-oracle{beam_suffix}.json",
                100,
            ),
            (
                "gef_classifier",
                f"results/gef-logs/{data_prefix}-classifier-knnv2{beam_suffix}.json",
                100,
            ),
            (
                "gef_classifier_i10",
                f"results/gef-logs/{data_prefix}-classifier-knnv2{beam_suffix}.json",
                10,
            ),
        ]:
            try:
                results[name] = load_summaries_from_logs(
                    loc, max_iterations=max_iterations
                )
            except Exception as e:
                print(f"[WARN]: Failed to load logs: `{name} / {loc}`")
                print(e)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", type=bool, default=False)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--test_set_offset", type=int, default=0)
    parser.add_argument("--entity_label_match", type=str, default="strict_all")
    parser.add_argument(
        "--data_subsets", type=str, default="bart-test-extrinsic,pegasus-test-extrinsic"
    )
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--print_first_n", type=int, default=0)
    parser.add_argument("--model_filter", type=str, default="")
    parser.add_argument("--count_skips", type=bool, default=False)
    args = parser.parse_args()

    baseline_metadata = get_summary_metrics("xsum", "facebook-bart-large-xsum")
    gold_sums, gold_metadata = get_gold_xsum_data()
    xsum_test = load_xsum_dict("test")
    beam_suffix = "" if args.num_beams == 4 else f"-beams-{args.num_beams}"
    unique_sums = set()
    unique_sum_ent_count = 0

    for data_subset in args.data_subsets.split(","):
        test_set_ids = set(
            load_shuffled_test_split(
                xsum_test, data_subset, args.test_size, args.test_set_offset
            ).keys()
        )
        print(f"Test results for {len(test_set_ids)} summaries")
        MODEL_RESULTS = load_model_results_for_subset(data_subset, beam_suffix)

        bart_models = [
            ("facebook-bart-large-xsum", "baseline-bart"),
            ("rl-fact", "rl-fact"),  # Hallucinated, but factual! Paper
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
        for model_label, (
            sums_by_id,
            sum_ents_by_id,
            failed_sums_by_id,
        ) in MODEL_RESULTS.items():
            if args.model_filter == "" or args.model_filter in model_label:
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
                for sum_id, sum in filtered_sums_by_id.items():
                    if sum not in unique_sums:
                        # Detect entities if they're not cached
                        if sum_id not in filtered_ents_by_id:
                            filtered_ents_by_id[sum_id] = detect_entities(
                                sum, xsum_test[sum_id]["document"]
                            )
                        unique_sums.add(sum)
                        unique_sum_ent_count += len(filtered_ents_by_id[sum_id])
                print(f"Model: {model_label}")
                agg_metrics, summaries = evaluate_factuality(
                    filtered_sums_by_id,
                    filtered_ents_by_id,
                    failed_sums_by_id,
                    gold_sums,
                    gold_metadata,
                    xsum_test,
                    args.annotate and "gold" not in model_label.lower(),
                    args.entity_label_match,
                    args.print_first_n,
                    is_gef="gef" in model_label.lower(),
                    is_oracle="oracle" in model_label.lower(),
                    count_skips=args.count_skips,
                    compute_rouge=False,
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
                        agg_metrics["entities"]["extrinsic_factuality_ratio"]["mean"],
                        agg_metrics["entities"]["count_extrinsic"],
                        agg_metrics["summaries"]["unknown"],
                        agg_metrics["summaries"]["skipped"],
                        agg_metrics["summaries"]["failed"],
                        agg_metrics["summaries"]["ents_per_sum"],
                        agg_metrics["summaries"]["sum_with_extrinsic"],
                        agg_metrics["summaries"]["sum_with_extrinsic_factual"],
                    ]
                )
                for sum_id, sum_metrics in summaries.items():
                    if sum_id not in summary_results:
                        summary_results[sum_id] = {}
                    for metric, value in sum_metrics.items():
                        summary_results[sum_id][f"{model_label}_{metric}"] = value
        
        if args.test_set_offset != 0:
            break
        df_aggregated = pd.DataFrame(
            aggregated_results,
            columns=[
                "model",
                "total",
                "factual",
                "non_factual",
                "non_factual_extrinsic",
                "non_factual_intrinsic",
                "extrinsic_factuality_ratio",
                "extrinsic_entity_count",
                "unknown",
                "skipped",
                "failed",
                "ents_per_sum",
                "sum_with_extrinsic",
                "sum_with_extrinsic_factual",
            ],
        ).set_index("model")
        eval_filename = f"{data_subset}-{args.test_size}{beam_suffix}"
        df_aggregated.to_csv(f"results/evaluation/{eval_filename}.csv")
        df_summaries = pd.DataFrame.from_dict(summary_results, orient="index")
        with open(f"results/evaluation/{eval_filename}-summaries.json", "w") as f:
            json.dump(df_summaries.to_dict(orient="index"), f, indent=2, sort_keys=True)

        # Export to latex
        model_mapping = [
            ("gef_oracle", "GEFOracle"),
            ("gef_classifier", "GEFClassifier"),
            (
                "baseline-bart" if "bart" in data_subset else "baseline-pegasus",
                "Baseline",
            ),
            ("corrector", "Corrector"),
            ("pinocchio", "Pinocchio"),
            ("rl-fact", "RLFact"),
        ]
        label_mapping = [
            ("factual", "Factual"),
            ("non_factual", "NonFactual"),
            ("non_factual_extrinsic", "NonFactualExtrinsic"),
            ("non_factual_intrinsic", "NonFactualIntrinsic"),
            ("extrinsic_factuality_ratio", "FactualExtrinsicRatio"),
            ("extrinsic_entity_count", "ExtrinsicEntityCount"),
            ("sum_with_extrinsic", "SummaryWithExtrinsic"),
            ("sum_with_extrinsic_factual", "SummaryWithExtrinsicFactual"),
            ("skipped", "Skipped"),
        ]
        with open(f"results/latex/{eval_filename}.tex", "w") as f:
            for model_index, model_label in model_mapping:
                if model_index in df_aggregated.index:
                    if "pegasus" in data_subset:
                        model_label = "pegasus" + model_label
                    elif "bart" in data_subset:
                        model_label = "bart" + model_label
                    for metric_index, metric_label in label_mapping:
                        metric = df_aggregated.loc[model_index][metric_index]
                        str = (
                            f"\\newcommand{{\\{model_label}{metric_label}}}{{{metric:.0%}}}"
                            + "\n"
                        )
                        f.write(str.replace("%", "\\%"))
    print(f"{unique_sum_ent_count} entities in {len(unique_sums)} unique summaries")
