from src.data_utils import (
    load_shuffled_test_split,
    load_summaries_from_logs,
    get_gold_xsum_data,
    load_xsum_dict,
)
from sumtool.storage import get_summaries
from src.evaluation.factuality import evaluate_factuality

TEST_SIZE = 100


def load_data(results_path: str):
    if "results" in results_path:
        # load gef results
        sums_by_id, sum_ents_by_id, failed_sums_by_id = load_summaries_from_logs(
            results_path,
            max_iterations=100
        )
    else:
        # load from sumtool
        sums_by_id = {
            sum_id: x["summary"]
            for sum_id, x in get_summaries("xsum", results_path).items()
        }
        sum_ents_by_id = {}
        failed_sums_by_id = {}
    gold_sums, gold_metadata = get_gold_xsum_data()
    xsum_test = load_xsum_dict("test")
    test_set_ids = set(
        load_shuffled_test_split(xsum_test, "bart-test-extrinsic", 100).keys()
    )

    filtered_sums_by_id = {
        sum_id: x for sum_id, x in sums_by_id.items() if sum_id in test_set_ids
    }
    filtered_ents_by_id = {
        sum_id: x for sum_id, x in sum_ents_by_id.items() if sum_id in test_set_ids
    }
    return (
        filtered_sums_by_id,
        filtered_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    )


def test_evaluate_factuality_oracle():
    (
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    ) = load_data("results/gef-logs/bart-test-extrinsic-oracle.json")
    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_all",
        print_first_n=0,
        is_gef=True,
        is_oracle=True,
        count_skips=True,
        compute_rouge=False
    )
    # These metrics might change if the dataset changes
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 0.67
    assert agg_metrics["summaries"]["non_factual"] == 0.11
    assert agg_metrics["summaries"]["skipped"] == 0.22
    assert agg_metrics["summaries"]["failed"] == 12
    assert agg_metrics["entities"]["Unknown"] == 0
    assert agg_metrics["entities"]["Factual Hallucination"] == 87
    assert agg_metrics["entities"]["Intrinsic Hallucination"] == 13
    assert agg_metrics["entities"]["Non-hallucinated"] == 150
    assert agg_metrics["entities"]["total"] == 87 + 13 + 150

    # CRUCIAL: Oracle should have 0 non factual hallucinations because they're skipped
    assert agg_metrics["summaries"]["non_factual_extrinsic"] == 0
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 0

    # This also means that extrinsic factuality ratio should always be 1.
    assert agg_metrics["entities"]["extrinsic_factuality_ratio"]["mean"] == 1
    assert agg_metrics["entities"]["extrinsic_factuality_ratio"]["stdev"] == 0

    # All non factual errors stem from intrinsic hallucinations
    assert (
        agg_metrics["summaries"]["non_factual"]
        == agg_metrics["summaries"]["non_factual_intrinsic"]
    )
    # Should sum to 1
    assert (
        sum(
            [
                agg_metrics["summaries"]["factual"],
                agg_metrics["summaries"]["non_factual"],
                agg_metrics["summaries"]["skipped"],
            ]
        )
        == 1
    )


def test_evaluate_factuality_oracle_no_skips():
    (
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    ) = load_data("results/gef-logs/bart-test-extrinsic-oracle.json")
    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_all",
        print_first_n=0,
        is_gef=True,
        is_oracle=True,
        count_skips=False,
        compute_rouge=False
    )
    # These metrics might change if the dataset changes
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 0.67
    assert agg_metrics["summaries"]["non_factual"] == 0.33
    assert agg_metrics["summaries"]["failed"] == 12
    assert agg_metrics["entities"]["Unknown"] == 0
    assert agg_metrics["entities"]["Factual Hallucination"] == 104
    assert agg_metrics["entities"]["Intrinsic Hallucination"] == 15
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 38
    assert agg_metrics["entities"]["Non-hallucinated"] == 187
    assert agg_metrics["entities"]["total"] == 104 + 15 + 38 + 187

    assert (
        agg_metrics["entities"]["extrinsic_factuality_ratio"]["mean"]
        == 0.9354166666666666
    )


def test_evaluate_factuality_classifier():
    (
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    ) = load_data("results/gef-logs/bart-test-extrinsic-classifier-knnv1.json")
    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_all",
        print_first_n=0,
        is_gef=True,
        is_oracle=False,
        count_skips=True,
        compute_rouge=False
    )
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 0.5
    assert agg_metrics["summaries"]["non_factual"] == 0.31
    assert agg_metrics["summaries"]["non_factual_extrinsic"] == 0.24
    assert agg_metrics["summaries"]["non_factual_intrinsic"] == 0.11
    assert agg_metrics["summaries"]["skipped"] == 0.19
    assert agg_metrics["summaries"]["failed"] == 18
    assert agg_metrics["summaries"]["unknown"] == 0
    assert agg_metrics["entities"]["Unknown"] == 0
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 30
    assert agg_metrics["entities"]["Factual Hallucination"] == 65
    assert agg_metrics["entities"]["Intrinsic Hallucination"] == 12
    assert agg_metrics["entities"]["Non-hallucinated"] == 159
    assert agg_metrics["entities"]["total"] == 30 + 65 + 12 + 159

    assert (
        agg_metrics["entities"]["extrinsic_factuality_ratio"]["mean"]
        == 0.9226415094339623
    )

    # Should sum to 1
    assert (
        sum(
            [
                agg_metrics["summaries"]["factual"],
                agg_metrics["summaries"]["non_factual"],
                agg_metrics["summaries"]["skipped"],
            ]
        )
        == 1
    )


def test_evaluate_factuality_classifier_no_skips():
    (
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    ) = load_data("results/gef-logs/bart-test-extrinsic-classifier-knnv1.json")
    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_all",
        print_first_n=0,
        is_gef=True,
        is_oracle=False,
        count_skips=False,
        compute_rouge=False
    )
    # These metrics might change if the dataset changes
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 0.52
    assert agg_metrics["summaries"]["non_factual"] == 0.48
    assert agg_metrics["summaries"]["non_factual_extrinsic"] == 0.41
    assert agg_metrics["summaries"]["non_factual_intrinsic"] == 0.14
    assert agg_metrics["summaries"]["skipped"] == 0
    assert agg_metrics["summaries"]["failed"] == 18
    assert agg_metrics["summaries"]["unknown"] == 0
    assert agg_metrics["entities"]["Unknown"] == 0
    assert agg_metrics["entities"]["Factual Hallucination"] == 80
    assert agg_metrics["entities"]["Intrinsic Hallucination"] == 15
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 63
    assert agg_metrics["entities"]["Non-hallucinated"] == 184
    assert agg_metrics["entities"]["total"] == 80 + 15 + 63 + 184

    assert (
        agg_metrics["entities"]["extrinsic_factuality_ratio"]["mean"]
        == 0.8614583333333333
    )


def test_evaluate_factuality_baseline():
    (
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    ) = load_data("facebook-bart-large-xsum")
    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_all",
        print_first_n=0,
        is_gef=False,
        is_oracle=False,
        compute_rouge=False
    )
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 0.42
    assert agg_metrics["summaries"]["non_factual"] == 0.58
    assert agg_metrics["summaries"]["non_factual_extrinsic"] == 0.52
    assert agg_metrics["summaries"]["non_factual_intrinsic"] == 0.12
    assert agg_metrics["summaries"]["skipped"] == 0
    assert agg_metrics["summaries"]["failed"] == 0
    assert agg_metrics["summaries"]["unknown"] == 0
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 72
    assert agg_metrics["entities"]["Unknown"] == 0
    assert agg_metrics["entities"]["Factual Hallucination"] == 90
    assert agg_metrics["entities"]["Intrinsic Hallucination"] == 13
    assert agg_metrics["entities"]["Non-hallucinated"] == 186
    assert agg_metrics["entities"]["total"] == 72 + 90 + 13 + 186

    assert (
        agg_metrics["entities"]["extrinsic_factuality_ratio"]["mean"]
        == 0.8222222222222222
    )

    # Should sum to 1
    assert (
        sum(
            [
                agg_metrics["summaries"]["factual"],
                agg_metrics["summaries"]["non_factual"],
                agg_metrics["summaries"]["skipped"],
            ]
        )
        == 1
    )


def test_evaluate_factuality_gold():
    (
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    ) = load_data("gold")
    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        failed_sums_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_all",
        print_first_n=0,
        is_gef=False,
        is_oracle=False,
        compute_rouge=False
    )
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 1
    assert agg_metrics["summaries"]["non_factual"] == 0
    assert agg_metrics["summaries"]["non_factual_extrinsic"] == 0
    assert agg_metrics["summaries"]["non_factual_intrinsic"] == 0
    assert agg_metrics["summaries"]["skipped"] == 0
    assert agg_metrics["summaries"]["failed"] == 0
    assert agg_metrics["summaries"]["unknown"] == 0
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 0
    assert agg_metrics["entities"]["Unknown"] == 0
    assert agg_metrics["entities"]["Factual Hallucination"] == 192
    assert agg_metrics["entities"]["Intrinsic Hallucination"] == 0
    assert agg_metrics["entities"]["Non-hallucinated"] == 169
    assert agg_metrics["entities"]["total"] == 192 + 169

    assert agg_metrics["entities"]["extrinsic_factuality_ratio"]["mean"] == 1
    assert agg_metrics["entities"]["extrinsic_factuality_ratio"]["stdev"] == 0

    # Should sum to 1
    assert (
        sum(
            [
                agg_metrics["summaries"]["factual"],
                agg_metrics["summaries"]["non_factual"],
                agg_metrics["summaries"]["skipped"],
            ]
        )
        == 1
    )
