from src.data_utils import (
    load_extrinsic_test_set,
    load_summaries_from_logs,
    get_gold_data,
    load_xsum_dict,
)
from src.evaluation.factuality import evaluate_factuality
from sumtool.storage import get_summary_metrics

TEST_SIZE = 100


def load_data(results_path: str):
    sums_by_id, sum_ents_by_id = load_summaries_from_logs(results_path)
    gold_sums, gold_metadata = get_gold_data()
    baseline_metadata = get_summary_metrics("xsum", "facebook-bart-large-xsum")
    xsum_test = load_xsum_dict("test")
    test_set = load_extrinsic_test_set(
        xsum_test, baseline_metadata, gold_metadata, TEST_SIZE
    )
    test_set_ids = {k for (k, v) in test_set}

    filtered_sums_by_id = {
        sum_id: x for sum_id, x in sums_by_id.items() if sum_id in test_set_ids
    }
    filtered_ents_by_id = {
        sum_id: x for sum_id, x in sum_ents_by_id.items() if sum_id in test_set_ids
    }
    return (
        filtered_sums_by_id,
        filtered_ents_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
    )


# TODO: add tests for sub-routines?


def test_evaluate_factuality_oracle():
    (sums_by_id, sum_ents_by_id, gold_sums, gold_metadata, xsum_test) = load_data(
        "results/test-extrinsic-oracle.json"
    )

    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_intrinsic",
        print_first_n=0,
        is_fbs=True,
        is_gold=False,
    )
    # These metrics might change if the dataset changes
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 0.65
    assert agg_metrics["summaries"]["non_factual"] == 0.12
    assert agg_metrics["summaries"]["skipped"] == 0.23
    assert agg_metrics["summaries"]["failed"] == 7
    assert agg_metrics["rouge1"] > 0.46
    assert agg_metrics["rouge2"] > 0.22

    # CRUCIAL: Oracle should have 0 non factual hallucinations because they're skipped
    assert agg_metrics["summaries"]["non_factual_extrinsic"] == 0
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 0
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


def test_evaluate_factuality_classifier():
    (sums_by_id, sum_ents_by_id, gold_sums, gold_metadata, xsum_test) = load_data(
        "results/test-extrinsic-classifier-knnv1.json"
    )
    gold_sums, gold_metadata = get_gold_data()
    xsum_test = load_xsum_dict("test")
    agg_metrics, summaries = evaluate_factuality(
        sums_by_id,
        sum_ents_by_id,
        gold_sums,
        gold_metadata,
        xsum_test,
        should_annotate=False,
        entity_match_type="strict_intrinsic",
        print_first_n=0,
        is_fbs=True,
        is_gold=False,
    )
    assert agg_metrics["summaries"]["total"] == 100
    assert agg_metrics["summaries"]["factual"] == 0.48
    assert agg_metrics["summaries"]["non_factual"] == 0.26
    assert agg_metrics["summaries"]["non_factual_extrinsic"] == 0.19
    assert agg_metrics["summaries"]["non_factual_intrinsic"] == 0.1
    assert agg_metrics["summaries"]["skipped"] == 0.26
    assert agg_metrics["summaries"]["failed"] == 14
    assert agg_metrics["summaries"]["unknown"] == 0
    assert agg_metrics["entities"]["Non-factual Hallucination"] == 24

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
