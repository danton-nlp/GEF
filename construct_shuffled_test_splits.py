import random
from typing import List
from sumtool.storage import get_summary_metrics
import json
from src.data_utils import load_xsum_dict, get_gold_xsum_data


def construct_test_split(xsum_test, filter_fn) -> List[str]:
    """
    Construct shuffled test set from xent test,
    filter_fn indicates how to filter xsum test.
    """
    filtered_sum_ids = [sum_id for sum_id in xsum_test.keys() if filter_fn(sum_id)]
    rng_data_split = random.Random(42)
    return rng_data_split.sample(filtered_sum_ids, len(filtered_sum_ids))


if __name__ == "__main__":
    xsum_test = load_xsum_dict("test")
    baseline_metadata = get_summary_metrics("xsum", "facebook-bart-large-xsum")
    gold_sums, gold_metadata = get_gold_xsum_data()

    def filter_xsum(sum_id):
        return "xent-train" not in gold_metadata[sum_id]

    def filter_xent_test(sum_id):
        return "xent-test" in gold_metadata[sum_id]

    def filter_xsum_extrinsic(sum_id):
        return (
            "xent-train" not in gold_metadata[sum_id]
            and len(
                [
                    ent
                    for ent in baseline_metadata[sum_id]["entities"]
                    if not ent["in_source"]
                ]
            )
            > 0
        )

    test_subset_ids = {
        "xsum-test": construct_test_split(xsum_test, filter_xsum),
        "xent-test": construct_test_split(xsum_test, filter_xent_test),
        "test-extrinsic": construct_test_split(xsum_test, filter_xsum_extrinsic),
    }

    with open("./data/xsum_shuffled_test_splits.json", "w") as f:
        json.dump(test_subset_ids, f, indent=2)
