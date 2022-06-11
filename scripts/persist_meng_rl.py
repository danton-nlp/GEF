from datasets import load_dataset
from sumtool.storage import store_model_summaries
import re


if __name__ == "__main__":
    xsum_test = load_dataset("xsum")["test"]

    meng_sums_by_id = {}
    n_empty = 0
    with open("./data/results-meng-rl/test.hypo", "r") as f:
        for summary, sum_id in zip(f, xsum_test["id"]):
            # Replace dots at the end
            meng_sums_by_id[sum_id] = re.sub("[\\.]+$", ".", summary.strip())
            if summary.strip() == "":
                n_empty += 1

    print(f"{n_empty} empty summaries!")

    store_model_summaries(
        "xsum",
        "meng-rl",
        {
            "Title": "Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization",
            "Paper": "https://arxiv.org/abs/2109.09784",
            "Github": "https://github.com/mcao516/EntFA",
        },
        meng_sums_by_id,
    )
