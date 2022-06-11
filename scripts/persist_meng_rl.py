from datasets import load_dataset
from sumtool.storage import store_model_summaries
import re


if __name__ == "__main__":
    xsum_test = load_dataset("xsum")["test"]

    for split in ["test", "test_last"]:
        meng_sums_by_id = {}
        n_empty = 0
        n_henko = 0
        with open(f"./data/results-meng-rl/{split}.hypo", "r") as f:
            for summary, sum_id in zip(f, xsum_test["id"]):
                if summary.strip() == "":
                    n_empty += 1
                if summary.startswith("henko"):
                    n_henko += 1
                
                # Replace dots at the end
                meng_sums_by_id[sum_id] = re.sub("[\\.]+$", ".", summary.strip())

        print(f"Split: {split}, sums: {len(meng_sums_by_id)}")
        print(f"{n_empty} empty summaries!")
        print(f"{n_henko} henko summaries!")

        store_model_summaries(
            "xsum",
            f"meng-{split}",
            {
                "Title": "Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization",
                "Paper": "https://arxiv.org/abs/2109.09784",
                "Github": "https://github.com/mcao516/EntFA",
            },
            meng_sums_by_id,
        )
