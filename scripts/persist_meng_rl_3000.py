from datasets import load_dataset
from sumtool.storage import store_model_summaries
import re


_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
        "Media playback is not supported on this device",
        "These are external links and will open in a new window."
    ]
)

REMOVE = "[\.,\?\s\()]"


def clean_source(source):
    for remove in _REMOVE_LINES:
        source = source.replace(remove.strip(), "")
    source = source.replace("\n", " ").strip().lower()
    source = re.sub(REMOVE, "", source)
    return source[:60]

if __name__ == "__main__":
    xsum_test = load_dataset("xsum")["test"]

    id_by_source = {
        clean_source(source): sum_id
        for (source, sum_id) in zip(
            xsum_test["document"], xsum_test["id"]
        )
    }

    meng_sums_by_id = {}
    n_missing = 0
    n_dups = 0
    with open("./data/results-meng-rl/test_3000.hypo", "r") as f_hypo:
        with open("./data/results-meng-rl/test_3000.source", "r") as f_source:
            for summary, source in zip(f_hypo, f_source):
                if clean_source(source) in id_by_source:
                    sum_id = id_by_source[clean_source(source)]
                    if sum_id in meng_sums_by_id:
                        n_dups += 1
                    meng_sums_by_id[sum_id] = summary.strip()
                else:
                    n_missing += 1
                    print()
                    print(source)

        print(f"{n_missing} missing summaries!")
        print(f"{n_dups} duplicates!")

        store_model_summaries(
            "xsum",
            "meng-3000",
            {
                "Title": "Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization",
                "Paper": "https://arxiv.org/abs/2109.09784",
                "Github": "https://github.com/mcao516/EntFA",
            },
            meng_sums_by_id,
        )
