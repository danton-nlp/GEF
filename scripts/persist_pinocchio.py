from sumtool.storage import store_model_summaries
import json


if __name__ == "__main__":
    sums_by_id = {}
    n_missing = 0
    with open("./data/results-pinocchio/xsum-pinocchio-results.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            summary = obj["pinocchio_predicted"].strip()
            if summary == "":
                n_missing += 1
                summary = obj["bart_predicted"].strip()
            sums_by_id[obj["id"]] = summary
    print(f"{n_missing} reverted to baseline")
    store_model_summaries(
        "xsum",
        "pinocchio",
        {
            "Title": "Don’t Say What You Don’t Know: Improving the Consistency of Abstractive Summarization by Constraining Beam Search",
            "Paper": "https://arxiv.org/pdf/2203.08436.pdf",
        },
        sums_by_id,
    )
    print(f"Persisted {len(sums_by_id)}")
