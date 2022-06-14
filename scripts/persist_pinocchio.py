from sumtool.storage import store_model_summaries
import json


if __name__ == "__main__":
    sums_by_id = {}
    with open("./data/results-pinocchio/xsum-pinocchio-results.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            sums_by_id[obj["id"]] = obj["pinocchio_predicted"]
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
