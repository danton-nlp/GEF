from sumtool.storage import store_model_summaries, get_summaries
import json


if __name__ == "__main__":
    sums_by_id_fallback_ours = {}
    sums_by_id_fallback_theirs = {}
    sums_by_id_with_skip = {}
    n_missing = 0
    n_pinocchio_zero = 0
    baseline_sums = get_summaries("xsum", "facebook-bart-large-xsum")
    with open("./data/results-pinocchio/xsum-pinocchio-results.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            sum_id = obj["id"]
            fallback = False
            pinocchio_summary = obj["pinocchio_predicted"].strip()

            if pinocchio_summary == "":
                n_missing += 1
                fallback = True
            elif obj["num_pinocchio_fires"] == 0:
                n_pinocchio_zero += 1
                fallback = True

            if fallback:
                sums_by_id_fallback_theirs[sum_id] = obj["bart_predicted"].strip()
                sums_by_id_fallback_ours[sum_id] = baseline_sums[sum_id]["summary"]
            else:
                sums_by_id_fallback_theirs[sum_id] = pinocchio_summary
                sums_by_id_fallback_ours[sum_id] = pinocchio_summary
                sums_by_id_with_skip[sum_id] = pinocchio_summary

    print(f"{n_missing} missing reverted to baseline")
    print(f"{n_pinocchio_zero} with pinocchio fire 0 reverted to baseline")
    store_model_summaries(
        "xsum",
        "pinocchio-fallback-ours",
        {
            "Title": "Don’t Say What You Don’t Know: Improving the Consistency of Abstractive Summarization by Constraining Beam Search",
            "Paper": "https://arxiv.org/pdf/2203.08436.pdf",
        },
        sums_by_id_fallback_ours,
    )
    store_model_summaries(
        "xsum",
        "pinocchio-fallback-theirs",
        {
            "Title": "Don’t Say What You Don’t Know: Improving the Consistency of Abstractive Summarization by Constraining Beam Search",
            "Paper": "https://arxiv.org/pdf/2203.08436.pdf",
        },
        sums_by_id_fallback_theirs,
    )
    store_model_summaries(
        "xsum",
        "pinocchio-with-skip",
        {
            "Title": "Don’t Say What You Don’t Know: Improving the Consistency of Abstractive Summarization by Constraining Beam Search",
            "Paper": "https://arxiv.org/pdf/2203.08436.pdf",
        },
        sums_by_id_with_skip,
    )
    print(f"Persisted ({len(sums_by_id_fallback_ours), len(sums_by_id_fallback_theirs), len(sums_by_id_with_skip)}) summaries")
