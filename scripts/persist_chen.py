from sumtool.storage import get_summaries, store_model_summaries


if __name__ == "__main__":
    gold_summaries = get_summaries("xsum", "gold")
    baseline_sums = get_summaries("xsum", "facebook-bart-large-xsum")

    id_by_ref = {x["summary"]: sum_id for sum_id, x in gold_summaries.items()}

    chen_summary = []
    chen_target = []
    with open("./data/results-chen-corrector/target.part.txt", "r") as f:
        for line in f:
            chen_target.append(line.strip())
    with open("./data/results-chen-corrector/corrected.part.txt", "r") as f:
        for line in f:
            chen_summary.append(line.strip())

    missing = 0
    chen_sums_by_id = {}
    for summary, target in zip(chen_summary, chen_target):
        if target in id_by_ref:
            sum_id = id_by_ref[target]
            chen_sums_by_id[sum_id] = summary
        else:
            missing += 1
    for sum_id, data in baseline_sums.items():
        if sum_id not in chen_sums_by_id:
            chen_sums_by_id[sum_id] = data["summary"]

    store_model_summaries(
        "xsum",
        "chen-corrector",
        {
            "Title": "Improving Faithfulness in Abstractive Summarization with Contrast Candidate Generation and Selection",
            "Paper": "https://www.seas.upenn.edu/~sihaoc/static/pdf/CZSR21.pdf",
            "Github": "https://github.com/CogComp/faithful_summarization",
        },
        chen_sums_by_id
    )
    print(f"Missing: {missing}")
