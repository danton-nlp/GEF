from sumtool.storage import store_model_summaries, get_summaries
import json


if __name__ == "__main__":
    n_total = 0
    n_empty = 0
    n_no_fires = 0
    n_pinocchio = 0
    with open("./data/results-pinocchio/xsum-pinocchio-results.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            sum_id = obj["id"]
            pinocchio_summary = obj["pinocchio_predicted"].strip()
            fallback_summary = obj["bart_predicted"].strip()

            n_total += 1
            if pinocchio_summary == "":
                n_empty += 1
            else:
                if obj["num_pinocchio_fires"] == 0:
                    n_no_fires += 1
                n_pinocchio += 1

    print(f"total: {n_total}")
    print(f"empty: {n_empty}")
    print(f"no fires: {n_no_fires}")
    print(f"pinocchio sums: {n_pinocchio}")
