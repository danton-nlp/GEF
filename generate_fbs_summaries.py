
"""
  Run FBS generation for a data subset
  with BART, Pegasus (Oracle & Classifier).
"""
import argparse
import os
import subprocess
from src.misc_utils import get_new_log_path, Timer


LOGGING_PATH = "logs-iterative"
TEST_SIZE = 200


def run_command(cmd: str):
    with Timer(f"[Command]: {cmd}"):
        return subprocess.run(cmd.split(" "), check=True)


def run_iterative_constraints(args, model_summarization, is_oracle):
    model_prefix = "pegasus" if "pegasus" in model_summarization else "bart"
    fbs_suffix = "oracle" if is_oracle else args.classifier_results_suffix
    if args.data_subset == "test_extrinsic":
        data_subset = f"{model_prefix}-{args.data_subset}"
        results_path = f"results/fbs-logs/{data_subset}-{fbs_suffix}.json"
    else:
        data_subset = args.data_subset
        results_path = f"results/fbs-logs/{model_prefix}-{data_subset}-{fbs_suffix}.json"
    logging_path = get_new_log_path(LOGGING_PATH) + ".json"
    print(f"Will write {logging_path} to {results_path} upon completion")
    run_command(
        " ".join(
            x
            for x in [
                f"python iterative_constraints.py --data_subset {data_subset}",
                f"--batch_size {args.batch_size}",
                f"--classifier_batch_size {args.classifier_batch_size}",
                f"--model_summarization {model_summarization}",
                "--max_iterations 10",
                f"--test_size {TEST_SIZE}",
                (
                    f"--pickled_classifier {args.pickled_classifier}"
                    if not is_oracle
                    else ""
                ),
            ]
            if x != ""
        )
    )
    os.system(f"mv {logging_path} {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_subset", type=str, default="test-extrinsic")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--classifier_batch_size", type=int, default=16)
    parser.add_argument("--pickled_classifier", type=str, default="factuality-classifiers/v1-knn.pickle")
    parser.add_argument("--classifier_results_suffix", type=str, default="classifier-knnv1")
    args = parser.parse_args()

    for model_path in ["facebook/bart-large-xsum", "google/pegasus-xsum"]:
        run_iterative_constraints(args, model_path, is_oracle=True)
        run_iterative_constraints(args, model_path, is_oracle=False)
