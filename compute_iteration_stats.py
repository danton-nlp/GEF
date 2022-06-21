from typing import List, Tuple
import pandas as pd
import json
from src.data_utils import load_xsum_dict
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def read_logs(path):
    iteration_logs = json.load(open(path, "r"))

    data = []
    for iteration in iteration_logs:
        result_obj = {
            "iteration": iteration["iteration"],
            "summaries_generated": iteration['summary_generated']
        }
        data.append(result_obj)

    return pd.DataFrame(data).set_index("iteration")


def normalize_df(df):
    return df / df.sum()


def compute_examples_converged_during_iteration(raw_stats, iteration_stat):
    next_iteration_idx = iteration_stat.name + 1
    try:
        next_iteration_summaries_generated = raw_stats.loc[next_iteration_idx, 'summaries_generated']
    except KeyError:
        next_iteration_summaries_generated = 0

    return iteration_stat['summaries_generated'] - next_iteration_summaries_generated


def convert_named_series_to_plot_df(named_series: List[Tuple[str, pd.Series]]):
    dfs = []

    for name, series in named_series:
        df = normalize_df(
            pd.DataFrame(series, columns=['examples_converged'])
        ).reset_index()
        df['system'] = name
        df = df.rename(columns={'index': 'iteration_index'})
        dfs.append(df)

    plot_df = pd.concat(dfs).reset_index().drop(columns=['index'])

    # add by 1 to convert interation index to "number of interations ran"
    plot_df['number_of_iterations_ran'] = plot_df['iteration_index'] + 1

    return plot_df


if __name__ == "__main__":
    description = "Compute and plot GEF convergence statistics. Plots are written to plots/"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("base_model", type=str, choices=['bart', 'pegasus'])
    parser.add_argument("--extrinsic_only", action='store_true', default=False, help="Compute ONLY on subset that includes extrinsic hallucinations.")
    args = parser.parse_args()

    xsum_test = load_xsum_dict("test")

    if args.base_model == 'bart':
        full_iteration_stats = read_logs("results/iteration-changes/bart-full-classifier-knnv1.json")
        oracle_iteration_stats = read_logs("results/iteration-changes/bart-test-extrinsic-100-oracle.json")
        knn_iteration_stats = read_logs("results/iteration-changes/bart-test-extrinsic-100-classifier-knnv1.json")
    else:
        full_iteration_stats = read_logs("results/iteration-changes/pegasus-full-classifier-knnv1.json")
        oracle_iteration_stats = read_logs("results/iteration-changes/pegasus-test-extrinsic-75-oracle.json")
        knn_iteration_stats = read_logs("results/iteration-changes/pegasus-test-extrinsic-75-classifier-knnv1.json")

    experiment_raw_data = {
        'full': full_iteration_stats,
        'oracle': oracle_iteration_stats,
        'knn': knn_iteration_stats
    }

    iteration_convergence_across_experiments: List[Tuple[str, pd.Series]] = []

    for experiment_name, raw_stats in experiment_raw_data.items():
        examples_converged_across_iterations = []
        for index, iteration_stat in raw_stats.iterrows():
            examples_converged = compute_examples_converged_during_iteration(raw_stats, iteration_stat)
            examples_converged_across_iterations.append(examples_converged)

        examples_converged_across_iterations = pd.Series(examples_converged_across_iterations)
        iteration_convergence_across_experiments.append((experiment_name, examples_converged_across_iterations))

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=convert_named_series_to_plot_df(iteration_convergence_across_experiments),
        x='number_of_iterations_ran',
        y='examples_converged',
        hue='system'
    )
    ax.set_xlabel("Number of Iterations", fontsize=12)
    ax.set_ylabel('Examples Converged', fontsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(f'Convergence over Iterations for {args.base_model.upper()} GEF', fontsize=12)
    plt.savefig(f"plots/{args.base_model}_gef_convergence_over_iterations.pdf")
    print(f'distribution plot written to plots/{args.base_model}_gef_convergence_over_iterations.pdf')

    for name, counts in iteration_convergence_across_experiments:
        frequencies_by_iteration = counts / counts.sum()

        # add by 1 to convert interation index to "number of interations ran"
        distrib = stats.rv_discrete(name=name, values=(frequencies_by_iteration.index + 1, frequencies_by_iteration.values))
        print(f"{name}: {distrib.mean()} ± {distrib.std()}")
