import streamlit as st
import json
import os
import pandas as pd


def st_select_results(folder="results/fbs-logs/"):
    return os.path.join(
        folder,
        st.selectbox("Select run", options=sorted(os.listdir(folder)))
    )


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def render_explore_logs():
    st.title("Explore Logs")
    results_path = st_select_results()
    results = load_results(results_path)

    with st.expander("Result args"):
        st.write(results["args"])

    sorted_keys = sorted([int(x) for x in results["iterations"].keys()])
    summary_stats = []
    entity_stats = []
    entity_stats_by_type = []
    for iteration_idx in sorted_keys:
        stats = results["iterations"][str(iteration_idx)]["stats"]
        summary_stats.append(
            {
                "iteration": iteration_idx,
                "factual": stats["summary"]["factual"],
                "non_factual": stats["summary"]["non_factual"],
                "unknown": stats["summary"]["unknown"],
                "failed": stats["summary"]["failed"],
                "completed": stats["summary"]["completed"],
                "total": stats["summary"]["total"],
            }
        )
        entity_stats.append(
            {
                "iteration": iteration_idx,
                "factual": stats["entity"]["label"]["Factual Hallucination"],
                "non_factual": stats["entity"]["label"]["Non-factual Hallucination"],
                "unknown": stats["entity"]["label"]["Unknown"],
                "non_hallucinated": stats["entity"]["label"]["Non-hallucinated"],
            }
        )
        for type in stats["entity"]["type"].keys():
            entity_stats_by_type.append(
                {
                    "iteration": iteration_idx,
                    "type": type,
                    "factual": stats["entity"]["type"][type]["Factual Hallucination"],
                    "non_factual": stats["entity"]["type"][type][
                        "Non-factual Hallucination"
                    ],
                    "unknown": stats["entity"]["type"][type]["Unknown"],
                    "non_hallucinated": stats["entity"]["type"][type][
                        "Non-hallucinated"
                    ],
                }
            )

    st.subheader("Summary-level stats")
    df_summary = pd.DataFrame(summary_stats).set_index("iteration")
    first_iter = df_summary.iloc[0]
    last_iter = df_summary.iloc[-1]
    n_summaries_corrected = sum([1 for x in results["iterations"]["1"]["summaries"].values() if len(x["banned_phrases"]) > 0])
    st.markdown(
f"""
Summaries corrected: {n_summaries_corrected} ({n_summaries_corrected/first_iter.total:.2%})

**Factual:** {first_iter.factual/first_iter.total:.2%} --> {last_iter.factual/last_iter.total:.2%}  \\
**Non-factual:** {first_iter.non_factual/first_iter.total:.2%} --> {last_iter.non_factual/last_iter.total:.2%}  \\
**Unknown:** {first_iter.unknown/first_iter.total:.2%} --> {last_iter.unknown/last_iter.total:.2%} \\
**Failed:** {first_iter.failed/first_iter.total:.2%} --> {last_iter.failed/last_iter.total:.2%}
"""
    )

    st.dataframe(df_summary)

    st.subheader("Entity-level stats")
    df_entity = pd.DataFrame(entity_stats).set_index("iteration")
    df_entity["total"] = (
        df_entity["factual"]
        + df_entity["non_factual"]
        + df_entity["unknown"]
        + df_entity["non_hallucinated"]
    )
    st.dataframe(df_entity)

    st.subheader("Entity-level stats by type")
    df_entity_type = pd.DataFrame(entity_stats_by_type).set_index("iteration")
    df_entity_type["total"] = (
        df_entity_type["factual"]
        + df_entity_type["non_factual"]
        + df_entity_type["unknown"]
        + df_entity_type["non_hallucinated"]
    )
    st.dataframe(df_entity_type)
