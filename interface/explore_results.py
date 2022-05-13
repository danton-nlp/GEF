import streamlit as st
import json
import os
import pandas as pd


def st_select_results(folder="results"):
    return os.path.join(
        "results", st.selectbox("Select run", options=os.listdir(folder))
    )


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def render_explore_results():
    st.title("Explore Results")
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
    st.dataframe(pd.DataFrame(summary_stats).set_index("iteration"))

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
