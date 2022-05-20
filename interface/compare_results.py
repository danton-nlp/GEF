import streamlit as st
import json
import os
import pandas as pd


filename = "results/test-extrinsic-100"
MODELS = []


def render_compare_results():

    df_aggregated = (
        pd.read_csv(f"{filename}.csv")
        .sort_values("model", ascending=True)
        .set_index("model")
    )
    df_sums = pd.read_json(f"{filename}-summaries.json")
    st.title("Compare Results")

    st.subheader("Human annotation")
    st.dataframe(df_aggregated)
    st.subheader("Compare models")
    col1, col2 = st.columns(2)
    non_factual_model = col1.selectbox(
        "This model is non-factual: ", options=df_aggregated.index, index=0
    )
    factual_model = col2.selectbox(
        "...and this model is factual: ", options=df_aggregated.index, index=2
    )

    df_inspect = df_sums[
        df_sums[f"{non_factual_model}_is_non_factual"]
        & df_sums[f"{factual_model}_is_factual"]
    ]

    st.table(df_inspect[[
        f"{non_factual_model}_summary",
        f"{factual_model}_summary"
    ]])
