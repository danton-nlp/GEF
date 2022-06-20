import streamlit as st
import json
import pandas as pd

MODELS = []


def load_summaries_df(path: str):
    with open(path, "r") as f:
        json_sums = json.load(f)
        df = pd.DataFrame(json_sums.values(), index=list(json_sums.keys()))
        return df


def render_compare_results():
    col1, col2 = st.columns(2)
    data_subset = col1.selectbox(
        "Data subset",
        options=[
            "bart-test-extrinsic",
            "pegasus-test-extrinsic",
            "xent-test",
            "xsum-test",
        ],
    )
    test_size = col2.number_input("Test size", value=100)
    filename = f"results/evaluation/{data_subset}-{test_size}"

    df_aggregated = (
        pd.read_csv(f"{filename}.csv")
        .sort_values("model", ascending=True)
        .set_index("model")
    )
    df_sums = load_summaries_df(f"{filename}-summaries.json")
    st.title("Compare Results")

    selected_models = st.multiselect(
        "Select models",
        options=df_aggregated.index,
        default=[x for x in [
            "fbs_classifier",
            "fbs_oracle",
            "baseline-pegasus",
            "baseline-bart"
        ] if x in df_aggregated.index],
    )
    selected_columns = st.multiselect(
        "Select columns",
        options=df_aggregated.columns,
        default=[
            "factual",
            "non_factual",
            "non_factual_intrinsic",
            "non_factual_extrinsic",
            # "skipped"
            "extrinsic_factuality_ratio",
            "extrinsic_entity_count",
            "sum_with_extrinsic",
            "sum_with_extrinsic_factual"
        ],
    )

    st.subheader("Human annotation")
    st.dataframe(df_aggregated[selected_columns].loc[selected_models])
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

    st.table(df_inspect[[f"{non_factual_model}_summary", f"{factual_model}_summary"]])
