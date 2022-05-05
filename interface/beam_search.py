from datasets import load_dataset
import streamlit as st
from src.beam_validators import DictionaryValidator
from src.word_logits_processor import WordLogitsProcessor
from src.generation_utils import generate_summaries, load_model_and_tokenizer
import pandas as pd
import numpy as np


@st.experimental_memo
def cached_model_and_tokenizer():
    return load_model_and_tokenizer("facebook/bart-large-xsum")


@st.experimental_memo
def cached_generate_summaries(docs_to_summarize, excluded_dictionary, num_beams):
    model, tokenizer = cached_model_and_tokenizer()

    return generate_summaries(
        model,
        tokenizer,
        docs_to_summarize,
        WordLogitsProcessor(
            tokenizer, num_beams, DictionaryValidator(excluded_dictionary)
        ),
        return_beam_metadata=True,
        num_beams=num_beams
    )


def render_beam_search():
    st.title("Beam Search Visualization")

    st.markdown("""
This app shows how dictionary constraints can be added to beam search (on a word-level).

The first two xsum articles are summarized.
    """)

    str_constraints = st.text_area(
        "Comma-separated dictionary to exclude from the summary",
        value="Edinburgh, Wales",
    )
    excluded_dictionary = {x.strip() for x in str_constraints.split(",") if len(x.strip()) > 0}
    st.write(excluded_dictionary)

    model, tokenizer = cached_model_and_tokenizer()
    xsum_test = load_dataset("xsum")["test"]
    num_beams = st.number_input("Number of beams", value=4)

    if st.button("Generate summaries"):

        st.subheader("Without constraints")
        summaries, beam_metadata = cached_generate_summaries(xsum_test["document"][:2], {}, num_beams)
        data = []
        for i, summary in enumerate(summaries):
            data.append((
                summary,
                np.exp(beam_metadata[i]["score"]),
                len(beam_metadata[i]["dropped_seqs"]),
                beam_metadata[i]["n_words_checked"]
            ))
        
        st.table(
            pd.DataFrame(data, columns=[
                "Summary",
                "Score",
                "Invalid sequences dropped",
                "Words checked"
            ])
        )
        st.subheader("With constraints")
        summaries, beam_metadata = cached_generate_summaries(xsum_test["document"][:2], excluded_dictionary, num_beams)
        data = []
        for i, summary in enumerate(summaries):
            data.append((
                summary,
                np.exp(beam_metadata[i]["score"]),
                len(beam_metadata[i]["dropped_seqs"]),
                beam_metadata[i]["n_words_checked"]
            ))
        
        st.table(
            pd.DataFrame(data, columns=[
                "Summary",
                "Score",
                "Invalid sequences dropped",
                "Words checked"
            ])
        )

        st.subheader("Top dropped sequences")

        data = []
        for i, metadata in enumerate(beam_metadata):
            for dropped_seq in metadata["dropped_seqs"][:5]:
                data.append((
                    i,
                    tokenizer.decode(dropped_seq[0]),
                    tokenizer.decode(dropped_seq[1])
                ))
        
        st.table(
            pd.DataFrame(data, columns=[
                "Summary idx",
                "Summary",
                "Next token"
            ])
        )


if __name__ == "__main__":
    render_beam_search()
