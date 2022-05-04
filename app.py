import streamlit as st
from interface.beam_search import render_beam_search


def render():
    pages = {"Dictionary Constraints (word-level)": render_beam_search}

    st.sidebar.title("Factual Beam Search")
    selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

    pages[selected_page]()


if __name__ == "__main__":
    render()
