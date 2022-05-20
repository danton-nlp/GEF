import streamlit as st
from interface.beam_search import render_beam_search
from interface.explore_logs import render_explore_logs
from interface.compare_results import render_compare_results


def render():
    pages = {
        "Compare Results": render_compare_results,
        "Explore Logs": render_explore_logs,
        "Phrase-level Beam Search Constraints": render_beam_search
    }

    st.sidebar.title("Factual Beam Search")
    selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

    pages[selected_page]()


if __name__ == "__main__":
    render()
