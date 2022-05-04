from datasets import load_dataset
import pytest
from src.word_logits_processor import WordLogitsProcessor
from src.beam_validators import DictionaryValidator
from src.generation_utils import generate_summaries, load_model_and_tokenizer


@pytest.fixture(scope="session")
def bart_xsum():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("facebook/bart-large-xsum")
    return model, tokenizer


@pytest.fixture(scope="module")
def docs_to_summarize():
    xsum_test = load_dataset("xsum")["test"]
    return xsum_test["document"][0:1]


def test_without_constraints(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum

    summary = generate_summaries(model, tokenizer, docs_to_summarize, None)[0]

    assert "Wales" in summary


def test_with_constraints(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        DictionaryValidator(
            {"Edinburgh", "Wales", "prison", "charity", "homeless", "man", "a"}
        ),
    )

    summary = generate_summaries(
        model, tokenizer, docs_to_summarize, factuality_enforcer, num_beams
    )[0]

    assert "Wales" not in summary
