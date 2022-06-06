from datasets import load_dataset
import pytest
from src.word_logits_processor import WordLogitsProcessor
from src.beam_validators import BannedPhrases
from src.generation_utils import generate_summaries, load_model_and_tokenizer


@pytest.fixture(scope="session")
def bart_xsum():
    print("Loading BART model...")
    model, tokenizer = load_model_and_tokenizer("facebook/bart-large-xsum")
    return model, tokenizer


@pytest.fixture(scope="session")
def pegasus_xsum():
    print("Loading Pegasus model...")
    model, tokenizer = load_model_and_tokenizer("google/pegasus-xsum")
    return model, tokenizer


@pytest.fixture(scope="module")
def docs_to_summarize():
    xsum_test = load_dataset("xsum")["test"]
    return xsum_test["document"][0:1]


def test_no_constraints(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    summary = generate_summaries(model, tokenizer, docs_to_summarize, None, num_beams)[
        0
    ]

    assert "Wales" in summary
    assert "former prison" in summary


def test_banned_word(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases(
            {
                "Wales",
            }
        ),
    )

    summaries, metadata = generate_summaries(
        model, tokenizer, docs_to_summarize, factuality_enforcer, num_beams,
        return_beam_metadata=True
    )
    dropped_seq = tokenizer.decode(metadata[0]["dropped_seqs"][0][0])

    assert "Wales" not in summaries[0]
    assert dropped_seq.split(" ")[-1] == "Wales"
    


def test_banned_phrase(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases({"former prison"}),
    )

    summary = generate_summaries(
        model, tokenizer, docs_to_summarize, factuality_enforcer, num_beams
    )[0]

    assert "former prison" not in summary


def test_failed_generation_one_beam(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 1

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases({"prison"}),
    )

    summary = generate_summaries(
        model, tokenizer, docs_to_summarize, factuality_enforcer, num_beams
    )[0]

    assert summary == "<Failed generation: blocked all beams>"
    assert 0 in factuality_enforcer.failed_sequences


def test_failed_generation_multiple_beams(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases(
            {"Wales", "prison", "accommodation", "charity", "housing", "former", "more"}
        ),
    )

    summary = generate_summaries(
        model, tokenizer, docs_to_summarize, factuality_enforcer, num_beams
    )[0]

    assert summary == "<Failed generation: blocked all beams>"
    assert 0 in factuality_enforcer.failed_sequences


def test_pegasus_no_constraints(pegasus_xsum, docs_to_summarize):
    model, tokenizer = pegasus_xsum
    num_beams = 4

    summary = generate_summaries(model, tokenizer, docs_to_summarize, None, num_beams)[
        0
    ]

    assert "More" in summary
    assert "homeless people" in summary


def test_pegasus_word(pegasus_xsum, docs_to_summarize):
    model, tokenizer = pegasus_xsum
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases({"More"}),
    )

    summary = generate_summaries(
        model, tokenizer, docs_to_summarize, factuality_enforcer, num_beams
    )[0]

    assert "More" not in summary


def test_pegasus_phrase(pegasus_xsum, docs_to_summarize):
    model, tokenizer = pegasus_xsum
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        BannedPhrases({"homeless people"}),
    )

    summary = generate_summaries(
        model, tokenizer, docs_to_summarize, factuality_enforcer, num_beams
    )[0]

    assert "homeless people" not in summary
