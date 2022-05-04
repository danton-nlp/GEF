from datasets import load_dataset
import pytest
from transformers import LogitsProcessorList
from beam_search import WordLogitsProcessor, load_model_and_tokenizer
from beam_validator import DictionaryValidator


@pytest.fixture(scope="session")
def bart_xsum():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("facebook/bart-large-xsum")
    return model, tokenizer

@pytest.fixture(scope="module")
def docs_to_summarize():
    xsum_test = load_dataset("xsum")["test"]
    return xsum_test["document"][0:1]


def generate_summaries(
    model,
    tokenizer,
    docs_to_summarize,
    logits_processor_list,
    num_beams=4
):
    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    model_output = model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        logits_processor=logits_processor_list
    )
    return [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]
    

def test_without_constraints(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum

    summary = generate_summaries(
        model,
        tokenizer,
        docs_to_summarize,
        []
    )[0]

    assert "Wales" in summary

def test_with_constraints(bart_xsum, docs_to_summarize):
    model, tokenizer = bart_xsum
    num_beams = 4

    factuality_enforcer = WordLogitsProcessor(
        tokenizer, 
        num_beams, 
        DictionaryValidator({
            "Edinburgh",
            "Wales",
            "prison",
            "charity",
            "homeless",
            "man",
            "a"
        })
    )

    summary = generate_summaries(
        model,
        tokenizer,
        docs_to_summarize,
        LogitsProcessorList([factuality_enforcer]),
        num_beams
    )[0]

    assert "Wales" not in summary