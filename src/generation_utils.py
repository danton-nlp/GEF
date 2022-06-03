import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList


def load_prior_model_and_tokenizer():
    return load_model_and_tokenizer("facebook/bart-large")


def load_model_and_tokenizer(
    path: str, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    return (
        AutoModelForSeq2SeqLM.from_pretrained(path).to(device),
        AutoTokenizer.from_pretrained(path),
    )


def load_bart_xsum_cmlm(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model = AutoModelForSeq2SeqLM.from_pretrained("model-checkpoints/entfa-cmlm").to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-xsum", mask_token="###"
    )

    return model, tokenizer
