from typing import List
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)
import math
import torch
import torch.nn as nn
from transformers import BartTokenizer
import torch.nn.functional as F

def prior_causal_probability(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target: str
):
  with torch.no_grad():
    target_tokens = tokenizer(
        target, 
        return_tensors='pt'
    ).input_ids

    logits = model(target_tokens).logits
    probs = logits.squeeze(0).softmax(dim=1)

    preds = []              
    for step_probs, target_token in zip(probs, target_tokens[0]):
        preds.append((
            tokenizer.decode(target_token),
            target_token.item(),
            step_probs[target_token].item()
        ))

  return preds


def forceful_conditional_generation(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    target_str: str,
    docs_to_summarize: List[str]
):
    target_tokens = list(tokenizer.encode(target_str))[1:]
    def prefix_allowed_tokens_fn(input_idx, input_ids):
        current_step = len(input_ids) - 1
        return [target_tokens[current_step]]


    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    model_output = model.generate(
        inputs.input_ids,
        num_beams=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
    )

    generated_summaries = [
        tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for idx, ids in enumerate(model_output.sequences)
    ]

    preds = []
    
    for token_scores, target_token in zip(model_output.scores, target_tokens):
        probs = F.softmax(token_scores, dim=1)
        preds.append((
            tokenizer.decode(target_token),
            target_token,
            probs[:, target_token]
        ))



    return preds, generated_summaries

if __name__ == "__main__":
  from transformers import (
      AutoModelForCausalLM,
      AutoTokenizer,
      AutoModel
  )

  bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
  causal_prior_model = AutoModelForCausalLM.from_pretrained("facebook/bart-large")
  bart_model = AutoModel.from_pretrained('facebook/bart-large')

  prior_causal_probability(
      causal_prior_model,
      bart_tokenizer,
      "Sydney has marked the first anniversary of the siege at the Waverley cafe in which two women were killed by a gunman in the Australian city."
  )

