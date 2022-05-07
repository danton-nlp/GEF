from typing import Union, List
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)
import math
import torch
import torch.nn as nn
from transformers import BartTokenizer
import torch.nn.functional as F

class ConditionalSequenceGenerator:
    """Conditional sequence generator for calculating prior and posterior probability."""
    def __init__(self, bart):
        self.bart = bart
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large", local_files_only=False)
        
        self.encode_func = bart.encode
        self.decode_func = bart.decode
        self.max_positions = bart.max_positions
        if type(self.max_positions) == int:
            self.max_positions = [self.max_positions]
        self.encode_line = bart.task.source_dictionary.encode_line
        
        self._initialize()
    
    def _initialize(self):
        """Set BART model to evaluation mode."""
        self.bart.cuda()
        self.bart.eval()
        self.bart.half()
        
    def tokenize_target(self, input_str, left_pad=False, append_eos=False):
        """BPE-encode a sentence (or multiple sentences).
        Args:
            input_str (str or List[str]): input sentence to be tokenized.
            left_pad (bool): self-explained.
        Return:
            prev_output_tokens (torch.Tensor): [batch_size, length]
            target (torch.Tensor): [batch_size, length]
            tgt_lengths (torch.Tensor): [batch_size]
            
        """
        if type(input_str) == type(''):
            input_str = [input_str]

        prev_ids, tgt_ids = [], []

        for ins in input_str:
            tokens = self.bart.bpe.encode(ins)  # <mask>: 1279 27932 29
            calibration = 1
            if len(tokens.split(" ")) > min(self.max_positions) - calibration:
                tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - calibration])
            
            if append_eos:
                tokens = "<s> " + tokens
            prev_tokens = "</s> " + tokens
            tgt_tokens = tokens + " </s>"
            
            tgt_ids.append(self.encode_line(tgt_tokens, append_eos=False).long())
            prev_ids.append(self.encode_line(prev_tokens, append_eos=False).long())

        prev_output_tokens = collate_tokens(prev_ids, pad_idx=1, left_pad=left_pad).cuda()
        target = collate_tokens(tgt_ids, pad_idx=1, left_pad=left_pad).cuda()
        tgt_lengths = torch.sum(target != 1, dim=1).cuda()

        return prev_output_tokens, target, tgt_lengths
        
    def tokenize(self, input_str, append_bos=False, append_eos=True, left_pad=True):
        """BPE-encode a sentence (or multiple sentences).
        Args:
            input_str (str or List[str]): input sentence to be tokenized.
            append_bos (bool): self-explained.
            append_eos (bool): self-explained.
        Return:
            input_ids (torch.Tensor): [batch_size, length]
            src_lengths (torch.Tensor): [batch_size]
        """
        if type(input_str) == type(''):
            input_str = [input_str]

        input_ids = []
        for ins in input_str:
            tokens = self.bart.bpe.encode(ins)  # <mask>: 1279 27932 29
            calibration = sum([append_bos, append_eos])
            if len(tokens.split(" ")) > min(self.max_positions) - calibration:
                tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - calibration])

            tokens = "<s> " + tokens if append_bos else tokens
            tokens = tokens + " </s>" if append_eos else tokens
            ids = self.encode_line(tokens, append_eos=False).long()
            input_ids.append(ids)

        input_ids = collate_tokens(input_ids, pad_idx=1, left_pad=left_pad).cuda()
        input_lengths = torch.sum(input_ids != 1, dim=1).cuda()

        return input_ids, input_lengths
    
    def tokenize_with_mask(self, input_str):
        """Tokenize sentence with a special <mask> token in it.
        Args:
            input_str (str or List[str]): input sentence to be tokenized.
        Return:
            input_ids (torch.Tensor): [batch_size, length]
            src_lengths (torch.Tensor): [batch_size]
        """
        input_ids = self.tokenizer(input_str, return_tensors='pt', padding=True)['input_ids'].cuda()
        input_lengths = torch.sum(input_ids != 1, dim=1).cuda()
        return input_ids, input_lengths
    
    def encode_decode(self, src_input, tgt_input, mask_filling=False):
        """
        Args:
            src_input: (List[str])
            tgt_input: (List[str])
            
        """
        if mask_filling:
            src_tokens, src_lengths = self.tokenize_with_mask(src_input)
            prev_output_tokens, target, tgt_lengths = self.tokenize_target(tgt_input, left_pad=False, append_eos=True)
        else:
            src_tokens, src_lengths = self.tokenize(src_input, append_bos=False)
            prev_output_tokens, target, tgt_lengths = self.tokenize_target(tgt_input, left_pad=False)
        
        with torch.no_grad():
            encoder_out = self.bart.model.encoder(src_tokens, src_lengths=src_lengths)
            decoder_out = self.bart.model.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=False)

            probs = nn.functional.softmax(decoder_out[0], dim=-1)
            tgt_token_probs = torch.gather(probs, 2, target.unsqueeze(-1)).squeeze(2)

            # mask <pad> with probability 1.0
            max_tgt_length = tgt_lengths.max().item()
            tgt_lengths = tgt_lengths - 1
            tgt_mask = torch.arange(max_tgt_length)[None, :].cuda() < tgt_lengths[:, None]
            tgt_token_probs.masked_fill_(tgt_mask == False, 1.0)

        return tgt_token_probs, target
    
    def generate(self, src_input, tgt_input=None):
        """Conditional generation.
        
        Args:
            src_input (str or List[str]): input source sentence to be tokenized.
            tgt_input (str or List[str]): input target sentence to be tokenized.
        """
        input_ids, lengths = self.tokenize(src_input, append_bos=False) 
        
        target_ids = None
        if tgt_input is not None:
            assert len(src_input) == len(tgt_input), "source & target length should match."
            target_ids, _ = self.tokenize(tgt_input, append_bos=False, left_pad=False)
        
        with torch.no_grad():
            encoder_output = self.encode_sequence(input_ids, lengths)
            decoder_output = self.decode_sequence(encoder_output, 
                                                  target_ids=target_ids,
                                                  prefix_tokens=[2])
        return decoder_output
    
    def mask_filling(self, src_input, tgt_input=None):
        """
        Filling the mask in sentence(s).
        """
        input_ids, lengths = self.tokenize_with_mask(src_input)
        
        target_ids = None
        if tgt_input is not None:
            assert len(src_input) == len(tgt_input), "source & target length should match."
            target_ids, _ = self.tokenize(tgt_input, left_pad=False)

        with torch.no_grad():
            encoder_output = self.encode_sequence(input_ids, lengths)
            decoder_output = self.decode_sequence(encoder_output, 
                                                  target_ids=target_ids,
                                                  prefix_tokens=[2, 0])
        return decoder_output
    
    def encode_sequence(self, input_ids, lengths):
        return self.bart.model.encoder(input_ids, src_lengths=lengths)
        
    def decode_sequence(
        self,
        encoder_out,
        target_ids=None,
        min_decode_step=3,
        max_decode_step=100,
        pad_id=1,
        eos_id=2,
        prefix_tokens=[2, 0],
    ):
        batch_size = encoder_out['encoder_padding_mask'][0].shape[0]
        init_input = torch.tensor([prefix_tokens] * batch_size, dtype=torch.long).cuda()
        token_probs, tokens = None, [[] for i in range(batch_size)]
        end_mask = torch.tensor([False] * batch_size).cuda()

        softmax = nn.Softmax(dim=1)
        for step in range(max_decode_step):
            decoder_outputs = self.bart.model.decoder(init_input, encoder_out, features_only=False)
            logits = decoder_outputs[0][:, -1, :]  # logits: [batch_size, vocab]
            attn = decoder_outputs[1]['attn'][0]  # [batch_size, prev_token_len, src_token_len]

            if step + 1 < min_decode_step:
                logits[:, eos_id] = -math.inf  # mask <EOS> token when within minimal step
            logits[:, pad_id], logits[:, 0] = -math.inf, -math.inf  # never select <PAD> & <BOS> token
            probs = softmax(logits)  # probs: [batch_size, vocab]

            # select tokens
            if target_ids is not None:
                selected_token = target_ids[:, step]
            else:
                value, indices = torch.topk(probs, 5, dim=1)
                selected_token = indices[:, 0]

            selected_token = selected_token.masked_fill(end_mask, pad_id)
            init_input = torch.cat([init_input, selected_token.unsqueeze(1)], dim=-1)
            
            probs = torch.gather(probs, 1, selected_token.unsqueeze(1)).detach()
            probs = probs.masked_fill(end_mask.unsqueeze(1), 1.0)
            
            # str & probability
            token_probs = probs if token_probs is None else torch.cat([token_probs, probs], dim=-1)
            for t, s in zip(tokens, selected_token):
                t.append(self.decode_func(s.unsqueeze(0)) if s.item() != pad_id else '<pad>')
            
            # stop generation when all finished
            end_mask = torch.logical_or(end_mask, selected_token == eos_id) 
            if end_mask.sum().item() == batch_size:
                break

        return init_input, tokens, token_probs


def prior_masked_probability(
    model: Union[AutoModelForCausalLM, AutoModelForMaskedLM],
    tokenizer: AutoTokenizer,
    masked_input: str,
    mask_target: str
):
  with torch.no_grad():
    masked_input_tokenized = tokenizer(
        masked_input, 
        return_tensors='pt'
    )['input_ids']
  
    # remove end of sentence special token...seems to lead to higher probs.
    masked_input_tokenized = masked_input_tokenized[0, 0:-1].unsqueeze(0)

    logits = model(masked_input_tokenized)['logits']
    masked_prob_distrib = logits[0, -1].softmax(dim=0)
                      
    probs = []
    true_tokens_idx = tokenizer.encode(f'{mask_target}', add_special_tokens=False)

    # can prob remove this clone
    running_tokens = masked_input_tokenized.clone()
    
    for token_idx in true_tokens_idx:
      probs.append(masked_prob_distrib[token_idx])
    
      running_tokens = torch.hstack((running_tokens, torch.Tensor([[token_idx]])))
      running_tokens = running_tokens.type('torch.IntTensor')
    
      logits = model(running_tokens)['logits']
      masked_prob_distrib = logits[0, -1].softmax(dim=0)
  
  probs = torch.Tensor(probs)
  joint_prob = probs.prod(dim=0)

  return probs, joint_prob


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

  prior_masked_probability(
      causal_prior_model,
      bart_tokenizer,
      masked_input="Sydney has marked the first anniversary of the siege at the",
      mask_target=" Waverley"
  )

