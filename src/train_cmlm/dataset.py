# Dataset with random entities masked


from datasets import Dataset, load_dataset
import torch
from transformers import AutoTokenizer
from src.detect_entities import detect_entities
from src.entity_utils import mask_entity


def tokenize_and_mask(tokenizer):
    def inner_fn(batch_examples):
        exploded_docs = []
        exploded_sums = []
        exploded_ids = []
        inputs_masked = []
        input_ids = []
        for example in batch_examples:
            summary = example["summary"]
            source_doc = example["document"]
            sum_entities = detect_entities(summary, source_doc)
            for entity in sum_entities:
                input_masked = f"<s>{mask_entity(summary, entity, mask_token=tokenizer.mask_token)}</s>{source_doc}"
                tokenized = tokenizer.encode(
                    input_masked,
                    add_special_tokens=False,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                # need to recreate these because we expand
                # the number of examples
                exploded_docs.append(example["id"])
                exploded_sums.append(summary)
                exploded_ids.append(summary)
                inputs_masked.append(input_masked)
                input_ids.append(tokenized)

        return {
            "input_ids": input_ids,
            "inputs_masked": inputs_masked,
            "document": exploded_docs,
            "summary": exploded_sums,
            "id": exploded_ids
        }

    return inner_fn


def create_masked_dataset(xsum_train: Dataset, tokenizer) -> Dataset:
    dataset_masked = xsum_train.select(range(100)).map(
        tokenize_and_mask(tokenizer), batched=True, remove_columns=[]
    )
    dataset_masked.set_format(type="torch", columns=["input_ids"])
    return dataset_masked


if __name__ == "__main__":
    xsum_train = load_dataset("xsum")["validation"]
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", mask_token="###")
    create_masked_dataset(xsum_train, tokenizer)
