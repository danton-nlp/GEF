import argparse
from typing import Tuple
from tqdm import tqdm
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from sumtool.storage import store_model_summaries


def load_pegasus_model_and_tokenizer(device) -> Tuple[PegasusForConditionalGeneration, PegasusTokenizer]:
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
    model.eval()

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict XSum summaries from Pegasus. Writes to data/xsum."
    )

    parser.add_argument(
        "--data_split",
        type=str,
        required=True,
        choices=["train", "test", "validation"],
        default='test',
        help="xsum data split generate summaires for",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="inference batch size"
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xsum_data = load_dataset("xsum", split=args.data_split)
    model, tokenizer = load_pegasus_model_and_tokenizer(device)

    data_loader = DataLoader(xsum_data, batch_size=args.batch_size)

    generated_summary_ids = []
    generated_summaries = []

    for batch_idx, batch_data in enumerate(tqdm(data_loader)):
        batch_inputs = tokenizer(
            batch_data['document'],
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)

        batch_predicted_summaries = tokenizer.batch_decode(
            model.generate(batch_inputs.input_ids)
        )

        generated_summary_ids.extend(batch_data['id'])
        generated_summaries.extend(batch_predicted_summaries)

    xsumidx_to_generated_summary = dict(zip(generated_summary_ids, generated_summaries))

    store_model_summaries(
        "xsum",
        model.config.name_or_path,
        model.config.to_dict(),
        xsumidx_to_generated_summary
    )
