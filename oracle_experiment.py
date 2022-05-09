from datasets import load_dataset
import argparse
from src.generation_utils import load_model_and_tokenizer, generate_summaries
import json
from src.beam_validators import BannedWords
from src.word_logits_processor import WordLogitsProcessor


def load_annotations(fname):
    with open(fname, "r") as f:
        return json.load(f)

def write_results(fname, results):
    with open(fname, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    parser.add_argument(
        "--oracle_data", 
        type=str, 
        default="data/bart-large-xsum/test_annotations.json"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data/bart-large-xsum/test_annotations_constrained.json"
    )
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    xsum_test = load_dataset("xsum")["test"]
    xsum_test_by_id = {doc["id"]: doc for doc in xsum_test}
    oracle_annotations = load_annotations(args.oracle_data)

    docs_to_summarize = []
    banned_words_by_input_idx = {}
    for j, (xsum_id, annotation) in enumerate(oracle_annotations.items()):
        docs_to_summarize.append(
            xsum_test_by_id[xsum_id]["document"]
        )
        banned_words_by_input_idx[j] = {
            word for word in annotation["non_factual_hallucinations"]
        }

    num_beams = 4
    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        num_beams,
        BannedWords(banned_words_by_input_idx=banned_words_by_input_idx),
    )
    summaries, metadata = generate_summaries(
        model, 
        tokenizer, 
        docs_to_summarize, 
        factuality_enforcer,
        num_beams=num_beams,
        return_beam_metadata=True
    )

    results = {}
    for j, (xsum_id, annotation) in enumerate(oracle_annotations.items()):
        results[xsum_id] = {
            "original_summary": annotation["summary"],
            "corrected_summary": summaries[j],
            "search_metadata": {
                "n_words_checked": metadata[j]["n_words_checked"],
                "dropped_seqs": [
                    tokenizer.decode(dropped_seq[0]) for dropped_seq in metadata[j]["dropped_seqs"]
                ]
            } 
        }

    write_results(
        args.output_file,
        results
    )