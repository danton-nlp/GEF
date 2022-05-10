import argparse
from src.data_utils import load_xsum_dict
from src.generation_utils import load_model_and_tokenizer, generate_summaries
import json
from src.beam_validators import BannedPhrases
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
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument(
        "--oracle_data",
        type=str,
        default="data/oracle-experiment/test_annotations.json",
    )
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    xsum_test_by_id = load_xsum_dict("test")
    oracle_annotations = load_annotations(args.oracle_data)

    docs_to_summarize = []
    banned_phrases_by_input_idx = {}
    for j, (xsum_id, annotation) in enumerate(oracle_annotations.items()):
        docs_to_summarize.append(xsum_test_by_id[xsum_id]["document"])
        banned_phrases_by_input_idx[j] = {
            word for word in annotation["non_factual_hallucinations"]
        }

    factuality_enforcer = WordLogitsProcessor(
        tokenizer,
        args.num_beams,
        BannedPhrases(banned_phrases_by_input_idx=banned_phrases_by_input_idx),
    )
    summaries, metadata = generate_summaries(
        model,
        tokenizer,
        docs_to_summarize,
        factuality_enforcer,
        num_beams=args.num_beams,
        return_beam_metadata=True,
    )

    results = {}
    for j, (xsum_id, annotation) in enumerate(oracle_annotations.items()):
        results[xsum_id] = {
            "original_summary": annotation["summary"],
            "corrected_summary": summaries[j],
            "search_metadata": {
                "n_words_checked": metadata[j]["n_words_checked"],
                "dropped_seqs": [
                    tokenizer.decode(dropped_seq[0])
                    for dropped_seq in metadata[j]["dropped_seqs"]
                ],
            },
        }
    output_file = (
        args.oracle_data.replace(".json", "")
        + f"_constrained-{args.num_beams}beams.json"
    )
    write_results(output_file, results)
