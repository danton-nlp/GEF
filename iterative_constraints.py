from collections import defaultdict
import argparse
from src.data_utils import load_xsum_dict
from src.generation_utils import generate_summaries, load_model_and_tokenizer
from src.beam_validators import BannedPhrases
from src.word_logits_processor import WordLogitsProcessor
from src.detect_entities import detect_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/bart-large-xsum")
    parser.add_argument("--max_iterations", type=int, default=5)
    # parser.add_argument("--sumtool_path", type=str, default="")
    # parser.add_argument("--data_subset", type=int, default=0)
    args = parser.parse_args()
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    xsum_test = load_xsum_dict("test")
    num_beams = 4

    # 1) Select subset of data to work on
    docs_to_summarize = [x["document"] for x in [
        xsum_test["36396523"],
        xsum_test["39368095"],
    ]]

    # initialize with no constraints
    banned_phrases_by_input_idx = defaultdict(lambda: set)

    # ...until convergence / max iterations
    n_iterations = 0
    while n_iterations < args.max_iterations:
        # ) Generate
        factuality_enforcer = WordLogitsProcessor(
            tokenizer,
            num_beams,
            BannedPhrases(
                banned_phrases_by_input_idx=dict(banned_phrases_by_input_idx)
            ),
        )
        summaries, metadata = generate_summaries(
            model,
            tokenizer,
            docs_to_summarize,
            factuality_enforcer,
            num_beams=num_beams,
            return_beam_metadata=True,
        )

        # 3) Detect NER
        summary_entities = []
        for summary, source in zip(summaries, docs_to_summarize):
            summary_entities.append(detect_entities(summary, source))

        print(
            summaries,
            summary_entities
        )
        # 4) See if NER is factual according to model (oracle / classification)
        labeled_summary_entities = label_entities(model, summary_entities)

        # 5) Add constraints
        # break if no new constriants

        n_iterations += 1

        # 6) Save results
        pass

    # Persist results
    pass
    # Evaluate?
    pass

    print(summaries)
