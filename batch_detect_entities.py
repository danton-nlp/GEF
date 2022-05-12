from sumtool.storage import get_summaries, store_summary_metrics
import argparse
from src.data_utils import load_xsum_dict
from src.detect_entities import detect_entities
from tqdm import tqdm


SUMTOOL_DATASET = "xsum"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    args = parser.parse_args()

    summaries = get_summaries(SUMTOOL_DATASET, args.model)
    xsum_test = load_xsum_dict("test")

    summary_metadata = {}

    for sum_id, sum_data in tqdm(list(summaries.items())):
        summary_metadata[sum_id] = {"entities": detect_entities(
            sum_data["summary"], 
            xsum_test[sum_id]["document"]
        )}

    store_summary_metrics(SUMTOOL_DATASET, args.model, summary_metadata)
