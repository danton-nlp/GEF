from src.data_utils import get_gold_xsum_data, load_shuffled_test_split, load_xsum_dict
import json


if __name__ == "__main__":
    xsum_test = load_xsum_dict("test")
    gold_summaries, gold_metadata = get_gold_xsum_data()
    bart_test_ids = load_shuffled_test_split(xsum_test, "bart-test-extrinsic", 200)
    pegasus_test_ids = load_shuffled_test_split(
        xsum_test, "pegasus-test-extrinsic", 100
    )

    n_ents = 0
    n_sums = 0

    bart_test = []
    pegasus_test = []
    train = []

    for sum_id, metadata in gold_metadata.items():
        if "our_annotations" in metadata:
            for gen_summary, entities in metadata["our_annotations"].items():
                sum_object = {
                    "source": xsum_test[sum_id]["document"],
                    "reference": gold_summaries[sum_id]["summary"],
                    "prediction": gen_summary,
                    "entities": entities,
                }
                n_sums += 1
                n_ents += len(entities)

                if sum_id in bart_test_ids:
                    bart_test.append(sum_object)
                if sum_id in pegasus_test_ids:
                    pegasus_test.append(sum_object)
                if sum_id not in pegasus_test_ids and sum_id not in bart_test_ids:
                    train.append(sum_object)

    with open("./data/xent-extended/train.json", "w") as f:
        json.dump(train, f, indent=2)

    with open("./data/xent-extended/test_bart.json", "w") as f:
        json.dump(bart_test, f, indent=2)

    with open("./data/xent-extended/test_pegasus.json", "w") as f:
        json.dump(pegasus_test, f, indent=2)

    print(f"Persisted {n_ents} entity annotations for {n_sums} summaries")
    print(
        f"Train: {len(train)}, test_bart: {len(bart_test)}, test_pegasus: {len(pegasus_test)}"
    )
