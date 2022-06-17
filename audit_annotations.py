from src.data_utils import get_gold_xsum_data, load_xsum_dict, load_shuffled_test_split
import pprint
import json

if __name__ == "__main__":
    _, gold_metadata = get_gold_xsum_data()
    test_summaries = load_xsum_dict("test")

    out_of_source_possible_bad_annotations = set()
    in_source_possible_bad_annotations = set()
    count_summaries_annotated = 0

    for sum_id, sum_metadata in gold_metadata.items():
        if "our_annotations" not in sum_metadata.keys():
            continue

        our_annotations = sum_metadata["our_annotations"].items()
        for generated_summary, annotation_sets in our_annotations:
            count_summaries_annotated += 1
            for annotation in annotation_sets:
                if annotation["in_source"]:
                    if annotation["label"] not in [
                        "Non-hallucinated",
                        "Intrinsic Hallucination",
                    ]:
                        in_source_possible_bad_annotations.add(
                            (sum_id, generated_summary, json.dumps(annotation))
                        )
                else:
                    if annotation["label"] not in [
                        "Non-factual Hallucination",
                        "Factual Hallucination",
                    ]:
                        out_of_source_possible_bad_annotations.add(
                            (sum_id, generated_summary, json.dumps(annotation))
                        )

    print('count of out-of-source possible bad annotations:', len(out_of_source_possible_bad_annotations))
    print('count of in-source possible bad annotations:', len(in_source_possible_bad_annotations))
    print('count of total summaries annotated:', count_summaries_annotated)

    print('---')
    print('IN SOURCE POSSIBLE BAD ANNOTATIONS')
    pprint.pprint(in_source_possible_bad_annotations, indent=2)

    print('---')
    print('OUT OF SOURCE POSSIBLE BAD ANNOTATIONS')
    pprint.pprint(out_of_source_possible_bad_annotations, indent=2)
