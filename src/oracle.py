from src.detect_entities import is_entity_contained
from src.entity_utils import MarkedEntityLookup
from src.entity_factuality import ANNOTATION_LABELS


def get_entity_annotations(sum_ids, metadata):
    annotations = {}
    for sum_id in sum_ids:
        annots = []
        for key in ["xent-train", "xent-test", "our_annotations"]:
            if key in metadata[sum_id]:
                for ents in metadata[sum_id][key].values():
                    annots += ents
        annotations[sum_id] = annots

    return annotations


def oracle_label_entities(
    summary_entities: MarkedEntityLookup,
    annotations: MarkedEntityLookup,
) -> MarkedEntityLookup:
    labeled_entities: MarkedEntityLookup = {}
    for bbc_id, marked_entities in summary_entities.items():
        to_be_labeled = [x.copy() for x in marked_entities]
        for x in to_be_labeled:
            x["label"] = (
                "Unknown"
                if not x["in_source"]
                else ANNOTATION_LABELS["Non-hallucinated"]
            )
        for unlabeled_entity in to_be_labeled:
            for annotated_entity in annotations[bbc_id]:
                if is_entity_contained(
                    unlabeled_entity["ent"], annotated_entity["ent"]
                ):
                    unlabeled_entity["label"] = annotated_entity["label"]
        labeled_entities[bbc_id] = to_be_labeled
    return labeled_entities
