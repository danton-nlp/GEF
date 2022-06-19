from typing import List, Literal
from src.detect_entities import is_entity_contained
from src.entity_utils import MarkedEntity, MarkedEntityLookup
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


EntityMatchType = Literal[
    "contained", "strict_all", "strict_intrinsic", "strict_extrinsic"
]


def is_entity_match(
    entity: MarkedEntity, annotation: MarkedEntity, match_type: EntityMatchType
) -> bool:
    entity_contained = is_entity_contained(entity["ent"], annotation["ent"])
    if (
        match_type == "strict_all"
        or (
            annotation["label"]
            in [ANNOTATION_LABELS["Intrinsic"], ANNOTATION_LABELS["Non-hallucinated"]]
        )
        or (
            match_type == "strict_extrinsic"
            and annotation["label"]
            in [ANNOTATION_LABELS["Non-factual"], ANNOTATION_LABELS["Factual"]]
        )
    ):
        entity_span_match = (
            entity["start"] == annotation["start"]
            and entity["end"] == annotation["end"]
            and entity["ent"] == annotation["ent"]
        )
        return entity_span_match and entity_contained
    else:
        return entity_contained


def oracle_label_entities(
    summary_entities: MarkedEntityLookup,
    annotations: MarkedEntityLookup,
    entity_match_type: EntityMatchType = "contained",
) -> MarkedEntityLookup:
    labeled_entities: MarkedEntityLookup = {}
    for bbc_id, marked_entities in summary_entities.items():
        to_be_labeled = [x.copy() for x in marked_entities]
        for x in to_be_labeled:
            x["label"] = (
                "Unknown"
                if not x["in_source"]
                or entity_match_type in ["strict_intrinsic", "strict_all"]
                else ANNOTATION_LABELS["Non-hallucinated"]
            )
        for unlabeled_entity in to_be_labeled:
            for annotated_entity in annotations[bbc_id]:
                if is_entity_match(
                    unlabeled_entity, annotated_entity, entity_match_type
                ):
                    unlabeled_entity["label"] = annotated_entity["label"]
        labeled_entities[bbc_id] = to_be_labeled
    return labeled_entities
