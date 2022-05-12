from typing import Callable, Dict, List, TypedDict
import re


MarkedEntity = TypedDict(
    "MarkedEntity",
    {"ent": str, "type": str, "start": int, "end": int, "in_source": bool},
)

LabeledEntity = TypedDict(
    "LabeledEntity",
    {
        "ent": str,
        "type": str,
        "start": int,
        "end": int,
        "in_source": bool,
        "label": str,
    },
)
MarkedEntityLookup = Dict[str, List[MarkedEntity]]
LabeledEntityLookup = Dict[str, List[LabeledEntity]]


def count_entities(entity_lookup: LabeledEntityLookup):
    return sum([len(x) for x in entity_lookup.values()])


def filter_entities(
    predicate_fn: Callable[[LabeledEntity], bool], entity_lookup: LabeledEntityLookup
) -> LabeledEntityLookup:
    return {
        sum_id: [x for x in labeled_entities if predicate_fn(x)]
        for sum_id, labeled_entities in entity_lookup.items()
    }



def is_entity_contained(entity, text):
    return re.search(re.escape(entity), text, re.IGNORECASE) is not None
