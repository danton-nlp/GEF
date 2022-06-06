from typing import Callable, Dict, List, TypedDict, Union
import re

from regex import D


MarkedEntity = TypedDict(
    "MarkedEntity",
    {
        "ent": str,
        "type": str,
        "start": int,
        "end": int,
        "in_source": bool,
        "label": Union[str, None],
        "predicted_label": Union[str, None],
    },
)

MarkedEntityLookup = Dict[str, List[MarkedEntity]]


def count_entities(entity_lookup: MarkedEntityLookup):
    return sum([len(x) for x in entity_lookup.values()])


def filter_entities(
    predicate_fn: Callable[[MarkedEntity], bool], entity_lookup: MarkedEntityLookup
) -> MarkedEntityLookup:
    return {
        sum_id: [x for x in labeled_entities if predicate_fn(x)]
        for sum_id, labeled_entities in entity_lookup.items()
    }


def is_entity_contained(entity, text):
    if entity.endswith("'s"):
        entity = entity.replace("'s", "")
    return re.search(re.escape(entity), text, re.IGNORECASE) is not None


def mask_entity(summary: str, entity: MarkedEntity, mask_token="<mask>"):
    return summary[0 : entity["start"]] + mask_token + summary[entity["end"] :]
