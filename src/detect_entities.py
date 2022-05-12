from typing import List, TypedDict
import spacy
import re


nlp = spacy.load("en_core_web_lg")

MarkedEntity = TypedDict(
    "MarkedEntity",
    {"ent": str, "type": str, "start": int, "end": int, "in_source": bool},
)


def is_entity_contained(entity, text):
    return re.search(re.escape(entity), text, re.IGNORECASE) is not None


def detect_entities(summary: str, source: str) -> List[MarkedEntity]:
    nlp_summary = nlp(summary)

    marked_entities: List[MarkedEntity] = []
    for entity in nlp_summary.ents:
        marked_entities.append(
            {
                "ent": entity.text,
                "type": entity[0].ent_type_,
                "start": entity.start_char,
                "end": entity.end_char,
                "in_source": is_entity_contained(entity.text, source),
            }
        )

    return marked_entities
