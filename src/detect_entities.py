from typing import List, TypedDict
import spacy
import re


nlp = spacy.load("en_core_web_lg")

MarkedEntity = TypedDict(
    "MarkedEntity",
    {"text": str, "type": str, "start": int, "end": int, "in_source": bool},
)


def source_contains_entity(source: str, entity: str) -> bool:
    return re.search(re.escape(entity), source, re.IGNORECASE) is not None


def detect_entities(summary: str, source: str) -> List[MarkedEntity]:
    nlp_summary = nlp(summary)

    marked_entities: List[MarkedEntity] = []
    for entity in nlp_summary.ents:
        marked_entities.append(
            {
                "text": entity.text,
                "type": entity[0].ent_type_,
                "start": entity.start_char,
                "end": entity.end_char,
                "in_source": source_contains_entity(source, entity.text),
            }
        )

    return marked_entities
