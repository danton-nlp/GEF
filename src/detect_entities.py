from typing import List, TypedDict
import spacy
from src.entity_utils import MarkedEntity, is_entity_contained


nlp = spacy.load("en_core_web_lg")


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
