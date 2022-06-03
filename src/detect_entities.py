from typing import List
import spacy
from spacy.tokens import Span
from src.entity_utils import MarkedEntity, is_entity_contained

nlp = spacy.load("en_core_web_lg")


def split_person_entity(entity: Span, source: str) -> List[MarkedEntity]:
    split_entity: List[MarkedEntity] = []
    ent_parts: List[str] = entity.text.split(" ")
    start_idx = entity[0].idx
    for sub_ent in ent_parts:
        split_entity.append(
            {
                "ent": sub_ent,
                "type": "PERSON",
                "start": start_idx,
                "end": start_idx + len(sub_ent),
                "in_source": is_entity_contained(sub_ent, source),
            }
        )
        start_idx += len(sub_ent) + 1
    return split_entity


def detect_entities(summary: str, source: str) -> List[MarkedEntity]:
    nlp_summary = nlp(summary)

    marked_entities: List[MarkedEntity] = []
    for entity in nlp_summary.ents:
        # split person entities
        if (
            entity[0].ent_type_ == "PERSON"
            and len(entity) > 1
            and entity[0].text != "St"
        ):
            marked_entities += split_person_entity(entity, source)
        else:
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
