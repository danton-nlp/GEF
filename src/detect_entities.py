from typing import List, TypedDict
import spacy
from src.entity_utils import MarkedEntity, is_entity_contained


nlp = spacy.load("en_core_web_lg")


def detect_entities(summary: str, source: str) -> List[MarkedEntity]:
    nlp_summary = nlp(summary)

    marked_entities: List[MarkedEntity] = []
    for entity in nlp_summary.ents:
        # split person entities
        # hard-coding "St" case because it's not split in entfa
        if entity[0].ent_type_ == "PERSON" and len(entity) > 1 and entity[0].text != "St":
            for entity_token in entity:
                # NB: this is handled slightly different 
                # from EntFA, they include 's in the last token
                # but there's only a few cases so I presume we're good
                if entity_token.text != "'s":
                    marked_entities.append(
                        {
                            "ent": entity_token.text,
                            "type": entity_token.ent_type_,
                            "start": entity_token.idx,
                            "end": entity_token.idx + len(entity_token.text),
                            "in_source": is_entity_contained(entity_token.text, source),
                        }
                    )
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
