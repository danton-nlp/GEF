from src.detect_entities import detect_entities


def test_ner_detection():
    summary = "A search is under way for the remains of a Pembrokeshire village which was wiped out by storms in the 1970s."
    source = "Pembrokeshire"
    entities = detect_entities(summary, source)

    assert len(entities) == 2

    assert entities[0]["ent"] == "Pembrokeshire"
    assert entities[0]["type"] == "GPE"
    assert entities[0]["in_source"]
    assert summary[entities[0]["start"] : entities[0]["end"]] == entities[0]["ent"]

    assert entities[1]["ent"] == "the 1970s"
    assert entities[1]["type"] == "DATE"
    assert not entities[1]["in_source"]
    assert summary[entities[1]["start"] : entities[1]["end"]] == entities[1]["ent"]


def test_split_person():
    summary = "A search is under way for Daniel Levenson"
    source = "Pembrokeshire"
    entities = detect_entities(summary, source)

    assert len(entities) == 2

    assert entities[0]["ent"] == "Daniel"
    assert entities[0]["type"] == "PERSON"
    assert not entities[0]["in_source"]
    assert summary[entities[0]["start"] : entities[0]["end"]] == entities[0]["ent"]

    assert entities[1]["ent"] == "Levenson"
    assert entities[1]["type"] == "PERSON"
    assert not entities[1]["in_source"]
    assert summary[entities[1]["start"] : entities[1]["end"]] == entities[1]["ent"]
