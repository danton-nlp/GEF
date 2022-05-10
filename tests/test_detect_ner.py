from src.detect_ner import detect_ner


def test_ner_detection():
    summary = "A search is under way for the remains of a Pembrokeshire village which was wiped out by storms in the 1970s."
    source = "Pembrokeshire"
    entities = detect_ner(summary, source)

    assert len(entities) == 2

    assert entities[0]["text"] == "Pembrokeshire"
    assert entities[0]["type"] == "GPE"
    assert entities[0]["in_source"]
    assert summary[entities[0]["start"] : entities[0]["end"]] == entities[0]["text"]

    assert entities[1]["text"] == "the 1970s"
    assert entities[1]["type"] == "DATE"
    assert not entities[1]["in_source"]
    assert summary[entities[1]["start"] : entities[1]["end"]] == entities[1]["text"]
