import json


def load_EntFA(split: str):
    with open(f"./data/EntFA/{split}.json", "r") as f:
        return json.load(f)

def lookup_entfa_by_prediction(data, pred):
    for x in data:
        if x["prediction"] == pred:
            return x