from typing import List
from src.entity_utils import (
    MarkedEntity,
    MarkedEntityLookup,
)
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle


ANNOTATION_LABELS = {
    "Non-factual": "Non-factual Hallucination",
    "Factual": "Factual Hallucination",
    "Non-hallucinated": "Non-hallucinated",
    "Unknown": "Unknown",
    "Intrinsic": "Intrinsic Hallucination"
}

PickledClassifier = KNeighborsClassifier


class EntityFactualityClassifier:
    """
    Wrapper for sklearn entity factuality classifier.

    Responsible for loading a pickled model, extracting features,
    classifying & returning label.
    """

    def __init__(self, pickled_model_path):
        with open(pickled_model_path, "rb") as f:
            self.clf: PickledClassifier = pickle.load(f)
        self.label_mapping = {
            0: ANNOTATION_LABELS["Factual"],
            1: ANNOTATION_LABELS["Non-factual"],
        }

    def extract_features(
        self,
        gen_summary: str,
        entities: List[MarkedEntity],
    ):
        features = np.array(
            list(
                zip(
                    [0] * len(entities),  # priors
                    [0] * len(entities),  # posteriors
                    [1.0 if ent["in_source"] else 0.0 for ent in entities],
                )
            )
        )

        return features

    def classify_entities(
        self,
        marked_entities: MarkedEntityLookup,
        gen_summaries_by_id,
    ) -> MarkedEntityLookup:
        classified_entities: MarkedEntityLookup = {}
        for sum_id, summary in gen_summaries_by_id.items():
            features = self.extract_features(summary, marked_entities[sum_id])
            predictions = self.clf.predict(features)
            entities_with_preds = [x.copy() for x in marked_entities[sum_id]]
            for entitiy, pred in zip(entities_with_preds, predictions):
                entitiy["predicted_label"] = self.label_mapping[pred]
            classified_entities[sum_id] = entities_with_preds
        return classified_entities
