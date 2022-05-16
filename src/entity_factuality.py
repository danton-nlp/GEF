from typing import List
from compute_probs import compute_probs_for_summary
from src.entity_utils import (
    MarkedEntity,
    MarkedEntityLookup,
)
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from src.misc_utils import Timer
import pickle
from src.generation_utils import load_bart_xsum_cmlm, load_model_and_tokenizer, load_prior_model_and_tokenizer
from src.prob_computation_utils import build_masked_inputs_and_targets_for_inference
import pandas as pd


ANNOTATION_LABELS = {
    "Non-factual": "Non-factual Hallucination",
    "Factual": "Factual Hallucination",
    "Non-hallucinated": "Non-hallucinated",
    "Unknown": "Unknown",
    "Intrinsic": "Intrinsic Hallucination",
}

PickledClassifier = KNeighborsClassifier


class EntityFactualityClassifier:
    """
    Wrapper for sklearn entity factuality classifier.

    Responsible for loading a pickled model, extracting features,
    classifying & returning label.
    """

    def __init__(self, pickled_model_path, batch_size=4):
        with Timer("Initializing entity factuality classifier"):
            with open(pickled_model_path, "rb") as f:
                self.clf: PickledClassifier = pickle.load(f)
            self.label_mapping = {
                0: ANNOTATION_LABELS["Factual"],
                1: ANNOTATION_LABELS["Non-factual"],
            }
            self.batch_size = batch_size
            self.prior_model_and_tokenizer = load_prior_model_and_tokenizer()
            self.posterior_model_and_tokenizer = load_bart_xsum_cmlm()

    def extract_features(
        self,
        gen_summary: str,
        source: str,
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
        (
            inputs,
            targets,
            masked_entities,
            sources,
        ) = build_masked_inputs_and_targets_for_inference(
            {
                "source": source,
                "prediction": gen_summary,
                "entities": entities,
            }
        )

        entity_probs = compute_probs_for_summary(
            masked_inputs=inputs,
            targets=targets,
            sources=sources,
            entities=masked_entities,
            batch_size=self.batch_size,
            prior_model_and_tokenizer=self.prior_model_and_tokenizer,
            posterior_model_and_tokenizer=self.posterior_model_and_tokenizer,
        )

        for i, (prior, posterior) in enumerate(entity_probs):
            features[i][0] = prior
            features[i][1] = posterior

        return pd.DataFrame(
            features, columns=["prior_prob", "posterior_prob", "overlaps_source"]
        )

    def classify_entities(
        self,
        marked_entities: MarkedEntityLookup,
        gen_summaries_by_id,
        sources_by_id,
    ) -> MarkedEntityLookup:
        classified_entities: MarkedEntityLookup = {}
        for sum_id, summary in gen_summaries_by_id.items():
            if len(marked_entities[sum_id]) > 0:
                features = self.extract_features(
                    summary, sources_by_id[sum_id], marked_entities[sum_id]
                )
                predictions = self.clf.predict(features)
                entities_with_preds = [x.copy() for x in marked_entities[sum_id]]
                for entitiy, pred in zip(entities_with_preds, predictions):
                    entitiy["predicted_label"] = self.label_mapping[pred]
                classified_entities[sum_id] = entities_with_preds
            else:
                classified_entities[sum_id] = []
        return classified_entities
