from typing import List, Tuple
from compute_probs import compute_probs_for_summary
from src.entity_utils import (
    MarkedEntity,
    MarkedEntityLookup,
)
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from src.misc_utils import Timer
import pickle
from src.generation_utils import (
    load_posterior_model_and_tokenizer,
    load_prior_model_and_tokenizer,
)
from src.prob_computation_utils import (
    InferenceInput,
    build_masked_inputs_and_targets_for_inference,
)
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

    def __init__(
        self, pickled_model_path, prior_model_path, posterior_model_path, batch_size=4
    ):
        with Timer("Initializing entity factuality classifier"):
            with open(pickled_model_path, "rb") as f:
                self.clf: PickledClassifier = pickle.load(f)
            self.label_mapping = {
                0: ANNOTATION_LABELS["Factual"],
                1: ANNOTATION_LABELS["Non-factual"],
            }
            self.batch_size = batch_size
            self.prior_model_and_tokenizer = load_prior_model_and_tokenizer(
                prior_model_path
            )
            self.posterior_model_and_tokenizer = load_posterior_model_and_tokenizer(
                posterior_model_path
            )

    def extract_features(self, ents_to_classify: InferenceInput):
        features = []
        for (_, _, ents) in ents_to_classify:
            for ent in ents:
                # prior / posterior / in_source
                features.append([0, 0, 1.0 if ent["in_source"] else 0.0])
        features = np.array(features)
        (
            inputs,
            targets,
            masked_entities,
            sources,
        ) = build_masked_inputs_and_targets_for_inference(ents_to_classify)

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
        # Build a list of (sum, source, ents[]) to enable
        # batching across summaries when extracting feaatures
        ents_to_classify: InferenceInput = []
        for sum_id, summary in gen_summaries_by_id.items():
            updated_entities = [x.copy() for x in marked_entities[sum_id]]
            for ent in updated_entities:
                if ent["in_source"]:
                    ent["predicted_label"] = ANNOTATION_LABELS["Non-hallucinated"]

            ents_not_in_source = [ent for ent in updated_entities if not ent["in_source"]]
            if len(ents_not_in_source) > 0:
                ents_to_classify.append(
                    (
                        summary,
                        sources_by_id[sum_id],
                        ents_not_in_source,
                    )
                )
            classified_entities[sum_id] = updated_entities

        if len(ents_to_classify) > 0:
            features = self.extract_features(ents_to_classify)
            predictions = self.clf.predict(features)
            idx = 0
            for (_, _, ents) in ents_to_classify:
                for entity in ents:
                    entity["predicted_label"] = self.label_mapping[predictions[idx]]
                    idx += 1

        return classified_entities
