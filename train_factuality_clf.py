import argparse
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def preprocess_summary(example):
    """
    Extract prior prob, posterior prob, overlap bool and label for kNN
    classification training.
    """

    prior_probs, posterior_probs, overlaps, labels = [], [], [], []
    for entity in example["entities"]:
        prior_probs.append(entity["prior_prob"])
        posterior_probs.append(entity["posterior_prob"])
        overlaps.append(True if entity["label"] == "Non-hallucinated" else False)
        labels.append(entity["label"])

    return prior_probs, posterior_probs, overlaps, labels


def preprocess_data(examples):
    prior_probs, posterior_probs, overlaps, labels = [], [], [], []

    for example in examples:
        has_no_probs = all(
            "prior_prob" not in entity.keys() for entity in example["entities"]
        )
        if has_no_probs:
            next
        else:
            (
                example_prior_probs,
                example_posterior_probs,
                example_overlaps,
                example_labels,
            ) = preprocess_summary(example)

            prior_probs.extend(example_prior_probs)
            posterior_probs.extend(example_posterior_probs)
            overlaps.extend(example_overlaps)
            labels.extend(example_labels)

    return pd.DataFrame(
        {
            "prior_prob": prior_probs,
            "posterior_prob": posterior_probs,
            "overlaps_source": overlaps,
            "entity_label": labels,
        }
    )


def to_nonfactual_label(example):
    if example["entity_label"] == "Non-factual Hallucination":
        return 1
    else:
        return 0


def build_features_and_targets(data):
    targets = data.apply(to_nonfactual_label, axis=1)
    features = data[["prior_prob", "posterior_prob", "overlaps_source"]]

    return features, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train kNN classifier on Xent-probs dataset"
    )
    parser.add_argument("--input_filepath", type=str)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--pickled_clf_path", type=str)
    args = parser.parse_args()

    train_data = json.load(open(args.input_filepath))
    Xy_train = preprocess_data(train_data)
    X_train, y_train = build_features_and_targets(Xy_train)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=args.n_neighbors)),
        ]
    )

    model.fit(X_train, y_train)

    print("train score:", model.score(X_train, y_train))

    print(
        classification_report(
            model.predict(X_train),
            y_train,
            target_names=["Factual", "Non-Factual"],
            digits=4,
        )
    )

    with open(args.pickled_clf_path, "wb") as handle:
        pickle.dump(model, handle)

    print(f"saved model to {args.pickled_clf_path}")