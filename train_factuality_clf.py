import argparse
import json
import pickle
from typing import List, Tuple
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


def filter_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    methods: List[str],
):
    if ("extrinsic_only" in methods) and (len(methods) > 1):
        raise ValueError("not sensible to specify extrinsic_only and other filters")

    if methods == ["extrinsic_only"]:
        print(
            "\n -- Only training and testing on extrinsic entity hallucinations -- \n"
        )

        train_data = train_data[
            train_data["entity_label"].isin(
                ["Factual Hallucination", "Non-factual Hallucination"]
            )
        ]

        test_data = test_data[
            test_data["entity_label"].isin(
                ["Factual Hallucination", "Non-factual Hallucination"]
            )
        ]

    else:
        if "no_intrinsic" in methods:
            print("\n -- Ignoring intrinsic hallucations in train and test data -- \n")

            train_data = train_data[
                train_data["entity_label"] != "Intrinsic Hallucination"
            ]
            test_data = test_data[
                test_data["entity_label"] != "Intrinsic Hallucination"
            ]

        if "extrinsic_test_only" in methods:
            print("\n -- Testing only on entities that are hallucinated -- \n")

            test_data = test_data[
                test_data["entity_label"].isin(
                    ["Factual Hallucination", "Non-factual Hallucination"]
                )
            ]

    return train_data, test_data


def build_features_and_targets(
    datasets: Tuple[pd.DataFrame, pd.DataFrame],
    no_overlaps_source: bool = False,
):
    features_and_targets: List[Tuple] = []

    if no_overlaps_source:
        print("\n -- No including overlaps_source as a feature -- \n")

    for dataset in datasets:
        targets = dataset.apply(to_nonfactual_label, axis=1)

        if no_overlaps_source:
            features = dataset[["prior_prob", "posterior_prob"]]
        else:
            features = dataset[["prior_prob", "posterior_prob", "overlaps_source"]]

        features_and_targets.append((features, targets))

    return features_and_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train kNN classifier on Xent-probs dataset"
    )
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--pickled_clf_path", type=str)
    parser.add_argument("--train_data_filepath", type=str)
    parser.add_argument("--test_data_filepath", type=str)
    parser.add_argument("--no_overlaps_source", default=False, action='store_true')
    parser.add_argument(
        "--data_filters",
        action="append",
        choices=["extrinsic_only", "extrinsic_test_only", "no_intrinsic"],
        help="Specific EITHER extrinsic_only (only test and train on extrinsic hallucinations, ignore factual named entities and intrinsic hallucinations)"
        + "OR any combination of extrinsic_test_only (test only on extrinsic train on everything unless specified otherwise) or no_instrinic"
        + "(don't train or test on entities labeled as intrinsic)",
    )

    args = parser.parse_args()

    train_data = json.load(open(args.train_data_filepath))
    test_data = json.load(open(args.test_data_filepath))

    Xy_train = preprocess_data(train_data)
    Xy_test = preprocess_data(test_data)

    Xy_train, Xy_test = filter_data(Xy_train, Xy_test, args.data_filters)

    (X_train, y_train), (X_test, y_test) = build_features_and_targets(
        (Xy_train, Xy_test),
        no_overlaps_source=args.no_overlaps_source
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=args.n_neighbors)),
        ]
    )

    model.fit(X_train, y_train)

    print(
        "Test Results\n",
        f"test file: {args.test_data_filepath}\n",
        classification_report(
            y_test,
            model.predict(X_test),
            target_names=["Factual", "Non-Factual"],
            digits=4,
        ),
    )

    if args.pickled_clf_path:
        with open(args.pickled_clf_path, "wb") as handle:
            pickle.dump(model, handle)

        print(f"saved model to {args.pickled_clf_path}")
