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


def build_train_features_and_targets(data, ignore_intrinsic: bool):
    if ignore_intrinsic:
        data = data[data["entity_label"] != "Intrinsic Hallucination"]

    targets = data.apply(to_nonfactual_label, axis=1)
    features = data[["prior_prob", "posterior_prob", "overlaps_source"]]

    return features, targets


def build_test_features_and_targets(data, ignore_intrinsic: bool, test_only_on_hallucinated: bool):
    if ignore_intrinsic:
        data = data[data["entity_label"] != "Intrinsic Hallucination"]

    if test_only_on_hallucinated:
        print("\n -- Testing only on entities that are hallucinated -- \n")
        data = data[data["entity_label"] != "Non-hallucinated"]

    targets = data.apply(to_nonfactual_label, axis=1)
    features = data[["prior_prob", "posterior_prob", "overlaps_source"]]

    return features, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train kNN classifier on Xent-probs dataset"
    )
    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--pickled_clf_path", type=str)
    parser.add_argument("--train_data_filepath", type=str)
    parser.add_argument("--test_data_filepath", type=str)
    parser.add_argument("--ignore_intrinsic", default=True)
    parser.add_argument(
        "--test_only_on_hallucinated",
        default=False,
        action="store_true",
        help="filter test set to only evaluate against hallucinated entitites."
        + "Enables a more fair comparison against xent-extended",
    )
    args = parser.parse_args()

    train_data = json.load(open(args.train_data_filepath))
    test_data = json.load(open(args.test_data_filepath))

    Xy_train = preprocess_data(train_data)
    Xy_test = preprocess_data(test_data)

    print("\n -- Ignoring intrinsic hallucations in train and test data -- \n")

    X_train, y_train = build_train_features_and_targets(Xy_train, args.ignore_intrinsic)
    X_test, y_test = build_test_features_and_targets(
        Xy_test,
        args.ignore_intrinsic,
        args.test_only_on_hallucinated
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
