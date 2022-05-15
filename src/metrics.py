from datasets import load_metric


_rouge_metric = load_metric("rouge")


def rouge(predictions: list, references: list):
    results = _rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    # types: rouge1, rouge2, rougeL
    # Use mid for every score because it is the mean of all inputs
    # high gives the 95th %ile score and low gives the 5th %ile
    scores = {
        "rouge1": {
            "precision": results["rouge1"].mid.precision,
            "recall": results["rouge1"].mid.recall,
            "f1": results["rouge1"].mid.fmeasure,
        },
        "rouge2": {
            "precision": results["rouge2"].mid.precision,
            "recall": results["rouge2"].mid.recall,
            "f1": results["rouge2"].mid.fmeasure,
        },
        "rougeL": {
            "precision": results["rougeL"].mid.precision,
            "recall": results["rougeL"].mid.recall,
            "f1": results["rougeL"].mid.fmeasure,
        },
    }
    return scores
