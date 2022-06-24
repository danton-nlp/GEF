# Generate via Entity Factuality (GEF)

## Evaluation & Annotation Scripts
- [Annotation workflow](annotation_demo.ipynb)
- [Evaluate generated summaries](evaluate_summaries.py)
- [Full dataset Rouge Score computation](compute_rouge_scores.py)
- [Statistical tests for evaluation](evaluation_statistical_tests.ipynb)
- [Test suite for validating results](tests/test_evaluation.py)

## Data
- [XEnt Annotations](data/xent/)
- [Our Annotations](data/xsum/gold-metrics.json)
- [Generation logs for GEF](results/gef-logs/)
- [Evaluation results for all models](results/evaluation/)
- [Summaries from various models (baselines, GEF, Entity Corrector, RL-Fact & Pinocchio)](data/xsum/)

## Running GEF
1. Train Factuality Classifier on XEnt data:
```
python train_factuality_clf.py \
    --train_data_filepath data/xent-probs/train-probs.json \
    --test_data_filepath data/xent-probs/test.json \
    --factuality-classifiers/knn-20n.pickle \
    --n_neighbors 20
``` 
2. Run GEF with Pegasus & BART on evaluation subset
```
python generate_fbs_summaries.py --test_size 100
```
3. Evaluate resuts
```
python evaluate_summaries.py --test_size 100
```

## Running GEF on all of XSUM Test (takes about 2 hours on a single RTX5000)
#### BART-Large
```
python iterative_constraints.py --data_subset full --batch_size 16 --classifier_batch_size 16 --max_iterations 100 --pickled_classifier factuality-classifiers/v2-knn-20n.pickle
```

#### PEGASUS
```
python iterative_constraints.py --data_subset full --batch_size 16 --classifier_batch_size 16 --max_iterations 100 --pickled_classifier factuality-classifiers/v2-knn-20n.pickle --model_summarization google/pegasus-xsum
```

## Compute rouge scores
```
python compute_rouge_scores.py
```

## Preprocessing Scripts
- [Compute probabilities for named entities](compute_probs.py)
- [Batch detect NER in generated summaries](batch_detect_entities.py.py)

# Development
## Dev Setup
```
conda create -n factual-beam-search python=3.8
conda activate factual-beam-search
pip install -r requirements.txt
```

## Run tests
```
pytests tests
```

### Iterative pipeline with oracle on test/debug
```
python iterative_constraints.py --data_subset test|debug --batch_size 4 --verbose 1
```

### Iterative pipeline with KNN on test/debug
```
python iterative_constraints.py --data_subset test|debug --batch_size 4 --verbose 1 --pickled_classifier factuality-classifiers/v0-knn.pickle
```
#### Annotate data
```
python annotate_summaries.py --test_size 100
```

#### Compare results in Streamlit app
```
streamlit run app.py
```

### Prior and Posterior Named Entity Probability Computation

To create dataset of prior and posterior probabilities of XEnt named entities,
run compute_probs.py. For example, to run on train

```bash
python compute_probs.py \
    --xent_split train \
    --output_filepath data/xent-probs/train.json
```

### Factuality Classification Model

As a proof of concept of a non-oracle named entity factuality classifier, we
train a non-parameteric model on prior and posterior probability, and whether
the name entitity overlaps with the source, to serve as a discriminator between
factual and non-factual entity.

To train, pickle and test this classification, run the following:

```bash
$ python train_factuality_clf.py \
    --train_data_filepath data/xent-probs/train.json \
    --test_data_filepath data/xent-probs/test.json \
    --pickled_clf_path optional/path/to/newly/train/knn.pickle
```
