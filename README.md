# factual-beam-search
```
conda create -n factual-beam-search python=3.8
conda activate factual-beam-search
```

## Run tests
```
pytests tests
```

## Scripts
- `batch_detect_ner.py <sumtool-model>`: batch detect NER & write to sumtool metadata
- `oracle_experiment.py`: run oracle experiment, writes & reads from `/data/oracle-experiment`
- `persist_xent_annotations.py`: persits xent annotations in `gold` sumtool index for easy access by ID

### Iterative pipeline with oracle on test/debug
```
python iterative_constraints.py --data_subset test|debug --batch_size 4 --verbose 1
```

### Iterative pipeline with KNN on test/debug
```
python iterative_constraints.py --data_subset test|debug --batch_size 4 --verbose 1 --pickled_classifier factuality-classifiers/v0-knn.pickle
```
#### with annotation
```
python iterative_constraints.py --annotate 1
```

#### Annotate Daniel's data
```
python iterative_constraints.py --data_subset test-daniel --batch_size 2 --verbose 1 --annotate 1
```

#### Annotate Anton's data
```
python iterative_constraints.py --data_subset test-anton --batch_size 2 --verbose 1 --annotate 1
```

#### Evaluate results
1. Move `logs-iterative/<desired-result>` to `results/` and name it appropriately
2. Open streamlit app to inspect outputs
3. Run `python evaluate_summaries.py` (this only checks `results/test.json` but can be updated)
### Prior and Posterior Named Entity Probability Computation

To create dataset of prior and posterior probabilities of Xent named entities,
run compute_probs.py. For example, to run on train

```bash
python compute_probs.py --xent_split train --output_filepath data/xent-probs/train.json
```
