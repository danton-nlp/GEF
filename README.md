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

### Iterative pipeline (`iterative_constraints.py`)
```
python iterative_constraints.py --data_subset debug|xent|full
```

#### with annotation
```
python iterative_constraints.py --annotate 1
```