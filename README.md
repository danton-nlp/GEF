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
