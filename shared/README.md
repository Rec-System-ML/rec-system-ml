# recsys-shared

Shared utilities for both `cds524-ml-project` and `cds525-dl-project`.

## What this repo contains

- `data_pipeline.py`: shared KuaiRand preprocessing pipeline
- `evaluation.py`: unified evaluation wrappers (with optional MS Recommenders backend)
- `experiment.py`: MLflow tracking wrappers
- `reranker.py`: time-decay reranker
- `shap_utils.py`: SHAP analysis helpers
- `tag_display.py`: tag id to display-name mapping helpers
- `visualization.py`: reusable plotting functions

## Installation

```bash
pip install -U pandas numpy scikit-learn matplotlib seaborn shap mlflow recommenders
```

## Quick usage

```python
from data_pipeline import preprocess_kuairand
from reranker import TimeDecayReranker
from evaluation import evaluate_ranking

train, val, test, encoders, tag_mlb, tag_matrix = preprocess_kuairand(
    data_path="/path/to/kuairand",
)

reranker = TimeDecayReranker(gamma=0.7, beta=1.0)
```

## Notes

- This repository should be maintained once and reused via git submodule in both course projects.
- Keep interfaces stable to reduce integration overhead for both teams.
