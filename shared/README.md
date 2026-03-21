# Shared toolkit (bundled in this project)

This directory is the in-repo copy of the former **recsys-shared** utilities. The project root is on `sys.path` so you import as **`shared.*`** (not a separate checkout).

## Contents

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
from shared.data_pipeline import preprocess_kuairand
from shared.reranker import TimeDecayReranker
from shared.evaluation import evaluate_ranking

train, val, test, encoders, tag_mlb, tag_matrix = preprocess_kuairand(
    data_path="/path/to/kuairand",
)

reranker = TimeDecayReranker(gamma=0.7, beta=1.0)
```

## KuaiRand data

Default training path is `shared/data/KuaiRand-1K/data/` (create or symlink if missing). Override with `python main.py --data-dir /path/to/data` or set `RECSYS_SHARED_PATH` to a directory that contains `data/KuaiRand-1K/data`.


