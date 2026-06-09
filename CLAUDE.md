# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training (Kaggle notebook)
Open and run `notebook/main.ipynb`. Configure the top-level variables before running:
```python
MODEL_NAME = "itransformer"   # arima | xgboost | itransformer | timemixer | tsmamba
DATA_NAME  = "beijing_air_quality"
PRED_LEN   = 24               # 12 | 24 | 48
SEQ_LEN    = 48
SCALER     = "standard"       # standard | minmax | none
```

### Gradio demo
```powershell
cd time-series-prediction-demo
pip install -r requirements.txt
python app.py
# opens at http://127.0.0.1:7860
```

### Demo tests
```powershell
cd time-series-prediction-demo
python -m pytest tests/
```

## Architecture

### Training pipeline (root)

```
data_loader.py   →  train.py   →  evaluate.py
TimeSeriesDataset   Trainer        metrics + file I/O
TSWindowDataset     XGBoostTrainer
                    ARIMATrainer
```

`data_loader.py` handles CSV loading, temporal feature engineering (`hour_sin/cos`, `month_sin/cos`), train/val/test splitting, scaling, and sliding-window construction. All dataset schemas live in `config/data_config.py`.

`src/factory.py` is the single entry point for instantiating both models and trainers from a string key. Add new models there.

Model hyperparameters are centralised in `config/model_config.py`. The three PyTorch models (iTransformer, TimeMixer, TSMamba) save `.pt` checkpoints; XGBoost and ARIMA save `.pkl`.

Outputs written by a training run:
- `checkpoints/<model>/<dataset>/pred_<N>.pt` (or `.pkl`)
- CSV loss logs and `.npz` prediction arrays (managed by `evaluate.py`)

### Gradio demo (`time-series-prediction-demo/`)

The demo is **config-driven**. Three YAML files control everything:
- `configs/datasets.yaml` — dataset paths and column schemas
- `configs/models.yaml` — display names, framework (`torch`/`sklearn`), adapter
- `configs/artifacts.yaml` — registered checkpoints (model + dataset + pred_len + seq_len + scaler → checkpoint path)

`core/registry.py` loads these YAMLs at startup into `DatasetSpec`, `ModelSpec`, `ArtifactSpec` dataclasses.

`core/inference.py` loads the checkpoint via `adapters/training_project.py`, which bridges into `vendor/training_project/` — a vendored copy of the root training code. **When you change model or data-loader logic in the root, mirror those changes into `vendor/training_project/`**.

The UI (`demo/ui.py`) is stateless; `demo/state.py` holds mutable Gradio state.

### Adding a checkpoint (most common task)
1. Drop the `.pt` / `.pkl` file under `time-series-prediction-demo/artifacts/checkpoints/<model>/<dataset>/`.
2. Add one entry to `configs/artifacts.yaml` with matching `model`, `dataset`, `pred_len`, `seq_len`, `scaler` fields.
3. Restart `python app.py`.

### Model input/output contract
All PyTorch models follow: `(batch, seq_len, n_features) → (batch, pred_len, n_targets)`. XGBoost and ARIMA operate per-target with a flattened feature vector.

### Prediction horizons and datasets
Supported horizons: `12`, `24`, `48` steps.  
Datasets: `sunspots`, `appliances_energy`, `beijing_air_quality`, `hanoi_air_quality`, `bitcoin`.  
Dataset CSVs are **not committed** (too large). Place them in `time-series-prediction-demo/data/raw/` using the filenames in `configs/datasets.yaml`.
