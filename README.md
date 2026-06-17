<div align="center">

# Time Series Prediction
<a href="https://huggingface.co/spaces/nvtkienn/time-series-prediction-demo">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue" alt="Live Demo">
</a>
&nbsp;
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
&nbsp;
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
&nbsp;
<img src="https://img.shields.io/badge/License-MIT-22c55e" alt="License MIT">

</div>

## Overview
**Applied Statistics & Experimental Design - Group 10 | Hanoi University of Science and Technology, 2026**. Benchmarks five forecasting approaches (ARIMA, XGBoost, iTransformer, TimeMixer, TSMamba) across five real-world datasets and three forecast horizons (12 / 24 / 48 steps). All models are implemented **from scratch** - no external ML libraries beyond NumPy and PyTorch. Ships with a [live interactive demo](https://huggingface.co/spaces/nvtkienn/time-series-prediction-demo).

---

## Models

| Model | Family | Paper | Key Config |
|---|---|---|---|
| ARIMA | Statistical | - | p=3, d=0, q=3; rolling 1-step forecast per target |
| XGBoost | Gradient Boosting | [Chen & Guestrin (2016)](https://arxiv.org/abs/1603.02754) | n_estimators=200, max_depth=6, lr=0.05 |
| iTransformer | Transformer | [Liu et al. (2024)](https://arxiv.org/abs/2310.06625) | d_model=32, n_heads=4, e_layers=2, d_ff=64 |
| TimeMixer | MLP-Mixer | [Wang et al. (2024)](https://arxiv.org/abs/2405.14616) | d_model=32, e_layers=2, moving_avg=6, down_sampling_window=2 |
| TSMamba | SSM / Mamba | [Ma et al. (2024)](https://arxiv.org/abs/2411.02941) | d_model=32, d_state=8, n_layers=1, patch_len=8 |

All deep learning models use RevIN instance normalization, seq_len=48, and early stopping (patience=20) on validation loss.

---

## Datasets

| Dataset | Freq | Rows | Targets | Source |
|---|---|---|---|---|
| Sunspots | Monthly | 3,265 | 1 | [Kaggle](https://www.kaggle.com/datasets/robervalt/sunspots) |
| Appliances Energy | 10-min | 19,735 | 2 (Appliances, lights) | [Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction) |
| Beijing Air Quality | Hourly | 35,064 | 6 (PM2.5, PM10, SO2, NO2, CO, O3) | [UKaggle](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set) |
| Hanoi Air Quality | Hourly | 26,280 | 7 (PM2.5, PM10, AQI, CO, NO2, O3, SO2) | [Github](https://github.com/namanhnt/Hanoi-Air-Quality-Analysis) |
| Bitcoin | Hourly | 125,833 | 1 (Open price) | [Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) |

Train / Val / Test split: 70 / 10 / 20. Cyclical time encodings (hour_sin/cos, month_sin/cos) are added automatically.

---

## Project Structure

```
time-series-prediction/
├── requirements.txt         # full dependency list (training + demo)
├── setup.py                 # set_seed(), clear_memory(), seed_worker
├── data_loader.py           # TimeSeriesDataset, TSWindowDataset
├── train.py                 # Trainer, XGBoostTrainer, ARIMATrainer
├── evaluate.py              # MAE, RMSE, WMAPE, R2; save/load history, checkpoint, predictions
├── plot.py                  # loss curve and prediction visualization
│
├── config/
│   ├── data_config.py       # DATASET_CONFIG - feature/target columns per dataset
│   └── model_config.py      # MODEL_CONFIG - hyperparameters per model
│
├── src/
│   ├── factory.py           # build_model(), build_trainer()
│   └── models/
│       ├── arima.py         # ARIMAModel, ARIMAForecaster
│       ├── xgboost.py       # _Node, _XGBoostTree, XGBoost, loader_to_numpy
│       ├── itransformer.py  # RevIN, FeedForward, EncoderBlock, iTransformer
│       ├── timemixer.py     # MovingAvg, SeriesDecomp, PastDecomposableMixing, TimeMixer
│       └── tsmamba.py       # selective_scan, MambaBlock, RevIN, TSMamba
│
├── weights/                 # 75 pre-computed .npz files (5 models x 5 datasets x 3 horizons)
├── checkpoints/             # trained model weights
├── notebook/
│   └── main.ipynb           # Kaggle-ready training notebook
│
└── demo/                    # Gradio + React interactive demo
    ├── app.py               # entry point: python app.py
    ├── batch_export.py      # .npz -> predictions.js
    ├── requirements.txt     # demo-only minimal requirements
    └── ui/
        ├── ui.py            # Gradio Blocks builder
        └── web/             # static React frontend (no build step needed)
```

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py --model itransformer --dataset sunspots --pred_len 24
```

| Argument | Options |
|---|---|
| `--model` | `arima` `xgboost` `itransformer` `timemixer` `tsmamba` |
| `--dataset` | `sunspots` `appliances_energy` `beijing_air_quality` `hanoi_air_quality` `bitcoin` |
| `--pred_len` | `12` `24` `48` |

Output is saved to `predictions/` as `{model}_{pred_len}_{dataset}_predictions.npz`.

---

## Demo

### Run locally

```bash
cd demo
python app.py
```

Opens at `http://localhost:7860`. Four tabs:

- **Forecast** - single model, full test set, dual-range slider, 3 reconstruction modes
- **Compare** - side-by-side model comparison with leaderboard
- **Datasets** - dataset overview with feature breakdown
- **Theory** - model cards with formulas, strengths, and limitations

Or open the [live HuggingFace Space](https://huggingface.co/spaces/nvtkienn/time-series-prediction-demo) directly.

### Regenerate predictions.js

After retraining or adding new `.npz` files:

```bash
cd demo
python batch_export.py
# reads from ../predictions, writes to ui/web/predictions.js
```

Custom paths:

```bash
python batch_export.py --dir ../predictions --out ui/web/predictions.js
```

---

## Notebook (Kaggle)

Edit the four variables at the top of `notebook/main.ipynb`, then **Save and Run All**:

```python
MODEL_NAME  = "timemixer"   # arima | itransformer | timemixer | tsmamba | xgboost
DATA_NAME   = "bitcoin"     # sunspots | appliances_energy | beijing_air_quality | hanoi_air_quality | bitcoin
PRED_LEN    = 12            # 12 | 24 | 48
DATASET_DIR = "/kaggle/input/"
```

Each run produces four files: `train_loss.csv`, `val_loss.csv`, `predictions.npz`, `checkpoint.pt` / `checkpoint.pkl`.