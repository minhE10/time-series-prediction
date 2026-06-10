# Time Series Prediction

Applied Statistics & Experimental Design — Group 10 - 168274  
Hanoi University of Science and Technology, 2025

Benchmarks five forecasting approaches (ARIMA, XGBoost, iTransformer, TimeMixer, TSMamba) across five real-world datasets and three forecast horizons, with a Gradio-based interactive demo.

---

## Model weights
Model weights are available [here](https://drive.google.com/drive/folders/1YCiRHWmAjzIc0HxIJySsRy9HfAVxWWKx?usp=sharing)

---

## Project Structure

```
time-series-prediction/
├── train.py                 # Training entry point
├── evaluate.py              # Metrics: MAE, RMSE, WMAPE, R²
├── data_loader.py           # Dataset loading, splitting, scaling, windowing
├── plot.py                  # Loss curve visualization
├── setup.py
│
├── config/
│   ├── data_config.py       # Feature/target columns per dataset
│   └── model_config.py      # Hyperparameters per model
│
├── src/
│   ├── factory.py           # Model and trainer builder
│   └── models/
│       ├── arima.py
│       ├── xgboost.py
│       ├── itransformer.py
│       ├── timemixer.py
│       └── tsmamba.py
│
├── predictions/             # Pre-computed .npz outputs (75 files)
├── checkpoints/             # Trained model weights
├── notebook/
│   └── main.ipynb
│
└── demo/                    # Gradio web demo
    ├── app.py               # Launch: python app.py
    ├── batch_export.py      # .npz → predictions.js
    ├── requirements.txt
    └── ui/
        ├── ui.py            # Gradio Blocks builder
        └── web/             # Static frontend (React/JSX, no build step)
```

---

## Models

| Model | Family | Key Config |
|---|---|---|
| ARIMA | Statistical | p=3, d=0, q=3 — rolling 1-step forecast per target |
| XGBoost | Gradient Boosting | n_estimators=200, max_depth=6, lr=0.05 |
| iTransformer | Deep Learning | d_model=64, n_heads=4, e_layers=3, d_ff=128 |
| TimeMixer | Deep Learning | d_model=32, e_layers=2, moving_avg=25, down_sampling_window=2 |
| TSMamba | Deep Learning | d_model=32, d_state=8, n_layers=1, patch_len=8 |

All deep learning models use RevIN instance normalization, seq_len=48, early stopping on validation loss.

---

## Datasets

| Dataset | Freq | Rows | Targets | Source |
|---|---|---|---|---|
| Sunspots | Monthly | 3,265 | 1 | SILSO / WDC-SILSO, Brussels |
| Appliances Energy | 10-min | 19,735 | 2 (Appliances, lights) | UCI ML Repository |
| Beijing Air Quality | Hourly | 35,064 | 6 (PM2.5, PM10, SO2, NO2, CO, O3) | UCI — Beijing Multi-Site |
| Hanoi Air Quality | Hourly | 26,280 | 7 (PM2.5, PM10, AQI, CO, NO2, O3, SO2) | IQAir / OpenWeather |
| Bitcoin | Hourly | 125,833 | 1 (Open price) | Kaggle / CoinDesk |

All datasets use cyclical time encodings (hour_sin/cos, month_sin/cos where applicable).  
Train / Val / Test split: 70 / 10 / 20.

---

## Training

```bash
python train.py --model itransformer --dataset sunspots --pred_len 24
```

Supported values:
- `--model`: `arima`, `xgboost`, `itransformer`, `timemixer`, `tsmamba`
- `--dataset`: `sunspots`, `appliances_energy`, `beijing_air_quality`, `hanoi_air_quality`, `bitcoin`
- `--pred_len`: `12`, `24`, `48`

Outputs are saved to `predictions/` as `{model}_{pred_len}_{dataset}_predictions.npz`.

---

## Demo

### Run

```bash
cd demo
pip install -r requirements.txt   # gradio, numpy
python app.py
```

Opens a Gradio app embedding the static web UI. The demo has four tabs:

- **Forecast** — single model, full test set, dual-range slider, 3 reconstruction modes
- **Compare** — side-by-side model comparison with leaderboard
- **Datasets** — dataset overview with feature breakdown
- **Theory** — model cards with formulas, strengths, and limitations

### Regenerate predictions.js

After retraining or adding new `.npz` files:

```bash
cd demo
python batch_export.py
# uses ../predictions as source and writes to ui/web/predictions.js
```

Custom paths:

```bash
python batch_export.py --dir ../predictions --out ui/web/predictions.js
```

---

## Requirements

**Training** (`setup.py` / root env):

```
numpy, pandas, scikit-learn, torch, xgboost, statsmodels, matplotlib
```

**Demo only** (`demo/requirements.txt`):

```
gradio>=4.44
numpy>=1.26
```

---

## Results

Pre-computed predictions cover all 75 combinations (5 models × 5 datasets × 3 horizons).  
Metrics (MAE, RMSE, WMAPE, R²) are computed over the full test set and displayed in the demo.

---

© 2025 Group 10 - 168274 · Hanoi University of Science and Technology