import numpy as np
import pandas as pd


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    err = actual - predicted
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    wmape = float(np.sum(np.abs(err)) / (np.sum(np.abs(actual)) + 1e-8))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return pd.DataFrame(
        [
            {"metric": "MAE", "value": mae},
            {"metric": "RMSE", "value": rmse},
            {"metric": "WMAPE", "value": wmape},
            {"metric": "R2", "value": r2},
        ]
    )
