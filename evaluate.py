import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def wmape(y_true, y_pred, eps=1e-8):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps)


def compute_metrics(y_true, y_pred, scaled=True):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(np.sqrt(mse)),
        "WMAPE": float(wmape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "scaled": scaled,
    }


def compute_metrics_per_target(y_true, y_pred, target_cols, scaled=True):
    n_targets = len(target_cols)
    y_true_2d = np.array(y_true).reshape(-1, n_targets)
    y_pred_2d = np.array(y_pred).reshape(-1, n_targets)

    per_target = {
        col: compute_metrics(y_true_2d[:, i], y_pred_2d[:, i], scaled=scaled)
        for i, col in enumerate(target_cols)
    }
    overall = compute_metrics(y_true_2d, y_pred_2d, scaled=scaled)

    return {"overall": overall, "per_target": per_target}


def format_metrics(m):
    return (
        f"MSE: {m['MSE']:.4f} | MAE: {m['MAE']:.4f} | "
        f"RMSE: {m['RMSE']:.4f} | WMAPE: {m['WMAPE']:.4f} | R2: {m['R2']:.4f}"
        + (" [scaled]" if m.get("scaled") else " [original]")
    )


def print_metrics(result, title="EVALUATION"):
    print(f"=== {title} ===")
    if "per_target" in result and len(result["per_target"]) > 1:
        for col, m in result["per_target"].items():
            print(f"  {col:>12s} | {format_metrics(m)}")
        print("-" * 90)
    print(f"  {'OVERALL':>12s} | {format_metrics(result['overall'])}")
    print("=" * 90)