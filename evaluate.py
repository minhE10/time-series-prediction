import os
import csv
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


def _run_prefix(model_name, pred_len, data_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{model_name}_{pred_len}_{data_name}")


def save_history(history, model_name, pred_len, data_name, out_dir="."):
    prefix = _run_prefix(model_name, pred_len, data_name, out_dir)
    for key in ("train_loss", "val_loss"):
        path = f"{prefix}_{key}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for epoch, v in enumerate(history[key], 1):
                writer.writerow([epoch, v])
        print(f"Saved: {path}")
    return prefix


def load_history(model_name, pred_len, data_name, out_dir="."):
    prefix = _run_prefix(model_name, pred_len, data_name, out_dir)
    history = {}
    for key in ("train_loss", "val_loss"):
        path = f"{prefix}_{key}.csv"
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            history[key] = [float(row["loss"]) for row in reader]
    return history


def save_predictions(y_true, y_pred, target_cols, model_name, pred_len, data_name, out_dir="."):
    prefix = _run_prefix(model_name, pred_len, data_name, out_dir)
    path = f"{prefix}_predictions.npz"
    np.savez(path,
             y_true=np.array(y_true),
             y_pred=np.array(y_pred),
             target_cols=np.array(target_cols))
    print(f"Saved: {path}")
    return path


def load_predictions(model_name, pred_len, data_name, out_dir="."):
    prefix = _run_prefix(model_name, pred_len, data_name, out_dir)
    path = f"{prefix}_predictions.npz"
    data = np.load(path, allow_pickle=True)
    return data["y_true"], data["y_pred"], list(data["target_cols"])


def save_checkpoint(trainer, model_name, pred_len, data_name, out_dir="."):
    import pickle
    prefix = _run_prefix(model_name, pred_len, data_name, out_dir)

    if model_name in ("itransformer", "timemixer", "tsmamba"):
        import torch
        path = f"{prefix}_checkpoint.pt"
        state = trainer.best_state if trainer.best_state is not None else trainer.model.state_dict()
        torch.save(state, path)

    elif model_name == "xgboost":
        path = f"{prefix}_checkpoint.pkl"
        with open(path, "wb") as f:
            pickle.dump(trainer.models, f)

    elif model_name == "arima":
        path = f"{prefix}_checkpoint.pkl"
        with open(path, "wb") as f:
            pickle.dump(trainer.forecaster, f)

    else:
        raise ValueError(f"Unknown model_name for checkpoint: {model_name}")

    print(f"Saved: {path}")
    return path


def load_checkpoint(trainer, model_name, pred_len, data_name, out_dir="."):
    import pickle
    prefix = _run_prefix(model_name, pred_len, data_name, out_dir)

    if model_name in ("itransformer", "timemixer", "tsmamba"):
        import torch
        path = f"{prefix}_checkpoint.pt"
        trainer.model.load_state_dict(
            torch.load(path, map_location=trainer.device, weights_only=True)
        )
        trainer.best_state = trainer.model.state_dict()

    elif model_name == "xgboost":
        path = f"{prefix}_checkpoint.pkl"
        with open(path, "rb") as f:
            trainer.models = pickle.load(f)

    elif model_name == "arima":
        path = f"{prefix}_checkpoint.pkl"
        with open(path, "rb") as f:
            trainer.forecaster = pickle.load(f)

    else:
        raise ValueError(f"Unknown model_name for checkpoint: {model_name}")

    print(f"Loaded: {path}")
    return trainer