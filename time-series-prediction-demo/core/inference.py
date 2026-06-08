from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adapters.training_project import build_dataset, build_torch_model, load_pickle_checkpoint
from core.metrics import regression_metrics


class ForecastService:
    def __init__(self, registry, device=None):
        self.registry = registry
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def predict(
        self,
        model,
        dataset_name,
        pred_len,
        target_col,
        sample_index=-1,
        csv_path=None,
        checkpoint_path=None,
        seq_len=48,
        scaler="standard",
    ):
        artifact = self.registry.get_optional_artifact(model, dataset_name, pred_len)
        dataset_spec = self.registry.datasets[dataset_name]
        source_csv = csv_path or dataset_spec.path
        uploaded_checkpoint = checkpoint_path is not None
        if not Path(source_csv).exists():
            raise FileNotFoundError(
                f"CSV file not found: {source_csv}. "
                "Place the dataset CSV at this path or upload a CSV with CSV Override."
            )

        if artifact and not uploaded_checkpoint:
            seq_len = artifact.seq_len
            scaler = artifact.scaler
            checkpoint_path = artifact.checkpoint
            artifact_id = artifact.id
        elif uploaded_checkpoint:
            checkpoint_path = Path(checkpoint_path)
            artifact_id = f"uploaded_{model}_{dataset_name}_{pred_len}"
        else:
            raise ValueError(
                f"No registered checkpoint for {model}/{dataset_name}/pred_len={pred_len}. "
                "Upload a compatible .pt/.pkl checkpoint or add it to configs/artifacts.yaml."
            )

        if csv_path or uploaded_checkpoint:
            ts_dataset = self._build_dataset_uncached(
                dataset_name=dataset_name,
                csv_path=source_csv,
                seq_len=int(seq_len),
                pred_len=int(pred_len),
                scaler=_normalize_scaler(scaler),
            )
            model_obj = self._load_model(model, checkpoint_path, ts_dataset, int(seq_len), int(pred_len))
        else:
            ts_dataset, model_obj = self._default_bundle(artifact.id)

        model_spec = self.registry.models[model]
        if model_spec.framework == "torch":
            result_df, index = self._predict_torch(model_obj, ts_dataset, sample_index)
        elif model == "xgboost":
            result_df, index = self._predict_xgboost(model_obj, ts_dataset, sample_index, int(pred_len))
        elif model == "arima":
            result_df, index = self._predict_arima(model_obj, ts_dataset, sample_index)
        else:
            raise ValueError(f"Model '{model}' is registered but does not have an inference adapter yet.")

        if target_col not in ts_dataset.target_cols:
            target_col = ts_dataset.target_cols[0]
        target_df = result_df[result_df["target"] == target_col]
        metrics_df = regression_metrics(target_df["actual"], target_df["predicted"])

        info = {
            "artifact_id": artifact_id,
            "checkpoint": str(checkpoint_path),
            "checkpoint_source": "uploaded" if uploaded_checkpoint else "registered",
            "dataset_rows": len(ts_dataset.df),
            "test_windows": len(ts_dataset.test_dataset),
            "sample_index": index,
            "device": self.device,
            "model": model,
            "dataset": dataset_name,
            "seq_len": int(seq_len),
            "pred_len": int(pred_len),
            "scaler": _normalize_scaler(scaler) or "none",
            "targets": ts_dataset.target_cols,
        }
        return result_df, metrics_df, info

    @lru_cache(maxsize=8)
    def _default_bundle(self, artifact_id):
        artifact = self.registry.artifacts[artifact_id]
        dataset_spec = self.registry.datasets[artifact.dataset]
        ts_dataset = self._build_dataset_uncached(
            dataset_name=artifact.dataset,
            csv_path=dataset_spec.path,
            seq_len=artifact.seq_len,
            pred_len=artifact.pred_len,
            scaler=artifact.scaler,
        )
        model_obj = self._load_model(artifact.model, artifact.checkpoint, ts_dataset, artifact.seq_len, artifact.pred_len)
        return ts_dataset, model_obj

    def _build_dataset_uncached(self, dataset_name, csv_path, seq_len, pred_len, scaler):
        return build_dataset(
            dataset_name=dataset_name,
            csv_path=csv_path,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
        )

    def _load_model(self, model_name, checkpoint_path, ts_dataset, seq_len, pred_len):
        model_spec = self.registry.models[model_name]
        if model_spec.framework == "torch":
            return build_torch_model(
                model_name=model_name,
                dataset=ts_dataset,
                seq_len=seq_len,
                pred_len=pred_len,
                checkpoint_path=checkpoint_path,
                device=self.device,
            )
        if model_name in ("xgboost", "arima"):
            return load_pickle_checkpoint(checkpoint_path)
        raise ValueError(f"Unsupported framework '{model_spec.framework}' for model '{model_name}'.")

    def _predict_torch(self, model_obj, ts_dataset, sample_index):
        X, y_true_scaled, index = self._get_window(ts_dataset, sample_index)
        with torch.no_grad():
            y_pred_scaled = model_obj(X.unsqueeze(0).to(self.device)).cpu().numpy()[0]

        y_true = ts_dataset.inverse_transform_target(y_true_scaled.unsqueeze(0))[0]
        y_pred = ts_dataset.inverse_transform_target(y_pred_scaled[np.newaxis, :, :])[0]
        return self._make_result_frame(ts_dataset, y_true, y_pred, index), index

    def _predict_xgboost(self, model_obj, ts_dataset, sample_index, pred_len):
        models = _extract_xgboost_models(model_obj)
        if len(models) != ts_dataset.n_targets:
            raise ValueError(
                f"XGBoost checkpoint has {len(models)} target models, "
                f"but dataset expects {ts_dataset.n_targets} targets."
            )

        X, y_window_scaled, index = self._get_window(ts_dataset, sample_index)
        X_flat = X.numpy().reshape(1, -1)
        y_pred_scaled = np.array([[model.predict(X_flat)[0] for model in models]], dtype=np.float32)
        y_true_scaled = y_window_scaled[-1:].numpy()

        y_true = ts_dataset.inverse_transform_target(y_true_scaled[np.newaxis, :, :])[0]
        y_pred = ts_dataset.inverse_transform_target(y_pred_scaled[np.newaxis, :, :])[0]
        result_df = self._make_result_frame(
            ts_dataset,
            y_true,
            y_pred,
            index,
            timestamp_offset=pred_len - 1,
            step_offset=pred_len - 1,
        )
        return result_df, index

    def _predict_arima(self, model_obj, ts_dataset, sample_index):
        forecaster = _extract_arima_forecaster(model_obj)
        forecaster.dataset = ts_dataset
        y_true_all, y_pred_all = forecaster.predict_rolling(ts_dataset.X_test, ts_dataset.y_test)

        total = len(ts_dataset.test_dataset)
        if total <= 0:
            raise ValueError("Dataset does not contain enough rows for seq_len + pred_len.")
        index = int(sample_index)
        if index < 0:
            index = total - 1
        index = max(0, min(index, total - 1))
        row_index = min(index + ts_dataset.seq_len, len(y_true_all) - 1)

        y_true = y_true_all[row_index:row_index + 1]
        y_pred = y_pred_all[row_index:row_index + 1]
        timestamp = _test_timestamp(ts_dataset, row_index)
        result_df = self._make_one_step_frame(ts_dataset, y_true[0], y_pred[0], timestamp)
        return result_df, index

    def _get_window(self, ts_dataset, sample_index):
        total = len(ts_dataset.test_dataset)
        if total <= 0:
            raise ValueError("Dataset does not contain enough rows for seq_len + pred_len.")
        index = int(sample_index)
        if index < 0:
            index = total - 1
        index = max(0, min(index, total - 1))
        X, y = ts_dataset.test_dataset[index]
        return X, y, index

    def _make_result_frame(self, ts_dataset, y_true, y_pred, sample_index, timestamp_offset=0, step_offset=0):
        timestamps = _forecast_timestamps(ts_dataset, sample_index, len(y_true), timestamp_offset)
        rows = []
        for step in range(len(y_true)):
            for target_idx, target in enumerate(ts_dataset.target_cols):
                rows.append(
                    {
                        "step": step + 1 + step_offset,
                        "timestamp": timestamps[step] if timestamps is not None else step + 1,
                        "target": target,
                        "actual": float(y_true[step, target_idx]),
                        "predicted": float(y_pred[step, target_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def _make_one_step_frame(self, ts_dataset, y_true, y_pred, timestamp):
        rows = []
        for target_idx, target in enumerate(ts_dataset.target_cols):
            rows.append(
                {
                    "step": 1,
                    "timestamp": timestamp,
                    "target": target,
                    "actual": float(y_true[target_idx]),
                    "predicted": float(y_pred[target_idx]),
                }
            )
        return pd.DataFrame(rows)


def _forecast_timestamps(ts_dataset, sample_index, pred_len, offset=0):
    if "datetime" not in ts_dataset.df.columns:
        return None
    n = len(ts_dataset.df)
    train_end = int(n * ts_dataset.train_ratio)
    val_end = int(n * (ts_dataset.train_ratio + ts_dataset.val_ratio))
    test_start = max(0, val_end - ts_dataset.seq_len)
    start = test_start + sample_index + ts_dataset.seq_len + offset
    values = ts_dataset.df.iloc[start:start + pred_len]["datetime"]
    if len(values) != pred_len:
        return None
    return values.reset_index(drop=True)


def _normalize_scaler(scaler):
    if not scaler or scaler == "none":
        return None
    return scaler


def _test_timestamp(ts_dataset, row_index):
    if "datetime" not in ts_dataset.df.columns:
        return row_index + 1
    n = len(ts_dataset.df)
    val_end = int(n * (ts_dataset.train_ratio + ts_dataset.val_ratio))
    test_start = max(0, val_end - ts_dataset.seq_len)
    global_index = min(test_start + row_index, len(ts_dataset.df) - 1)
    return ts_dataset.df.iloc[global_index]["datetime"]


def _extract_xgboost_models(checkpoint):
    if isinstance(checkpoint, dict):
        if "models" in checkpoint:
            return checkpoint["models"]
        if "xgboost_models" in checkpoint:
            return checkpoint["xgboost_models"]
    if isinstance(checkpoint, list):
        return checkpoint
    raise ValueError("XGBoost checkpoint must be a list of target models or a dict containing 'models'.")


def _extract_arima_forecaster(checkpoint):
    if isinstance(checkpoint, dict):
        if "forecaster" in checkpoint:
            return checkpoint["forecaster"]
        if "arima_forecaster" in checkpoint:
            return checkpoint["arima_forecaster"]
    if hasattr(checkpoint, "predict_rolling"):
        return checkpoint
    raise ValueError("ARIMA checkpoint must be an ARIMAForecaster or a dict containing 'forecaster'.")
