import sys
import pickle

import torch

from core.paths import TRAINING_PROJECT_DIR


def ensure_training_project_on_path():
    project_path = str(TRAINING_PROJECT_DIR)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def build_dataset(dataset_name, csv_path, seq_len, pred_len, batch_size=32, scaler="standard"):
    ensure_training_project_on_path()
    from data_loader import TimeSeriesDataset

    dataset = TimeSeriesDataset(
        name=dataset_name,
        path=str(csv_path),
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        num_workers=0,
    )
    if scaler:
        dataset.apply_scaling(scaler)
    return dataset


def build_torch_model(model_name, dataset, seq_len, pred_len, checkpoint_path, device):
    ensure_training_project_on_path()
    from src.factory import build_model

    model = build_model(
        model_name=model_name,
        dataset=dataset,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    state = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_pickle_checkpoint(checkpoint_path):
    ensure_training_project_on_path()
    with open(checkpoint_path, "rb") as file:
        return pickle.load(file)


def _load_state_dict(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint
