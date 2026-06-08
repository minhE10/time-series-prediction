from dataclasses import dataclass
from pathlib import Path

import yaml

from core.paths import CONFIG_DIR, resolve_path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    path: Path
    time_col: str | list[str]
    feature_cols: list[str]
    target_cols: list[str]
    drop_cols: list[str]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    framework: str
    adapter: str
    config: dict


@dataclass(frozen=True)
class ArtifactSpec:
    id: str
    model: str
    dataset: str
    pred_len: int
    seq_len: int
    scaler: str
    checkpoint: Path


class Registry:
    DEFAULT_HORIZONS = [12, 24, 48]

    def __init__(self, datasets, models, artifacts):
        self.datasets = datasets
        self.models = models
        self.artifacts = artifacts

    @classmethod
    def load(cls, config_dir=CONFIG_DIR):
        config_dir = Path(config_dir)
        datasets_raw = _read_yaml(config_dir / "datasets.yaml")["datasets"]
        models_raw = _read_yaml(config_dir / "models.yaml")["models"]
        artifacts_raw = _read_yaml(config_dir / "artifacts.yaml")["artifacts"]

        datasets = {
            name: DatasetSpec(
                name=name,
                display_name=raw.get("display_name", name),
                path=resolve_path(raw["path"]),
                time_col=raw["time_col"],
                feature_cols=list(raw["feature_cols"]),
                target_cols=list(raw["target_cols"]),
                drop_cols=list(raw.get("drop_cols", [])),
            )
            for name, raw in datasets_raw.items()
        }

        models = {
            name: ModelSpec(
                name=name,
                display_name=raw.get("display_name", name),
                framework=raw.get("framework", ""),
                adapter=raw.get("adapter", ""),
                config=dict(raw.get("config", {})),
            )
            for name, raw in models_raw.items()
        }

        artifacts = {
            raw["id"]: ArtifactSpec(
                id=raw["id"],
                model=raw["model"],
                dataset=raw["dataset"],
                pred_len=int(raw["pred_len"]),
                seq_len=int(raw["seq_len"]),
                scaler=raw.get("scaler", "standard"),
                checkpoint=resolve_path(raw["checkpoint"]),
            )
            for raw in artifacts_raw
        }
        return cls(datasets=datasets, models=models, artifacts=artifacts)

    def model_choices(self, framework=None, registered_only=False):
        if registered_only:
            names = sorted({artifact.model for artifact in self.artifacts.values()})
        else:
            names = sorted(self.models)
        if framework:
            names = [name for name in names if self.models[name].framework == framework]
        return [(self.models[name].display_name, name) for name in names]

    def dataset_choices(self, model=None, registered_only=False):
        if registered_only:
            artifacts = self.find_artifacts(model=model)
            names = sorted({artifact.dataset for artifact in artifacts})
        else:
            names = sorted(self.datasets)
        return [(self.datasets[name].display_name, name) for name in names]

    def horizon_choices(self, model=None, dataset=None, include_defaults=False):
        artifacts = self.find_artifacts(model=model, dataset=dataset)
        horizons = sorted({artifact.pred_len for artifact in artifacts})
        if horizons:
            return horizons
        if include_defaults:
            return self.DEFAULT_HORIZONS.copy()
        return []

    def find_artifacts(self, model=None, dataset=None, pred_len=None):
        result = list(self.artifacts.values())
        if model:
            result = [artifact for artifact in result if artifact.model == model]
        if dataset:
            result = [artifact for artifact in result if artifact.dataset == dataset]
        if pred_len:
            result = [artifact for artifact in result if artifact.pred_len == int(pred_len)]
        return sorted(result, key=lambda a: (a.model, a.dataset, a.pred_len, a.id))

    def get_artifact(self, model, dataset, pred_len):
        artifacts = self.find_artifacts(model=model, dataset=dataset, pred_len=pred_len)
        if not artifacts:
            raise ValueError(f"No checkpoint registered for {model}/{dataset}/pred_len={pred_len}.")
        return artifacts[0]

    def get_optional_artifact(self, model, dataset, pred_len):
        artifacts = self.find_artifacts(model=model, dataset=dataset, pred_len=pred_len)
        return artifacts[0] if artifacts else None


def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
