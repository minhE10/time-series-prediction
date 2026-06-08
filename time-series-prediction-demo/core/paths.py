from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = DEMO_ROOT / "configs"
TRAINING_PROJECT_DIR = DEMO_ROOT / "vendor" / "training_project"


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (DEMO_ROOT / path).resolve()
