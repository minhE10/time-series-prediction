from core.registry import Registry


def test_registry_loads_current_itransformer_artifacts():
    registry = Registry.load()

    assert "itransformer" in registry.models
    assert "bitcoin" in registry.datasets
    assert registry.get_artifact("itransformer", "bitcoin", 12).seq_len == 48


def test_registered_checkpoint_files_exist():
    registry = Registry.load()

    for dataset in registry.datasets.values():
        assert str(dataset.path).endswith(".csv")

    for artifact in registry.artifacts.values():
        assert artifact.checkpoint.exists(), artifact.checkpoint
