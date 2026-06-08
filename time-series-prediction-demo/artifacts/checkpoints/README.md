# Checkpoint Layout

Put trained checkpoints under:

```text
artifacts/checkpoints/<model>/<dataset>/
```

Current folders:

```text
artifacts/checkpoints/
|-- arima/
|-- itransformer/
|-- timemixer/
|-- tsmamba/
`-- xgboost/
```

Recommended file names:

```text
pred_12.pt
pred_24.pt
pred_48.pt
```

For non-PyTorch models:

```text
pred_12.pkl
pred_24.pkl
pred_48.pkl
```

After adding a stable checkpoint, register it in `configs/artifacts.yaml`.

For quick testing of PyTorch `.pt` weights, you can also use `Weight Override` in the UI without editing YAML.
