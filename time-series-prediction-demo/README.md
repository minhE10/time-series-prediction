# Time Series Prediction Demo

Gradio demo for trained time-series forecasting checkpoints. The app is self-contained and config-driven, so new datasets and checkpoints can be added mostly by editing YAML files instead of changing UI code.

## Project Structure

```text
time-series-prediction-demo/
|-- app.py
|-- configs/
|   |-- artifacts.yaml
|   |-- datasets.yaml
|   `-- models.yaml
|-- data/
|   `-- raw/
|-- artifacts/
|   `-- checkpoints/
|-- core/
|   |-- inference.py
|   |-- registry.py
|   |-- plotting.py
|   `-- metrics.py
|-- adapters/
|   `-- training_project.py
|-- vendor/
|   `-- training_project/
`-- demo/
    `-- ui.py
```

Registered checkpoint files and model/data-loader code are inside this folder. Full dataset CSV files are intentionally not committed because they are too large.

## Install

Open PowerShell in this folder:

```powershell
cd F:\HUST\20252\TKUD\time-series-prediction-demo
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Share With Teammates

You can send the whole `time-series-prediction-demo` folder to another machine. The folder already contains:

- Gradio app code.
- Vendored model/data-loader code in `vendor/training_project`.
- Registered checkpoint files in `artifacts/checkpoints`.
- Config files in `configs`.

Full CSV files are not included in Git. On another machine, your teammate should either:

- Copy the required CSV files into `data/raw` with the names used by `configs/datasets.yaml`.
- Or upload a compatible CSV through `CSV Override` in the UI.

Then install dependencies and run `python app.py` from inside this folder.

Expected default CSV paths:

```text
data/raw/appliances_energy_dataset.csv
data/raw/beijing_air_quality_dataset.csv
data/raw/bitcoin_dataset.csv
data/raw/hanoi_air_quality_dataset.csv
data/raw/sunspots_dataset.csv
```

## Run The Demo

Start Gradio:

```powershell
python app.py
```

Then open this URL in your browser:

```text
http://127.0.0.1:7860
```

To stop the demo, return to the terminal and press:

```text
Ctrl + C
```

## How To Use The UI

1. Select `Model`.
   - Currently, the registered trained checkpoints use `iTransformer`.

2. Select `Dataset`.
   - Available datasets include `Sunspots`, `Appliances Energy`, `Beijing Air Quality`, `Bitcoin`, and `Hanoi Air Quality`.

3. Select `Horizon`.
   - This is the number of future time steps to predict.
   - Current horizons are `12`, `24`, and `48`, depending on which checkpoints exist for the selected dataset.

4. Select `Target`.
   - For single-target datasets such as `Bitcoin` or `Sunspots`, there is only one target.
   - For multi-target datasets such as air quality, choose a column such as `PM2.5`, `PM10`, `CO`, or `O3`.

5. Set `Test Window Index`.
   - Use `-1` to predict the last available test window.
   - Use `0` for the first test window.
   - Use any positive integer to inspect another test window.
   - If the number is too large, the app clamps it to the last valid window.

6. Set `Sequence Length`.
   - Use the same `seq_len` value that was used when training the checkpoint.
   - The current registered checkpoints were trained with `seq_len = 48`.

7. Select `Scaler`.
   - Use the same scaler that was used during training.
   - The current registered checkpoints use `standard`.
   - Use `none` only if the model was trained without scaling.

8. Optionally upload a CSV in `CSV Override`.
   - The uploaded CSV must have the same schema as the selected dataset.
   - For example, if you choose `Beijing Air Quality`, the CSV must contain the same required columns used by that dataset config.
   - Leave this empty to use the default dataset file registered in `configs/datasets.yaml`.
   - If the default CSV has not been copied into `data/raw`, prediction will ask you to upload or place the file manually.

9. Optionally upload a checkpoint in `Weight Override`.
   - Use this when your group has a new `.pt` or `.pkl` checkpoint that is not yet registered in `configs/artifacts.yaml`.
   - Use `.pt` for `iTransformer`, `TimeMixer`, and `TSMamba`.
   - Use `.pkl` for `XGBoost` and `ARIMA`.
   - Select the correct `Model`, `Dataset`, `Horizon`, `Sequence Length`, and `Scaler` before clicking `Predict`.
   - The uploaded checkpoint architecture must match the selected model and settings.

10. Click `Predict`.

The app will show:

- A forecast chart for the selected target.
- A table of actual and predicted values.
- Metrics for the selected target: `MAE`, `RMSE`, `WMAPE`, and `R2`.
- Run information, including artifact id, checkpoint path, row count, test window count, and device.

## Add A New Checkpoint

1. Put the checkpoint under `artifacts/checkpoints/<model>/<dataset>/` or keep it anywhere accessible.

2. Add one entry to `configs/artifacts.yaml`.

Example:

```yaml
- id: itransformer_beijing_air_quality_48
  model: itransformer
  dataset: beijing_air_quality
  pred_len: 48
  seq_len: 48
  scaler: standard
  checkpoint: artifacts/checkpoints/itransformer/beijing_air_quality/pred_48.pt
```

3. Make sure these fields are correct:

- `id`: unique name for this trained artifact.
- `model`: model key from `configs/models.yaml`.
- `dataset`: dataset key from `configs/datasets.yaml`.
- `pred_len`: forecast horizon used during training.
- `seq_len`: input sequence length used during training.
- `scaler`: scaler used during training, usually `standard`.
- `checkpoint`: path to the checkpoint, usually `.pt` for PyTorch models or `.pkl` for XGBoost/ARIMA.

4. Restart the app so Gradio reloads the updated registry.

## Use An Unregistered Weight

Use `Weight Override` in the UI when you want to test a checkpoint without editing YAML.

1. Select the model architecture, for example `iTransformer`, `TimeMixer`, `TSMamba`, `XGBoost`, or `ARIMA`.
2. Select the dataset.
3. Select the horizon used during training.
4. Set `Sequence Length` to the training `seq_len`.
5. Set `Scaler` to the training scaler.
6. Upload the checkpoint file in `Weight Override`.
   - Use `.pt` for `iTransformer`, `TimeMixer`, and `TSMamba`.
   - Use `.pkl` for `XGBoost` and `ARIMA`.
7. Click `Predict`.

This mode is useful for quick team testing. For stable sharing, register the checkpoint in `configs/artifacts.yaml`.

## Add A New Model Architecture

1. Make sure the model implementation exists in `vendor/training_project`.

2. Add the model to `configs/models.yaml`.

Example:

```yaml
new_model:
  display_name: New Model
  framework: torch
  adapter: training_project
```

3. Add a compatible checkpoint to `configs/artifacts.yaml`, or upload its checkpoint file through `Weight Override`.

## Add A New Dataset

1. Put the CSV under `data/raw/` or keep it anywhere accessible.

2. Add one entry to `configs/datasets.yaml`.

Example:

```yaml
my_dataset:
  display_name: My Dataset
  path: data/raw/my_dataset.csv
  time_col: datetime
  feature_cols:
    - value
    - hour_sin
    - hour_cos
  target_cols:
    - value
```

3. Register at least one compatible checkpoint in `configs/artifacts.yaml`.

4. Restart the app.

## Troubleshooting

If the browser cannot open the app, make sure the terminal still shows Gradio running and open:

```text
http://127.0.0.1:7860
```

If port `7860` is busy, Gradio may choose another port. Check the terminal output for the actual local URL.

If a checkpoint does not appear in the UI, check that it is registered in `configs/artifacts.yaml`, the checkpoint path exists, and the app has been restarted.

If prediction fails after uploading a CSV, make sure the uploaded file has the same columns and time format as the selected dataset.
