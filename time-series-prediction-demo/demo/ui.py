import gradio as gr

from core.plotting import forecast_figure
from demo.components.selectors import first_value, target_choices
from demo.state import create_demo_state


CSS = """
.app-shell {max-width: 1280px; margin: 0 auto;}
.toolbar button {min-height: 40px;}
"""


def build_app():
    registry, service = create_demo_state()

    model_choices = registry.model_choices()
    default_model = first_value(model_choices)
    dataset_choices = registry.dataset_choices(model=default_model)
    default_dataset = first_value(dataset_choices)
    horizon_choices = registry.horizon_choices(
        model=default_model,
        dataset=default_dataset,
        include_defaults=True,
    )
    default_horizon = horizon_choices[0] if horizon_choices else None
    default_targets = target_choices(registry, default_dataset)
    default_target = default_targets[0] if default_targets else None

    with gr.Blocks(title="Time Series Prediction Demo") as app:
        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown("# Time Series Prediction Demo")

            with gr.Row():
                model = gr.Dropdown(
                    label="Model",
                    choices=model_choices,
                    value=default_model,
                    interactive=True,
                )
                dataset = gr.Dropdown(
                    label="Dataset",
                    choices=dataset_choices,
                    value=default_dataset,
                    interactive=True,
                )
                horizon = gr.Dropdown(
                    label="Horizon",
                    choices=horizon_choices,
                    value=default_horizon,
                    interactive=True,
                )
                target = gr.Dropdown(
                    label="Target",
                    choices=default_targets,
                    value=default_target,
                    interactive=True,
                )

            with gr.Row():
                sample_index = gr.Number(
                    label="Test Window Index",
                    value=-1,
                    precision=0,
                    interactive=True,
                )
                seq_len = gr.Number(
                    label="Sequence Length",
                    value=48,
                    precision=0,
                    interactive=True,
                )
                scaler = gr.Dropdown(
                    label="Scaler",
                    choices=["standard", "minmax", "none"],
                    value="standard",
                    interactive=True,
                )

            with gr.Row():
                csv_file = gr.File(
                    label="CSV Override",
                    file_types=[".csv"],
                    type="filepath",
                )
                weight_file = gr.File(
                    label="Weight Override",
                    file_types=[".pt", ".pkl"],
                    type="filepath",
                )
                run = gr.Button("Predict", variant="primary", elem_classes=["toolbar"])

            status = gr.Markdown()
            plot = gr.Plot(label="Forecast")
            with gr.Row():
                metrics = gr.Dataframe(label="Metrics", interactive=False)
                info = gr.JSON(label="Run Info")
            result_table = gr.Dataframe(label="Forecast Values", interactive=False)

        model.change(
            fn=lambda model_name: _on_model_change(registry, model_name),
            inputs=model,
            outputs=[dataset, horizon, target],
        )
        dataset.change(
            fn=lambda model_name, dataset_name: _on_dataset_change(registry, model_name, dataset_name),
            inputs=[model, dataset],
            outputs=[horizon, target],
        )
        horizon.change(
            fn=lambda dataset_name: _target_update(registry, dataset_name),
            inputs=dataset,
            outputs=target,
        )
        run.click(
            fn=lambda m, d, h, t, i, s, sc, c, w: _predict(service, m, d, h, t, i, s, sc, c, w),
            inputs=[model, dataset, horizon, target, sample_index, seq_len, scaler, csv_file, weight_file],
            outputs=[plot, result_table, metrics, info, status],
        )

    return app


def _on_model_change(registry, model_name):
    datasets = registry.dataset_choices(model=model_name)
    dataset_value = first_value(datasets)
    horizons = registry.horizon_choices(model=model_name, dataset=dataset_value, include_defaults=True)
    horizon_value = horizons[0] if horizons else None
    targets = target_choices(registry, dataset_value)
    target_value = targets[0] if targets else None
    return (
        gr.update(choices=datasets, value=dataset_value),
        gr.update(choices=horizons, value=horizon_value),
        gr.update(choices=targets, value=target_value),
    )


def _on_dataset_change(registry, model_name, dataset_name):
    horizons = registry.horizon_choices(model=model_name, dataset=dataset_name, include_defaults=True)
    horizon_value = horizons[0] if horizons else None
    targets = target_choices(registry, dataset_name)
    target_value = targets[0] if targets else None
    return (
        gr.update(choices=horizons, value=horizon_value),
        gr.update(choices=targets, value=target_value),
    )


def _target_update(registry, dataset_name):
    targets = target_choices(registry, dataset_name)
    return gr.update(choices=targets, value=targets[0] if targets else None)


def _predict(service, model, dataset, horizon, target, sample_index, seq_len, scaler, csv_file, weight_file):
    if not model or not dataset or not horizon:
        raise gr.Error("Model, dataset, and horizon are required.")
    if not seq_len or int(seq_len) <= 0:
        raise gr.Error("Sequence Length must be a positive integer.")

    csv_path = _file_path(csv_file)
    checkpoint_path = _file_path(weight_file)
    try:
        result_df, metrics_df, run_info = service.predict(
            model=model,
            dataset_name=dataset,
            pred_len=int(horizon),
            target_col=target,
            sample_index=sample_index,
            csv_path=csv_path,
            checkpoint_path=checkpoint_path,
            seq_len=int(seq_len),
            scaler=scaler,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    selected_target = target if target in run_info["targets"] else run_info["targets"][0]
    fig = forecast_figure(result_df, selected_target)
    status = (
        f"Loaded `{run_info['artifact_id']}` from `{run_info['checkpoint_source']}` on `{run_info['device']}`. "
        f"Rows: `{run_info['dataset_rows']}`, test windows: `{run_info['test_windows']}`."
    )
    return fig, result_df, metrics_df, run_info, status


def _file_path(file_value):
    if not file_value:
        return None
    if isinstance(file_value, str):
        return file_value
    return getattr(file_value, "name", None)
