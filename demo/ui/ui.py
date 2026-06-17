# Gradio shell that embeds the static React SPA (demo/web/) via an iframe.
# Gradio serves allowed files at /gradio_api/file=<abs-path>; every asset
# (data.js, charts.jsx, predictions.js, ...) resolves automatically from there.
from pathlib import Path

import gradio as gr

WEB_DIR = (Path(__file__).parent / "web").resolve()
INDEX = WEB_DIR / "index.html"

# Gradio 4.x serves allowed files at  /file=<abs-path>
# Gradio 5.x+ serves them at          /gradio_api/file=<abs-path>
FILE_ROUTE = "/gradio_api/file="

_CSS = """
html, body { margin: 0; padding: 0; background: #eef1f5 !important; overflow: hidden; }
.gradio-container, .contain, .gap, .block { padding: 0 !important; margin: 0 !important; background: transparent !important; border: none !important; box-shadow: none !important; }
footer { display: none !important; }
#tkud-frame { position: fixed; top: 0; left: 0; width: 100%; height: 100%; border: 0; z-index: 1000; }
"""


def build_app():
    src = f"{FILE_ROUTE}{INDEX.as_posix()}"
    iframe = f'<iframe id="tkud-frame" src="{src}" title="Time Series Prediction Demo"></iframe>'
    with gr.Blocks(title="Time Series Prediction Demo — Group 10", css=_CSS) as app:
        gr.HTML(iframe)
    return app