from pathlib import Path

from ui.ui import build_app

WEB_DIR = (Path(__file__).parent / "ui" / "web").resolve()


if __name__ == "__main__":
    app = build_app()
    app.launch(
        share=True,
        allowed_paths=[str(WEB_DIR)],  # required: lets /file= serve demo/web/*
    )
