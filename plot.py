import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

if sns is not None:
    sns.set_theme(style="whitegrid")


def plot_loss(history, title="Training & Validation Loss", save_path=None):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    selection_label = history.get("selection_name", "Val")
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=1.5)
    ax.plot(epochs, val_loss, label=f"{selection_label} Loss", color="#ff7f0e", linewidth=1.5)
    best_epoch = int(np.argmin(val_loss)) + 1
    ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7, label=f"Best {selection_label} @ epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_predictions(y_true,
                     y_pred,
                     target_cols,
                     n_samples=500,
                     title="Predictions vs Ground Truth",
                     save_path=None,
                     mode="non_overlapping"):
    y_true = _prediction_timeline(y_true, len(target_cols), mode=mode)
    y_pred = _prediction_timeline(y_pred, len(target_cols), mode=mode)

    n = min(n_samples, len(y_true))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    x = np.arange(n)

    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 4 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        ax.plot(x, y_true[:, i], label="Ground Truth", color="#1f77b4", linewidth=1.2, alpha=0.8)
        ax.plot(x, y_pred[:, i], label="Prediction", color="#d62728", linewidth=1.2, alpha=0.8, linestyle="--")
        ax.set_ylabel(col)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def _prediction_timeline(values, n_targets, mode="non_overlapping"):
    values = np.asarray(values)
    if values.ndim == 2:
        return values.reshape(-1, n_targets)
    if values.ndim != 3:
        return values.reshape(-1, n_targets)

    if mode == "flatten":
        return values.reshape(-1, n_targets)
    if mode == "last_step":
        return values[:, -1, :]
    if mode == "first_step":
        return values[:, 0, :]
    if mode == "non_overlapping":
        pred_len = values.shape[1]
        return values[::pred_len].reshape(-1, n_targets)
    raise ValueError("mode must be one of: non_overlapping, last_step, first_step, flatten")


def plot_predictions_from_trainer(trainer, loader=None, n_samples=500, title=None, save_path=None,
                                  mode="non_overlapping"):
    if loader is None:
        loader = trainer.test_loader
    y_true, y_pred = trainer.predict(loader)
    plot_title = title or f"Predictions vs Ground Truth ({trainer.dataset.name})"
    plot_predictions(y_true, y_pred, trainer.dataset.target_cols, n_samples=n_samples,
                     title=plot_title, save_path=save_path, mode=mode)
