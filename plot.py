import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_loss(history, title="Training & Validation Loss", save_path=None):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=1.5)
    ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=1.5)
    best_epoch = int(np.argmin(val_loss)) + 1
    ax.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7, label=f"Best Val @ epoch {best_epoch}")
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
                     save_path=None):
    y_true = np.array(y_true).reshape(-1, len(target_cols))
    y_pred = np.array(y_pred).reshape(-1, len(target_cols))

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


def plot_predictions_from_trainer(trainer, loader=None, n_samples=500, title=None, save_path=None):
    if loader is None:
        loader = trainer.test_loader
    y_true, y_pred = trainer.predict(loader)
    plot_title = title or f"Predictions vs Ground Truth ({trainer.dataset.name})"
    plot_predictions(y_true, y_pred, trainer.dataset.target_cols, n_samples=n_samples,
                     title=plot_title, save_path=save_path)
