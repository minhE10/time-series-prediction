import matplotlib.pyplot as plt


def forecast_figure(result_df, target_col):
    subset = result_df[result_df["target"] == target_col]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = subset["timestamp"] if "timestamp" in subset else subset["step"]
    ax.plot(x, subset["actual"], marker="o", linewidth=1.8, label="Actual")
    ax.plot(x, subset["predicted"], marker="o", linewidth=1.8, label="Predicted")
    ax.set_title(target_col)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
