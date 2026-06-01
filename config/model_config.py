MODEL_CONFIG = {
    "arima": {
        "p": 3,
        "d": 0,
        "q": 3
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample": 0.8,
        "reg_lambda": 1.0,
        "reg_gamma": 0.0
    },
    "itransformer": {
        "d_model": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32
    },
    "timemixer": {
        "d_model": 64,
        "d_ff": 128,
        "e_layers": 2,
        "dropout": 0.1,
        "down_sampling_layers": 2,
        "down_sampling_window": 2,
        "down_sampling_method": "avg",
        "decomp_method": "moving_avg",
        "moving_avg": 25,
        "top_k": 5,
        "channel_independence": False,
        "use_norm": True,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32
    },
    "tsmamba": {
        "d_model": 64,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "n_layers": 2,
        "patch_len": 8,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32
    }
}