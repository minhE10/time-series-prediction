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
        "d_model": 32,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 64,
        "dropout": 0.3,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 32
    },

    "timemixer": {
        "d_model": 32,
        "d_ff": 64,
        "e_layers": 2,
        "dropout": 0.3,
        "down_sampling_layers": 2,
        "down_sampling_window": 2,
        "down_sampling_method": "avg",
        "decomp_method": "moving_avg",
        "moving_avg": 6,
        "top_k": 5,
        "channel_independence": False,
        "use_norm": True,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 32
    },

    "tsmamba": {
        "d_model": 32,
        "d_state": 8,
        "d_conv": 4,
        "expand": 2,
        "n_layers": 1,
        "patch_len": 8,
        "dropout": 0.3,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 32
    }
}