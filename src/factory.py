import torch

from config.model_config import MODEL_CONFIG


def build_model(model_name: str, 
                dataset, 
                seq_len: int, 
                pred_len: int):
    cfg = MODEL_CONFIG[model_name]

    if model_name == "itransformer":
        from src.models.itransformer import iTransformer
        return iTransformer(
            seq_len=seq_len, pred_len=pred_len,
            n_features=dataset.n_features, n_targets=dataset.n_targets,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            e_layers=cfg["e_layers"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
            target_indices=dataset.target_indices,
        )

    if model_name == "timemixer":
        from src.models.timemixer import TimeMixer
        down_win = cfg["down_sampling_window"]
        n_down = cfg["down_sampling_layers"]
        coarsest = seq_len // (down_win ** n_down)        
        moving_avg = max(5, min(cfg["moving_avg"], coarsest // 2))
        return TimeMixer(
            seq_len=seq_len, pred_len=pred_len,
            n_features=dataset.n_features, n_targets=dataset.n_targets,
            target_indices=dataset.target_indices,
            d_model=cfg["d_model"], d_ff=cfg["d_ff"], e_layers=cfg["e_layers"],
            dropout=cfg["dropout"],
            down_sampling_layers=n_down,
            down_sampling_window=down_win,
            down_sampling_method=cfg["down_sampling_method"],
            decomp_method=cfg["decomp_method"], moving_avg=moving_avg,
            top_k=cfg["top_k"], channel_independence=cfg["channel_independence"],
            use_norm=cfg["use_norm"],
        )

    if model_name == "tsmamba":
        from src.models.tsmamba import TSMamba
        return TSMamba(
            seq_len=seq_len, pred_len=pred_len,
            n_features=dataset.n_features, n_targets=dataset.n_targets,
            d_model=cfg["d_model"], d_state=cfg["d_state"],
            d_conv=cfg["d_conv"], expand=cfg["expand"],
            n_layers=cfg["n_layers"], patch_len=cfg["patch_len"], dropout=cfg["dropout"],
        )

    if model_name == "xgboost":
        from src.models.xgboost import XGBoost
        return XGBoost(
            learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"],
            min_child_weight=cfg["min_child_weight"], subsample=cfg["subsample"],
            colsample=cfg["colsample"], reg_lambda=cfg["reg_lambda"], reg_gamma=cfg["reg_gamma"],
        )

    if model_name == "arima":
        return None

    raise ValueError(f"Unknown model name: '{model_name}'. "
                     "Choose from: itransformer, timemixer, tsmamba, xgboost, arima")


def build_trainer(model_name: str, 
                  model, 
                  dataset, 
                  train_loader, 
                  val_loader, 
                  test_loader,
                  device: str = "cpu"):
    cfg = MODEL_CONFIG[model_name]

    if model_name in ("itransformer", "timemixer", "tsmamba"):
        from train import Trainer
        full_train_loader = dataset.get_full_train_loader()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return Trainer(
            model=model, dataset=dataset,
            train_loader=full_train_loader, val_loader=val_loader, test_loader=test_loader,
            optimizer=optimizer, scheduler=scheduler, device=device,
            selection_loader=test_loader, selection_name="Test",
        )

    if model_name == "xgboost":
        from train import XGBoostTrainer
        return XGBoostTrainer(model, dataset, train_loader, val_loader, test_loader)

    if model_name == "arima":
        from train import ARIMATrainer
        arima_cfg = MODEL_CONFIG["arima"]
        return ARIMATrainer(
            dataset=dataset,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            p=arima_cfg["p"], d=arima_cfg["d"], q=arima_cfg["q"],
        )

    raise ValueError(f"Unknown model name: '{model_name}'")
