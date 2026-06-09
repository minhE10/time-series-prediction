import copy
import numpy as np
import torch
import torch.nn as nn

from evaluate import compute_metrics_per_target, print_metrics


class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 train_loader,
                 val_loader,
                 test_loader,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 device="cpu",
                 grad_clip=1.0,
                 selection_loader=None,
                 selection_name="Val"):
        self.model = model.to(device)
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.selection_loader = selection_loader if selection_loader is not None else val_loader
        self.selection_name = selection_name
        self.best_state = None
        self.history = {"train_loss": [], "val_loss": [], "selection_name": selection_name}

    def _train_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X), y)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            bs = X.size(0); total += loss.item() * bs; n += bs
        return total / n

    @torch.no_grad()
    def _eval_loss(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            loss = self.criterion(self.model(X), y)
            bs = X.size(0); total += loss.item() * bs; n += bs
        return total / n

    @torch.no_grad()
    def predict(self, loader=None):
        loader = loader or self.test_loader
        self.model.eval()
        preds, trues = [], []
        for X, y in loader:
            preds.append(self.model(X.to(self.device)).cpu())
            trues.append(y)
        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(trues).numpy()
        if self.dataset.is_scaled:
            y_pred = self.dataset.inverse_transform_target(y_pred)
            y_true = self.dataset.inverse_transform_target(y_true)
        return y_true, y_pred

    def fit(self, num_epochs=200, verbose=True, log_every=10):
        best_selection = float("inf")
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch()
            selection_loss = self._eval_loss(self.selection_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(selection_loss)
                else:
                    self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(selection_loss)

            if selection_loss < best_selection:
                best_selection = selection_loss
                self.best_state = copy.deepcopy(self.model.state_dict())

            if verbose and epoch % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch [{epoch:03d}/{num_epochs}] "
                      f"Train: {train_loss:.6f} | {self.selection_name}: {selection_loss:.6f} | "
                      f"Best {self.selection_name}: {best_selection:.6f} | lr: {lr:.2e}")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        print(f"Done. Best {self.selection_name} loss: {best_selection:.6f}")
        return self.history

    def evaluate(self, loader=None, scaled=False, title="TEST"):
        y_true, y_pred = self.predict(loader)
        result = compute_metrics_per_target(y_true, y_pred, self.dataset.target_cols, scaled=scaled)
        print_metrics(result, title=title)
        return result


class XGBoostTrainer:
    def __init__(self, model_template, dataset, train_loader, val_loader, test_loader):
        self.template = model_template
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.models = []
        self.history = {"train_loss": [], "val_loss": []}

    def fit(self, num_epochs=200, verbose=True, log_every=10):
        from src.models.xgboost import XGBoost, loader_to_numpy

        X_tr,  y_tr  = loader_to_numpy(self.train_loader)  
        X_val, y_val = loader_to_numpy(self.val_loader)

        # Predict the last step of the pred window
        y_tr_1d  = y_tr[:, -1, :]  
        y_val_1d = y_val[:, -1, :]

        histories, self.models = [], []
        for i, col in enumerate(self.dataset.target_cols):
            if verbose:
                print(f"  [{i+1}/{self.dataset.n_targets}] Fitting XGBoost for '{col}' ...")
            t = self.template
            m = XGBoost(
                learning_rate=t.learning_rate, max_depth=t.max_depth,
                min_child_weight=t.min_child_weight, subsample=t.subsample,
                colsample=t.colsample, reg_lambda=t.reg_lambda, reg_gamma=t.reg_gamma,
            ).fit(X_tr, y_tr_1d[:, i], n_estimators=num_epochs,
                  X_val=X_val, y_val=y_val_1d[:, i], log_every=log_every)
            self.models.append(m)
            histories.append(m.history)

        n_pts = len(histories[0]["train_loss"])
        self.history = {
            "train_loss": [np.mean([h["train_loss"][j] for h in histories]) for j in range(n_pts)],
            "val_loss": [np.mean([h["val_loss"][j]   for h in histories]) for j in range(n_pts)],
        }
        best_val = min(self.history["val_loss"])
        print(f"Best val MSE: {best_val:.6f}")
        return self.history

    def predict(self, loader=None):
        from src.models.xgboost import loader_to_numpy
        ldr = loader or self.test_loader
        X, y_true_raw = loader_to_numpy(ldr)                  
        preds = np.column_stack([m.predict(X) for m in self.models]) 
        y_true = y_true_raw[:, -1, :]                               

        if self.dataset.is_scaled:
            preds = self.dataset.inverse_transform_target(preds[:, np.newaxis, :])[:, 0, :]
            y_true = self.dataset.inverse_transform_target(y_true[:, np.newaxis, :])[:, 0, :]
        return y_true, preds

    def evaluate(self, loader=None, scaled=False, title="XGBOOST TEST"):
        y_true, y_pred = self.predict(loader)
        result = compute_metrics_per_target(y_true, y_pred, self.dataset.target_cols, scaled=scaled)
        print_metrics(result, title=title)
        return result


class ARIMATrainer:
    def __init__(self, dataset, train_loader, val_loader, test_loader, p=3, d=0, q=3):
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.p = p; self.d = d; self.q = q
        self.forecaster = None
        self.history = {"train_loss": [], "val_loss": []}

    def fit(self, num_epochs=None, verbose=True, log_every=None):
        from src.models.arima import ARIMAForecaster
        self.forecaster = ARIMAForecaster(
            dataset=self.dataset, p=self.p, d=self.d, q=self.q
        )
        self.forecaster.fit_all(verbose=verbose)

        y_true_val, y_pred_val = self.forecaster.predict_rolling(
            self.dataset.X_val, self.dataset.y_val
        )
        val_loss = float(np.mean((y_true_val - y_pred_val) ** 2))
        self.history = {"train_loss": [val_loss], "val_loss": [val_loss]}
        print(f"Done. Val MSE: {val_loss:.6f}")
        return self.history

    def predict(self, loader=None):
        return self.forecaster.predict_rolling(self.dataset.X_test, self.dataset.y_test)

    def evaluate(self, loader=None, scaled=False, title="ARIMA TEST"):
        y_true, y_pred = self.predict()
        result = compute_metrics_per_target(y_true, y_pred, self.dataset.target_cols, scaled=scaled)
        print_metrics(result, title=title)
        return result
