import numpy as np


class _Node:
    def __init__(self, depth, idxs, gradient, hessian, X,
                 max_depth, min_child_weight, reg_lambda, reg_gamma, colsample):
        self.depth = depth
        self.idxs = idxs
        self.gradient = gradient
        self.hessian = hessian
        self.X = X
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.colsample = colsample

        self.is_leaf = False
        self.left = self.right = None
        self.split_col = self.split_val = None

        g, h = gradient[idxs], hessian[idxs]
        self.leaf_value = -g.sum() / (h.sum() + reg_lambda)

        if depth < max_depth and len(idxs) >= 2 * min_child_weight:
            self._find_split()
        if self.left is None:
            self.is_leaf = True

    def _gain(self, lhs, rhs):
        g, h = self.gradient[self.idxs], self.hessian[self.idxs]
        GL, HL = g[lhs].sum(), h[lhs].sum()
        GR, HR = g[rhs].sum(), h[rhs].sum()
        lam = self.reg_lambda
        return (
            GL**2 / (HL + lam) + GR**2 / (HR + lam)
            - (GL + GR)**2 / (HL + HR + lam)
        ) * 0.5 - self.reg_gamma

    def _find_split(self):
        n_feat = self.X.shape[1]
        n_try = max(1, int(n_feat * self.colsample))
        feats = np.random.choice(n_feat, n_try, replace=False)

        best_gain = 0.0
        best_col = best_val = best_lhs = best_rhs = None

        for col in feats:
            x_col = self.X[self.idxs, col]
            h_local = self.hessian[self.idxs]
            for thr in np.unique(x_col)[:-1]:
                lhs = x_col <= thr
                rhs = ~lhs
                if h_local[lhs].sum() < self.min_child_weight:
                    continue
                if h_local[rhs].sum() < self.min_child_weight:
                    continue
                gain = self._gain(lhs, rhs)
                if gain > best_gain:
                    best_gain, best_col, best_val = gain, col, thr
                    best_lhs, best_rhs = lhs, rhs

        if best_col is None:
            return

        self.split_col = best_col
        self.split_val = best_val
        kw = dict(
            gradient=self.gradient, hessian=self.hessian, X=self.X,
            max_depth=self.max_depth, min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda, reg_gamma=self.reg_gamma, colsample=self.colsample,
        )
        self.left = _Node(self.depth + 1, self.idxs[best_lhs], **kw)
        self.right = _Node(self.depth + 1, self.idxs[best_rhs], **kw)

    def predict_row(self, x):
        if self.is_leaf:
            return self.leaf_value
        return (self.left if x[self.split_col] <= self.split_val else self.right).predict_row(x)


class _XGBoostTree:
    def fit(self, X, gradient, hessian, max_depth, min_child_weight, reg_lambda, reg_gamma, colsample):
        self.root = _Node(0, np.arange(len(gradient)), gradient, hessian, X,
                          max_depth, min_child_weight, reg_lambda, reg_gamma, colsample)
        return self

    def predict(self, X):
        return np.array([self.root.predict_row(row) for row in X])


class XGBoost:
    def __init__(self, learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.8,
                 colsample=0.8, reg_lambda=1.0, reg_gamma=0.0, random_state=42):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample = colsample
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.random_state = random_state
        self.trees = []
        self.base_pred = None
        self.history = {"train_loss": [], "val_loss": []}

    def fit(self, X_train, y_train, n_estimators=200, X_val=None, y_val=None, log_every=10):
        np.random.seed(self.random_state)
        n = len(y_train)
        self.base_pred = float(y_train.mean())

        pred_tr = np.full(n, self.base_pred)
        pred_val = np.full(len(y_val), self.base_pred) if y_val is not None else None

        self.trees = []
        self.history = {"train_loss": [], "val_loss": []}

        for i in range(n_estimators):
            g = 2.0 * (pred_tr - y_train)
            h = np.full(n, 2.0)

            idx = (
                np.random.choice(n, int(n * self.subsample), replace=False)
                if self.subsample < 1.0 else np.arange(n)
            )

            tree = _XGBoostTree().fit(
                X_train[idx], g[idx], h[idx],
                max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda, reg_gamma=self.reg_gamma,
                colsample=self.colsample,
            )
            pred_tr += self.learning_rate * tree.predict(X_train)
            if pred_val is not None:
                pred_val += self.learning_rate * tree.predict(X_val)

            self.trees.append(tree)

            if (i + 1) % log_every == 0 or i == n_estimators - 1:
                self.history["train_loss"].append(float(np.mean((pred_tr - y_train) ** 2)))
                if pred_val is not None:
                    self.history["val_loss"].append(float(np.mean((pred_val - y_val) ** 2)))

        return self

    def predict(self, X):
        pred = np.full(len(X), self.base_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


def loader_to_numpy(loader):
    Xs, ys = [], []
    for X_b, y_b in loader:
        Xs.append(X_b.numpy().reshape(X_b.size(0), -1))
        ys.append(y_b.numpy())
    return np.vstack(Xs), np.concatenate(ys, axis=0)