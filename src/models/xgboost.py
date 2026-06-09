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
            self._find_split(g, h)
        if self.left is None:
            self.is_leaf = True

    def _find_split(self, g, h):
        # h = 2.0 per sample (MSE), so H_cumsum[i] = 2*(i+1) — no cumsum needed for H
        n = len(g)
        G_total = g.sum()
        H_total = 2.0 * n
        lam = self.reg_lambda
        base = G_total**2 / (H_total + lam)

        n_feat = self.X.shape[1]
        feats = np.random.choice(n_feat, max(1, int(n_feat * self.colsample)), replace=False)
        n_try = len(feats)

        # Sort all sampled features simultaneously — one argsort call instead of n_try
        X_local  = self.X[self.idxs][:, feats]        # (n, n_try)
        orders   = np.argsort(X_local, axis=0)         # (n, n_try)
        col_idx  = np.arange(n_try)
        X_sorted = X_local[orders, col_idx]            # (n, n_try)

        # Cumulative GL for all features at once; GR by complement
        GL = np.cumsum(g[orders], axis=0)[:-1]         # (n-1, n_try)
        GR = G_total - GL

        # HL/HR from constant H=2.0 — analytical, no cumsum
        counts = np.arange(1, n, dtype=np.float64)
        HL = (2.0 * counts)[:, None]                   # (n-1, 1)  broadcasts over n_try
        HR = (2.0 * (n - counts))[:, None]

        # Mask: split must change value AND satisfy min_child_weight
        ok = (X_sorted[:-1] != X_sorted[1:])           # (n-1, n_try)
        mw = self.min_child_weight
        ok &= (HL >= mw) & (HR >= mw)

        if not ok.any():
            return

        gains = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - base) - self.reg_gamma
        gains[~ok] = -np.inf

        best_flat = int(np.argmax(gains))
        row, ci = divmod(best_flat, n_try)
        if gains[row, ci] <= 0.0:
            return

        self.split_col = int(feats[ci])
        self.split_val = float(X_sorted[row, ci])
        x_col = self.X[self.idxs, self.split_col]
        lhs = x_col <= self.split_val
        kw = dict(gradient=self.gradient, hessian=self.hessian, X=self.X,
                  max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                  reg_lambda=self.reg_lambda, reg_gamma=self.reg_gamma, colsample=self.colsample)
        self.left  = _Node(self.depth + 1, self.idxs[lhs],  **kw)
        self.right = _Node(self.depth + 1, self.idxs[~lhs], **kw)


class _XGBoostTree:
    def fit(self, X, gradient, hessian, max_depth, min_child_weight, reg_lambda, reg_gamma, colsample):
        root = _Node(0, np.arange(len(gradient)), gradient, hessian, X,
                     max_depth, min_child_weight, reg_lambda, reg_gamma, colsample)
        # Flatten recursive tree into parallel arrays for O(depth) vectorized predict
        feat, thr, val, left, right, is_leaf = [], [], [], [], [], []

        def _build(node):
            idx = len(feat)
            is_leaf.append(node.is_leaf)
            val.append(node.leaf_value)
            if node.is_leaf:
                feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1)
            else:
                feat.append(node.split_col); thr.append(node.split_val)
                left.append(-1); right.append(-1)  # filled in below
                left[idx]  = _build(node.left)
                right[idx] = _build(node.right)
            return idx

        _build(root)
        self._feat    = np.array(feat,    dtype=np.int32)
        self._thr     = np.array(thr,     dtype=np.float64)
        self._val     = np.array(val,     dtype=np.float64)
        self._left    = np.array(left,    dtype=np.int32)
        self._right   = np.array(right,   dtype=np.int32)
        self._is_leaf = np.array(is_leaf, dtype=bool)
        return self

    def predict(self, X):
        n = len(X)
        nodes = np.zeros(n, dtype=np.int32)
        # walk all rows simultaneously; each iteration = one tree level
        while True:
            at_leaf = self._is_leaf[nodes]
            if at_leaf.all():
                break
            active = np.where(~at_leaf)[0]
            ni = nodes[active]
            # fancy indexing: each active row picks its own split feature
            go_left = X[active, self._feat[ni]] <= self._thr[ni]
            nodes[active[go_left]]  = self._left[ni[go_left]]
            nodes[active[~go_left]] = self._right[ni[~go_left]]
        return self._val[nodes]


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

        pred_tr  = np.full(n, self.base_pred)
        pred_val = np.full(len(y_val), self.base_pred) if y_val is not None else None

        self.trees   = []
        self.history = {"train_loss": [], "val_loss": []}

        for i in range(n_estimators):
            g = 2.0 * (pred_tr - y_train)
            h = np.full(n, 2.0)

            idx = (np.random.choice(n, int(n * self.subsample), replace=False)
                   if self.subsample < 1.0 else np.arange(n))

            tree = _XGBoostTree().fit(
                X_train[idx], g[idx], h[idx],
                max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda, reg_gamma=self.reg_gamma, colsample=self.colsample,
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
