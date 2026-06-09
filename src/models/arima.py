import numpy as np
from itertools import product


def difference(series, d=1):
    y = np.array(series, dtype=float)
    for _ in range(d):
        y = np.diff(y)
    return y


def inverse_difference(diff_vals, history, d=1):
    if d == 0:
        return np.array(diff_vals)
    y = np.array(diff_vals, dtype=float)
    for _ in range(d):
        y = np.concatenate([[history[-1]], y]).cumsum()[1:]
    return y


def acf(series, max_lag):
    mu = series.mean()
    var = ((series - mu) ** 2).mean()
    if var == 0:
        return np.zeros(max_lag + 1)
    r = [1.0]
    for lag in range(1, max_lag + 1):
        cov = ((series[lag:] - mu) * (series[:-lag] - mu)).mean()
        r.append(cov / var)
    return np.array(r)


def adf_test(series, max_lag=1):
    y = np.array(series, dtype=float)
    n = len(y)
    dy = np.diff(y)
    y_lag = y[:-1]

    X = [np.ones(n - 1 - max_lag), y_lag[max_lag:]]
    for j in range(1, max_lag + 1):
        X.append(dy[max_lag - j: n - 1 - j])
    X = np.column_stack(X)
    Y = dy[max_lag:]

    try:
        coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ coef
        sigma2 = (resid ** 2).sum() / (len(Y) - X.shape[1])
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(XtX_inv))
        t_stat = coef[1] / (se[1] + 1e-10)
    except Exception:
        return 0.0, 1.0

    p_approx = 0.01 if t_stat < -3.43 else (0.05 if t_stat < -2.86 else 0.10 if t_stat < -2.57 else 0.50)
    return float(t_stat), p_approx


def auto_d(series, max_d=2):
    y = series.copy()
    for d in range(max_d + 1):
        _, p = adf_test(y)
        if p <= 0.05:
            return d
        y = np.diff(y)
    return max_d


def fit_ar(y, p):
    n = len(y)
    if p == 0:
        return np.array([]), y.copy()
    X = np.column_stack([y[p - i - 1: n - i - 1] for i in range(p)])
    Y = y[p:]
    phi, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ phi
    return phi, resid


def fit_ma(resid, q):
    n = len(resid)
    if q == 0:
        return np.array([])
    X = np.column_stack([resid[q - i - 1: n - i - 1] for i in range(q)])
    Y = resid[q:]
    theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return theta


def aic_bic(resid, n_params, n_obs):
    sigma2 = (resid ** 2).mean()
    if sigma2 <= 0:
        return np.inf, np.inf
    log_lik = -0.5 * n_obs * (np.log(2 * np.pi * sigma2) + 1)
    aic = -2 * log_lik + 2 * n_params
    bic = -2 * log_lik + n_params * np.log(n_obs)
    return aic, bic


class ARIMAModel:
    def __init__(self, p=1, d=0, q=0):
        self.p = p
        self.d = d
        self.q = q
        self.phi = None
        self.theta = None
        self._orig = None
        self._y_diff = None
        self._last_y = None
        self._last_resid = None

    def fit(self, series):
        y = np.array(series, dtype=float)
        self._orig = y.copy()

        y_diff = difference(y, self.d)
        self._y_diff = y_diff

        phi, resid_ar = fit_ar(y_diff, self.p)
        theta = fit_ma(resid_ar, self.q)

        self.phi = phi
        self.theta = theta

        p_eff = max(self.p, 1)
        q_eff = max(self.q, 1)
        self._last_y = list(y_diff[-p_eff:]) if self.p > 0 else []
        self._last_resid = list(resid_ar[-q_eff:]) if self.q > 0 else []
        return self

    def predict_one_step(self):
        ar_term = sum(self.phi[i] * self._last_y[-(i + 1)] for i in range(self.p))
        ma_term = sum(self.theta[j] * self._last_resid[-(j + 1)] for j in range(self.q))
        return ar_term + ma_term

    def forecast(self, steps=1):
        preds_diff = []
        last_y = list(self._last_y)
        last_resid = list(self._last_resid)

        for _ in range(steps):
            ar = sum(self.phi[i] * last_y[-(i + 1)] for i in range(self.p))
            ma = sum(self.theta[j] * last_resid[-(j + 1)] for j in range(self.q))
            yhat = ar + ma
            preds_diff.append(yhat)
            last_y.append(yhat)
            last_resid.append(0.0)

        return inverse_difference(np.array(preds_diff), self._orig, self.d)

    @staticmethod
    def grid_search(series, p_range=(0, 3), d_range=(0, 1), q_range=(0, 3)):
        y = np.array(series, dtype=float)
        best_aic = np.inf
        best_order = (1, 0, 0)

        for p, d, q in product(range(*p_range), range(*d_range), range(*q_range)):
            try:
                m = ARIMAModel(p, d, q).fit(y)
                theta = m.theta
                y_diff = difference(y, d)
                _, resid_ar = fit_ar(y_diff, p)
                if q > 0:
                    n = len(resid_ar)
                    X_ma = np.column_stack([resid_ar[q - i - 1: n - i - 1] for i in range(q)])
                    resid = resid_ar[q:] - X_ma @ theta
                else:
                    resid = resid_ar
                n_params = p + q + 1
                aic, _ = aic_bic(resid, n_params, len(resid))
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except Exception:
                continue

        return best_order


class ARIMAForecaster:
    def __init__(self, dataset, p=3, d=0, q=3, auto_order=False):
        self.dataset = dataset
        self.p = p
        self.d = d
        self.q = q
        self.auto_order = auto_order
        self.models = {}
        self.history = {"train_loss": [], "val_loss": []}

    def fit_all(self, verbose=True):
        for col in self.dataset.target_cols:
            col_idx = self.dataset.target_cols.index(col)
            train_series = self.dataset.y_train[:, col_idx]

            if self.auto_order:
                order = ARIMAModel.grid_search(train_series)
                p, d, q = order
                if verbose:
                    print(f"  {col}: best order = ARIMA({p},{d},{q})")
            else:
                p, d, q = self.p, self.d, self.q
                if verbose:
                    print(f"  Fitting ARIMA({p},{d},{q}) for '{col}' ...")

            m = ARIMAModel(p, d, q).fit(train_series)
            self.models[col] = m

    def predict_rolling(self, y_arr):
        y_pred = np.zeros_like(y_arr)

        for i, col in enumerate(self.dataset.target_cols):
            m = self.models[col]
            preds = []
            last_y = list(m._last_y)
            last_resid = list(m._last_resid)

            for t in range(len(y_arr)):
                ar = sum(m.phi[j] * last_y[-(j + 1)] for j in range(m.p))
                ma = sum(m.theta[j] * last_resid[-(j + 1)] for j in range(m.q))
                yhat = ar + ma
                preds.append(yhat)
                last_y.append(y_arr[t, i])
                last_resid.append(y_arr[t, i] - yhat)

            y_pred[:, i] = np.array(preds)

        y_true = y_arr.copy()

        if self.dataset.is_scaled:
            y_pred = self.dataset.inverse_transform_target(y_pred[:, np.newaxis, :])[:, 0, :]
            y_true = self.dataset.inverse_transform_target(y_true[:, np.newaxis, :])[:, 0, :]

        return y_true, y_pred