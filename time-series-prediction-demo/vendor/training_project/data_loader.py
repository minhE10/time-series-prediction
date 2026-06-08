import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.data_config import DATASET_CONFIG


class TSWindowDataset(Dataset):
    def __init__(self, feature_arr, target_arr, seq_len, pred_len):
        self.X = torch.tensor(feature_arr, dtype=torch.float32)
        self.y = torch.tensor(target_arr, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]
        y = self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return x, y


class TimeSeriesDataset:
    def __init__(self,
                 name,
                 path,
                 seq_len=48,
                 pred_len=24,
                 train_ratio=0.7,
                 val_ratio=0.2,
                 batch_size=32,
                 num_workers=0,
                 worker_init_fn=None,
                 generator=None):
        if name not in DATASET_CONFIG:
            raise ValueError(f"Dataset '{name}' not in DATASET_CONFIG")

        self.name = name
        self.path = path
        self.config = DATASET_CONFIG[name].copy()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.worker_init_fn = worker_init_fn
        self.generator = generator

        self.feature_cols = None
        self.target_cols = None
        self.feature_scaler = None
        self.target_scaler = None
        self.is_scaled = False

        self._load()
        self._split()
        self._make_datasets()

    def _load(self):
        df = pd.read_csv(self.path)

        for col in self.config.get("drop_cols", []):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        for col in ["No", "station"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        time_col = self.config["time_col"]
        if isinstance(time_col, list):
            df["datetime"] = pd.to_datetime(df[time_col])
            df.drop(columns=time_col, inplace=True)
        elif self.name == "bitcoin":
            df["datetime"] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
        else:
            df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")

        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

        if self.name == "bitcoin":
            df.set_index("datetime", inplace=True)
            df = df.resample("1h").last().dropna(subset=["Open"])
            df.reset_index(inplace=True)

        df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)

        if "wd" in df.columns:
            df["wd"] = df["wd"].fillna(df["wd"].mode()[0])
            df = pd.get_dummies(df, columns=["wd"], prefix="wd", dtype=np.float32)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        feature_cols = list(self.config["feature_cols"])
        target_cols = list(self.config["target_cols"])

        for c in [c for c in df.columns if c.startswith("wd_")]:
            if c not in feature_cols:
                feature_cols.append(c)

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.df = df

    def _split(self):
        df = self.df
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        val_start = max(0, train_end - self.seq_len)
        test_start = max(0, val_end - self.seq_len)

        def to_float(sub, cols):
            arr = sub[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
            return arr.ffill().bfill().to_numpy()

        self.X_train = to_float(df.iloc[:train_end], self.feature_cols)
        self.X_val = to_float(df.iloc[val_start:val_end], self.feature_cols)
        self.X_test = to_float(df.iloc[test_start:], self.feature_cols)

        self.y_train = to_float(df.iloc[:train_end], self.target_cols)
        self.y_val = to_float(df.iloc[val_start:val_end], self.target_cols)
        self.y_test = to_float(df.iloc[test_start:], self.target_cols)

    def apply_scaling(self, scaler_type="standard"):
        if scaler_type == "standard":
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'.")

        self.X_train = self.feature_scaler.fit_transform(self.X_train)
        self.X_val = self.feature_scaler.transform(self.X_val)
        self.X_test = self.feature_scaler.transform(self.X_test)

        self.y_train = self.target_scaler.fit_transform(self.y_train)
        self.y_val = self.target_scaler.transform(self.y_val)
        self.y_test = self.target_scaler.transform(self.y_test)

        self.is_scaled = True
        self._make_datasets()

    def inverse_transform_target(self, y_arr):
        if self.target_scaler is None:
            raise ValueError("apply_scaling() has not been called.")
        if isinstance(y_arr, torch.Tensor):
            y_arr = y_arr.detach().cpu().numpy()
        shape = y_arr.shape
        return self.target_scaler.inverse_transform(
            y_arr.reshape(-1, len(self.target_cols))
        ).reshape(shape)

    def _make_datasets(self):
        self.train_dataset = TSWindowDataset(self.X_train, self.y_train, self.seq_len, self.pred_len)
        self.val_dataset = TSWindowDataset(self.X_val,   self.y_val,   self.seq_len, self.pred_len)
        self.test_dataset = TSWindowDataset(self.X_test,  self.y_test,  self.seq_len, self.pred_len)

    def get_loaders(self):
        kw = dict(
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
        )
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,  **kw)
        val_loader = DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False, **kw)
        test_loader = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False, **kw)
        return train_loader, val_loader, test_loader

    @property
    def n_features(self):
        return len(self.feature_cols)

    @property
    def n_targets(self):
        return len(self.target_cols)

    @property
    def target_indices(self):
        return [self.feature_cols.index(c) for c in self.target_cols if c in self.feature_cols]