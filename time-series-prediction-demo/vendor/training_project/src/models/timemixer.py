import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        back = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x = torch.cat([front, x, back], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.ma = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.ma(x)
        return x - trend, trend


class DFTSeriesDecomp(nn.Module):
    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        amp = torch.abs(xf)
        amp[:, 0, :] = 0
        _, idx = torch.topk(amp, self.top_k, dim=1)
        mask = torch.zeros_like(xf)
        mask.scatter_(1, idx, 1)
        seasonal = torch.fft.irfft(xf * mask, n=x.size(1), dim=1)
        return seasonal, x - seasonal


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, seq_len, down_window, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_window ** i), seq_len // (down_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(seq_len // (down_window ** (i + 1)), seq_len // (down_window ** (i + 1))),
            )
            for i in range(n_layers)
        ])

    def forward(self, season_list):
        out = [season_list[0]]
        cur = season_list[0]
        for i, layer in enumerate(self.layers):
            cur = layer(cur) + season_list[i + 1]
            out.append(cur)
        return out


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, seq_len, down_window, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_window ** (i + 1)), seq_len // (down_window ** i)),
                nn.GELU(),
                nn.Linear(seq_len // (down_window ** i), seq_len // (down_window ** i)),
            )
            for i in reversed(range(n_layers))
        ])

    def forward(self, trend_list):
        rev = list(reversed(trend_list))
        out = [rev[0]]
        cur = rev[0]
        for i, layer in enumerate(self.layers):
            cur = layer(cur) + rev[i + 1]
            out.append(cur)
        out.reverse()
        return out


class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, dropout, down_window, n_down_layers,
                 decomp_method="moving_avg", moving_avg=25, top_k=5, channel_independence=False):
        super().__init__()
        self.channel_independence = channel_independence

        if decomp_method == "moving_avg":
            self.decomp = SeriesDecomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decomp = DFTSeriesDecomp(top_k)
        else:
            raise ValueError(f"decomp_method '{decomp_method}' not valid.")

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        if not channel_independence:
            self.cross = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
            )

        self.season_mix = MultiScaleSeasonMixing(seq_len, down_window, n_down_layers)
        self.trend_mix = MultiScaleTrendMixing(seq_len, down_window, n_down_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x_list):
        lengths = [x.size(1) for x in x_list]
        s_list, t_list = [], []
        for x in x_list:
            s, t = self.decomp(x)
            if not self.channel_independence:
                s = self.cross(s)
                t = self.cross(t)
            s_list.append(s.permute(0, 2, 1))
            t_list.append(t.permute(0, 2, 1))

        s_out = self.season_mix(s_list)
        t_out = self.trend_mix(t_list)

        result = []
        for ori, s, t, l in zip(x_list, s_out, t_out, lengths):
            merged = s.permute(0, 2, 1) + t.permute(0, 2, 1)
            out = ori + self.drop(self.out_proj(merged))
            result.append(out[:, :l, :])
        return result


class Normalize(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, non_norm=False):
        super().__init__()
        self.eps = eps
        self.non_norm = non_norm
        self.affine = affine
        if affine:
            self.w = nn.Parameter(torch.ones(num_features))
            self.b = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=tuple(range(1, x.ndim - 1)), keepdim=True).detach()
            self.std = (x.var(dim=tuple(range(1, x.ndim - 1)), keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            if not self.non_norm:
                x = (x - self.mean) / self.std
                if self.affine:
                    x = x * self.w + self.b
        elif mode == "denorm":
            if not self.non_norm:
                if self.affine:
                    x = (x - self.b) / (self.w + self.eps ** 2)
                x = x * self.std + self.mean
        return x


class TimeMixer(nn.Module):
    def __init__(self,
                 seq_len,
                 pred_len,
                 n_features,
                 n_targets=None,
                 target_indices=None,
                 d_model=64,
                 d_ff=128,
                 e_layers=2,
                 dropout=0.1,
                 down_sampling_layers=2,
                 down_sampling_window=2,
                 down_sampling_method="avg",
                 decomp_method="moving_avg",
                 moving_avg=25,
                 top_k=5,
                 channel_independence=False,
                 use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_targets = n_targets if n_targets is not None else n_features
        self.target_indices = target_indices
        self.n_down = down_sampling_layers
        self.down_window = down_sampling_window
        self.channel_independence = channel_independence
        self.use_norm = use_norm

        embed_in = 1 if channel_independence else n_features
        self.embed = nn.Sequential(nn.Linear(embed_in, d_model), nn.Dropout(dropout))

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len, d_model, d_ff, dropout, down_sampling_window,
                down_sampling_layers, decomp_method, moving_avg, top_k, channel_independence
            )
            for _ in range(e_layers)
        ])

        if down_sampling_method == "avg":
            self.down_pool = nn.AvgPool1d(kernel_size=down_sampling_window)
        elif down_sampling_method == "max":
            self.down_pool = nn.MaxPool1d(kernel_size=down_sampling_window)
        elif down_sampling_method == "conv":
            self.down_pool = nn.Conv1d(
                n_features, n_features, kernel_size=3, padding=1,
                stride=down_sampling_window, padding_mode="circular", bias=False,
            )
        else:
            raise ValueError(f"down_sampling_method '{down_sampling_method}' not valid.")

        self.predict_layers = nn.ModuleList([
            nn.Linear(seq_len // (down_sampling_window ** i), pred_len)
            for i in range(down_sampling_layers + 1)
        ])

        if channel_independence:
            self.proj = nn.Linear(d_model, 1, bias=True)
        else:
            self.proj = nn.Linear(d_model, self.n_targets, bias=True)

        if use_norm:
            self.norm_layers = nn.ModuleList([
                Normalize(n_features, affine=True)
                for _ in range(down_sampling_layers + 1)
            ])

    def _multiscale(self, x):
        x_list = [x]
        cur = x.permute(0, 2, 1)
        for _ in range(self.n_down):
            cur = self.down_pool(cur)
            x_list.append(cur.permute(0, 2, 1))
        return x_list

    def forward(self, x):
        x_list = self._multiscale(x)

        if self.use_norm:
            x_list = [self.norm_layers[i](xi, "norm") for i, xi in enumerate(x_list)]

        if self.channel_independence:
            x_list = [
                xi.permute(0, 2, 1).reshape(xi.size(0) * xi.size(2), xi.size(1), 1)
                for xi in x_list
            ]

        enc = [self.embed(xi) for xi in x_list]

        for pdm in self.pdm_blocks:
            enc = pdm(enc)

        preds = []
        for i, e in enumerate(enc):
            p = self.predict_layers[i](e.permute(0, 2, 1)).permute(0, 2, 1)
            preds.append(self.proj(p))

        out = torch.stack(preds, dim=-1).sum(-1)

        if self.channel_independence:
            B = x.size(0)
            out = out.reshape(B, self.n_features, self.pred_len).permute(0, 2, 1)

        if self.use_norm:
            if not self.channel_independence:
                if out.size(-1) != self.n_features:
                    full = torch.zeros(out.size(0), out.size(1), self.n_features, device=out.device)
                    idx = self.target_indices if self.target_indices else slice(None, out.size(-1))
                    if isinstance(idx, list):
                        full[:, :, idx] = out
                    else:
                        full[:, :, idx] = out
                    out = self.norm_layers[0](full, "denorm")
                    out = out[:, :, (self.target_indices if self.target_indices else slice(None, self.n_targets))]
                else:
                    out = self.norm_layers[0](out, "denorm")
                    out = out[:, :, : self.n_targets]
            else:
                out = self.norm_layers[0](out, "denorm")
                out = out[:, :, : self.n_targets]
        else:
            out = out[:, :, : self.n_targets]

        return out