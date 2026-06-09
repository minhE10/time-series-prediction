import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_variates, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_variates))
            self.beta = nn.Parameter(torch.zeros(num_variates))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = x.std(dim=1, keepdim=True, unbiased=False).detach() + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.std + self.mean
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class iTransformer(nn.Module):
    def __init__(self,
                 seq_len,
                 pred_len,
                 n_features,
                 n_targets=None,
                 d_model=64,
                 n_heads=4,
                 e_layers=2,
                 d_ff=128,
                 dropout=0.1,
                 use_revin=True,
                 target_indices=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_targets = n_targets if n_targets is not None else n_features
        self.use_revin = use_revin
        self.target_indices = target_indices

        if use_revin:
            self.revin = RevIN(n_features)

        self.embed = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(dropout),
        )

        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])

        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        if self.use_revin:
            x = self.revin(x, "norm")

        h = self.embed(x.permute(0, 2, 1))

        for block in self.encoder:
            h = block(h)

        out = self.projection(h).permute(0, 2, 1)

        if self.use_revin:
            out = self.revin(out, "denorm")

        if self.target_indices is not None:
            return out[:, :, self.target_indices]
        return out[:, :, : self.n_targets]
