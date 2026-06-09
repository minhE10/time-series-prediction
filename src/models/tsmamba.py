import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(u, delta, A, B, C, D):
    B_batch, L, d_inner = u.shape
    d_state = A.shape[1]

    delta_A = torch.exp(delta.unsqueeze(-1) * A)         
    delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1) 

    h = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = delta_A[:, t] * h + delta_B_u[:, t]           
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1)         
        ys.append(y_t)

    y = torch.stack(ys, dim=1) + u * D                 
    return y


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        _, L, _ = x.shape

        xz = self.in_proj(x)                                
        x_in, z = xz.chunk(2, dim=-1)                      

        # Causal conv
        x_conv = self.conv1d(x_in.permute(0, 2, 1))[:, :, :L].permute(0, 2, 1)
        x_conv = F.silu(x_conv)

        # Compute SSM parameters
        x_dbl = self.x_proj(x_conv)                         
        delta, B_param, C_param = x_dbl.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))               
        A = -torch.exp(self.A_log)                               

        y = selective_scan(x_conv, delta, A, B_param, C_param, self.D)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.drop(y)


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = self.beta = None

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(1, keepdim=True).detach()
            self.std = x.std(1, keepdim=True, unbiased=False).detach() + self.eps
            x = (x - self.mean) / self.std
            if self.gamma is not None:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.gamma is not None:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.std + self.mean
        return x


class TSMamba(nn.Module):
    def __init__(self,
                 seq_len,
                 pred_len,
                 n_features,
                 n_targets=None,
                 d_model=64,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 n_layers=2,
                 patch_len=8,
                 dropout=0.1,
                 use_revin=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_targets = n_targets if n_targets is not None else n_features
        self.patch_len = patch_len
        self.use_revin = use_revin

        n_patches = seq_len // patch_len

        if use_revin:
            self.revin = RevIN(n_features)

        self.patch_embed = nn.Linear(patch_len * n_features, d_model)

        self.fwd_encoder = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), MambaBlock(d_model, d_state, d_conv, expand, dropout))
            for _ in range(n_layers)
        ])
        self.bwd_encoder = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), MambaBlock(d_model, d_state, d_conv, expand, dropout))
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_patches * d_model, pred_len * self.n_targets)

    def _encode(self, x):
        h = x
        for block in self.fwd_encoder:
            norm_layer, mamba = block
            h = h + mamba(norm_layer(h))
        return h

    def _encode_bwd(self, x):
        h = x.flip(1)
        for block in self.bwd_encoder:
            norm_layer, mamba = block
            h = h + mamba(norm_layer(h))
        return h.flip(1)

    def forward(self, x):
        B, T, C = x.shape

        if self.use_revin:
            x = self.revin(x, "norm")

        n_patches = T // self.patch_len
        x_patch = x[:, : n_patches * self.patch_len, :]         
        x_patch = x_patch.reshape(B, n_patches, self.patch_len * C) 
        h = self.patch_embed(x_patch)                                

        h_fwd = self._encode(h)
        h_bwd = self._encode_bwd(h)
        h = self.norm(h_fwd + h_bwd)                                 

        h_flat = h.reshape(B, -1)
        out = self.head(self.head_drop(h_flat))
        out = out.reshape(B, self.pred_len, self.n_targets)

        if self.use_revin:
            dummy = torch.zeros(B, self.pred_len, C, device=x.device, dtype=x.dtype)
            dummy[:, :, : self.n_targets] = out
            dummy = self.revin(dummy, "denorm")
            out = dummy[:, :, : self.n_targets]

        return out