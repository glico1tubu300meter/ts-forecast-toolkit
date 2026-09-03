"""
時系列予測モデル群(DLinear / TSMixer / PatchTST / TimeMixer)。

出典: I:\\マイドライブ\\Claude2\\output\\2609\\02-014-DLinear-TSMixer-PatchTST比較
での実験(USD/JPY等の金融時系列、1期先・N期先予測)で使用した実装を
再利用可能な形に整理したもの。

いずれも入力 (batch, lookback) の単変量系列を受け取り、(batch,) の
スカラー予測(horizon分の累積対数リターンなど)を返すインターフェースで統一している。
DLinearWithNews のみニュースセンチメント系列 (batch, lookback) を追加入力に取る。
"""
import torch
import torch.nn as nn


class DLinear(nn.Module):
    """移動平均によるトレンド/季節性分解 + 線形層のみ(Zeng et al. 2022の簡易再現)"""

    def __init__(self, lookback: int, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)
        self.linear_trend = nn.Linear(lookback, 1)
        self.linear_seasonal = nn.Linear(lookback, 1)

    def forward(self, x: torch.Tensor, x_tone: torch.Tensor = None) -> torch.Tensor:  # x: (N, L)
        # x_toneは受け取るが使わない(DLinearWithNewsと呼び出しインターフェースを揃えるため)
        x_ = x.unsqueeze(1)
        trend = self.avg_pool(x_)[:, :, :x.shape[1]].squeeze(1)
        seasonal = x - trend
        return (self.linear_trend(trend) + self.linear_seasonal(seasonal)).squeeze(-1)


class TSMixerBlock(nn.Module):
    def __init__(self, lookback: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(lookback)
        self.time_mlp = nn.Sequential(
            nn.Linear(lookback, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, lookback),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.time_mlp(self.norm(x))


class TSMixer(nn.Module):
    """時間方向のMLP混合(LayerNorm+MLP+残差)を複数ブロック積んだ簡易TSMixer"""

    def __init__(self, lookback: int, hidden: int = 64, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([TSMixerBlock(lookback, hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(lookback, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class PatchTST(nn.Module):
    """lookbackをパッチに分割しTransformer Encoderへ入力(Nie et al. 2023の簡易再現)"""

    def __init__(self, lookback: int, patch_len: int = 6, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        assert lookback % patch_len == 0, "lookbackはpatch_lenで割り切れる必要がある"
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len
        self.input_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * self.n_patches, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        patches = x.view(n, self.n_patches, self.patch_len)
        h = self.input_proj(patches) + self.pos_emb
        h = self.encoder(h)
        return self.head(h.reshape(n, -1)).squeeze(-1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor):
        trend = self.avg_pool(x.unsqueeze(1))[:, :, :x.shape[1]].squeeze(1)
        seasonal = x - trend
        return trend, seasonal


class TimeMixer(nn.Module):
    """複数時間スケール(ダウンサンプル)でのトレンド/季節性分解+スケール間ミキシング
    (Wang et al. 2024 "TimeMixer" の簡易再現。公式実装そのものではない)
    """

    def __init__(self, lookback: int, scales=(1, 2, 3), kernel_size: int = 5):
        super().__init__()
        self.scales = scales
        self.decomp = nn.ModuleList([SeriesDecomp(kernel_size) for _ in scales])
        self.downsample = nn.ModuleList([
            nn.AvgPool1d(s, stride=s) if s > 1 else nn.Identity() for s in scales
        ])
        lens = [lookback // s for s in scales]
        self.season_mix = nn.ModuleList([nn.Linear(lens[i], lens[i + 1]) for i in range(len(scales) - 1)])
        self.trend_mix = nn.ModuleList([nn.Linear(lens[i + 1], lens[i]) for i in range(len(scales) - 1)])
        self.heads = nn.ModuleList([nn.Linear(length, 1) for length in lens])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trends, seasonals = [], []
        for down, decomp in zip(self.downsample, self.decomp):
            xs = down(x.unsqueeze(1)).squeeze(1) if not isinstance(down, nn.Identity) else x
            t, s = decomp(xs)
            trends.append(t)
            seasonals.append(s)
        for i in range(len(self.scales) - 1):
            seasonals[i + 1] = seasonals[i + 1] + self.season_mix[i](seasonals[i])
        for i in reversed(range(len(self.scales) - 1)):
            trends[i] = trends[i] + self.trend_mix[i](trends[i + 1])
        preds = []
        for trend, seasonal, head in zip(trends, seasonals, self.heads):
            preds.append(head(trend + seasonal))
        return torch.stack(preds, dim=0).mean(dim=0).squeeze(-1)


class DLinearWithNews(nn.Module):
    """DLinearに、ニュースセンチメント系列(同じlookback長)の線形項を加えたもの"""

    def __init__(self, lookback: int, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)
        self.linear_trend = nn.Linear(lookback, 1)
        self.linear_seasonal = nn.Linear(lookback, 1)
        self.linear_tone = nn.Linear(lookback, 1)

    def forward(self, x_ret: torch.Tensor, x_tone: torch.Tensor) -> torch.Tensor:
        x_ = x_ret.unsqueeze(1)
        trend = self.avg_pool(x_)[:, :, :x_ret.shape[1]].squeeze(1)
        seasonal = x_ret - trend
        out = self.linear_trend(trend) + self.linear_seasonal(seasonal) + self.linear_tone(x_tone)
        return out.squeeze(-1)


MODEL_REGISTRY = {
    "dlinear": lambda lookback: DLinear(lookback),
    "tsmixer": lambda lookback: TSMixer(lookback),
    "patchtst": lambda lookback: PatchTST(lookback),
    "timemixer": lambda lookback: TimeMixer(lookback),
}
