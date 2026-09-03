"""学習・評価の共通ループとNaiveベースライン・評価指標。"""
import time

import numpy as np
import torch
import torch.nn as nn


def price_metrics(actual_price: np.ndarray, pred_price: np.ndarray) -> dict:
    err = actual_price - pred_price
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "mape_pct": float((np.abs(err) / actual_price).mean() * 100),
    }


def naive_metrics(price_at_origin_minus1: np.ndarray, actual_price: np.ndarray) -> dict:
    """「horizon日後も価格は変わらない」と仮定するNaive予測の評価指標"""
    return price_metrics(actual_price, price_at_origin_minus1)


def train_and_predict(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    device: torch.device,
    n_epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """標準化済みのX,yで学習し、テストの正規化予測値(numpy)と学習・推論時間を返す。

    X_train/X_test はモデルによって (N, L) または (N, L, ...) の形を取りうるが、
    このヘルパーはDLinear/TSMixer/PatchTST/TimeMixerのような単一テンソル入力を想定。
    ニュース入力など複数テンソルを使うモデルは、このヘルパーを使わず個別に学習ループを書く
    (16_experiment_with_news.py参照)。
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_train = X_train.shape[0]

    train_start = time.perf_counter()
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - train_start

    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_start = time.perf_counter()
        pred_norm = model(X_test).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - infer_start

    n_params = sum(p.numel() for p in model.parameters())
    timing = {
        "train_sec": train_time,
        "infer_per_sample_ms": infer_time / X_test.shape[0] * 1000,
        "n_params": n_params,
    }
    return pred_norm, timing
