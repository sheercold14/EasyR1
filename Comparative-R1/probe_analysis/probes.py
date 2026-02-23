from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ProbeOutput:
    name: str
    y_pred: np.ndarray


def _standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_torch_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    model: nn.Module,
    seed: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    steps: int = 500,
    batch_size: int = 256,
    verbose: bool = False,
    log_every: int = 100,
) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_tr = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    x_te = torch.from_numpy(x_test.astype(np.float32)).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = x_tr.shape[0]
    for step in range(steps):
        perm = torch.randperm(n, device=device)
        last_loss = None
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x_tr[idx])
            loss = criterion(logits, y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
        if verbose and (step == 0 or (step + 1) % max(1, log_every) == 0 or step + 1 == steps):
            print(f"[probe-train] step={step+1}/{steps} loss={last_loss:.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(x_te).argmax(dim=-1).cpu().numpy()
    return pred


def run_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    verbose: bool = False,
    log_every: int = 100,
) -> ProbeOutput:
    x_train, x_test = _standardize(x_train, x_test)
    d = x_train.shape[1]
    c = int(np.max(y_train) + 1)
    model = nn.Linear(d, c)
    y_pred = _fit_torch_classifier(
        x_train,
        y_train,
        x_test,
        model=model,
        seed=seed,
        lr=1e-2,
        steps=300,
        verbose=verbose,
        log_every=log_every,
    )
    return ProbeOutput(name="linear", y_pred=y_pred)


def run_knn_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    k: int,
    verbose: bool = False,
) -> ProbeOutput:
    x_train, x_test = _standardize(x_train, x_test)
    k = max(1, min(k, len(x_train)))

    # [N_test, N_train]
    d2 = np.sum((x_test[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    y_pred = np.zeros((x_test.shape[0],), dtype=np.int64)
    for i in range(x_test.shape[0]):
        labels = y_train[nn_idx[i]]
        vals, cnts = np.unique(labels, return_counts=True)
        y_pred[i] = vals[np.argmax(cnts)]

    if verbose:
        print(f"[probe-train] knn uses no gradient training, k={k}")
    return ProbeOutput(name=f"knn_k{k}", y_pred=y_pred)


def run_mlp_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    verbose: bool = False,
    log_every: int = 100,
) -> ProbeOutput:
    x_train, x_test = _standardize(x_train, x_test)
    d = x_train.shape[1]
    c = int(np.max(y_train) + 1)
    model = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, c))
    y_pred = _fit_torch_classifier(
        x_train,
        y_train,
        x_test,
        model=model,
        seed=seed,
        lr=1e-3,
        steps=500,
        verbose=verbose,
        log_every=log_every,
    )
    return ProbeOutput(name=f"mlp_h{hidden_dim}", y_pred=y_pred)
