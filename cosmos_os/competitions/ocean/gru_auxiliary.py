"""A GRU auxiliary for drift and offset, the two types the tree is weakest on.

Trained on those two alone and blended one-directionally - it may promote a row,
never demote one - because a model taught that noise rows are negative pushes them
down. Every window is centred on its own median, so a warmer year yields the same
tensor. Trained from scratch on the distributed data; no pretrained weights.

Runs as `python -m cosmos_os.competitions.ocean.gru_auxiliary` because importing
torch alongside LightGBM deadlocks their OpenMP runtimes on macOS.

Rejected on the leaderboard (Phase 33); kept off by default.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SEQ_LEN = 128
THREADS = 1
TRAIN_STRIDE = 4
HIDDEN = 64
EPOCHS = 3
BATCH = 512
LEARNING_RATE = 1e-3
TARGET_TYPES = ("drift", "offset")
SERIES_COLUMNS = ["station_layer", "time", "temp", "temp_dev_layers"]


def build_windows(
    frame: pd.DataFrame, stride: int, with_labels: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Causal windows ending at each endpoint, centred on their own median.

    Centring is what makes this year-invariant.
    """
    missing = set(SERIES_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if stride < 1:
        raise ValueError("stride must be positive")

    ordered = frame.sort_values(["station_layer", "time"]).reset_index()
    kinds = (
        ordered["anomaly_type"].fillna("").astype(str).str.split("+").str[0]
        if with_labels
        else None
    )
    blocks: list[np.ndarray] = []
    labels: list[int] = []
    positions: list[int] = []

    for _, group in ordered.groupby("station_layer", sort=False):
        temp = group["temp"].to_numpy(dtype=np.float32)
        deviation = group["temp_dev_layers"].to_numpy(dtype=np.float32)
        original = group["index"].to_numpy()
        if len(temp) < SEQ_LEN:
            continue
        label = group["label"].to_numpy() if with_labels else None
        kind = kinds.loc[group.index].to_numpy() if with_labels else None

        for end in range(SEQ_LEN, len(temp) + 1, stride):
            start = end - SEQ_LEN
            window_temp = temp[start:end]
            window_deviation = deviation[start:end]
            if not (np.isfinite(window_temp).all() and np.isfinite(window_deviation).all()):
                continue
            blocks.append(
                np.stack(
                    [
                        window_temp - np.median(window_temp),
                        window_deviation - np.median(window_deviation),
                        np.abs(np.diff(window_temp, prepend=window_temp[0])),
                    ]
                )
            )
            positions.append(original[end - 1])
            if with_labels:
                index = end - 1
                labels.append(
                    1 if (label[index] == 1 and kind[index] in TARGET_TYPES) else 0
                )

    if not blocks:
        raise ValueError("no usable windows; series are shorter than SEQ_LEN")
    return (
        np.asarray(blocks, dtype=np.float32),
        np.asarray(labels, dtype=np.float32) if with_labels else None,
        np.asarray(positions),
    )


def _model(torch):
    from torch import nn

    class Auxiliary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(3, HIDDEN, num_layers=1, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(
                nn.Linear(HIDDEN * 2, 32), nn.ReLU(), nn.Linear(32, 1)
            )

        def forward(self, x):
            out, _ = self.gru(x.transpose(1, 2))
            return self.head(out[:, -1]).squeeze(-1)

    return Auxiliary()


def score_frames(
    train: pd.DataFrame, target: pd.DataFrame, seed: int = 0, device: str = "cpu"
) -> np.ndarray:
    """Train on `train` and return one score per row of `target`.

    CPU by default: the submission's hash is verified on every rebuild and MPS
    does not reproduce bit-for-bit.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Must be one: raising it segfaults once LightGBM's OpenMP runtime is loaded in
    # the parent, and a 128-step GRU is sequential enough that nothing is lost.
    torch.set_num_threads(THREADS)
    np.random.seed(seed)

    x_train, y_train, _ = build_windows(train, TRAIN_STRIDE, with_labels=True)
    x_target, _, positions = build_windows(target, 1, with_labels=False)
    LOGGER.info(
        "GRU auxiliary: %d training windows (%d positive), %d target windows",
        len(x_train),
        int(y_train.sum()),
        len(x_target),
    )
    if y_train.sum() == 0:
        raise ValueError("no drift/offset windows in the training frame")

    unit = torch.device(device)
    model = _model(torch).to(unit)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Positives are a few percent of windows; unweighted, the model predicts all zero.
    positive_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], device=unit
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    tensor_x = torch.from_numpy(x_train)
    tensor_y = torch.from_numpy(y_train)
    generator = np.random.default_rng(seed)
    order = np.arange(len(tensor_x))
    model.train()
    for epoch in range(EPOCHS):
        generator.shuffle(order)
        total = 0.0
        for start in range(0, len(order), BATCH):
            index = order[start : start + BATCH]
            optimiser.zero_grad()
            loss = criterion(model(tensor_x[index].to(unit)), tensor_y[index].to(unit))
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(index)
        LOGGER.info("GRU auxiliary: epoch %d loss %.4f", epoch, total / len(order))

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_target), BATCH):
            batch = torch.from_numpy(x_target[start : start + BATCH]).to(unit)
            chunks.append(torch.sigmoid(model(batch)).cpu().numpy())

    # Rows before their series' first full window get the training base rate rather
    # than a score the model never produced for them.
    scores = np.full(len(target), float(y_train.mean()), dtype=float)
    scores[positions] = np.concatenate(chunks)
    return scores


def blend_one_directional(
    baseline: np.ndarray, auxiliary: np.ndarray, weight: float
) -> np.ndarray:
    """Let the auxiliary raise a row's rank but never lower it.

    A symmetric blend demoted noise rows the auxiliary was trained to call negative,
    costing more than it gained. The maximum keeps flatline and noise where the tree
    put them.
    """
    if len(baseline) != len(auxiliary):
        raise ValueError("baseline and auxiliary must have the same length")
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    base_rank = pd.Series(baseline).rank(pct=True).to_numpy()
    if weight == 0:
        return base_rank
    auxiliary_rank = pd.Series(auxiliary).rank(pct=True).to_numpy()
    return np.maximum(base_rank, (1 - weight) * base_rank + weight * auxiliary_rank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    scores = score_frames(
        pd.read_parquet(args.train),
        pd.read_parquet(args.target),
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, scores)
    LOGGER.info("wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
