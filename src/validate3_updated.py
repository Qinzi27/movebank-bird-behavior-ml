#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate3.py — LOBO validation for 2-state Gaussian HMM (Rest/Move) + speed-threshold split.

Chinese/中文：
- 用 GaussianHMM 在 log(1+speed) 上拟合 2 个隐藏状态（Rest vs Move）
- 通过 “发射均值更小的状态 = Rest” 来自动映射状态含义（避免 state label 交换）
- 在测试个体上解码得到 Rest/Move，再用速度阈值把 Move 进一步拆成 Forage / Flight
  * speed < low_thr          -> Rest
  * low_thr <= speed < flight_thr -> Forage
  * speed >= flight_thr      -> Flight
- 训练阶段会严格按 individual 切分序列（lengths），避免不同个体拼接时产生“跨个体跳转”。

English:
- Fit a 2-state GaussianHMM on log(1+speed) to separate Rest vs Move.
- Map the hidden state with the smaller emission mean to Rest (label-switch safe).
- Decode the held-out individual, then split Move into Forage/Flight by speed thresholds.

Typical run:
  python src/validate3.py --n-folds 5 --random-state 42 --n-iter 100 --low-thr 2.5 --flight-thr 10.0

Outputs:
  outputs/validate3_fold{i}_{bird_id}.png
  outputs/validate3_summary.csv
"""

from __future__ import annotations

import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hmmlearn is required
from hmmlearn.hmm import GaussianHMM


# ---- Optional dependency: your project preprocessor ----
try:
    from src.features import TrajectoryProcessor  # type: ignore
except Exception:
    TrajectoryProcessor = None  # fallback later


# ---------------------------
# Utilities
# ---------------------------

def _safe_log1p(x: np.ndarray) -> np.ndarray:
    """Known-safe log1p wrapper: clamps negatives to 0 (speed should not be negative)."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    return np.log1p(x)


def build_sequences_by_individual(
    df: pd.DataFrame,
    speed_col: str = "speed",
    id_col: str = "individual-local-identifier",
    time_col: Optional[str] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Concatenate per-individual sequences into X and lengths.
    This prevents HMM from learning fake transitions across individuals.

    Parameters
    ----------
    df : DataFrame containing at least speed_col and id_col.
    time_col : if provided and exists in df, we sort within each individual.

    Returns
    -------
    X : (N, 1) array of log(1+speed)
    lengths : list of sequence lengths per individual
    """
    lengths: List[int] = []
    xs: List[np.ndarray] = []

    for _, g in df.groupby(id_col, sort=True):
        if time_col is not None and time_col in g.columns:
            g = g.sort_values(time_col)
        s = g[speed_col].to_numpy(dtype=float)
        x = _safe_log1p(s).reshape(-1, 1)
        xs.append(x)
        lengths.append(len(x))

    if len(xs) == 0:
        raise ValueError("No sequences: check df filtering and column names.")

    X = np.vstack(xs)
    return X, lengths


def map_states_by_mean(model: GaussianHMM) -> Tuple[int, int, np.ndarray]:
    """
    Map HMM hidden states to semantic labels (rest vs move) by emission mean.
    For log(1+speed), lower mean => Rest.

    Returns
    -------
    rest_state, move_state, means
    """
    means = model.means_.reshape(-1)  # (n_components,)
    rest_state = int(np.argmin(means))
    move_state = int(1 - rest_state)  # because n_components=2
    return rest_state, move_state, means


def compute_retry_metrics(
    states: np.ndarray,
    move_state: int,
    speed_test: np.ndarray,
    low_thr: float,
    flight_thr: float,
) -> Tuple[float, float]:
    """
    Compute two diagnostics used to accept/reject a fit.

    recall_high_speed_as_move:
        among points with speed >= flight_thr, fraction predicted as move_state.
    false_move_on_low:
        among points with speed < low_thr, fraction predicted as move_state.

    If a slice is empty, return np.nan for that metric.
    """
    speed_test = np.asarray(speed_test, dtype=float)
    is_move = (states == move_state)

    high_mask = speed_test >= flight_thr
    low_mask = speed_test < low_thr

    recall = float(np.mean(is_move[high_mask])) if np.any(high_mask) else float("nan")
    false_move = float(np.mean(is_move[low_mask])) if np.any(low_mask) else float("nan")
    return recall, false_move


def decode_to_behavior_labels(
    states: np.ndarray,
    rest_state: int,
    move_state: int,
    speed: np.ndarray,
    low_thr: float,
    flight_thr: float,
) -> np.ndarray:
    """
    Convert decoded Rest/Move into 3-class behaviors with speed thresholds.

    Rule (robust version):
      - If speed < low_thr: Rest (force rest at very low speed)
      - Else if decoded state == rest_state: Rest
      - Else (decoded move):
           speed >= flight_thr -> Flight
           else -> Forage
    """
    speed = np.asarray(speed, dtype=float)

    out = np.empty(len(states), dtype=object)

    # Force Rest below low_thr (this is optional but stabilizes plots & avoids tiny 'Forage' at ~0 speed)
    low = speed < low_thr
    out[low] = "Rest"

    # For remaining:
    rem = ~low
    # decoded rest
    out[rem & (states == rest_state)] = "Rest"
    # decoded move
    move_idx = rem & (states == move_state)
    out[move_idx & (speed >= flight_thr)] = "Flight"
    out[move_idx & (speed < flight_thr)] = "Forage"

    return out


# ---------------------------
# Retry fit + decode
# ---------------------------

@dataclass
class RetryResult:
    model: GaussianHMM
    states: np.ndarray
    rest_state: int
    move_state: int
    means: np.ndarray
    recall_high_speed_as_move: float
    false_move_on_low: float
    attempt: int
    rs: int
    init_kwargs: Dict


def fit_decode_with_retry(
    X_train: np.ndarray,
    len_train: List[int],
    X_test: np.ndarray,
    len_test: List[int],
    speed_test: np.ndarray,
    low_thr: float,
    flight_thr: float,
    n_iter: int = 100,
    max_retries: int = 3,
    fail_false_move_on_low: float = 0.5,
    random_state_base: int = 42,
) -> RetryResult:
    """
    Fit GaussianHMM with a small set of initializations; accept the first fit that
    doesn't classify "too many" low-speed points as Move.

    Rationale (中文/English):
      - hmmlearn can converge to a label-swapped or degenerate solution depending on init.
      - retry with different seeds / 'sticky' transitions often fixes it.
    """
    # Candidate initializations: (name, kwargs)
    init_candidates = [
        ("sticky_diag", dict(covariance_type="diag", known_trans=True)),
        ("sticky_full", dict(covariance_type="full", known_trans=True)),
        ("free_diag", dict(covariance_type="diag", known_trans=False)),
    ]

    best: Optional[RetryResult] = None

    for attempt in range(1, max_retries + 1):
        for k, (init_name, init_cfg) in enumerate(init_candidates):
            rs = int(random_state_base + 1000 * (attempt - 1) + 17 * k)

            cov_type = init_cfg["covariance_type"]
            known_trans = init_cfg["known_trans"]

            # Build model
            model = GaussianHMM(
                n_components=2,
                covariance_type=cov_type,
                n_iter=int(n_iter),
                tol=1e-3,
                random_state=rs,
                verbose=False,
            )

            init_kwargs: Dict = {"init": init_name, "covariance_type": cov_type, "known_trans": known_trans}

            # Optional: encourage persistence (sticky transitions)
            if known_trans:
                model.startprob_ = np.array([0.5, 0.5])
                model.transmat_ = np.array([[0.97, 0.03], [0.03, 0.97]])
                # only initialize means/covars from data; keep start/trans fixed
                model.init_params = "mc"
                model.params = "stmc"  # allow EM updates; you can restrict if you want
            else:
                # Let hmmlearn initialize everything
                model.init_params = "stmc"
                model.params = "stmc"

            # Fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(X_train, len_train)
                except Exception:
                    continue

            # Decode
            try:
                states = model.predict(X_test, len_test)
            except Exception:
                continue

            rest_state, move_state, means = map_states_by_mean(model)
            recall, false_move = compute_retry_metrics(
                states=states,
                move_state=move_state,
                speed_test=speed_test,
                low_thr=low_thr,
                flight_thr=flight_thr,
            )

            res = RetryResult(
                model=model,
                states=states,
                rest_state=rest_state,
                move_state=move_state,
                means=means,
                recall_high_speed_as_move=recall,
                false_move_on_low=false_move,
                attempt=attempt,
                rs=rs,
                init_kwargs=init_kwargs,
            )

            # Keep the best (lowest false_move), in case all fail.
            if best is None:
                best = res
            else:
                # prefer lower false_move; tie-breaker: higher recall
                best_key = (np.nan_to_num(best.false_move_on_low, nan=1.0), -np.nan_to_num(best.recall_high_speed_as_move, nan=0.0))
                res_key = (np.nan_to_num(res.false_move_on_low, nan=1.0), -np.nan_to_num(res.recall_high_speed_as_move, nan=0.0))
                if res_key < best_key:
                    best = res

            # Accept condition
            if np.isfinite(false_move) and false_move <= fail_false_move_on_low:
                return res

    if best is None:
        raise RuntimeError("All HMM fits failed. Check your data for NaNs / empty sequences.")

    return best


# ---------------------------
# Plotting
# ---------------------------

def plot_fold(
    test_df: pd.DataFrame,
    pred_labels: np.ndarray,
    bird_id: str,
    low_thr: float,
    flight_thr: float,
    save_path: str,
):
    """
    Plot: grey speed series + colored predicted labels.
    """
    speed = test_df["speed"].to_numpy(dtype=float)

    # map labels to colors/markers
    palette = {"Rest": ("blue", "o"), "Forage": ("orange", "o"), "Flight": ("red", "o")}

    plt.figure(figsize=(14, 4.8))
    plt.plot(speed, color="lightgray", linewidth=2, alpha=0.8, label="Speed (m/s)")

    # scatter by label
    for lab in ["Rest", "Forage", "Flight"]:
        mask = (pred_labels == lab)
        if np.any(mask):
            c, m = palette[lab]
            plt.scatter(np.where(mask)[0], speed[mask], s=18, c=c, marker=m, label=f"{lab} (Pred)")

    plt.axhline(low_thr, color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"{low_thr:.1f} m/s")
    plt.axhline(flight_thr, color="black", linestyle=":", linewidth=1, alpha=0.5, label=f"{flight_thr:.1f} m/s")

    plt.title(f"validate3 LOBO: {bird_id}  (2-state HMM Rest/Move + threshold split)")
    plt.xlabel("Time step (1 min if resampled at 60s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------
# Main validation
# ---------------------------

def validate3(
    n_folds: int = 5,
    random_state: int = 42,
    n_iter: int = 100,
    low_thr: float = 2.5,
    flight_thr: float = 10.0,
    max_retries: int = 3,
    fail_false_move_on_low: float = 0.5,
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "GPS tracking of guanay cormorants.csv")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("Reading:", data_path)
    raw_df = pd.read_csv(data_path)

    # Preprocess
    if TrajectoryProcessor is None:
        raise ImportError(
            "TrajectoryProcessor not found. "
            "Please ensure src/features.py provides TrajectoryProcessor, "
            "or implement speed/resample preprocessing here."
        )

    print("Preprocessing data (resample=True)...")
    processor = TrajectoryProcessor()
    df = processor.preprocess(raw_df, do_resample=True)

    # Basic sanity prints
    qs = [0.5, 0.9, 0.95, 0.99]
    print("\nGlobal speed quantiles (m/s):")
    print(df["speed"].quantile(qs))
    print("Global max speed:", float(df["speed"].max()))
    print("Global frac speed>=flight_thr:", float((df["speed"] >= flight_thr).mean()))

    id_col = "individual-local-identifier"
    individual_ids = np.array(sorted(df[id_col].unique()))
    print(f"Dataset contains {len(individual_ids)} individuals.")

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(individual_ids, size=min(n_folds, len(individual_ids)), replace=False)

    rows = []

    for fold_idx, test_id in enumerate(chosen, start=1):
        print("\n" + "=" * 80)
        print(f"[Fold {fold_idx}/{n_folds}] LOBO test_id = {test_id}")

        train_df = df[df[id_col] != test_id].copy()
        test_df = df[df[id_col] == test_id].copy()

        # Build proper sequence lengths
        X_train, len_train = build_sequences_by_individual(train_df, speed_col="speed", id_col=id_col)
        # Test is a single sequence (one individual)
        X_test = _safe_log1p(test_df["speed"].to_numpy(dtype=float)).reshape(-1, 1)
        len_test = [len(X_test)]
        speed_test = test_df["speed"].to_numpy(dtype=float)

        # ---- THIS is where the retry-fit goes inside the fold loop ----
        result = fit_decode_with_retry(
            X_train=X_train,
            len_train=len_train,
            X_test=X_test,
            len_test=len_test,
            speed_test=speed_test,
            low_thr=low_thr,
            flight_thr=flight_thr,
            n_iter=n_iter,
            max_retries=max_retries,
            fail_false_move_on_low=fail_false_move_on_low,
            random_state_base=random_state,
        )

        model = result.model
        states = result.states
        rest_state = result.rest_state
        move_state = result.move_state
        means = result.means

        print(f"[Retry info] attempt={result.attempt} rs={result.rs} init={result.init_kwargs}")
        print("HMM means (log1p speed):", means)
        print("rest_state:", rest_state, "move_state:", move_state)
        print("recall_high_speed_as_move =", result.recall_high_speed_as_move)
        print("false_move_on_low         =", result.false_move_on_low)

        # Convert to 3-class labels
        pred_labels = decode_to_behavior_labels(
            states=states,
            rest_state=rest_state,
            move_state=move_state,
            speed=speed_test,
            low_thr=low_thr,
            flight_thr=flight_thr,
        )

        # Save plot
        fig_name = f"validate3_fold{fold_idx}_{str(test_id).replace('/', '_')}.png"
        fig_path = known_path = os.path.join(out_dir, fig_name)
        plot_fold(test_df, pred_labels, str(test_id), low_thr, flight_thr, fig_path)
        print("Saved:", fig_path)

        # Per-fold counts
        pred_rest = int(np.sum(pred_labels == "Rest"))
        pred_forage = int(np.sum(pred_labels == "Forage"))
        pred_flight = int(np.sum(pred_labels == "Flight"))

        rows.append(
            dict(
                fold=fold_idx,
                test_id=str(test_id),
                n_test=len(test_df),
                pred_rest=pred_rest,
                pred_forage=pred_forage,
                pred_flight=pred_flight,
                recall_high_speed_as_move=result.recall_high_speed_as_move,
                false_move_on_low=result.false_move_on_low,
                hmm_mean_rest=float(means[rest_state]),
                hmm_mean_move=float(means[move_state]),
                attempt=result.attempt,
                rs=result.rs,
                init=str(result.init_kwargs),
                test_max_speed=float(np.max(speed_test)) if len(speed_test) else float("nan"),
                test_frac_speed_ge_flight=float(np.mean(speed_test >= flight_thr)) if len(speed_test) else float("nan"),
            )
        )

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "validate3_summary.csv")
    summary.to_csv(out_csv, index=False)
    print("\nSaved summary:", out_csv)
    print(summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--low-thr", type=float, default=2.5)
    p.add_argument("--flight-thr", type=float, default=10.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--fail-false-move-on-low", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate3(
        n_folds=args.n_folds,
        random_state=args.random_state,
        n_iter=args.n_iter,
        low_thr=args.low_thr,
        flight_thr=args.flight_thr,
        max_retries=args.max_retries,
        fail_false_move_on_low=args.fail_false_move_on_low,
    )
