# src/validate3.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from hmmlearn.hmm import GaussianHMM
except Exception as e:
    raise ImportError("hmmlearn is required. Please `pip install hmmlearn`.") from e

# Allow running both as `python -m src.validate3` and `python src/validate3.py`
try:
    from src.features import TrajectoryProcessor
except ModuleNotFoundError:
    from features import TrajectoryProcessor


# ----------------------------
# Utilities
# ----------------------------
def ensure_log_speed(df: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    """
    Ensure df has a column 'log_speed' defined as log(speed + eps).
    中文：保证存在 log_speed 列（对 speed 做 log(speed+eps)）。
    """
    if "speed" not in df.columns:
        raise ValueError("DataFrame must contain 'speed' column.")
    if "log_speed" not in df.columns:
        df = df.copy()
        # speed may include 0; eps avoids -inf
        df["log_speed"] = np.log(np.maximum(df["speed"].to_numpy(), 0.0) + eps)
    return df


def init_means_from_quantiles(X_train: np.ndarray) -> np.ndarray:
    """
    Initialize 2-state means from quantiles of log_speed.
    中文：用训练 log_speed 分位数初始化两类均值，帮助 EM 不乱跑。
    """
    x = X_train.reshape(-1)
    q_low = np.quantile(x, 0.20)
    q_high = np.quantile(x, 0.80)
    means = np.array([[q_low], [q_high]], dtype=float)  # shape (2,1)
    return means


def fit_decode_with_retry(
    X_train: np.ndarray,
    len_train: list,
    X_test: np.ndarray,
    len_test: list,
    speed_test: np.ndarray,
    low_thr: float = 2.5,
    flight_thr: float = 10.0,
    n_iter: int = 100,
    tol: float = 1e-3,
    max_retries: int = 5,
    fail_false_move_on_low: float = 0.5,
    random_state_base: int = 42,
):
    """
    Train 2-state GaussianHMM and decode states with retries if sanity checks fail.

    Sanity checks (无真值稳定性检查):
      - recall_high_speed_as_move: among speed >= flight_thr, predicted Move proportion (want high).
      - false_move_on_low        : among speed <  low_thr, predicted Move proportion (want low).
    We retry if false_move_on_low > fail_false_move_on_low.

    Returns dict with model, states, rest_state, move_state, metrics, attempt info.
    """

    # Candidate init configs to try (English/中文：不同初始化策略，防止落入坏局部最优)
    init_configs = [
        # (init_params, params)
        ("stc", "stmc"),   # do NOT re-init means; allow EM to update means
        ("stmc", "stmc"),  # let hmmlearn init everything
    ]

    # Pre-compute a “good” mean init from training data
    means_init = init_means_from_quantiles(X_train)

    best = None
    for attempt in range(1, max_retries + 1):
        rs = random_state_base + 97 * (attempt - 1)

        # pick init config round-robin
        init_params, params = init_configs[(attempt - 1) % len(init_configs)]

        model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=n_iter,
            tol=tol,
            random_state=rs,
            init_params=init_params,
            params=params,
        )

        # If init_params excludes 'm', we can set means_ ourselves
        init_kwargs = {"init_params": init_params, "params": params}
        if "m" not in init_params:
            # must also set covars_ shape (2,1) for diag
            # initialize with variance of training data
            var = np.var(X_train.reshape(-1))
            if not np.isfinite(var) or var <= 1e-6:
                var = 1.0
            model.means_ = means_init.copy()
            model.covars_ = np.array([[var], [var]], dtype=float)

            # startprob_ and transmat_ optional; give mild persistence
            model.startprob_ = np.array([0.5, 0.5], dtype=float)
            model.transmat_ = np.array([[0.95, 0.05],
                                        [0.05, 0.95]], dtype=float)

        # ---- fit ----
        try:
            model.fit(X_train, lengths=len_train)
        except Exception:
            # if training fails, continue retry
            continue

        # ---- decode ----
        states = model.predict(X_test, lengths=len_test)

        # Decide which state is Rest/Move by emission mean (higher mean => faster => Move)
        means = model.means_.reshape(-1)
        order = np.argsort(means)  # low->high
        rest_state = int(order[0])
        move_state = int(order[-1])

        # Raw predicted move mask (before enforcing low_thr)
        pred_move_raw = (states == move_state)

        # metrics for retry decision
        high_mask = (speed_test >= flight_thr)
        low_mask = (speed_test < low_thr)

        recall_high_speed_as_move = float(np.mean(pred_move_raw[high_mask])) if np.any(high_mask) else float("nan")
        false_move_on_low = float(np.mean(pred_move_raw[low_mask])) if np.any(low_mask) else float("nan")

        result = {
            "model": model,
            "states": states,
            "rest_state": rest_state,
            "move_state": move_state,
            "means": means,
            "attempt": attempt,
            "rs": rs,
            "init_kwargs": init_kwargs,
            "recall_high_speed_as_move": recall_high_speed_as_move,
            "false_move_on_low": false_move_on_low,
        }

        # keep best (min false_move_on_low), in case all fail
        if best is None:
            best = result
        else:
            # compare with nan-safe
            a = best["false_move_on_low"]
            b = false_move_on_low
            if (np.isnan(a) and not np.isnan(b)) or (not np.isnan(b) and b < a):
                best = result

        # Stop early if acceptable
        if np.isnan(false_move_on_low) or false_move_on_low <= fail_false_move_on_low:
            return result

    return best


def plot_validate3(
    speed: np.ndarray,
    labels: np.ndarray,
    bird_id: str,
    save_path: str,
    low_thr: float,
    flight_thr: float,
    limit: int = 1000,
):
    """
    Plot speed over time with labels (Rest/Forage/Flight).
    中文：画 speed 时间序列，用颜色标注预测行为。
    """
    n = min(limit, len(speed))
    x = np.arange(n)
    speed_sub = speed[:n]
    lab_sub = labels[:n]

    plt.figure(figsize=(16, 6))
    plt.plot(x, speed_sub, color="gray", alpha=0.25, label="Speed (m/s)")

    palette = {"Rest": "blue", "Forage": "orange", "Flight": "red"}
    for name in ["Rest", "Forage", "Flight"]:
        mask = (lab_sub == name)
        if np.any(mask):
            plt.scatter(x[mask], speed_sub[mask], s=18, alpha=0.9, color=palette[name], label=f"{name} (Pred)")

    plt.axhline(low_thr, color="black", linestyle="--", linewidth=1, alpha=0.5, label=f"{low_thr} m/s")
    plt.axhline(flight_thr, color="black", linestyle=":", linewidth=1, alpha=0.5, label=f"{flight_thr} m/s")

    plt.title(f"validate3 LOBO: {bird_id}  (2-state HMM Rest/Move + threshold split)")
    plt.xlabel("Time step (1 min if resampled at 60s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# Main validation
# ----------------------------
def validate3(
    n_folds: int = 5,
    random_state: int = 42,
    n_iter: int = 100,
    eps: float = 1e-3,
    low_thr: float = 2.5,
    flight_thr: float = 10.0,
    max_retries: int = 5,
    fail_false_move_on_low: float = 0.5,
):
    # dynamic project root (…/movebank-bird-behavior-ml)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "GPS tracking of guanay cormorants.csv")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    raw_df = pd.read_csv(data_path)

    print("Preprocessing data (resample=True)...")
    processor = TrajectoryProcessor()
    df = processor.preprocess(raw_df, do_resample=True)
    df = ensure_log_speed(df, eps=eps)

    # Global sanity
    print("\nGlobal speed quantiles (m/s):")
    print(df["speed"].quantile([0.5, 0.9, 0.95, 0.99]))
    print("Global max speed:", float(df["speed"].max()))
    print("Global frac speed>=10:", float((df["speed"] >= flight_thr).mean()))

    # Individuals
    ids = np.array(sorted(df["individual-local-identifier"].unique()))
    print(f"Dataset contains {len(ids)} individuals.")
    if len(ids) < 2:
        raise ValueError("Need at least 2 individuals for LOBO validation.")

    rng = np.random.default_rng(random_state)
    n_folds = min(n_folds, len(ids))
    test_ids = rng.choice(ids, size=n_folds, replace=False)
    print(f"\n--- validate3 LOBO ({n_folds} folds) ---")
    print("Test bird IDs:", test_ids)

    rows = []

    for k, test_id in enumerate(test_ids, start=1):
        print(f"\n[Fold {k}/{n_folds}] Test bird: {test_id}")

        train_df = df[df["individual-local-identifier"] != test_id]
        test_df = df[df["individual-local-identifier"] == test_id].copy()

        if len(test_df) < 50:
            print("Test track too short; skipping.")
            continue

        # Per-bird speed diagnostics
        print("Test speed quantiles:", test_df["speed"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict())
        print("Test max speed:", float(test_df["speed"].max()))
        print("Test frac speed>=10:", float((test_df["speed"] >= flight_thr).mean()))

        # Build training sequences (by individual)
        X_train_list, len_train = [], []
        for _, g in train_df.groupby("individual-local-identifier"):
            if len(g) > 50:
                X_train_list.append(g["log_speed"].to_numpy().reshape(-1, 1))
                len_train.append(len(g))

        if not X_train_list:
            print("Not enough training data; skipping fold.")
            continue

        X_train = np.vstack(X_train_list)
        X_test = test_df["log_speed"].to_numpy().reshape(-1, 1)
        len_test = [len(test_df)]
        s = test_df["speed"].to_numpy()

        # ---- GaussianHMM with retry ----
        result = fit_decode_with_retry(
            X_train=X_train,
            len_train=len_train,
            X_test=X_test,
            len_test=len_test,
            speed_test=s,
            low_thr=low_thr,
            flight_thr=flight_thr,
            n_iter=n_iter,
            tol=1e-3,
            max_retries=max_retries,
            fail_false_move_on_low=fail_false_move_on_low,
            random_state_base=random_state,
        )

        model = result["model"]
        states = result["states"]
        rest_state = result["rest_state"]
        move_state = result["move_state"]
        means = result["means"]

        print(f"[Retry info] attempt={result['attempt']} rs={result['rs']} init={result['init_kwargs']}")
        print("Emission means (log_speed):", {0: float(means[0]), 1: float(means[1])})
        print("Rest state:", rest_state, "Move state:", move_state)
        print("recall_high_speed_as_move =", result["recall_high_speed_as_move"])
        print("false_move_on_low         =", result["false_move_on_low"])

        # ---- Apply thresholds to create final labels ----
        # Step A1: raw move prediction
        pred_move = (states == move_state)

        # Step A2 (important): enforce physics constraint at low speed
        # 中文：速度 < low_thr 的点强制判为 Rest，避免 Move 吞掉低速点
        pred_move = pred_move & (s >= low_thr)

        # Step A3: split Move into Forage vs Flight
        pred_flight = pred_move & (s >= flight_thr)
        pred_forage = pred_move & (s < flight_thr)
        pred_rest = ~pred_move

        # Build string labels for plotting
        labels = np.empty(len(s), dtype=object)
        labels[pred_rest] = "Rest"
        labels[pred_forage] = "Forage"
        labels[pred_flight] = "Flight"

        counts = {
            "Rest": int(np.sum(pred_rest)),
            "Forage": int(np.sum(pred_forage)),
            "Flight": int(np.sum(pred_flight)),
        }
        print("Counts:", counts)

        # Optional: post-split sanity
        # P(speed>=flight_thr | pred=Flight) should be 1.0 by construction
        # but we can report median speed for each label
        med = {}
        for name in ["Rest", "Forage", "Flight"]:
            mask = (labels == name)
            med[name] = float(np.median(s[mask])) if np.any(mask) else float("nan")
        print("Median speed (m/s) per label:", med)

        # Save plot
        fig_path = os.path.join(out_dir, f"validate3_fold{k}_{test_id}.png")
        plot_validate3(
            speed=s,
            labels=labels,
            bird_id=str(test_id),
            save_path=fig_path,
            low_thr=low_thr,
            flight_thr=flight_thr,
            limit=1000,
        )
        print("[Done] Saved:", fig_path)

        rows.append({
            "fold": k,
            "test_bird": str(test_id),
            "attempt": int(result["attempt"]),
            "random_state": int(result["rs"]),
            "mean_logspeed_state0": float(means[0]),
            "mean_logspeed_state1": float(means[1]),
            "rest_state": int(rest_state),
            "move_state": int(move_state),
            "recall_high_speed_as_move": float(result["recall_high_speed_as_move"]) if result["recall_high_speed_as_move"] == result["recall_high_speed_as_move"] else np.nan,
            "false_move_on_low": float(result["false_move_on_low"]) if result["false_move_on_low"] == result["false_move_on_low"] else np.nan,
            "pred_rest": counts["Rest"],
            "pred_forage": counts["Forage"],
            "pred_flight": counts["Flight"],
            "median_speed_rest": med["Rest"],
            "median_speed_forage": med["Forage"],
            "median_speed_flight": med["Flight"],
        })

    summary = pd.DataFrame(rows)
    print("\n=== validate3 Summary ===")
    print(summary)

    csv_path = os.path.join(out_dir, "validate3_summary.csv")
    summary.to_csv(csv_path, index=False)
    print("Saved:", csv_path)


if __name__ == "__main__":
    validate3(
        n_folds=5,
        random_state=42,
        n_iter=100,
        eps=1e-3,
        low_thr=2.5,
        flight_thr=10.0,
        max_retries=5,
        fail_false_move_on_low=0.5,
    )
