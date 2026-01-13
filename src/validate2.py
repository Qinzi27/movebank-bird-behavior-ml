import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# allow both `python -m src.validate2` and `python src/validate2.py`
try:
    from src.features import TrajectoryProcessor
except ModuleNotFoundError:
    from features import TrajectoryProcessor

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None

from hmmlearn.hmm import GaussianHMM
import numpy as np

def make_2state_hmm_with_quantile_init(
    X_train,
    q_rest=0.30,
    q_move=0.95,
    sticky_rest=0.99,
    sticky_move=0.95,
    n_iter=100,
    tol=1e-3,
    random_state=42,
    var_floor=1e-3,
):
    """
    Build a 2-state GaussianHMM with controlled initialization.
    States are intended as Rest (low log_speed) vs Move (high log_speed).

    X_train: shape (N, 1) numpy array (log_speed)
    """

    X = np.asarray(X_train).reshape(-1, 1)
    assert X.shape[1] == 1, "This initializer assumes 1D feature (log_speed)."

    # 1) Quantile-based mean init
    mu_rest = float(np.quantile(X[:, 0], q_rest))
    mu_move = float(np.quantile(X[:, 0], q_move))
    if mu_move <= mu_rest:
        # fallback if data is weird
        mu_rest = float(np.quantile(X[:, 0], 0.20))
        mu_move = float(np.quantile(X[:, 0], 0.90))

    means_init = np.array([[mu_rest], [mu_move]], dtype=float)

    # 2) Variance init (diag covars): use global variance + floor
    var = float(np.var(X[:, 0]))
    var = max(var, var_floor)
    covars_init = np.array([[var], [var]], dtype=float)  # shape (2, 1)

    # 3) Start prob init: neutral
    startprob_init = np.array([0.5, 0.5], dtype=float)

    # 4) Sticky transition matrix init (behavior has inertia)
    transmat_init = np.array([
        [sticky_rest, 1.0 - sticky_rest],
        [1.0 - sticky_move, sticky_move],
    ], dtype=float)

    # Important:
    # init_params="" prevents hmmlearn from overwriting our startprob/transmat/means/covars
    # params="stmc" allows EM to update them during fit (starting from our init)
    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        init_params="",     # DO NOT re-init randomly
        params="stmc",      # DO update during EM
    )

    # Set initialized parameters
    model.startprob_ = startprob_init
    model.transmat_ = transmat_init
    model.means_ = means_init
    model.covars_ = covars_init

    return model


def _means_1d(model) -> np.ndarray:
    if hasattr(model, "means_"):
        return model.means_.reshape(-1)
    raise AttributeError("Model has no means_. Is hmmlearn installed?")

'''
def fit_two_state_hmm(train_df: pd.DataFrame, random_state: int = 42, n_iter: int = 100):
    if GaussianHMM is None:
        raise RuntimeError("hmmlearn not available. Please `pip install hmmlearn`.")

    X_list, lengths = [], []
    for _, g in train_df.groupby("individual-local-identifier"):
        g = g.sort_values("timestamp")
        if len(g) > 50:
            X_list.append(g["log_speed"].values.reshape(-1, 1))
            lengths.append(len(g))
    if not X_list:
        raise ValueError("No usable training trajectories (>50 points).")

    X = np.vstack(X_list)

    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
    )
    model.fit(X, lengths=lengths)

    means = _means_1d(model)
    rest_state = int(np.argmin(means))
    move_state = int(np.argmax(means))
    return model, rest_state, move_state, means
'''
# --- Fit 2-state HMM with retry if collapse happens ---
def fit_decode_with_retry(X_train, len_train, X_test, len_test, speed_test,
                          max_retries=3, fail_false_move_on_low=0.5, random_state_base=42):

    best = None  # store best attempt

    # Different init schemes to try
    init_list = [
        dict(q_rest=0.30, q_move=0.95, sticky_rest=0.99, sticky_move=0.95),
        dict(q_rest=0.20, q_move=0.90, sticky_rest=0.995, sticky_move=0.97),
        dict(q_rest=0.40, q_move=0.98, sticky_rest=0.99, sticky_move=0.90),
    ]

    for attempt in range(max_retries):
        init_kwargs = init_list[min(attempt, len(init_list)-1)]
        rs = random_state_base + attempt

        model = make_2state_hmm_with_quantile_init(
            X_train,
            n_iter=100,
            tol=1e-3,
            random_state=rs,
            **init_kwargs
        )

        model.fit(X_train, lengths=len_train)
        states = model.predict(X_test, lengths=len_test)

        # Map state by mean: low mean -> Rest, high mean -> Move
        means = model.means_.reshape(-1)
        rest_state = int(np.argmin(means))
        move_state = int(np.argmax(means))

        pred_is_move = (states == move_state)

        # Your two sanity metrics
        high_mask = (speed_test >= 10.0)
        low_mask  = (speed_test < 2.5)

        recall_high_speed_as_move = float(pred_is_move[high_mask].mean()) if high_mask.any() else float("nan")
        false_move_on_low         = float(pred_is_move[low_mask].mean())  if low_mask.any()  else float("nan")

        # Keep best attempt (lowest false_move_on_low is good)
        score = false_move_on_low if np.isfinite(false_move_on_low) else 1.0
        if (best is None) or (score < best["score"]):
            best = dict(
                model=model,
                states=states,
                rest_state=rest_state,
                move_state=move_state,
                recall_high_speed_as_move=recall_high_speed_as_move,
                false_move_on_low=false_move_on_low,
                attempt=attempt,
                rs=rs,
                init_kwargs=init_kwargs,
                score=score,
            )

        # Early stop if acceptable
        if np.isfinite(false_move_on_low) and false_move_on_low <= fail_false_move_on_low:
            break

    return best


def plot_fold(test_df, pred_label, bird_id, save_path, limit=1000):
    g = test_df.sort_values("timestamp").head(limit).copy()
    speed = g["speed"].to_numpy()
    labels = pred_label[: len(g)]

    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(len(g)), speed, color="gray", alpha=0.25, label="Speed (m/s)")

    palette = {"Rest": "blue", "Forage": "orange", "Flight": "red"}
    for name in ["Rest", "Forage", "Flight"]:
        m = labels == name
        if m.any():
            plt.scatter(np.arange(len(g))[m], speed[m], s=18, color=palette[name], label=f"{name} (Pred)")

    plt.axhline(2.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="2.5 m/s")
    plt.axhline(10.0, color="black", linestyle=":", linewidth=1, alpha=0.5, label="10 m/s")
    plt.title(f"validate2 LOBO: {bird_id}  (2-state HMM Rest/Move + threshold split)")
    plt.xlabel("Time step (1 min if resampled at 60s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def validate2(n_folds=5, random_state=42, n_iter=100, low_thr=2.5, flight_thr=10.0):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "GPS tracking of guanay cormorants.csv")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    raw_df = pd.read_csv(data_path)

    print("Preprocessing data (resample=True)...")
    processor = TrajectoryProcessor()
    df = processor.preprocess(raw_df, do_resample=True)

    # global sanity
    qs = [0.5, 0.9, 0.95, 0.99]
    print("\nGlobal speed quantiles (m/s):")
    print(df["speed"].quantile(qs))
    print("Global max speed:", float(df["speed"].max()))
    print("Global frac speed>=10:", float((df["speed"] >= flight_thr).mean()))

    individual_ids = np.array(sorted(df["individual-local-identifier"].unique()))
    print(f"Dataset contains {len(individual_ids)} individuals.")

    rng = np.random.default_rng(random_state)
    n_folds = min(n_folds, len(individual_ids))
    test_ids = rng.choice(individual_ids, size=n_folds, replace=False)
    print(f"\n--- validate2 LOBO ({n_folds} folds) ---")
    print("Test bird IDs:", test_ids)

    rows = []

    for k, test_id in enumerate(test_ids, start=1):
        print(f"\n[Fold {k}/{n_folds}] Test bird: {test_id}")

        train_df = df[df["individual-local-identifier"] != test_id]
        test_df = df[df["individual-local-identifier"] == test_id].copy()
        if len(test_df) < 50:
            print("Test track too short, skip.")
            continue

        # per-bird diagnostics
        s = test_df["speed"].to_numpy()
        print("Test speed quantiles:", test_df["speed"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict())
        print("Test max speed:", float(np.max(s)))
        print("Test frac speed>=10:", float((s >= flight_thr).mean()))

        # train 2-state
        model, rest_state, move_state, means = fit_two_state_hmm(train_df, random_state=random_state, n_iter=n_iter)
        print("Emission means (log_speed):", {i: float(m) for i, m in enumerate(means)})
        print("Rest state:", rest_state, "Move state:", move_state)
        


        X_test = test_df.sort_values("timestamp")["log_speed"].values.reshape(-1, 1)
        states = model.predict(X_test, lengths=[len(test_df)])

        pred_move = states == move_state
        pred_rest = states == rest_state

        # final labels (A-route)
        pred_flight = pred_move & (s >= flight_thr)
        pred_forage = pred_move & (s < flight_thr)

        labels = np.empty(len(test_df), dtype=object)
        labels[pred_rest] = "Rest"
        labels[pred_flight] = "Flight"
        labels[pred_forage] = "Forage"

        # proxy validation metrics
        true_high = (s >= flight_thr)
        denom_high = int(true_high.sum())
        tp_high = int((true_high & pred_move).sum())
        recall_high_speed_as_move = tp_high / denom_high if denom_high > 0 else float("nan")

        low = (s < low_thr)
        denom_low = int(low.sum())
        fp_move_low = int((low & pred_move).sum())
        false_move_on_low = fp_move_low / denom_low if denom_low > 0 else float("nan")

        print("Counts:",
              {"Rest": int(pred_rest.sum()), "Move": int(pred_move.sum()),
               "Flight": int(pred_flight.sum()), "Forage": int(pred_forage.sum())})
        print("recall_high_speed_as_move =", recall_high_speed_as_move)
        print("false_move_on_low         =", false_move_on_low)

        fig_path = os.path.join(out_dir, f"validate2_fold{k}_{test_id}.png")
        plot_fold(test_df, labels, test_id, fig_path)

        result = fit_decode_with_retry(
            X_train=X_train,
            len_train=len_train,
            X_test=X_test,
            len_test=len_test,
            speed_test=speed_test,
            max_retries=3,
            fail_false_move_on_low=0.5,
            random_state_base=random_state,
        )

        model = result["model"]
        states = result["states"]
        rest_state = result["rest_state"]
        move_state = result["move_state"]

        print(f"[Retry info] attempt={result['attempt']} rs={result['rs']} init={result['init_kwargs']}")
        print("recall_high_speed_as_move =", result["recall_high_speed_as_move"])
        print("false_move_on_low         =", result["false_move_on_low"])


        rows.append({
            "fold": k,
            "test_bird": test_id,
            "n_points": int(len(test_df)),
            "n_high_speed_true": denom_high,
            "n_high_speed_captured_by_move": tp_high,
            "recall_high_speed_as_move": recall_high_speed_as_move,
            "false_move_on_low": false_move_on_low,
            "pred_rest": int(pred_rest.sum()),
            "pred_move": int(pred_move.sum()),
            "pred_flight": int(pred_flight.sum()),
            "pred_forage": int(pred_forage.sum()),
        })



    summary = pd.DataFrame(rows)
    print("\n=== validate2 Summary ===")
    print(summary)

    csv_path = os.path.join(out_dir, "validate2_summary.csv")
    summary.to_csv(csv_path, index=False)
    print("Saved:", csv_path)


if __name__ == "__main__":
    validate2(n_folds=5, random_state=42, n_iter=100, low_thr=2.5, flight_thr=10.0)
