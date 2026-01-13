import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.features import TrajectoryProcessor
except ModuleNotFoundError:
    from features import TrajectoryProcessor

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None


def _means_1d(model) -> np.ndarray:
    if hasattr(model, "means_"):
        return model.means_.reshape(-1)
    raise AttributeError("Model has no means_. Is hmmlearn installed?")


def fit_two_state_hmm(df: pd.DataFrame, random_state: int = 42, n_iter: int = 100):
    if GaussianHMM is None:
        raise RuntimeError("hmmlearn not available. Please `pip install hmmlearn`.")

    X_list, lengths = [], []
    for _, g in df.groupby("individual-local-identifier"):
        g = g.sort_values("timestamp")
        if len(g) > 50:
            X_list.append(g["log_speed"].values.reshape(-1, 1))
            lengths.append(len(g))

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


def main(flight_thr: float = 10.0):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "GPS tracking of guanay cormorants.csv")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading:", data_path)
    raw_df = pd.read_csv(data_path)

    print("Preprocessing (resample=True)...")
    processor = TrajectoryProcessor()
    df = processor.preprocess(raw_df, do_resample=True)

    print("Training 2-state HMM (Rest vs Move)...")
    model, rest_state, move_state, means = fit_two_state_hmm(df, random_state=42, n_iter=100)
    print("Emission means (log_speed):", {i: float(m) for i, m in enumerate(means)})
    print("Rest state:", rest_state, "Move state:", move_state)

    all_out = []
    for bird_id, g in df.groupby("individual-local-identifier"):
        g = g.sort_values("timestamp").copy()
        X = g["log_speed"].values.reshape(-1, 1)
        states = model.predict(X, lengths=[len(g)])

        speed = g["speed"].to_numpy()
        pred_rest = states == rest_state
        pred_move = states == move_state
        pred_flight = pred_move & (speed >= flight_thr)
        pred_forage = pred_move & (speed < flight_thr)

        label = np.empty(len(g), dtype=object)
        label[pred_rest] = "Rest"
        label[pred_flight] = "Flight"
        label[pred_forage] = "Forage"

        out = g[[
            "individual-local-identifier", "timestamp",
            "location-long", "location-lat",
            "speed", "log_speed"
        ]].copy()
        out["hmm_state"] = states
        out["behavior"] = label
        all_out.append(out)

    pred_df = pd.concat(all_out, ignore_index=True)
    csv_path = os.path.join(out_dir, "main3_predictions.csv")
    pred_df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # example plot: pick bird with most points
    top_bird = pred_df["individual-local-identifier"].value_counts().index[0]
    ex = pred_df[pred_df["individual-local-identifier"] == top_bird].sort_values("timestamp").head(1200)

    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(len(ex)), ex["speed"].to_numpy(), color="gray", alpha=0.25, label="Speed (m/s)")
    palette = {"Rest": "blue", "Forage": "orange", "Flight": "red"}
    for name in ["Rest", "Forage", "Flight"]:
        m = ex["behavior"].to_numpy() == name
        if m.any():
            plt.scatter(np.arange(len(ex))[m], ex["speed"].to_numpy()[m], s=18, color=palette[name], label=name)

    plt.axhline(2.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="2.5 m/s")
    plt.axhline(10.0, color="black", linestyle=":", linewidth=1, alpha=0.5, label="10 m/s")
    plt.title(f"main3 example bird: {top_bird} (2-state HMM + threshold)")
    plt.xlabel("Time step (1 min if resampled at 60s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "main3_example_plot.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved:", fig_path)


if __name__ == "__main__":
    main(flight_thr=10.0)
