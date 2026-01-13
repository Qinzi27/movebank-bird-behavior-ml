import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Allow running both as `python -m src.validate` and `python src/validate.py`
try:
    from src.features import TrajectoryProcessor
    from src.models import MovementHMM
except ModuleNotFoundError:
    from features import TrajectoryProcessor
    from models import MovementHMM

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None


# --- Helper functions for state mapping and speed reconstruction ---
def _get_hmm_means(hmm_obj):
    """Return emission means array of shape (n_components, n_features) from wrapper or raw hmmlearn model."""
    if hasattr(hmm_obj, "means_"):
        return hmm_obj.means_
    # common wrapper patterns
    if hasattr(hmm_obj, "model") and hasattr(hmm_obj.model, "means_"):
        return hmm_obj.model.means_
    if hasattr(hmm_obj, "hmm") and hasattr(hmm_obj.hmm, "means_"):
        return hmm_obj.hmm.means_
    raise AttributeError("Cannot find HMM means_. Please expose means_ on MovementHMM.")


def map_states_by_mean(hmm_obj):
    """Map arbitrary state ids to semantic names by ordering mean log-speed."""
    means = _get_hmm_means(hmm_obj).reshape(-1)
    order = np.argsort(means)  # low -> high
    mapping = {int(order[0]): "Rest", int(order[-1]): "Flight"}
    if len(order) == 3:
        mapping[int(order[1])] = "Forage"  # middle activity
    else:
        # for 2-state, only Rest/Flight
        pass
    return mapping, means


def inverse_log_speed(log_speed, eps=1e-3):
    """Recover speed from log(speed+eps)."""
    return np.maximum(0.0, np.exp(log_speed) - eps)

def validate_generalization(n_folds=5, random_state=42, n_components=3, n_iter=50):
    # 1. 动态获取数据路径 (兼容不同操作系统)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "GPS tracking of guanay cormorants.csv")

    print(f"Loading data from: {data_path}")
    try:
        raw_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    # 2. 数据预处理
    print("Preprocessing data (resample=True)...")
    processor = TrajectoryProcessor()
    df = processor.preprocess(raw_df, do_resample=True)

    # Quick data sanity check: do we even see high speeds after preprocessing?
    qs = [0.5, 0.9, 0.95, 0.99]
    print("\nGlobal speed quantiles after preprocessing (m/s):")
    print(df["speed"].quantile(qs))
    print("Global max speed:", float(df["speed"].max()))
    print("Global fraction speed>=2.5:", float((df["speed"] >= 2.5).mean()))
    print("Global fraction speed>=10 :", float((df["speed"] >= 10.0).mean()))

    # 获取所有个体的 ID
    individual_ids = np.array(sorted(df["individual-local-identifier"].unique()))
    print(f"Dataset contains {len(individual_ids)} individuals.")

    if len(individual_ids) < 2:
        print("Error: Need at least 2 individuals to perform cross-validation.")
        return

    # 3. 选择多个未见个体做验证（更稳定）
    rng = np.random.default_rng(random_state)
    n_folds = min(n_folds, len(individual_ids))
    test_ids = rng.choice(individual_ids, size=n_folds, replace=False)

    print(f"\n--- Validation Setup (Leave-One-Bird-Out, {n_folds} folds) ---")
    print(f"Test bird IDs (unseen): {test_ids}")

    # 输出目录
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    fold_rows = []

    for k, test_bird_id in enumerate(test_ids, start=1):
        print(f"\n[Fold {k}/{n_folds}] Test bird: {test_bird_id}")

        train_bird_ids = [x for x in individual_ids if x != test_bird_id]

        # --- 训练集 ---
        train_df = df[df["individual-local-identifier"].isin(train_bird_ids)]
        X_train, len_train = [], []
        for _, g in train_df.groupby("individual-local-identifier"):
            if len(g) > 50:
                X_train.append(g["log_speed"].values.reshape(-1, 1))
                len_train.append(len(g))

        if not X_train:
            print("Not enough training data for this fold.")
            continue

        X_train = np.vstack(X_train)

        # --- 测试集 ---
        test_df = df[df["individual-local-identifier"] == test_bird_id].copy()
        if len(test_df) < 50:
            print("Test track too short; skipping.")
            continue

        # Per-bird data diagnostics (before modeling)
        s = test_df["speed"].to_numpy()
        print("Test bird speed quantiles:", test_df["speed"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict())
        print("Test bird max speed:", float(np.max(s)))
        print("Test bird fraction speed>=2.5:", float((s >= 2.5).mean()))
        print("Test bird fraction speed>=10 :", float((s >= 10.0).mean()))

        X_test = test_df["log_speed"].values.reshape(-1, 1)
        len_test = [len(test_df)]

        # 4. 训练模型
        print("Training HMM...")
        if GaussianHMM is None:
            print("Warning: hmmlearn is not available; falling back to MovementHMM wrapper.")
            model = MovementHMM(n_components=n_components, n_iter=n_iter)
            model.fit(X_train, len_train)
            print("Decoding states on unseen bird...")
            states = model.predict(X_test, len_test)
        else:
            # Use raw hmmlearn to avoid wrapper-specific constraints (e.g., freezing means)
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="diag",
                n_iter=n_iter,
                random_state=random_state,
                tol=1e-3,
            )
            model.fit(X_train, lengths=len_train)
            print("Decoding states on unseen bird...")
            states = model.predict(X_test, lengths=len_test)

        # 6. 状态语义映射（用均值排序，解决 state id 任意性）
        state_name_map, means = map_states_by_mean(model)

        # 7. 定量 sanity checks（无真值情况下的稳定检查）
        speed_test = test_df["speed"].to_numpy()
        # 选出预测为 Flight 的点
        flight_state_ids = [sid for sid, name in state_name_map.items() if name == "Flight"]
        flight_mask = np.isin(states, flight_state_ids)

        # 统计：Flight 点中超过阈值的比例
        thr1, thr2 = 2.5, 10.0
        if flight_mask.any():
            p_thr1 = float((speed_test[flight_mask] >= thr1).mean())
            p_thr2 = float((speed_test[flight_mask] >= thr2).mean())
            flight_median = float(np.median(speed_test[flight_mask]))
        else:
            p_thr1 = p_thr2 = float("nan")
            flight_median = float("nan")

        # 各状态的速度中位数（应当 Rest < Forage < Flight）
        per_state_median = {}
        for sid in np.unique(states):
            per_state_median[int(sid)] = float(np.median(speed_test[states == sid]))
        print("Per-state median speed (m/s):", per_state_median)
        print(f"Flight points: {flight_mask.sum()} / {len(states)}")
        print(f"P(speed>=2.5 | pred=Flight) = {p_thr1:.3f}")
        print(f"P(speed>=10  | pred=Flight) = {p_thr2:.3f}")
        print(f"Median speed (pred=Flight)   = {flight_median:.3f} m/s")

        # Predicted state usage
        unique, counts = np.unique(states, return_counts=True)
        state_counts = {int(u): int(c) for u, c in zip(unique, counts)}
        print("Predicted state counts:", state_counts)

        # 8. 可视化
        save_path = os.path.join(out_dir, f"validation_fold{k}_{test_bird_id}.png")
        plot_validation(test_df, X_test, states, test_bird_id, state_name_map, save_path)

        fold_rows.append({
            "fold": k,
            "test_bird": test_bird_id,
            "flight_points": int(flight_mask.sum()),
            "total_points": int(len(states)),
            "p_speed_ge_2p5_given_flight": p_thr1,
            "p_speed_ge_10_given_flight": p_thr2,
            "median_speed_given_flight": flight_median,
            "pred_state_counts": str(state_counts),
        })

    if fold_rows:
        summary = pd.DataFrame(fold_rows)
        print("\n=== Validation Summary (across folds) ===")
        print(summary)
        csv_path = os.path.join(out_dir, "validation_summary.csv")
        summary.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    else:
        print("No folds were completed. Please check data length thresholds.")

def plot_validation(test_df, X_test, states, bird_id, state_name_map, save_path):
    """绘制验证结果图：用语义映射后的状态标签着色。"""
    plt.figure(figsize=(14, 6))

    # 为了看清细节，只画前 1000 个点
    limit = min(1000, len(test_df))

    eps = 1e-3
    # 还原真实速度：speed = exp(log_speed) - eps
    real_speed = inverse_log_speed(X_test[:limit].flatten(), eps=eps)
    subset_states = states[:limit]

    # 背景速度曲线
    plt.plot(range(limit), real_speed, color="gray", alpha=0.25, label="Speed (m/s)")

    # 为每个 state 自动分配颜色（固定调色盘，避免 state id 变化导致颜色语义错乱）
    palette = {
        "Rest": "blue",
        "Forage": "orange",
        "Flight": "red",
    }

    # 逐状态画点
    for sid in sorted(np.unique(subset_states)):
        name = state_name_map.get(int(sid), f"State {sid}")
        mask = subset_states == sid
        plt.scatter(
            np.arange(limit)[mask],
            real_speed[mask],
            color=palette.get(name, "black"),
            s=18,
            label=f"{name} (Pred)",
            zorder=5,
            alpha=0.9,
        )

    # 画两条参考阈值线（用于肉眼 sanity check）
    plt.axhline(2.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="2.5 m/s threshold")
    plt.axhline(10.0, color="black", linestyle=":", linewidth=1, alpha=0.5, label="10 m/s reference")

    plt.title(f"Leave-one-bird-out validation on unseen bird: {bird_id}\n(states mapped by emission mean: Rest < Forage < Flight)")
    plt.ylabel("Speed (m/s)")
    plt.xlabel("Time step (1 min if resampled at 60s)")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Done] Saved validation plot: {save_path}")

if __name__ == "__main__":
    validate_generalization(n_folds=5, random_state=42, n_components=3, n_iter=50)