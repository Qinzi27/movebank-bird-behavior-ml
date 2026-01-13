import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Allow running both as `python -m src.main` (package) and `python src/main.py` (script)
try:
    from src.features import TrajectoryProcessor
    from src.models import MovementHMM
except ModuleNotFoundError:
    # When executed as a script, `src/` is on sys.path, so import local modules directly
    from features import TrajectoryProcessor
    from models import MovementHMM

def main():
    # 1. 配置路径
    RAW_DATA_PATH = "data/raw/GPS tracking of guanay cormorants.csv"
    
    # 2. 数据加载
    print(f"Loading data from {RAW_DATA_PATH}...")
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print("错误：找不到文件")
        return

    # 3. 特征工程
    processor = TrajectoryProcessor()
    df = processor.preprocess(raw_df, do_resample=True)
    
    # [新增诊断]：检查是否有飞行数据
    max_speed = df["speed"].max()
    print(f"Data processed. Shape: {df.shape}")
    print(f"Max Speed in data: {max_speed:.2f} m/s")
    if max_speed < 5:
        print("警告：数据中最大速度过小，可能是重采样导致飞行片段丢失，或单位错误！")
    
    # 4. 准备 HMM 输入
    X_list = []
    lengths = []
    for ind_id, group in df.groupby("individual-local-identifier"):
        if len(group) > 50:
            X_list.append(group["log_speed"].values.reshape(-1, 1))
            lengths.append(len(group))
            
    if not X_list:
        print("没有足够的数据。")
        return

    X_train = np.vstack(X_list)
    
    # 5. 模型训练 - [修改] 尝试 3 个状态以捕捉飞行
    print("\nTraining HMM (trying 3 components to capture Flight)...")
    hmm = MovementHMM(n_components=3, n_iter=100) # 改为 3
    hmm.fit(X_train, lengths)
    
    # 6. 推断
    print("Predicting states...")
    states = hmm.predict(X_train, lengths)
    
    # 7. 简易可视化
    plt.figure(figsize=(12, 6))
    subset_len = min(1000, len(X_train)) # 画前1000个点
    subset_X = X_train[:subset_len].flatten()
    subset_states = states[:subset_len]
    
    plt.plot(subset_X, label="Log Speed", color='gray', alpha=0.5)
    plt.scatter(range(subset_len), subset_X, c=subset_states, cmap='viridis', s=10, label="State")
    plt.title(f"HMM Decoding (First {subset_len} points)")
    plt.xlabel("Time Step (60s)")
    plt.ylabel("Log Speed")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()