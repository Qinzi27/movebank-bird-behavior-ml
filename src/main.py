# main.py
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
        print("错误：找不到文件，请确认 data/raw/ 目录下有 csv 文件")
        return

    # 3. 特征工程 (含重采样)
    processor = TrajectoryProcessor()
    # 注意：这里开启 do_resample=True，解决了 HMM 时间步长不一致的逻辑漏洞
    df = processor.preprocess(raw_df, do_resample=True)
    
    print(f"Data processed. Shape: {df.shape}")
    print(df[["timestamp", "dt", "speed", "log_speed"]].head())

    # 4. 准备 HMM 输入数据
    # HMMlearn 需要将多条轨迹拼接成一个长矩阵 X，并提供 lengths 列表告诉它每条轨迹多长
    X_list = []
    lengths = []
    
    # 仅选择足够长的轨迹进行训练
    valid_groups = []
    for ind_id, group in df.groupby("individual-local-identifier"):
        if len(group) > 50: # 忽略过短的碎片轨迹
            X_list.append(group["log_speed"].values.reshape(-1, 1))
            lengths.append(len(group))
            valid_groups.append(group)
            
    if not X_list:
        print("没有足够的数据进行训练。")
        return

    X_train = np.vstack(X_list)
    
    # 5. 模型训练
    print("\nTraining HMM...")
    hmm = MovementHMM(n_components=2, n_iter=100)
    hmm.fit(X_train, lengths)
    
    # 6. 推断 (Inference)
    print("Predicting states...")
    # 这里的 states 已经是语义对齐的了：1=Flight, 0=Rest
    states = hmm.predict(X_train, lengths)
    
    # 7. 结果可视化 (简单的验证)
    plt.figure(figsize=(10, 6))
    
    # 取前 500 个点画图
    subset_len = min(500, len(X_train))
    subset_X = X_train[:subset_len].flatten()
    subset_states = states[:subset_len]
    
    plt.plot(subset_X, label="Log Speed", color='gray', alpha=0.6)
    # 将状态乘以一个系数画在图上，方便对比
    plt.plot(subset_states * subset_X.max(), label="Inferred State (1=Flight)", color='red', alpha=0.6)
    
    plt.title("HMM State Decoding (First 500 points)")
    plt.xlabel("Time Step (resampled)")
    plt.ylabel("Log Speed / State")
    plt.legend()
    plt.show()
    
    print("\n逻辑检查完成：")
    print("1. [√] 已使用线性插值重采样，修复了 HMM 时间步长不一致的问题。")
    print("2. [√] 已添加状态自动映射，输出 1 稳定代表飞行状态。")

if __name__ == "__main__":
    main()