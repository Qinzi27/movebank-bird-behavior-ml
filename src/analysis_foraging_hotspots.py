import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_density_map(x, y, title, output_path, cmap="Reds"):
    """
    使用 Gaussian KDE 绘制空间密度热力图
    """
    # 1. 计算点密度 (Gaussian KDE)
    # 这里的 xy 是数据点的坐标矩阵
    xy = np.vstack([x, y])
    try:
        # bw_method 控制平滑度，越小越细碎，越大越平滑
        kde = gaussian_kde(xy, bw_method=0.2) 
    except Exception as e:
        print(f"KDE 计算失败 (可能是数据点太少): {e}")
        return

    # 2. 创建网格 (Grid) 用于绘图
    # 在数据范围内生成 100x100 的网格
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    
    # 稍微向外扩展一点边界
    pad_x = (xmax - xmin) * 0.1
    pad_y = (ymax - ymin) * 0.1
    xmin -= pad_x; xmax += pad_x
    ymin -= pad_y; ymax += pad_y

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # 3. 计算网格上每个点的密度值 (Z)
    Z = np.reshape(kde(positions).T, X.shape)

    # 4. 绘图
    plt.figure(figsize=(10, 8))
    
    # 画热力图底色
    plt.imshow(np.rot90(Z), cmap=cmap, extent=[xmin, xmax, ymin, ymax], aspect='auto', alpha=0.9)
    
    # 叠加原始散点（可选，设为黑色微小点，增加真实感）
    plt.scatter(x, y, c='k', s=1, alpha=0.1)

    # 装饰
    plt.title(title, fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    cbar = plt.colorbar(label="Density")
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    # 1. 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, "outputs", "main3_predictions.csv")
    out_dir = os.path.join(base_dir, "outputs")
    
    if not os.path.exists(input_csv):
        print(f"Error: 找不到输入文件 {input_csv}")
        print("请先运行 python src/main3.py 生成预测结果。")
        return

    # 2. 读取数据
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 3. 筛选数据
    # 提取所有被标记为 'Forage' 的点
    forage_df = df[df["behavior"] == "Forage"]
    
    # 提取所有被标记为 'Rest' 的点 (作为对比)
    rest_df = df[df["behavior"] == "Rest"]

    print(f"Total points: {len(df)}")
    print(f"Forage points: {len(forage_df)}")
    print(f"Rest points: {len(rest_df)}")

    if len(forage_df) < 10:
        print("警告：觅食点太少，无法进行热点分析。")
        return

    # 4. 绘制并保存热力图
    # 图1: 觅食热点 (核心产出)
    plot_density_map(
        forage_df["location-long"], 
        forage_df["location-lat"], 
        title="Foraging Hotspots (Density of 'Forage' Behavior)",
        output_path=os.path.join(out_dir, "analysis_hotspot_forage.png"),
        cmap="Reds"  # 红色代表热点
    )

    # 图2: 休息热点 (对比组，通常在岛屿/岸边)
    if len(rest_df) > 10:
        plot_density_map(
            rest_df["location-long"], 
            rest_df["location-lat"], 
            title="Resting Locations (Roosting Sites)",
            output_path=os.path.join(out_dir, "analysis_hotspot_rest.png"),
            cmap="Blues" # 蓝色代表静止
        )

if __name__ == "__main__":
    main()