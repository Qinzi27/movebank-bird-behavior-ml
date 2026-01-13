# Guanay Cormorant Behavior Analysis: A Hybrid Physics-Informed HMM Approach
# 鸟类行为分析项目总结：基于物理约束的混合 HMM 方法

## 1. Project Overview (项目概览)
This project implements an unsupervised machine learning pipeline to decode behavioral states (Resting, Foraging, Flight) from GPS tracking data of Guanay Cormorants (*Leucocarbo bougainvillii*). 
本项目实现了一套无监督机器学习流程，用于从海鸟 GPS 轨迹数据中解码行为状态（休息、觅食、飞行）。

Unlike purely data-driven approaches, we adopt a **Hybrid Strategy** combining probabilistic modeling (Hidden Markov Models) with biophysical constraints to ensure ecological interpretability and model stability.
与纯数据驱动的方法不同，我们采用了一种**混合策略**，结合了概率建模（隐马尔可夫模型 HMM）与生物物理约束，以确保结果的生态学可解释性和模型稳定性。

---

## 2. Methodology (方法论)

### 2.1 Data Preprocessing (数据预处理)
* **Resampling (重采样)**: Irregular GPS fixes are linearly interpolated to a regular 60s interval to satisfy HMM assumptions.
    * *Implementation*: `src/features.py` (TrajectoryProcessor)
* **Feature Engineering (特征工程)**: 
    * Speed ($v$) calculated via Haversine distance.
    * Log-transformed speed ($\log(1+v)$) used as the emission feature to handle heavy-tailed distributions.

### 2.2 The Hybrid Model (混合模型架构)
We utilize a **2-stage classification system** instead of a direct 3-state HMM, addressing the issue where unsupervised models often fail to distinguish "Foraging" from "Flight" purely based on speed distributions.
我们采用**两阶段分类系统**取代直接的 3 状态 HMM，解决了无监督模型常因速度分布重叠而无法有效区分“觅食”与“飞行”的问题。

1.  **Stage 1: Latent State Discovery (HMM)**
    * **Model**: 2-state Gaussian HMM.
    * **States**: Automatically learns a "Low Energy/Stationary" state and a "High Energy/Active" state.
    * **Constraint**: The state with the lower emission mean is strictly mapped to "Rest".
    
2.  **Stage 2: Biophysical Thresholding (Rule-based)**
    * The "Active" state is further split based on an ecological speed threshold ($v_{flight} \approx 10 m/s$):
        * **Flight**: Active State + $v \ge 10 m/s$ (Commuting).
        * **Forage**: Active State + $v < 10 m/s$ (Area-restricted search).

### 2.3 Robust Validation (鲁棒性验证)
* **Method**: Leave-One-Bird-Out (LOBO) Cross-Validation.
* **Sanity Checks**: implemented in `src/validate3_updated.py`.
    * **Retry Mechanism**: If the HMM converges to a biologically impossible solution (e.g., classifying static points as "Moving"), the model automatically retries with different initializations.
    * **Correction**: Physics-based correction ensures points with $v \approx 0$ are never classified as movement.

---

## 3. Key Findings & Biological Insights (主要发现与生物学洞察)

Applying this pipeline (`src/main3.py`) yields the following insights:

1.  **Natural Dichotomy (行为的自然二分性)**:
    The dataset naturally clusters into two dominant modes (Stationary vs. Active), confirming that birds operate in distinct energetic phases.
    数据集自然聚类为两种主导模式（静止 vs 活跃），证实了鸟类在截然不同的能量阶段中切换。

2.  **Identification of Foraging (觅食行为识别)**:
    Foraging is successfully identified as an "intermediate" behavior—active movement that lacks the high directional speed of commuting. This aligns with "Area-Restricted Search" theory.
    觅食被成功识别为一种“中间态”行为——即活跃移动但缺乏通勤飞行的高速特征。这符合“区域限制搜索”理论。

3.  **Spatial Hotspots (空间热点)**:
    Using Kernel Density Estimation (KDE) in `src/analysis_foraging_hotspots.py`, we identified distinct marine zones used exclusively for foraging, separate from terrestrial resting sites.
    利用核密度估计 (KDE)，我们识别出了专门用于觅食的特定海域，其与陆地栖息地在空间上显著分离。

---

## 4. Code Structure (代码结构)

| File | Description | Role |
| :--- | :--- | :--- |
| `src/features.py` | Trajectory resampling & speed calculation. | **ETL** |
| `src/models.py` | (Legacy) Wrapper for standard HMMs. | **Model** |
| `src/main3.py` | **Main execution script**. Runs the 2-state HMM + Threshold pipeline and saves CSV results. | **Production** |
| `src/validate3_updated.py` | Performs LOBO validation with retry logic and stability metrics. | **Validation** |
| `src/analysis_foraging_hotspots.py` | Generates KDE heatmaps to visualize foraging grounds vs. resting sites. | **Analysis** |

---

## 5. Usage (使用说明)

### Step 1: Validation (验证模型性能)
Check if the model generalizes well to unseen individuals:
```bash
python src/validate3_updated.py --n-folds 5