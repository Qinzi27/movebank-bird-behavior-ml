# src/features.py
import numpy as np
import pandas as pd
from typing import Tuple

class TrajectoryProcessor:
    """处理轨迹数据的特征工程类"""
    
    def __init__(self, earth_radius: float = 6_371_000):
        self.R = earth_radius

    def haversine_distance(self, lat1: np.ndarray, lon1: np.ndarray, 
                          lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """向量化计算 Haversine 距离 (单位: 米)"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return self.R * c

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行完整的数据清洗管道"""
        df = df.copy()
        # 1. 时间处理
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["individual-local-identifier", "timestamp"]).reset_index(drop=True)
        
        # 2. 计算时空差分 (Groupby individual)
        # 注意：这里处理了边界条件，确保不会跨个体计算差分
        grouper = df.groupby("individual-local-identifier")
        df["dt"] = grouper["timestamp"].diff().dt.total_seconds()
        
        # Shift 操作获取下一个点坐标，用于计算当前段距离
        next_lat = grouper["location-lat"].shift(-1)
        next_lon = grouper["location-long"].shift(-1)
        
        df["dist"] = self.haversine_distance(
            df["location-lat"], df["location-long"],
            next_lat, next_lon
        )
        
        # 3. 计算速度 & 清洗异常值
        df = df[df["dt"] > 0]  # 移除重复时间戳
        df["speed"] = df["dist"] / df["dt"]
        
        # 4. 对数变换 (用于 HMM)
        df["log_speed"] = np.log(df["speed"] + 1e-3)
        
        return df.dropna()

# src/models.py
from hmmlearn.hmm import GaussianHMM
import joblib

class MovementHMM:
    """运动状态识别 HMM 模型封装"""
    
    def __init__(self, n_components: int = 2, n_iter: int = 200, random_state: int = 42):
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=n_iter, 
            random_state=random_state
        )
        self.is_fitted = False
        self.flight_state_idx = None

    def fit(self, X: np.ndarray, lengths: list):
        """训练模型"""
        self.model.fit(X, lengths)
        self.is_fitted = True
        # 自动识别哪个状态对应"飞行" (均值较大的那个)
        self.flight_state_idx = np.argmax(self.model.means_.flatten())
        print(f"Training Complete. Flight State Index: {self.flight_state_idx}")
        print(f"State Means: {self.model.means_.flatten()}")

    def predict(self, X: np.ndarray, lengths: list) -> np.ndarray:
        """预测状态序列"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.model.predict(X, lengths)

    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_fitted = True