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

    def resample_tracks(self, df: pd.DataFrame, time_step: str = '60s') -> pd.DataFrame:
        """
        [新增] 轨迹重采样：将不规则的 GPS 点位规整为固定时间间隔。
        这对 HMM 模型至关重要。
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        resampled_list = []
        
        # 按个体分组进行重采样
        for ind_id, group in df.groupby("individual-local-identifier"):
            # 1. 重采样并对经纬度进行线性插值
            # dropna() 是为了去除那些由于时间间隔过大导致的纯空行，
            # 实际科研中可能需要设定 max_gap 来截断轨迹，这里做简化处理。
            g_resampled = group[["location-lat", "location-long"]].resample(time_step).mean().interpolate(method='linear').dropna()
            
            # 补全 ID 信息
            g_resampled["individual-local-identifier"] = ind_id
            resampled_list.append(g_resampled)
            
        # 还原索引
        if not resampled_list:
            return df # 如果列表为空，返回原数据（或报错）
            
        df_resampled = pd.concat(resampled_list).reset_index()
        return df_resampled

    def preprocess(self, df: pd.DataFrame, do_resample: bool = True) -> pd.DataFrame:
        """执行完整的数据清洗管道"""
        # 1. 基础清洗
        df = df.dropna(subset=["location-lat", "location-long", "timestamp"])
        
        # 2. [关键修改] 重采样，确保 HMM 时间步长一致
        if do_resample:
            print("正在对轨迹进行重采样 (Interval: 60s)...")
            df = self.resample_tracks(df, time_step='60s')
        
        # 3. 排序
        df = df.sort_values(["individual-local-identifier", "timestamp"]).reset_index(drop=True)
        
        # 4. 计算时空差分 (Groupby individual)
        grouper = df.groupby("individual-local-identifier")
        
        # 计算当前点与下一点的距离和时间差
        # 使用 shift(-1) 向前看一个点
        next_lat = grouper["location-lat"].shift(-1)
        next_lon = grouper["location-long"].shift(-1)
        next_time = grouper["timestamp"].shift(-1)
        
        df["dist"] = self.haversine_distance(
            df["location-lat"], df["location-long"],
            next_lat, next_lon
        )
        
        df["dt"] = (next_time - df["timestamp"]).dt.total_seconds()
        # 5. 计算速度 & 清洗异常值
        # [修改] 增加 .copy() 解决 SettingWithCopyWarning
        df = df[df["dt"] > 0].copy()  
        df["speed"] = df["dist"] / df["dt"]
        
        # 6. 对数变换
    
        df["log_speed"] = np.log(df["speed"] + 1e-3)

        
        return df.dropna()