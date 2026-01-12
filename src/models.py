# src/models.py
from hmmlearn.hmm import GaussianHMM
import numpy as np
import joblib

class MovementHMM:
    """运动状态识别 HMM 模型封装"""
    
    def __init__(self, n_components: int = 2, n_iter: int = 200, random_state: int = 42):
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=n_iter, 
            random_state=random_state,
            verbose=False
        )
        self.is_fitted = False
        self.flight_state_idx = None # 内部记录哪个 ID 是飞行

    def fit(self, X: np.ndarray, lengths: list):
        """训练模型并自动识别飞行状态"""
        self.model.fit(X, lengths)
        self.is_fitted = True
        
        # 自动识别逻辑：速度均值较大的那个状态定义为“飞行”
        # model.means_ 形状为 (n_components, n_features)
        means = self.model.means_.flatten()
        self.flight_state_idx = np.argmax(means)
        
        print(f"HMM Training Complete.")
        print(f" - State 0 Mean Log-Speed: {means[0]:.2f}")
        print(f" - State 1 Mean Log-Speed: {means[1]:.2f}")
        print(f" - Identified Flight State Index: {self.flight_state_idx}")

    def predict(self, X: np.ndarray, lengths: list) -> np.ndarray:
        """
        预测状态序列。
        返回: 0=Rest, 1=Flight (已自动对齐语义)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
            
        raw_states = self.model.predict(X, lengths)
        
        # 如果模型内部认为 0 是飞行，我们需要把它反转成 1，或者保持原样
        # 目标：输出 1 为飞行，0 为非飞行
        if self.flight_state_idx == 1:
            return raw_states # 内部 1 就是飞行，直接返回
        else:
            return 1 - raw_states # 内部 0 是飞行，所以用 1-0 得到 1

    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_fitted = True