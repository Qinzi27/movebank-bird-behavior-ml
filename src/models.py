# src/models.py
from hmmlearn.hmm import GaussianHMM
import numpy as np
import joblib

class MovementHMM:
    def __init__(self, n_components: int = 3, n_iter: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=n_iter, 
            random_state=random_state,
            verbose=False,
            # 初始化时只随机生成 s(startprob) 和 t(transmat)
            # m(means) 和 c(covars) 稍后手动指定
            init_params="st"
        )
        self.is_fitted = False
        self.flight_state_idx = None

    def fit(self, X: np.ndarray, lengths: list):
        """
        训练模型。对于 3 状态模型，强制锁定均值以对抗数据不平衡。
        """
        if self.n_components == 3:
            print(">>>启用物理约束模式：锁定均值 (Freezing Means) <<<")
            
            # 1. 设定物理意义明确的均值 (Log Scale)
            # Rest: exp(-4) ~ 0.02 m/s
            # Forage: exp(0.5) ~ 1.65 m/s
            # Flight: exp(2.5) ~ 12.2 m/s
            self.model.means_ = np.array([[-4.0], [0.5], [2.5]])
            
            # 2. 设定初始方差 (允许模型微调方差)
            self.model.covars_ = np.array([[0.5], [1.0], [1.0]])
            
            # 3. [关键一步] 设置 params = "stc"
            # 去掉 'm'，告诉算法在 EM 迭代中不要更新 means_，强制保持我们设定的值
            self.model.params = "stc"
            
        else:
            # 普通模式：全参数更新
            self.model.init_params = "stmc"
            self.model.params = "stmc"

        # 训练
        self.model.fit(X, lengths)
        self.is_fitted = True
        
        # 结果分析
        means = self.model.means_.flatten()
        self.flight_state_idx = np.argmax(means)
        
        print(f"HMM Training Complete (n={self.n_components}).")
        
        labels = ["Low (Rest)", "Mid (Forage)", "High (Flight)"]
        sorted_indices = np.argsort(means)
        
        for i, state_idx in enumerate(sorted_indices):
            mean_log = means[state_idx]
            mean_speed = np.exp(mean_log)
            # 如果是 3 状态，尝试匹配我们预设的标签
            label = labels[i] if self.n_components == 3 else f"State {state_idx}"
            print(f" - State {state_idx} [{label}]: Mean Log-Speed = {mean_log:.2f} (Speed ≈ {mean_speed:.2f} m/s)")
            
        print(f" - Identified Flight State Index: {self.flight_state_idx}")

    def predict(self, X: np.ndarray, lengths: list) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.model.predict(X, lengths)

    def save(self, path: str):
        joblib.dump(self.model, path)