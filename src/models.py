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