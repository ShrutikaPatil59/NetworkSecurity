import os
import sys
import joblib
import numpy as np
import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class NetworkSecurityEstimator:
    """
    Wrapper class for managing trained models:
     - save model
     - load model
     - predict on new data
    """

    def __init__(self, model=None):
        self.model = model

    def save(self, file_path: str):
        """
        Save trained model to disk.
        """
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            joblib.dump(self.model, file_path)
            logging.info(f"✅ Model saved at: {file_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @classmethod
    def load(cls, file_path: str):
        """
        Load model from disk.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model not found at {file_path}")
            model = joblib.load(file_path)
            logging.info(f"✅ Model loaded from: {file_path}")
            return cls(model=model)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, X: pd.DataFrame | np.ndarray):
        """
        Generate predictions using the trained model.
        """
        try:
            if self.model is None:
                raise ValueError("No model is loaded or trained")
            preds = self.model.predict(X)
            logging.info(f"🔮 Prediction done on {len(preds)} samples")
            return preds
        except Exception as e:
            raise NetworkSecurityException(e, sys)
