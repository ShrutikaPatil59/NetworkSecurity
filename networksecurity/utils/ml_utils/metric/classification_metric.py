import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
import sys


def get_classification_score(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics and return as ClassificationMetricArtifact.
    Wraps errors with NetworkSecurityException.
    """
    try:
        return ClassificationMetricArtifact(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score=f1_score(y_true, y_pred, average="macro", zero_division=0),
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)
