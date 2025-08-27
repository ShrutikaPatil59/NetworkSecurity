from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str
    
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str

# New after transformation
@dataclass
class ModelTrainerArtifact:
    model_file_path: str
    train_score: float
    test_score: float

@dataclass
class ClassificationMetricArtifact:
    """
    Stores classification metrics after model evaluation.
    Returned by ModelTrainer after training & testing.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float

@dataclass
class ModelPusherArtifact:
    saved_model_path: str
    model_registry_path: str
    

from dataclasses import dataclass

@dataclass
class ClassificationMetricArtifact:
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def __str__(self):
        return (
            f"ClassificationMetricArtifact("
            f"accuracy={self.accuracy:.4f}, "
            f"precision={self.precision:.4f}, "
            f"recall={self.recall:.4f}, "
            f"f1_score={self.f1_score:.4f})"
        )


@dataclass
class ModelTrainerArtifact:
    model_path: str
    report_path: str
    metrics: ClassificationMetricArtifact

    def __str__(self):
        return (
            f"ModelTrainerArtifact(\n"
            f"  model_path={self.model_path},\n"
            f"  report_path={self.report_path},\n"
            f"  metrics={self.metrics}\n"
            f")"
        )
