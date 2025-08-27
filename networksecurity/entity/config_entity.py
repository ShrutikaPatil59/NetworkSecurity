from datetime import datetime
import os

from networksecurity.constant import training_pipeline

# Root directory for all artifacts
ARTIFACT_DIR = os.path.join(os.getcwd(), "artifacts")

# Timestamped subdirectory for each pipeline run
PIPELINE_TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACT_DIR = os.path.join(ARTIFACT_DIR, PIPELINE_TIMESTAMP)


print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_name=training_pipeline.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.timestamp:str=timestamp


        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path:str=os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR.FILE_NAME
        )
        self.training_file_path:str=os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TRAIN_FILE_NAME
        )
        self.testing_file_path:str=os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ration:float=training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str =training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name:str=training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Validate artifact directory
        """
        if not hasattr(training_pipeline_config, "artifact_dir") or not training_pipeline_config.artifact_dir:
            raise ValueError("❌ TrainingPipelineConfig must define a valid artifact_dir")
         """
        # Base validation directory
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # Validated data path
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        # Invalid data path
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        # Drift report directory + file
        self.drift_report_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR
        )

        self.drift_report_file_path: str = os.path.join(
            self.drift_report_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.trained_model_file_path = os.path.join(self.model_trainer_dir, "model.pkl")
        os.makedirs(self.model_trainer_dir, exist_ok=True)


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config):
        self.model_evaluation_dir = os.path.join(training_pipeline_config.artifact_dir, "model_evaluation")
        self.report_file_path = os.path.join(self.model_evaluation_dir, "evaluation_report.yaml")
        os.makedirs(self.model_evaluation_dir, exist_ok=True)


class ModelPusherConfig:
    def __init__(self, training_pipeline_config):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
        self.saved_model_path = os.path.join(self.model_pusher_dir, "saved_model.pkl")
        self.model_registry_path = os.path.join("saved_models", "latest_model.pkl")
        os.makedirs(self.model_pusher_dir, exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)

 