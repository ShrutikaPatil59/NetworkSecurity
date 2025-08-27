from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils import read_yaml_file

import sys
import os
if __name__=="__main__":
     try:
        trainingpipelineconfig=TrainingPipelineConfig
        data_ingestion=DataIngestionConfig(trainingpipelineconfig)
        dataingestionconfig=DataIngestion( DataIngestionConfig)
        logger.info("Initiate data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        schema = read_yaml_file(SCHEMA_FILE_PATH)

        # 3. Initialize Data Validation config
        data_validation_config = DataValidationConfig(DataValidationConfig)

        # 4. Run Data Validation
        data_validation = DataValidationConfig(
            dataingestionartifact=dataingestionartifact,
            data_validation_config=data_validation_config
        )

        data_validation_artifact = data_validation.initiate_data_validation()

        print("Data Validation Completed")
        print(data_validation_artifact)


     except Exception as e:
        raise NetworkSecurityException(e,sys)
     
     
from networksecurity.components.model_trainer import ModelTrainer

# 5. Run Model Trainer
"""
trainer = ModelTrainer()
train_csv = os.path.join(ARTIFACT_DIR, TRAIN_FILE_NAME)
test_csv = os.path.join(ARTIFACT_DIR, TEST_FILE_NAME)
model_trainer_artifact = trainer.initiate_model_trainer(train_csv, test_csv)
print("Model Training Completed")
print(model_trainer_artifact)
"""
# 5. Run Model Trainer
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import ModelTrainerConfig

# Initialize config
model_trainer_config = ModelTrainerConfig()

# Pass DataTransformationArtifact from previous step
trainer = ModelTrainer(
    data_transformation_artifact=data_transformation_artifact,
    model_trainer_config=model_trainer_config
)

# Run training
model_trainer_artifact = trainer.initiate_model_trainer()

print("Model Training Completed")
print(model_trainer_artifact)
