from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_ingestion import DataIngestion

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils import read_yaml_file

import sys

if __name__=="__main__":
     try:
        trainingpipelineconfig=TrainingPipelineConfig
        data_ingestion=DataIngestionConfig(trainingpipelineconfig)
        dataingestionconfig=DataIngestion( DataIngestionConfig)
        logging.info("Initiate data ingestion")
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

        print("✅ Data Validation Completed")
        print(data_validation_artifact)


     except Exception as e:
        raise NetworkSecurityException(e,sys)