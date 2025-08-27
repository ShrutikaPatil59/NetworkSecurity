import os
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN="RESULT"
PIPELINE_NAME:str ="NetworkSecurity"
ARTIFACT_DIR:str="Artifacts"
FILE_NAME:str="phishing_data.csv"

TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

SCHEMA_FILE_PATH=os.path.join("data_schema","schema.yaml")

"""data ingestion """
DATA_INGESTION_COLLECTION_NAME:str ="NetworkData"
DATA_INGESTION_DATABASE_NAME:str ="AIML Course"
DATA_INGESTION_DIR_NAME:str ="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str ="feature_store"
DATA_INGESTION_INGESTED_DIR:str ="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION:float= 0.2

"""data validation"""

DATA_VALIDATION_DIR_NAME:str ="data_validation"
DATA_VALIDATION_VALID_DIR:str ="validated"
DATA_VALIDATION_INVALID_DIR:str ="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str="report.yaml"

""" data transformation """

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORM_OBJECT_NAME: str = "transformer.joblib"
DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train_transformed.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test_transformed.npy"
DATA_TRANSFORMATION_PREPROCESSOR_REPORT_NAME: str = "preprocessor_report.yaml"

""" model trainer """

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_MODEL_NAME: str = "model.joblib"
MODEL_TRAINER_TRANSFORMER_NAME: str = "transformer.joblib"
MODEL_TRAINER_REPORT_NAME: str = "report.yaml"

# optional extra paths
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_models"
MODEL_TRAINER_FINAL_MODEL_FILE_NAME: str = "final_model.joblib"