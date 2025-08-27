import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array

class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self, schema: dict) -> ColumnTransformer:
        """
        Creates preprocessing pipeline for numerical and categorical columns
        """
        try:
            numerical_columns = schema["numerical_columns"]
            categorical_columns = schema["categorical_columns"]

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self, schema: dict) -> DataTransformationArtifact:
        """
        Performs data transformation and saves transformed arrays + preprocessing object
        """
        try:
            logging.info("🚀 Starting Data Transformation...")

            # Load train and test datasets
            train_df = pd.read_csv(self.data_ingestion_artifact.training_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.testing_file_path)

            target_column = schema["target_column"]

            # Split into features (X) and target (y)
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Get preprocessing object
            preprocessor = self.get_data_transformer_object(schema)

            # Fit on train, transform both train & test
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Concatenate with target
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save transformed arrays
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save preprocessing object
            joblib.dump(preprocessor, self.data_transformation_config.transformed_object_file_path)
            logging.info(f"Preprocessing object saved at {self.data_transformation_config.transformed_object_file_path}")

            # Return artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        

        except Exception as e:
            raise NetworkSecurityException(e, sys)


