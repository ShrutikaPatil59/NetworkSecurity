from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils import read_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
import yaml

class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = self.read_schema(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def read_schema(self, file_path: str):
        """Load schema.yaml"""
        try:
            with open(file_path, "r") as file:
                schema = yaml.safe_load(file)
            logging.info(f"✅ Schema loaded from {file_path}")
            return schema
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Check if all required columns are present in dataset"""
        try:
            required_columns = [col["name"] for col in self.schema["columns"]]
            df_columns = df.columns.tolist()

            missing_cols = [col for col in required_columns if col not in df_columns]
            if missing_cols:
                logging.error(f"Missing columns: {missing_cols}")
                return False

            logging.info("All required columns are present.")
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_domain_values(self, df: pd.DataFrame) -> bool:
        """Check if domain values are within allowed set"""
        try:
            if "domain_values" not in self.schema:
                return True

            for col, allowed_values in self.schema["domain_values"].items():
                if col in df.columns:
                    invalid = df[~df[col].isin(allowed_values)]
                    if not invalid.empty:
                        logging.error(f"Invalid values found in column {col}")
                        return False
            logging.info(" Domain values validated.")
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        """Check drift using KS-test"""
        try:
            drift_report = {}
            for col in base_df.columns:
                if col in current_df.columns and pd.api.types.is_numeric_dtype(base_df[col]):
                    d1 = base_df[col].dropna()
                    d2 = current_df[col].dropna()
                    if len(d1) > 0 and len(d2) > 0:
                        stat, pvalue = ks_2samp(d1, d2)
                        drift_report[col] = {
                            "p_value": float(pvalue),
                            "drift_detected": bool(pvalue < 0.05)
                        }
            return drift_report
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Main method to perform validation"""
        try:
            logging.info("🚀 Starting Data Validation...")

            # Load datasets
            train_df = pd.read_csv(self.data_ingestion_artifact.training_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.testing_file_path)

            # Step 1: Validate schema
            status_train = self.validate_columns(train_df)
            status_test = self.validate_columns(test_df)

            # Step 2: Validate domain values
            domain_status_train = self.validate_domain_values(train_df)
            domain_status_test = self.validate_domain_values(test_df)

            # Step 3: Drift detection
            drift_report = self.detect_data_drift(train_df, test_df)

            # Save drift report
            drift_report_file = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file), exist_ok=True)
            with open(drift_report_file, "w") as f:
                yaml.dump(drift_report, f)

            logging.info(f" Drift report saved at: {drift_report_file}")

            # Create artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=status_train and status_test and domain_status_train and domain_status_test,
                valid_train_file_path=self.data_ingestion_artifact.training_file_path if status_train else None,
                valid_test_file_path=self.data_ingestion_artifact.testing_file_path if status_test else None,
                invalid_train_file_path=None if status_train else self.data_ingestion_artifact.training_file_path,
                invalid_test_file_path=None if status_test else self.data_ingestion_artifact.testing_file_path,
                drift_report_file_path=drift_report_file
            )

            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
