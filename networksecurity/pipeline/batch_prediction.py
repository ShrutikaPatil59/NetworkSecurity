import sys
import os
import pandas as pd
import joblib
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger
from networksecurity.utils.ml_utils.model.estimator import NetworkSecurityModel


def start_batch_prediction(input_file_path: str, output_file_path: str):
    try:
        logger.info(f"Batch prediction started for file: {input_file_path}")

        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")

        # Load CSV
        df = pd.read_csv(input_file_path)
        logger.info(f"Input data shape: {df.shape}")

        # Load trained model
        model_path = "artifacts/model/trained_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Trained model not found. Please train the model first.")

        model: NetworkSecurityModel = joblib.load(model_path)

        # Predictions
        predictions = model.predict(df.values)
        df["prediction"] = predictions

        # Save output file
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, index=False)
        logger.info(f"Batch prediction completed. Output saved at: {output_file_path}")

        return output_file_path

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Prediction for Network Security ML Model")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV")

    args = parser.parse_args()

    result_path = start_batch_prediction(args.input, args.output)
    print(f"✅ Batch prediction saved at: {result_path}")
