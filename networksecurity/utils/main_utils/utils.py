import os
import sys
import yaml
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Load .env variables if present
load_dotenv()


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns contents as a dictionary.
    """
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    """
    Writes dictionary into a YAML file.
    If replace=True, overwrite the file, else append.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logging.info(f"📄 YAML file saved at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """
    Save a NumPy array to file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
        logging.info(f"✅ NumPy array saved at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Load a NumPy array from file.
    """
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_dataframe(file_path: str, df: pd.DataFrame) -> None:
    """
    Save pandas DataFrame as CSV.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"✅ DataFrame saved at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def get_mongo_client() -> MongoClient:
    """
    Create a MongoDB client using MONGO_DB_URL from .env
    """
    try:
        mongo_url = os.getenv("MONGO_DB_URL")
        if not mongo_url:
            raise ValueError("❌ MONGO_DB_URL not found in environment variables")

        client = MongoClient(mongo_url)
        logging.info("✅ MongoDB client created successfully.")
        return client
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def get_collection_as_dataframe(database_name: str, collection_name: str, mongo_client: MongoClient) -> pd.DataFrame:
    """
    Convert MongoDB collection into pandas DataFrame.
    """
    try:
        collection = mongo_client[database_name][collection_name]
        df = pd.DataFrame(list(collection.find()))
        if "_id" in df.columns:
            df.drop("_id", axis=1, inplace=True)  # drop MongoDB ObjectId field
        logging.info(f"✅ Loaded {len(df)} records from MongoDB collection: {collection_name}")
        return df
    except Exception as e:
        raise NetworkSecurityException(e, sys)
