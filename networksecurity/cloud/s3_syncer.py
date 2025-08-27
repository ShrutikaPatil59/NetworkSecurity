import boto3
import os
import sys
from networksecurity.logging import logging
from networksecurity.exception import NetworkSecurityException

class S3Sync:
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region_name="ap-south-1"):
        """
        Initialize S3 client
        """
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
            logging.info("✅ S3 client initialized successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_folder_to_s3(self, folder: str, bucket_name: str, s3_folder: str):
        """
        Upload local folder content to S3 bucket
        """
        try:
            logging.info(f"🚀 Syncing local folder {folder} to S3 bucket {bucket_name}/{s3_folder}")

            for root, _, files in os.walk(folder):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, folder)
                    s3_key = os.path.join(s3_folder, relative_path).replace("\\", "/")

                    logging.info(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
                    self.s3_client.upload_file(local_path, bucket_name, s3_key)

            logging.info("✅ Folder synced to S3 successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_s3_to_folder(self, bucket_name: str, s3_folder: str, local_folder: str):
        """
        Download folder content from S3 bucket to local
        """
        try:
            logging.info(f"⬇️ Syncing S3 folder {bucket_name}/{s3_folder} to local {local_folder}")

            if not os.path.exists(local_folder):
                os.makedirs(local_folder)

            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
            for obj in response.get("Contents", []):
                s3_key = obj["Key"]
                relative_path = os.path.relpath(s3_key, s3_folder)
                local_path = os.path.join(local_folder, relative_path)

                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                logging.info(f"Downloading s3://{bucket_name}/{s3_key} → {local_path}")
                self.s3_client.download_file(bucket_name, s3_key, local_path)

            logging.info("✅ S3 folder synced to local successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
