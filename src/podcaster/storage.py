import typing as t
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError
from loguru import logger

from podcaster.const import BUCKET_URL
from podcaster.parser import ParsedArticle


class S3BucketManager:
    def __init__(self, bucket_name: str, region_name: str = "auto"):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            service_name="s3",
            endpoint_url=BUCKET_URL,
            region_name=region_name,
        )

    def list_bucket_contents(self) -> t.List[str]:
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name)
            return [item["Key"] for item in response.get("Contents", [])]
        except NoCredentialsError:
            print("Credentials not available")
            return []

    def upload_files(self, files: t.List[t.Tuple[str, str]]) -> t.List[str]:
        uploaded_files = []
        for local_file_path, s3_file_name in files:
            try:
                self.s3.upload_file(
                    local_file_path, self.bucket_name, s3_file_name
                )
                uploaded_files.append(s3_file_name)
            except FileNotFoundError:
                print(f"The file {local_file_path} was not found")
            except NoCredentialsError:
                print("Credentials not available")
            logger.info(f"Uploaded {local_file_path} to cloud.")
        return uploaded_files

    def delete_files(self, file_names: t.List[str]) -> t.List[str]:
        deleted_files = []
        for file_name in file_names:
            try:
                self.s3.delete_object(Bucket=self.bucket_name, Key=file_name)
                deleted_files.append(file_name)
            except NoCredentialsError:
                print("Credentials not available")
        return deleted_files

    def get_untranscribed(
        self, articles: t.List[ParsedArticle]
    ) -> t.List[ParsedArticle]:
        bucket_contents = self.list_bucket_contents()
        transcribed_ids = [Path(f).stem for f in bucket_contents]
        return [a for a in articles if a.id not in transcribed_ids]

    def get_transcribed(
        self, articles: t.List[ParsedArticle]
    ) -> t.List[ParsedArticle]:
        bucket_contents = self.list_bucket_contents()
        transcribed_ids = [Path(f).stem for f in bucket_contents]
        return [a for a in articles if a.id in transcribed_ids]
