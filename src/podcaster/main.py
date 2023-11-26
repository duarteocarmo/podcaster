from pathlib import Path

from loguru import logger

from podcaster.const import BUCKET_NAME, FEED_URL
from podcaster.modal_functions import transcribe_to_file
from podcaster.parser import get_articles
from podcaster.storage import S3BucketManager

if __name__ == "__main__":
    s3 = S3BucketManager(bucket_name=BUCKET_NAME)

    all_articles = get_articles(FEED_URL)[-8:]
    logger.info(f"Found {len(all_articles)} articles")

    bucket_contents = s3.list_bucket_contents()
    transcribed_ids = [Path(f).stem for f in bucket_contents]
    aricles_to_transribe = [
        a for a in all_articles if a.id not in transcribed_ids
    ]
    logger.info(f"Found {len(aricles_to_transribe)} articles to transcribe")

    transcribed_files = transcribe_to_file(
        articles=aricles_to_transribe, remote=True
    )

    s3.upload_files(files=list(zip(transcribed_files, transcribed_files)))
