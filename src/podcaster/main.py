from loguru import logger

from podcaster.const import BUCKET_NAME, FEED_URL
from podcaster.modal_functions import transcribe_to_file
from podcaster.parser import get_articles
from podcaster.storage import S3BucketManager


class Podcaster:
    def __init__(self, feed_url: str, bucket_name: str):
        self.feed_url = feed_url
        self.bucket_name = bucket_name

    def scan(self):
        s3 = S3BucketManager(bucket_name=self.bucket_name)

        all_articles = get_articles(self.feed_url)[-18:]
        logger.info(f"Found {len(all_articles)} articles")

        articles_to_transcribe = s3.get_untranscribed(articles=all_articles)
        logger.info(
            f"Found {len(articles_to_transcribe)} articles to transcribe"
        )

        transcribed_files = transcribe_to_file(
            articles=articles_to_transcribe, remote=True
        )

        s3.upload_files(files=list(zip(transcribed_files, transcribed_files)))

    def upload(self):
        ...


if __name__ == "__main__":
    p = Podcaster(feed_url=FEED_URL, bucket_name=BUCKET_NAME)
    p.scan()
    p.upload()
