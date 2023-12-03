from loguru import logger

from podcaster.const import BUCKET_NAME, FEED_URL, TRANSCRIBE_LAST
from podcaster.modal_functions import transcribe_to_file
from podcaster.parser import (
    generate_podcast_feed_from,
    get_articles,
)
from podcaster.storage import S3BucketManager


class Podcaster:
    def __init__(self, feed_url: str, bucket_name: str):
        self.feed_url = feed_url
        self.bucket_name = bucket_name
        self.s3 = S3BucketManager(bucket_name=self.bucket_name)
        self.transcribe_last = TRANSCRIBE_LAST

    def scan(self):
        all_articles = get_articles(self.feed_url)[-self.transcribe_last :]
        logger.info(f"Found {len(all_articles)} articles")

        articles_to_transcribe = self.s3.get_untranscribed(
            articles=all_articles
        )
        logger.info(
            f"Found {len(articles_to_transcribe)} articles to transcribe"
        )

        transcribed_files = transcribe_to_file(
            articles=articles_to_transcribe, remote=True
        )

        self.s3.upload_files(
            files=list(zip(transcribed_files, transcribed_files))
        )

    def upload(self):
        all_articles = get_articles(self.feed_url)[-self.transcribe_last :]
        transcribed_articles = self.s3.get_transcribed(articles=all_articles)
        feed_file = generate_podcast_feed_from(articles=transcribed_articles)
        self.s3.upload_files(files=[(feed_file, feed_file)])


if __name__ == "__main__":
    p = Podcaster(feed_url=FEED_URL, bucket_name=BUCKET_NAME)
    p.scan()
    p.upload()
