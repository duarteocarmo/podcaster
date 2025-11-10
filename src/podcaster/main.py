import os

import requests
from loguru import logger

from podcaster.config import (
    BUCKET_NAME,
    FEED_URL,
    PREPROCESSING_MODEL,
    TRANSCRIBE_LAST,
)
from podcaster.parser import (
    generate_podcast_feed_from,
    get_articles,
)
from podcaster.storage import S3BucketManager
from podcaster.transcription import transcribe_to_file


class Podcaster:
    def __init__(self, feed_url: str, bucket_name: str):
        self.feed_url = feed_url
        self.bucket_name = bucket_name
        self.s3 = S3BucketManager(bucket_name=self.bucket_name)
        self.transcribe_last = TRANSCRIBE_LAST
        self.trigger_website_rebuild = False

    def scan(self):
        all_articles = get_articles(self.feed_url)[-self.transcribe_last :]
        logger.info(f"Found {len(all_articles)} articles")

        articles_to_transcribe = self.s3.get_untranscribed(
            articles=all_articles
        )
        for article in articles_to_transcribe:
            all_articles.remove(article)
        articles_to_transcribe = sorted(
            articles_to_transcribe, key=lambda x: x.date
        )

        if len(articles_to_transcribe) < 1:
            logger.info("No new articles to transcribe")
            return

        logger.info(
            f"Found {len(articles_to_transcribe)} articles to transcribe"
        )

        for article in articles_to_transcribe:
            assert article not in all_articles
            article.preprocess_with_llm(PREPROCESSING_MODEL)
            mp3_file_name = transcribe_to_file(
                article=article,
            )
            self.s3.upload_files(files=[(mp3_file_name, mp3_file_name)])
            logger.info(f"Uploaded file to S3: {mp3_file_name}")
            all_articles.append(article)
            feed_file = generate_podcast_feed_from(articles=all_articles)
            self.s3.upload_files(files=[(feed_file, feed_file)])
            logger.info("Updated podcast feed in S3")

        self.trigger_website_rebuild = True

    def rebuild(self):
        rebuild_trigger_url = os.getenv("REBUILD_TRIGGER_URL", None)

        if rebuild_trigger_url is None:
            logger.info("No rebuild trigger URL")
            return

        if self.trigger_website_rebuild is False:
            logger.info("No need to rebuild website")
            return

        r = requests.post(rebuild_trigger_url)

        if r.status_code == 200:
            logger.success("Website rebuilt")


def main():
    p = Podcaster(feed_url=FEED_URL, bucket_name=BUCKET_NAME)
    p.scan()
    p.rebuild()


if __name__ == "__main__":
    main()
