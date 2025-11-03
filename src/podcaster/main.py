import os

import requests
from loguru import logger

from podcaster.config import (
    BUCKET_NAME,
    FEED_URL,
    TRANSCRIBE_LAST,
)
from podcaster.modal_functions import transcribe_to_file
from podcaster.parser import (
    ParsedArticle,
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
        self.trigger_website_rebuild = False

    def scan(self):
        all_articles = get_articles(self.feed_url)[-self.transcribe_last :]
        logger.info(f"Found {len(all_articles)} articles")

        articles_to_transcribe = self.s3.get_untranscribed(
            articles=all_articles
        )

        if len(articles_to_transcribe) < 1:
            logger.info("No new articles to transcribe")
            return

        logger.info(
            f"Found {len(articles_to_transcribe)} articles to transcribe"
        )

        transcribed_files = transcribe_to_file(articles=articles_to_transcribe)

        self.s3.upload_files(
            files=list(zip(transcribed_files, transcribed_files))
        )
        self.trigger_website_rebuild = True

    def test(self):
        all_articles: list[ParsedArticle] = get_articles(self.feed_url)[
            -self.transcribe_last :
        ]
        last_article = all_articles[-1]
        last_article.preprocess_with_llm(model="openai/gpt-5")

    def upload(self):
        all_articles = get_articles(self.feed_url)[-self.transcribe_last :]
        transcribed_articles = self.s3.get_transcribed(articles=all_articles)
        feed_file = generate_podcast_feed_from(articles=transcribed_articles)
        self.s3.upload_files(files=[(feed_file, feed_file)])

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


def run():
    p = Podcaster(feed_url=FEED_URL, bucket_name=BUCKET_NAME)
    p.scan()
    p.upload()
    p.rebuild()


if __name__ == "__main__":
    run()
