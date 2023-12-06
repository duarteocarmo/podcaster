import hashlib
import re
import typing as t
from dataclasses import dataclass, field
from datetime import datetime

import feedparser
from bs4 import BeautifulSoup
from feedgen.feed import FeedGenerator
from loguru import logger

from podcaster.config import (
    PODCAST_AUTHOR,
    PODCAST_CATEGORIES,
    PODCAST_DESCRIPTION,
    PODCAST_FEED_NAME,
    PODCAST_IMAGE,
    PODCAST_NAME,
    PODCAST_WEBSITE,
    PUBLIC_BUCKET_URL,
    RESULTS_DIR,
)


@dataclass
class ParsedArticle:
    title: str
    summary: str
    link: str
    content: str
    date_as_str: str
    date: datetime = field(init=False)
    is_valid: bool = field(init=False, default=True)
    text_for_tts: str = field(init=False)
    id: str = field(init=False)
    number: int = field(init=False)
    podcast_url: str = field(init=False)

    def __post_init__(self):
        if len(self.content) <= 100:
            self.is_valid = False

        if not self.title.strip().endswith("."):
            self.title = self.title.strip() + "."

        tts_content = prepare_text_for_tts(html_string=self.content)
        self.text_for_tts = f"Article title: {self.title}\n{tts_content}"
        self.date = datetime.fromisoformat(self.date_as_str)
        self.id = hashlib.md5(self.link.encode()).hexdigest()
        self.podcast_url = f"{PUBLIC_BUCKET_URL}/{RESULTS_DIR}{self.id}.mp3"

    @property
    def podcast_title(self):
        return f"#{self.number} {self.title.removesuffix('.')}"


def prepare_text_for_tts(html_string: str) -> str:
    soup = BeautifulSoup(html_string, "html.parser")
    to_remove = ["pre", "figcaption", "img", "iframe", "script", "details"]

    for bad_tag in soup.find_all(to_remove):
        bad_tag.decompose()

    for a_tag in soup.find_all("a"):
        a_tag.unwrap()

    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if not header.text.strip().endswith("."):
            header.string = header.text.strip() + "."

    text = soup.get_text()

    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_articles(feed_url: str) -> t.List[ParsedArticle]:
    feed = feedparser.parse(feed_url)

    articles = [
        ParsedArticle(
            title=fe.title,
            summary=fe.summary,
            link=fe.link,
            content=fe.content[0].value,
            date_as_str=fe.published,
        )
        for fe in feed.entries
    ]
    articles = sorted(articles, key=lambda x: x.date, reverse=False)

    for index, article in enumerate(articles):
        article.number = index + 1

    articles = [a for a in articles if a.is_valid]

    return articles


def generate_podcast_feed_from(
    articles: t.List[ParsedArticle], podcast_feed_name: str = PODCAST_FEED_NAME
):
    fg = FeedGenerator()
    fg.load_extension("podcast")

    fg.podcast.itunes_category(PODCAST_CATEGORIES)
    fg.title(PODCAST_NAME)
    fg.description(PODCAST_DESCRIPTION)
    fg.link(href=PODCAST_WEBSITE)
    fg.logo(PODCAST_IMAGE)
    fg.author(PODCAST_AUTHOR)
    fg.podcast.itunes_author(PODCAST_AUTHOR.get("name"))

    for article in articles:
        fe = fg.add_entry()
        fe.id(article.podcast_url)
        fe.title(article.podcast_title)
        fe.link(href=article.link)
        fe.description(article.summary)
        fe.enclosure(article.podcast_url, 0, "audio/mpeg")
        fe.published(article.date)

    fg.rss_file(podcast_feed_name)
    logger.info(f"Generated podcast feed with {len(articles)} articles")

    return podcast_feed_name
