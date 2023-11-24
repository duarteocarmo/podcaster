import re
import typing as t
from dataclasses import dataclass, field
from datetime import datetime

import feedparser
from bs4 import BeautifulSoup
from loguru import logger


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

    def __post_init__(self):
        if len(self.content) <= 100:
            logger.info(f"Skipping article: {self.title}")
            self.is_valid = False

        if not self.title.strip().endswith("."):
            self.title = self.title.strip() + "."

        tts_content = prepare_text_for_tts(html_string=self.content)
        self.text_for_tts = f"Article title: {self.title}\n{tts_content}"
        self.date = datetime.fromisoformat(self.date_as_str)


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
    articles = [a for a in articles if a.is_valid]
    articles = sorted(articles, key=lambda x: x.date, reverse=False)

    return articles
