import hashlib
import typing as t
from dataclasses import dataclass, field
from datetime import datetime

import feedparser
from bs4 import BeautifulSoup
from feedgen.feed import FeedGenerator

# from litellm import completion
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

        self.date = datetime.fromisoformat(self.date_as_str)

        self.text_for_tts = prepare_text_for_tts(
            html_string=self.content,
            article_title=self.title,
            posted_date=self.date,
        )

        self.id = hashlib.md5(self.link.encode()).hexdigest()
        self.podcast_url = f"{PUBLIC_BUCKET_URL}/{RESULTS_DIR}{self.id}.mp3"

    def preprocess_with_llm(self):
        logger.info(f"Preprocessing article '{self.title}' with LLM...")
        self.text_for_tts = preprocess_text_with_llm(self.text_for_tts)
        logger.success("Preprocessing done.")

    @property
    def podcast_title(self):
        return f"#{self.number} {self.title.removesuffix('.')}"


def prepare_text_for_tts(
    html_string: str, article_title: str, posted_date: datetime
) -> str:
    soup = BeautifulSoup(html_string, "html.parser")
    to_remove = [
        "pre",
        "figcaption",
        "img",
        "iframe",
        "script",
        "details",
        "table",
    ]

    for bad_tag in soup.find_all(to_remove):
        bad_tag.decompose()

    for a_tag in soup.find_all("a"):
        a_tag.unwrap()

    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if not header.text.strip().endswith("."):
            header.string = f"\nArticle Section: {header.text.strip()}\n"

    text = soup.get_text()

    # Maybe you can prounounce this name better? Aha
    text_tts = f"""
This article was posted originally on Dwarteh's blog on {posted_date.strftime("%B %d, %Y")}.
This is a text-to-speech version of the article. The original article may contain images, links, and other elements not included in this audio version.

--------------------

Title: "{article_title}"

{text}""".strip()

    return text_tts


def preprocess_text_with_llm(original_text: str) -> str:
    return (
        original_text  # TODO: Uncomment this when you have a working LLM setup
    )
    # SYSTEM_PROMPT = dedent("""
    #     Please perform the following task:
    #
    #     * Translate the input into written word so a text-to-speech model can read it (things like fractions don't work well).
    #     * Examples include 1/4 to one quarter, 20-30 to twenty to thirty, or $1.5m to one point five million dollars. Most dollar signs should be converted. When given a sentence, just replace those.
    #     * Also replace things like emojis, special characters, code blocks, and other elements that don't work well with text-to-speech models.
    #     * If something looks like a table of contents for the article, you can remove it.
    #     * If something looks too dense as a table or code block, you can remove it, and just mention that it was removed.
    #     * You can specify the pronunciation of words based on their phoneme sequence. For example 'I enjoyed a day in Besiktas, Istanbul.', can be as 'I enjoyed a day in (B EH1 SH IH0 K T AA0 SH), (IH0 S T AA1 N B UH0 L).' Don't abuse this, only use when you are sure the model will mispronounce it.
    #     * Do not output anything else than the text to the speech-to-text model.
    # """).strip()
    #
    # response = completion(
    #     model=PREPROCESSING_MODEL,
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": original_text},
    #     ],
    #     stream=False,
    # )
    #
    # text = response["choices"][0]["message"]["content"]
    #
    # if not isinstance(text, str):
    #     logger.error(
    #         "Could not preprocess text with LLM, returning original text"
    #     )
    #     return original_text
    #
    # return text


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
    logger.success(f"Generated podcast feed with {len(articles)} articles")

    return podcast_feed_name
