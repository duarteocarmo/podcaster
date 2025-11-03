import hashlib
import os
import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from textwrap import dedent

import feedparser
from feedgen.feed import FeedGenerator
from loguru import logger
from openai import OpenAI

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
    text_for_tts: str | None = field(init=False)
    id: str = field(init=False)
    number: int = field(init=False)
    podcast_url: str = field(init=False)

    def __post_init__(self):
        if len(self.content) <= 100:
            self.is_valid = False

        if not self.title.strip().endswith("."):
            self.title = self.title.strip() + "."

        self.date = datetime.fromisoformat(self.date_as_str)
        self.text_for_tts = None

        self.id = hashlib.md5(self.link.encode()).hexdigest()
        self.podcast_url = f"{PUBLIC_BUCKET_URL}/{RESULTS_DIR}{self.id}.mp3"

    def preprocess_with_llm(self, model: str):
        logger.info(f"Preprocessing article '{self.title}' with LLM...")
        self.text_for_tts = get_tts_text(
            html_string=self.content,
            article_title=self.title,
            posted_date=self.date,
            openrouter_model=model,
        )
        logger.success("Preprocessing done.")

    @property
    def podcast_title(self):
        return f"#{self.number} {self.title.removesuffix('.')}"


def get_tts_text(
    html_string: str,
    article_title: str,
    posted_date: datetime,
    openrouter_model: str,
) -> str:
    sys_msg = dedent("""
    You will receive an article in markdown format from a blog post.
    Your task is to convert that article into a transcript suitable for reading aloud by a text-to-speech system.
    Follow these rules carefully:
        - Remove all metadata, HTML, and formatting not meant to be spoken.
        - Keep only text that should be read aloud — no stage directions, timestamps, or narrator labels.
        - Normalize headings: # → Section Title: <text>, ## → Subsection Title: <text>.
        - Always include four newlines before and after sections, subsections, and other major breaks.
        - Use line breaks for pacing: one newline = short pause; two newlines = long pause or paragraph break.
        - Break long paragraphs into sentences ≤ ~20–25 words when possible; split long sentences at clause boundaries.
        - Convert ordered lists to spoken enumerations (e.g., One: <text>), one item per line.
        - Convert unordered lists to one bullet per line, with a short pause between items.
        - Replace quotation marks with "quote" and "end quote" markers. Put them on the same line as the quoted text.
        - For nested quotes, nest "quote" / "end quote" explicitly.
        - For inline links [text](url), replace with: <text> (check article for link). If it's multiple links in a row, say "links in original article" instead of repeating too many times so that the listener isn't overwhelmed.
        - For ambiguous link anchors like “this” / “here”, resolve by context to "quote this article end quote" (link in original article) or "quote this video end quote" (link in original article).
        - Treat code blocks as omitted or summarized: "(code block omitted; main idea: <short summary>)". Even if the code block is only one line.
        - Render inline code as normal text without special formatting.
        - Read shell/command lines slowly; add extra newlines before and after; prefix with Command:.
        - Replace images with descriptive alt/caption if present: "(image of <alt/caption>)".
        - For videos/podcasts, say "quote this video end quote" (link in original article) or "quote this podcast episode end quote" (link in original article).
        - Summarize tables instead of reading dense rows: "(table that shows <one-line summary>)".
        - Expand acronyms and abbreviations to spoken form (e.g., LLMs → large language models).
        - Normalize dates and times (e.g., 4PM → four P.M., 2025 → twenty twenty-five)
        - When the text shows emphasis, you can use double newlines before and after the emphasized text for effect.
        - Convert parentheticals to short asides on separate lines: "(aside: ...)".
        - Convert footnotes to "(footnote: <text>)" or "(citation: see link in original article)".
        - Replace emojis with words or remove if irrelevant.
        - For dialogue, put each speaker line on its own line and optionally include Speaker:.
        - QA checks before output: ensure no raw markdown tokens remain; flag unresolved links; warn on very long sentences.
        - Always start with Article Title: <text> followed by two newlines.
        - Then say Date of publication: <text> followed by two newlines.
        - Then say "This transcript was automatically generated by a text to speech system." followed by two newlines.
        - Output the transcript in a markdown block starting with ```markdown and ending with ```.
        - Output only the transcript in the markdown block, nothing else.
    """).strip("\n")

    usr_msg = f"""
Article Title: {article_title}\n
Date of publication: {posted_date.strftime("%B %d, %Y")}\n
Content:\n{html_string}
    """.strip()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=openrouter_model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg},
        ],
        temperature=0.01,
    )
    result = completion.choices[0].message.content
    if not isinstance(result, str):
        raise ValueError("LLM did not return a string")

    inside_md_block = (
        result.removeprefix("```markdown").removesuffix("```").strip()
    )

    return inside_md_block


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
