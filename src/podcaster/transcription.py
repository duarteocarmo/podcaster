import io
import re

import modal
from loguru import logger
from pydub import AudioSegment

from podcaster.config import (
    CHATTERBOX_CFG_WEIGHT,
    CHATTERBOX_CROSSFADE_MS,
    CHATTERBOX_EXAGGERATION,
    CHATTERBOX_MAX_CHARS_PER_CHUNK,
    CHATTERBOX_TEMPERATURE,
    RESULTS_DIR,
)
from podcaster.modal_functions import app, transcribe
from podcaster.parser import ParsedArticle


def split_text_into_chunks(
    text: str, max_chars: int = CHATTERBOX_MAX_CHARS_PER_CHUNK
) -> list[str]:
    """Split text into sentence-based chunks of <= max_chars."""
    import nltk
    from nltk.tokenize import sent_tokenize

    nltk.download("punkt", quiet=True)
    text = re.sub(r"\s+", " ", text.strip())
    sentences = sent_tokenize(text)
    chunks, current = [], ""

    for sent in sentences:
        sent = sent.strip()
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


def concatenate_wav_bytes(audio_bytes_list: list[bytes]) -> bytes:
    from pydub import AudioSegment

    combined = AudioSegment.from_wav(io.BytesIO(audio_bytes_list[0]))
    for audio_bytes in audio_bytes_list[1:]:
        segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        combined = combined.append(segment, crossfade=CHATTERBOX_CROSSFADE_MS)

    output = io.BytesIO()
    combined.export(output, format="wav")
    logger.info(f"Merged {len(audio_bytes_list)} chunks into output.wav")
    return output.getvalue()


def text_to_bytes(text: str) -> bytes:
    chunks = split_text_into_chunks(
        text, max_chars=CHATTERBOX_MAX_CHARS_PER_CHUNK
    )
    generation_kwargs = {
        "exaggeration": CHATTERBOX_EXAGGERATION,
        "cfg_weight": CHATTERBOX_CFG_WEIGHT,
        "temperature": CHATTERBOX_TEMPERATURE,
    }
    print(f"Text split into {len(chunks)} chunks.")
    audio_bytes = None
    with modal.enable_output():
        with app.run():
            audio_bytes = list(
                transcribe.map(chunks, [generation_kwargs] * len(chunks))
            )
    if not audio_bytes:
        raise ValueError("No results returned from transcription.")

    final_audio = concatenate_wav_bytes(audio_bytes)
    return final_audio


def transcribe_to_file(
    article: ParsedArticle,
    target_dir: str = RESULTS_DIR,
) -> str:
    assert isinstance(article, ParsedArticle), "Input is not a ParsedArticle."
    if not article.text_for_tts:
        raise ValueError(
            "Article has no text for TTS. Did you preprocess it with LLM?"
        )

    article_bytes = text_to_bytes(article.text_for_tts)
    wav_file_name = f"{target_dir}{article.id}.wav"
    with open(wav_file_name, "wb") as f:
        f.write(article_bytes)

    mp3_file_name = convert_to_mp3(wav_file_name)
    logger.info(f"Transcription saved to {mp3_file_name}")

    return mp3_file_name


def convert_to_mp3(file_path: str):
    audio = AudioSegment.from_wav(file_path)
    file_path = file_path.replace("wav", "mp3")
    audio.export(file_path, format="mp3")
    return file_path
