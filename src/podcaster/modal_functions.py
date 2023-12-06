import os
import typing as t

import modal
import torch
from loguru import logger
from pydub import AudioSegment
from synchronicity.exceptions import UserCodeException
from TTS.api import TTS

from podcaster.config import (
    LANGUAGE,
    LOCAL_DATA_DIR,
    MODAL_GPU,
    MODAL_NAME,
    MODAL_REMOTE_DATA_DIR,
    MODEL_NAME,
    RESULTS_DIR,
    VOICE_FILE,
)
from podcaster.parser import ParsedArticle

MODAL_IMAGE = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject("pyproject.toml")
    .apt_install("ffmpeg")
)
stub = modal.Stub(MODAL_NAME, image=MODAL_IMAGE)


@stub.function(
    gpu=MODAL_GPU,
    mounts=[
        modal.Mount.from_local_dir(
            LOCAL_DATA_DIR, remote_path=MODAL_REMOTE_DATA_DIR
        )
    ],
)
def transcribe(
    article: ParsedArticle,
    voice_file: str = VOICE_FILE,
    model_name: str = MODEL_NAME,
    language: str = LANGUAGE,
) -> bytes:
    os.environ["COQUI_TOS_AGREED"] = "1"  # avoid confirmation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_file = f"{article.title}.wav"
    text_to_transcribe = article.text_for_tts
    num_chars = len(text_to_transcribe)
    logger.info(
        f"Using {device} to transcribe article with {num_chars} characters"
    )

    tts = TTS(model_name=model_name, progress_bar=True).to(device)

    def sentence_split_patcher(text: str) -> t.List[str]:
        sentences = tts.synthesizer.seg.segment(text)
        new_sentences = []
        for sentence in sentences:
            if len(sentence) < 200:
                new_sentences.append(sentence)
            else:
                to_extend = sentence.split(".")
                to_extend = [x for x in to_extend if len(x.strip()) > 0]
                new_sentences.extend(to_extend)

        return new_sentences

    tts.synthesizer.split_into_sentences = sentence_split_patcher

    tts.tts_to_file(
        text=text_to_transcribe,
        speaker_wav=voice_file,
        language=language,
        file_path=target_file,
    )

    mp3_file = convert_to_mp3(target_file)

    with open(mp3_file, "rb") as audio_file:
        audio_bytes = audio_file.read()

    logger.info("Transcription done.")

    return audio_bytes


def transcribe_to_file(
    articles: t.List[ParsedArticle],
    remote: bool,
    target_dir: str = RESULTS_DIR,
) -> t.List[str]:
    if len(articles) == 0:
        return []

    if remote is False:
        article_bytes_list = [transcribe.local(a) for a in articles]

    else:
        with stub.run():
            article_bytes_list: t.List[bytes] = list(
                transcribe.map(articles, return_exceptions=True)
            )

    file_names = []
    for a, b in zip(articles, article_bytes_list):
        if isinstance(b, UserCodeException):
            logger.warning(f"Could not transcribe article {a.id}")
            continue

        file_name = f"{target_dir}{a.id}.mp3"
        with open(file_name, "wb") as f:
            f.write(b)

        file_names.append(file_name)

    return file_names


def convert_to_mp3(file_path: str):
    audio = AudioSegment.from_wav(file_path)
    file_path.replace("wav", "mp3")
    audio.export(file_path, format="mp3")
    return file_path
