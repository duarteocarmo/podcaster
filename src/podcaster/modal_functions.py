import os
import typing as t

import modal
import torch
from loguru import logger
from TTS.api import TTS

from podcaster.const import (
    LANGUAGE,
    LOCAL_DATA_DIR,
    MODAL_GPU,
    MODAL_NAME,
    MODAL_REMOTE_DATA_DIR,
    MODAL_VOLUME_NAME,
    MODEL_NAME,
    RESULTS_DIR,
    VOICE_FILE,
)
from podcaster.parser import ParsedArticle

MODAL_IMAGE = modal.Image.debian_slim().pip_install_from_pyproject(
    "pyproject.toml"
)
MODAL_VOLUME = modal.NetworkFileSystem.persisted(MODAL_VOLUME_NAME)
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

    tts.tts_to_file(
        text=text_to_transcribe,
        speaker_wav=voice_file,
        language=language,
        file_path=target_file,
    )

    with open(target_file, "rb") as audio_file:
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
        for a in articles:
            _ = transcribe.local(a)

    with stub.run():
        article_bytes_list = list(transcribe.map(articles))

    file_names = []
    for a, b in zip(articles, article_bytes_list):
        file_name = f"{target_dir}{a.id}.wav"
        with open(file_name, "wb") as f:
            f.write(b)

        file_names.append(file_name)

    return file_names
