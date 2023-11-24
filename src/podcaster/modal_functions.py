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
    MODEL_NAME,
    VOICE_FILE,
)
from podcaster.parser import ParsedArticle

MODAL_IMAGE = modal.Image.debian_slim().pip_install_from_pyproject(
    "pyproject.toml"
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
def transcribe_in_modal(
    articles: t.List[ParsedArticle],
    voice_file: str = VOICE_FILE,
    model_name: str = MODEL_NAME,
    language: str = LANGUAGE,
) -> None:
    os.environ["COQUI_TOS_AGREED"] = "1"  # avoid confirmation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} for transcribing {len(articles)} articles.")
    logger.info(os.listdir("/root/data"))

    tts = TTS(model_name=model_name, progress_bar=True).to(device)

    for article in articles:
        tts.tts_to_file(
            article.text_for_tts,
            speaker_wav=voice_file,
            language=language,
            file_path=f"{article.title}.wav",
        )


def transcribe_list_of(articles: t.List[ParsedArticle], remote: bool):
    if remote is True:
        with stub.run():
            result = transcribe_in_modal.remote(articles)
    else:
        result = transcribe_in_modal.local(articles)

    print(result)
