import subprocess
import typing as t
from pathlib import Path

import modal
from loguru import logger
from pydub import AudioSegment
from synchronicity.exceptions import UserCodeException

from podcaster.config import (
    LOCAL_DATA_DIR,
    MODAL_GPU,
    MODAL_NAME,
    MODAL_REMOTE_DATA_DIR,
    MODEL_NAME,
    PREPROCESS_WITH_LLM,
    REFERENCE_TEXT,
    REFERENCE_VOICE,
    RESULTS_DIR,
)
from podcaster.parser import ParsedArticle

MODAL_IMAGE = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "git+https://github.com/SWivid/F5-TTS.git@3fcdbc70b4a9d4299e1ecd0b5a1c35209f23fd69",
    )
)
app = modal.App(MODAL_NAME, image=MODAL_IMAGE)


@app.function(
    gpu=MODAL_GPU,
    mounts=[
        modal.Mount.from_local_dir(
            LOCAL_DATA_DIR, remote_path=MODAL_REMOTE_DATA_DIR
        )
    ],
    timeout=400,
)
def transcribe(
    article: ParsedArticle,
    reference_voice_file: str = REFERENCE_VOICE,
    reference_text_file: str = REFERENCE_TEXT,
    model_name: str = MODEL_NAME,
) -> bytes:
    with open(reference_text_file, "r") as voice_text:
        reference_text = voice_text.read()

    command = [
        "f5-tts_infer-cli",
        "--model",
        model_name,
        "--ref_audio",
        reference_voice_file,
        "--ref_text",
        reference_text,
        "--gen_text",
        article.text_for_tts,
    ]

    result = subprocess.run(command, check=True)
    assert result.returncode == 0, f"Error: {result.stderr}, \n{result.stdout}"

    target_file = "tests/infer_cli_out.wav"
    assert Path(target_file).exists(), f"File {target_file} does not exist."

    mp3_file = convert_to_mp3(target_file)

    with open(mp3_file, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return audio_bytes


def transcribe_to_file(
    articles: t.List[ParsedArticle],
    remote: bool,
    target_dir: str = RESULTS_DIR,
) -> t.List[str]:
    if len(articles) == 0:
        return []

    if PREPROCESS_WITH_LLM is True:
        for article in articles:
            article.preprocess_with_llm()

    if remote is False:
        article_bytes_list = [transcribe.local(a) for a in articles]

    else:
        with modal.enable_output():
            with app.run():
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
