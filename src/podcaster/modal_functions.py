import io
import re
import typing as t

import modal

from podcaster.config import (
    MODAL_NAME,
    RESULTS_DIR,
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install("chatterbox-tts==0.1.1")
    .add_local_file(
        local_path="./data_raw/reference_enhanced.wav",
        remote_path="/root/reference.wav",
    )
)
app = modal.App(MODAL_NAME, image=image)
with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS


@app.function(gpu="a10g", max_containers=5)
def transcribe(text: str, generation_kwargs: dict = {}) -> bytes:
    model = ChatterboxTTS.from_pretrained(device="cuda")
    wav = model.generate(
        text, audio_prompt_path="reference.wav", **generation_kwargs
    )
    ta.save("temp.wav", wav, model.sr)
    with open("temp.wav", "rb") as f:
        audio_bytes = f.read()
    return audio_bytes


def split_text_into_chunks(text: str, max_chars: int = 300) -> list[str]:
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

    combined = AudioSegment.empty()
    for audio_bytes in audio_bytes_list:
        combined += AudioSegment.from_wav(io.BytesIO(audio_bytes))

    output = io.BytesIO()
    combined.export(output, format="wav")
    return output.getvalue()


def text_to_bytes(text: str) -> bytes:
    chunks = split_text_into_chunks(text)
    generation_kwargs = {
        "exaggeration": 0.2,
        "cfg_weight": 0.6,
        "temperature": 0.5,
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
    articles: list,
    llm_model: str = "openai/gpt-5",
    target_dir: str = RESULTS_DIR,
) -> t.List[str]:
    from podcaster.parser import ParsedArticle

    for article in articles:
        assert isinstance(article, ParsedArticle), (
            "Articles must be of type ParsedArticle."
        )

    if len(articles) == 0:
        return []

    file_names = []
    for article in articles:
        article.preprocess_with_llm(llm_model)
        assert article.text_for_tts is not None, (
            "Article text for TTS is None."
        )
        article_bytes = text_to_bytes(article.text_for_tts)
        wav_file_name = f"{target_dir}{article.id}.wav"
        with open(wav_file_name, "wb") as f:
            f.write(article_bytes)

        mp3_file_name = convert_to_mp3(wav_file_name)
        file_names.append(mp3_file_name)

    return file_names


def convert_to_mp3(file_path: str):
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(file_path)
    file_path.replace("wav", "mp3")
    audio.export(file_path, format="mp3")
    return file_path
