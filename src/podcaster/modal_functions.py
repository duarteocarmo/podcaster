import modal

from podcaster.config import (
    MODAL_NAME,
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


@app.function(gpu="a10g", max_containers=10)
def transcribe(text: str, generation_kwargs: dict = {}) -> bytes:
    model = ChatterboxTTS.from_pretrained(device="cuda")
    wav = model.generate(
        text, audio_prompt_path="reference.wav", **generation_kwargs
    )
    ta.save("temp.wav", wav, model.sr)
    with open("temp.wav", "rb") as f:
        audio_bytes = f.read()
    return audio_bytes
