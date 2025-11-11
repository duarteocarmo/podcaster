import modal

from podcaster.config import (
    MODAL_APP_NAME,
    MODAL_GPU,
    MODAL_LOCAL_REFERENCE_AUDIO_PATH,
    MODAL_MAX_CONTAINERS,
    MODAL_PYTHON_VERSION,
    MODAL_REMOTE_REFERENCE_AUDIO_PATH,
)

image = (
    modal.Image.debian_slim(python_version=MODAL_PYTHON_VERSION)
    .uv_pip_install("chatterbox-tts==0.1.1")
    .add_local_file(
        local_path=MODAL_LOCAL_REFERENCE_AUDIO_PATH,
        remote_path=MODAL_REMOTE_REFERENCE_AUDIO_PATH,
    )
)
app = modal.App(MODAL_APP_NAME, image=image)
with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS


@app.function(gpu=MODAL_GPU, max_containers=MODAL_MAX_CONTAINERS)
def transcribe(text: str, generation_kwargs: dict = {}) -> bytes:
    model = ChatterboxTTS.from_pretrained(device="cuda")
    wav = model.generate(
        text,
        audio_prompt_path=MODAL_REMOTE_REFERENCE_AUDIO_PATH.split("/")[-1],
        **generation_kwargs,
    )
    ta.save("temp.wav", wav, model.sr)
    with open("temp.wav", "rb") as f:
        audio_bytes = f.read()
    return audio_bytes
