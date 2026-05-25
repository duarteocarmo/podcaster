import io
import os
from typing import Any

import modal

from podcaster.config import (
    ENV_HF_XET_HIGH_PERFORMANCE,
    ENV_MODAL_IMAGE_BUILDER_VERSION,
    ENV_VLLM_WORKER_MULTIPROC_METHOD,
    HF_XET_HIGH_PERFORMANCE,
    MODAL_APP_NAME,
    MODAL_GPU,
    MODAL_IMAGE_BUILDER_VERSION,
    MODAL_MAX_CONTAINERS,
    QWEN_TTS_BATCH_SIZE,
    QWEN_TTS_LANGUAGE,
    QWEN_TTS_MAX_NEW_TOKENS,
    QWEN_TTS_MODAL_IMAGE,
    QWEN_TTS_MODEL,
    QWEN_TTS_SPEAKER,
    QWEN_TTS_TASK_TYPE,
    QWEN_TTS_VLLM_VERSION,
    VLLM_WORKER_MULTIPROC_METHOD,
)

os.environ.setdefault(
    ENV_MODAL_IMAGE_BUILDER_VERSION, MODAL_IMAGE_BUILDER_VERSION
)
os.environ.setdefault(
    ENV_VLLM_WORKER_MULTIPROC_METHOD, VLLM_WORKER_MULTIPROC_METHOD
)

image = (
    modal.Image.from_registry(QWEN_TTS_MODAL_IMAGE)
    .entrypoint([])
    .uv_pip_install(QWEN_TTS_VLLM_VERSION)
    .env(
        {
            ENV_HF_XET_HIGH_PERFORMANCE: HF_XET_HIGH_PERFORMANCE,
            ENV_VLLM_WORKER_MULTIPROC_METHOD: VLLM_WORKER_MULTIPROC_METHOD,
        }
    )
)
app = modal.App(MODAL_APP_NAME, image=image)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
prompt_len_cache: dict[str, tuple[Any, Any]] = {}


def _prompt_len_resources_for(model_name: str) -> tuple[Any, Any]:
    if model_name not in prompt_len_cache:
        from transformers import AutoTokenizer
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        config = Qwen3TTSConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        prompt_len_cache[model_name] = (
            tokenizer,
            getattr(config, "talker_config", None),
        )
    return prompt_len_cache[model_name]


def _estimate_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
) -> int:
    try:
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        tokenizer, talker_config = _prompt_len_resources_for(
            model_name=model_name,
        )
        task_type = (
            additional_information.get("task_type") or [QWEN_TTS_TASK_TYPE]
        )[0]

        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda text: tokenizer(text, padding=False)[
                "input_ids"
            ],
            codec_language_id=getattr(
                talker_config, "codec_language_id", None
            ),
            spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
            estimate_ref_code_len=lambda ref_audio: None,
        )
    except Exception as exc:
        raise RuntimeError("Failed to estimate prompt length") from exc


def _build_custom_voice_request(
    text: str,
    speaker: str,
    language: str,
    instructions: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    additional_information = {
        "task_type": [QWEN_TTS_TASK_TYPE],
        "text": [text],
        "language": [language],
        "speaker": [speaker],
        "instruct": [instructions],
        "max_new_tokens": [max_new_tokens],
    }
    return {
        "prompt_token_ids": [0]
        * _estimate_prompt_len(
            additional_information=additional_information,
            model_name=QWEN_TTS_MODEL,
        ),
        "additional_information": additional_information,
    }


def _audio_bytes_from_multimodal_output(
    multimodal_output: dict[str, Any],
) -> bytes:
    import soundfile
    import torch

    audio_data = multimodal_output["audio"]
    sample_rate_raw = multimodal_output["sr"]
    sample_rate_value = (
        sample_rate_raw[-1]
        if isinstance(sample_rate_raw, list) and sample_rate_raw
        else sample_rate_raw
    )
    sample_rate = (
        sample_rate_value.item()
        if hasattr(sample_rate_value, "item")
        else int(sample_rate_value)
    )
    audio_tensor = (
        torch.cat(audio_data, dim=-1)
        if isinstance(audio_data, list)
        else audio_data
    )
    output = io.BytesIO()
    soundfile.write(
        output,
        audio_tensor.float().cpu().numpy().flatten(),
        samplerate=sample_rate,
        format="WAV",
    )
    return output.getvalue()


def _request_index_from(request_id: str):
    prefix = request_id.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


@app.function(
    gpu=MODAL_GPU,
    max_containers=MODAL_MAX_CONTAINERS,
    timeout=20 * 60,
    scaledown_window=5 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
def transcribe_chunks(
    chunks: list[str],
    generation_kwargs=None,
) -> list[bytes]:
    from vllm_omni import Omni

    if not chunks:
        return []

    generation_kwargs = generation_kwargs or {}
    speaker = generation_kwargs.get("speaker", QWEN_TTS_SPEAKER)
    language = generation_kwargs.get("language", QWEN_TTS_LANGUAGE)
    instructions = generation_kwargs.get("instructions", "")
    max_new_tokens = generation_kwargs.get(
        "max_new_tokens",
        QWEN_TTS_MAX_NEW_TOKENS,
    )
    batch_size = generation_kwargs.get("batch_size", QWEN_TTS_BATCH_SIZE)

    omni = Omni(
        model=QWEN_TTS_MODEL,
        log_stats=True,
        stage_init_timeout=600,
        trust_remote_code=True,
    )
    audio_results = []
    try:
        requests = [
            _build_custom_voice_request(
                text=chunk,
                speaker=speaker,
                language=language,
                instructions=instructions,
                max_new_tokens=max_new_tokens,
            )
            for chunk in chunks
        ]
        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start : batch_start + batch_size]
            batch_audio = [None] * len(batch)
            fallback_index = 0
            for stage_outputs in omni.generate(batch):
                request_output = stage_outputs.request_output
                if request_output is None or not request_output.outputs:
                    continue

                multimodal_output = request_output.outputs[0].multimodal_output
                if not multimodal_output or "audio" not in multimodal_output:
                    continue

                request_index = _request_index_from(request_output.request_id)
                if request_index is None or request_index >= len(batch):
                    request_index = fallback_index
                batch_audio[request_index] = (
                    _audio_bytes_from_multimodal_output(multimodal_output)
                )
                fallback_index += 1

            missing = [
                i for i, audio in enumerate(batch_audio) if audio is None
            ]
            if missing:
                raise RuntimeError(
                    f"No audio returned for batch indexes: {missing}"
                )
            audio_results.extend(
                audio for audio in batch_audio if audio is not None
            )
    finally:
        close = getattr(omni, "close", None)
        if close is not None:
            close()

    return audio_results
