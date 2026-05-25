#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["modal==1.4.3"]
# ///

import argparse
import io
import logging
import os
from typing import Any

import modal

os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")

MODEL_ID = "duarteocarmo/qwen_tts_finetune_0.6B_e10_l1e6"
SPEAKER = "duarte"
SAMPLE_TEXT = """Hi, I'm Duarte du-art — a technologist — born and raised in sunny Lisbon, now based in Copenhagen. I work at the intersection of ML/AI, Data, Software, and People.
I've worked in Consumer Electronics, Public Institutions, Big Three Management Consulting, and YC-backed startups. The common thread? Solving hard problems end-to-end."""
OUTPUT_PATH = "modal_qwen_tts_output.wav"
GPU = "L40S"
MINUTES = 60

image = (
    modal.Image.from_registry("vllm/vllm-omni:v0.21.0rc1")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)

app = modal.App("qwen-tts-offline-poc", image=image)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
logger = logging.getLogger(__name__)


def _estimate_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
) -> int:
    try:
        import torch
        from transformers import AutoTokenizer
        from transformers.utils import cached_file
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizer,
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
        talker_config = getattr(config, "talker_config", None)

        speech_tokenizer = None
        try:
            speech_tokenizer_config_path = cached_file(
                model_name,
                "speech_tokenizer/config.json",
            )
            if speech_tokenizer_config_path:
                speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                    os.path.dirname(speech_tokenizer_config_path),
                    torch_dtype=torch.bfloat16,
                )
        except Exception as exc:
            logger.info("Skipped speech tokenizer load: %s", exc)

        def estimate_ref_code_len(ref_audio: object) -> int | None:
            if not isinstance(ref_audio, (str, list)):
                return None

            audio_path = (
                ref_audio[0] if isinstance(ref_audio, list) else ref_audio
            )
            if not isinstance(audio_path, str) or not audio_path.strip():
                return None

            try:
                import numpy
                from vllm.multimodal.media.audio import load_audio

                audio, sample_rate = load_audio(audio_path, sr=None, mono=True)
                wav = numpy.asarray(audio, dtype=numpy.float32)

                if speech_tokenizer is not None:
                    encoded = speech_tokenizer.encode(
                        wav,
                        sr=int(sample_rate),
                        return_dict=True,
                    )
                    ref_code = getattr(encoded, "audio_codes", None)
                    if isinstance(ref_code, list):
                        ref_code = ref_code[0] if ref_code else None
                    if ref_code is not None and hasattr(ref_code, "shape"):
                        shape = ref_code.shape
                        if len(shape) == 2:
                            return int(shape[0])
                        if len(shape) == 3:
                            return int(shape[1])

                codec_hz = (
                    getattr(talker_config, "codec_frame_rate", None) or 12
                )
                return int(len(audio) / sample_rate * codec_hz)
            except Exception:
                return None

        task_type = (
            additional_information.get("task_type") or ["CustomVoice"]
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
            estimate_ref_code_len=estimate_ref_code_len,
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
        "task_type": ["CustomVoice"],
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
            model_name=MODEL_ID,
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


@app.function(
    gpu=GPU,
    timeout=20 * MINUTES,
    scaledown_window=5 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
def synthesize(
    text: str = SAMPLE_TEXT,
    speaker: str = SPEAKER,
    language: str = "English",
    instructions: str = "",
    max_new_tokens: int = 2048,
) -> bytes:
    import os

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from vllm_omni import Omni

    request = _build_custom_voice_request(
        text=text,
        speaker=speaker,
        language=language,
        instructions=instructions,
        max_new_tokens=max_new_tokens,
    )
    omni = Omni(
        model=MODEL_ID,
        log_stats=True,
        stage_init_timeout=600,
        trust_remote_code=True,
    )
    try:
        for stage_outputs in omni.generate([request]):
            request_output = stage_outputs.request_output
            if request_output is None or not request_output.outputs:
                continue

            multimodal_output = request_output.outputs[0].multimodal_output
            if multimodal_output and "audio" in multimodal_output:
                return _audio_bytes_from_multimodal_output(multimodal_output)
    finally:
        close = getattr(omni, "close", None)
        if close is not None:
            close()

    raise RuntimeError("vLLM-Omni did not return audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen TTS once on Modal.")
    parser.add_argument("--text", default=SAMPLE_TEXT)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--speaker", default=SPEAKER)
    parser.add_argument("--language", default="English")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with modal.enable_output():
        with app.run():
            audio = synthesize.remote(
                text=args.text,
                speaker=args.speaker,
                language=args.language,
                instructions=args.instructions,
                max_new_tokens=args.max_new_tokens,
            )
    with open(args.output, "wb") as output_file:
        output_file.write(audio)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
