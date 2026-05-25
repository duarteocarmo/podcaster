import argparse
import gc
import glob
import json
import os
import random
import shutil

import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def _resample_wavs(folders):
    seen = set()
    for folder in folders:
        if not folder:
            continue
        for wav_path in glob.glob(os.path.join(folder, "*.wav")):
            if wav_path in seen:
                continue
            audio, _ = librosa.load(wav_path, sr=24000)
            sf.write(wav_path, audio, 24000)
            seen.add(wav_path)
    print("Resampled audio to 24kHz")


def _paired_entries(folder, ref_audio=None):
    entries = []
    for wav_path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            entry = {"audio": wav_path, "text": f.read().strip()}
        if ref_audio is not None:
            entry["ref_audio"] = ref_audio
        entries.append(entry)
    return entries


def _write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def prepare_data(args, experiment_dir):
    data_dirs = [args.train_dir, args.ref_dir]
    if args.test_dir:
        data_dirs.append(args.test_dir)
    _resample_wavs(data_dirs)

    if args.ref_audio:
        ref_audio = args.ref_audio
    else:
        ref_wavs = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
        assert ref_wavs, f"No .wav files in {args.ref_dir}"
        ref_audio = ref_wavs[0]
    print(f"Reference: {ref_audio}")

    entries = _paired_entries(args.train_dir, ref_audio=ref_audio)
    if not entries:
        raise ValueError(f"No paired .wav/.txt files in {args.train_dir}")

    rng = random.Random(args.seed)
    if args.test_dir:
        train_entries = entries
        if args.num_samples and args.num_samples < len(train_entries):
            train_entries = rng.sample(train_entries, args.num_samples)
            print(f"Subsampled to {args.num_samples} training entries (seed={args.seed})")
        test_entries = _paired_entries(args.test_dir)
    else:
        rng.shuffle(entries)
        train_count = min(args.num_samples, len(entries)) if args.num_samples else len(entries)
        train_entries = entries[:train_count]
        test_entries = entries[train_count:train_count + args.test_samples]
        if not test_entries:
            raise ValueError("No samples left for test split; lower --num_samples or add more samples")
        print(f"Split samples with seed={args.seed}")

    print(f"Training samples: {len(train_entries)}")
    print(f"Test samples: {len(test_entries)}")

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0"
    )
    prepared = []
    for i in range(0, len(train_entries), 32):
        batch = train_entries[i : i + 32]
        enc = tokenizer.encode([b["audio"] for b in batch])
        for code, entry in zip(enc.audio_codes, batch):
            entry["audio_codes"] = code.cpu().tolist()
            prepared.append(entry)

    del tokenizer
    torch.cuda.empty_cache()

    train_jsonl_path = os.path.join(experiment_dir, "train_with_codes.jsonl")
    _write_jsonl(train_jsonl_path, prepared)
    print(f"Wrote {len(prepared)} entries → {train_jsonl_path}")

    test_jsonl_path = os.path.join(experiment_dir, "test_split.jsonl")
    _write_jsonl(test_jsonl_path, test_entries)
    print(f"Wrote {len(test_entries)} test entries → {test_jsonl_path}")
    return train_jsonl_path


def _normalize(text):
    # lowercase, punctuation removed, whitespace normalized
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def evaluate(
    checkpoint_dir,
    test_jsonl,
    audio_dir,
    speaker_name,
    config,
    eval_batch_size=4,
    eval_max_new_tokens=480,
):
    import onnx_asr
    from jiwer import cer, wer

    with open(test_jsonl, encoding="utf-8") as f:
        test_entries = [json.loads(line) for line in f]
    if not test_entries:
        print("No test entries found, skipping evaluation.")
        return

    transcripts = [entry["text"] for entry in test_entries]
    basenames = [os.path.splitext(os.path.basename(entry["audio"]))[0] for entry in test_entries]
    ref_wavs = [entry["audio"] for entry in test_entries]

    # --- TTS inference ---
    tts = Qwen3TTSModel.from_pretrained(
        checkpoint_dir,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    gen_wavs = []
    gen_paths = []
    sr = None
    for start in range(0, len(transcripts), eval_batch_size):
        end = min(start + eval_batch_size, len(transcripts))
        batch_texts = transcripts[start:end]
        batch_basenames = basenames[start:end]
        print(
            f"Generating eval audio {start + 1}-{end}/{len(transcripts)}: "
            f"{', '.join(batch_basenames)}",
            flush=True,
        )
        cleanup_memory()
        batch_wavs, batch_sr = tts.generate_custom_voice(
            text=batch_texts,
            speaker=speaker_name,
            max_new_tokens=eval_max_new_tokens,
        )
        if sr is None:
            sr = batch_sr
        elif sr != batch_sr:
            raise ValueError(f"Inconsistent sample rates during eval: {sr} vs {batch_sr}")

        for basename, wav in zip(batch_basenames, batch_wavs):
            out_path = os.path.join(audio_dir, f"{basename}_finetuned.wav")
            sf.write(out_path, wav, sr)
            print(f"Saved: {out_path}", flush=True)
            gen_paths.append(out_path)
            gen_wavs.append(wav)

    del tts
    torch.cuda.empty_cache()

    # --- ASR transcription ---
    # onnxruntime-gpu requires CUDA 12; use CPU provider on CUDA 13+
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    asr_provider = "CUDAExecutionProvider" if cuda_major == 12 else "CPUExecutionProvider"
    print(f"ASR: using {asr_provider} (CUDA {torch.version.cuda})")
    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=[asr_provider])

    results = []
    for basename, transcript, ref_wav_path, gen_path, gen_wav in zip(
        basenames, transcripts, ref_wavs, gen_paths, gen_wavs
    ):
        asr_text = asr.recognize(gen_path)

        gen_duration = len(gen_wav) / sr
        ref_duration = None
        if os.path.exists(ref_wav_path):
            ref_audio, ref_sr = sf.read(ref_wav_path)
            ref_duration = len(ref_audio) / ref_sr

        ref_norm = _normalize(transcript)
        hyp_norm = _normalize(asr_text)
        sample_wer = wer(ref_norm, hyp_norm)
        sample_cer = cer(ref_norm, hyp_norm)
        dur_diff = round(gen_duration - ref_duration, 3) if ref_duration is not None else None

        print(
            f"[{basename}] WER={sample_wer:.2%}  CER={sample_cer:.2%}"
            + (f"  dur_diff={dur_diff:+.2f}s" if dur_diff is not None else "")
        )
        results.append({
            "file": basename,
            "reference": transcript,
            "asr": asr_text,
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4),
            "gen_duration_s": round(gen_duration, 3),
            "ref_duration_s": round(ref_duration, 3) if ref_duration is not None else None,
            "duration_diff_s": dur_diff,
        })

    del asr
    torch.cuda.empty_cache()

    # --- Summary ---
    avg_wer = sum(r["wer"] for r in results) / len(results)
    avg_cer = sum(r["cer"] for r in results) / len(results)
    dur_diffs = [r["duration_diff_s"] for r in results if r["duration_diff_s"] is not None]
    avg_dur_diff = sum(dur_diffs) / len(dur_diffs) if dur_diffs else None

    print(f"\n--- Eval Summary ({len(results)} samples) ---")
    print(f"Avg WER:          {avg_wer:.2%}")
    print(f"Avg CER:          {avg_cer:.2%}")
    if avg_dur_diff is not None:
        print(f"Avg duration diff: {avg_dur_diff:+.3f}s")

    eval_results = {
        "config": config,
        "summary": {
            "num_samples": len(results),
            "avg_wer": round(avg_wer, 4),
            "avg_cer": round(avg_cer, 4),
            "avg_duration_diff_s": round(avg_dur_diff, 3) if avg_dur_diff is not None else None,
        },
        "samples": results,
    }
    results_path = os.path.join(os.path.dirname(audio_dir), "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"Saved eval results → {results_path}")


def save_loss_plot(loss_history, experiment_dir, lr):
    steps = [e["global_step"] for e in loss_history]
    losses = [e["loss"] for e in loss_history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, linewidth=1.5)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training loss  (lr={lr})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(experiment_dir, "training_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot → {out_path}")


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--ref_audio", type=str, default=None, help="Specific reference wav (defaults to first file in ref_dir)")
    parser.add_argument("--test_dir", type=str, default=None, help="Optional explicit test dir; otherwise split from train_dir")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--speaker_name", type=str, default="duarte")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Number of test prompts to generate per eval batch")
    parser.add_argument("--eval_max_new_tokens", type=int, default=480,
                        help="Maximum codec tokens to generate per eval sample")
    parser.add_argument("--subcodec_input", action="store_true",
                        help="Include codec layers 1-15 in AR input (disabled by default — causes speech acceleration)")
    args = parser.parse_args()

    experiment_dir = os.path.join("experiments", args.experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    audio_dir = os.path.join(experiment_dir, "audio")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=experiment_dir,
    )

    if accelerator.is_main_process:
        prepare_data(args, experiment_dir)
    accelerator.wait_for_everyone()
    train_jsonl = os.path.join(experiment_dir, "train_with_codes.jsonl")
    test_jsonl = os.path.join(experiment_dir, "test_split.jsonl")
    cleanup_memory()

    from huggingface_hub import snapshot_download
    MODEL_PATH = snapshot_download(args.init_model_path, local_files_only=True)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    with open(train_jsonl, encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    model.train()
    loss_history = []
    global_step = 0

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.text_projection(model.talker.model.text_embedding(input_text_ids))
                    * text_embedding_mask
                )
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                if args.subcodec_input:
                    for i in range(1, 16):
                        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if step % 10 == 0:
                loss_val = loss.item()
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss_val:.4f}")
                if accelerator.is_main_process:
                    loss_history.append({"global_step": global_step, "loss": loss_val})

    if accelerator.is_main_process:
        shutil.copytree(MODEL_PATH, checkpoint_dir, dirs_exist_ok=True)

        config_file = os.path.join(checkpoint_dir, "config.json")
        with open(config_file) as f:
            config_dict = json.load(f)
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {args.speaker_name: 3000}
        talker_config["spk_is_dialect"] = {args.speaker_name: False}
        config_dict["talker_config"] = talker_config
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
        for k in [k for k in state_dict if k.startswith("speaker_encoder")]:
            del state_dict[k]

        weight = state_dict["talker.model.codec_embedding.weight"]
        state_dict["talker.model.codec_embedding.weight"][3000] = (
            target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        )
        save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
        print(f"Saved checkpoint → {checkpoint_dir}")

        save_loss_plot(loss_history, experiment_dir, args.lr)

    # Free all training state from VRAM before loading inference model
    target_speaker_embedding = None
    del model, optimizer, train_dataloader, qwen3tts
    cleanup_memory()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        cleanup_memory()
        print("\nEvaluating on test set...")
        evaluate(
            checkpoint_dir,
            test_jsonl,
            audio_dir,
            args.speaker_name,
            vars(args),
            eval_batch_size=args.eval_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
        )


if __name__ == "__main__":
    train()
