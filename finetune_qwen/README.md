# Qwen TTS Finetuning Files

Minimal files needed to run Qwen3-TTS finetuning.

## Files

- `train.py` - Main finetuning/evaluation script. Includes the 0.6B text projection fix, eval batching, max token cap, and CUDA memory cleanup.
- `dataset.py` - Dataset and collate logic used by `train.py`.
- `pyproject.toml` - Python dependencies.
- `uv.lock` - Locked dependency versions for `uv`.
- `run.sh` - Single-run example script.

## Required Data Layout

`run.sh` syncs the dataset from Hugging Face to `dataset/`:

```text
dataset/
  reference/
    *.wav
    *.txt
  samples/
    *.wav
    *.txt
```

Each `.wav` should have a matching `.txt` transcript with the same basename. Training uses `samples/`; evaluation uses up to 10 deterministic random samples left out of training.

## Setup

Install dependencies:

```bash
uv sync
```

Run one experiment:

```bash
export HF_TOKEN=...
./run.sh
```

Results are written under:

```text
experiments/
```
