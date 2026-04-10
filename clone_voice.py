#!/usr/bin/env python3
"""
clone_voice.py — Zero-shot audio-to-audio voice conversion using Seed-VC v2.

Converts SOURCE audio to sound like the TARGET voice.
No training required — just two audio samples.

Usage:
    uv run clone_voice.py source.wav target_voice.wav [options]

First run will automatically:
  1. Clone Seed-VC into ./seed-vc/
  2. Create an isolated virtual environment at ./.venv-vc/ (requires uv)
  3. Install all dependencies (~500 MB download)

Subsequent runs start in seconds.

Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
          git, Python 3.10+
"""

# ── Self-bootstrapping: set up isolated venv on first run ────────────────────
import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SEED_VC_DIR = SCRIPT_DIR / "seed-vc"
VENV_DIR = SCRIPT_DIR / ".venv-vc"
# Bump this string whenever the installed package list changes; the old
# sentinel will be ignored and deps will reinstall automatically.
DEPS_VERSION = "v2"
SENTINEL = VENV_DIR / f".deps_ok_{DEPS_VERSION}"


def _in_managed_venv() -> bool:
    return str(VENV_DIR) in sys.executable


def _patch_seed_vc() -> None:
    """
    Patch seed-vc source for two known API mismatches.
    Idempotent — skips silently where already patched.
    """
    import re

    # Patch 1 — vc_wrapper.py: remove kwargs CFM.inference() no longer accepts
    vc_wrapper = SEED_VC_DIR / "modules" / "v2" / "vc_wrapper.py"
    if vc_wrapper.exists():
        text = vc_wrapper.read_text()
        original = text
        for kwarg in ("sway_sampling", "amo_sampling"):
            text = re.sub(rf',\s*{kwarg}\s*=\s*[^,)\n]+', '', text)
            text = re.sub(rf'\b{kwarg}\s*=\s*[^,)\n]+,\s*', '', text)
        if text != original:
            vc_wrapper.write_text(text)
            print("[patch] Removed unsupported sway_sampling/amo_sampling kwargs from vc_wrapper.py")

    # Patch 2 — cfm.py: inference_cfg_rate is sometimes a plain float but
    # solve_euler uses it as a subscriptable sequence ([0], [1], iteration…).
    # Instead of patching each usage, inject a one-line normalisation at the
    # very top of solve_euler so all downstream code sees a list.
    cfm_file = SEED_VC_DIR / "modules" / "v2" / "cfm.py"
    if cfm_file.exists():
        text = cfm_file.read_text()
        original = text
        # Guard: skip if the normalisation line is already present
        if "inference_cfg_rate = [inference_cfg_rate, inference_cfg_rate]" not in text:
            # Match "def solve_euler(self...):⏎<indent>" and insert normalisation
            text = re.sub(
                r"(def solve_euler\(self[^\n]*\n)(\s+)",
                lambda m: (
                    m.group(1)
                    + m.group(2)
                    + "if isinstance(inference_cfg_rate, (int, float)):\n"
                    + m.group(2) + "    "
                    + "inference_cfg_rate = [inference_cfg_rate, inference_cfg_rate]\n"
                    + m.group(2)
                ),
                text,
                count=1,
            )
        if text != original:
            cfm_file.write_text(text)
            print("[patch] Normalised inference_cfg_rate in cfm.py solve_euler")


def _bootstrap() -> None:
    # 1. Clone Seed-VC
    if not SEED_VC_DIR.exists():
        print("[setup] Cloning Seed-VC…")
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/Plachtaa/seed-vc.git",
                str(SEED_VC_DIR),
            ],
            check=True,
        )

    # 2. Patch seed-vc source for API mismatches
    _patch_seed_vc()

    # 3. Create virtual environment
    if not VENV_DIR.exists():
        print("[setup] Creating virtual environment (Python 3.10)…")
        subprocess.run(
            ["uv", "venv", str(VENV_DIR), "--python", "3.10"],
            check=True,
        )

    # 3. Install dependencies (once)
    if not SENTINEL.exists():
        print("[setup] Installing dependencies (this may take several minutes)…")
        env = {**os.environ, "VIRTUAL_ENV": str(VENV_DIR)}

        # Step A: PyTorch — requirements-mac.txt uses pip-only inline flags that uv
        # rejects, so we install torch separately with the correct index URL.
        # PyTorch 2.x stable includes MPS support for Apple Silicon.
        if sys.platform == "darwin":
            print("[setup] Installing PyTorch (MPS-compatible)…")
            subprocess.run(
                [
                    "uv", "pip", "install",
                    "--extra-index-url", "https://download.pytorch.org/whl/nightly/cpu",
                    "torch", "torchvision", "torchaudio",
                ],
                env=env,
                check=True,
            )
        else:
            print("[setup] Installing PyTorch (CUDA)…")
            subprocess.run(
                [
                    "uv", "pip", "install",
                    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
                    "torch", "torchvision", "torchaudio",
                ],
                env=env,
                check=True,
            )

        # Step B: All other Seed-VC dependencies (torch lines skipped).
        # gradio, FreeSimpleGUI, sounddevice are excluded — they are only used
        # by the web-UI and real-time apps, not by the conversion pipeline, and
        # gradio in particular creates large temp directories on every import.
        print("[setup] Installing remaining dependencies…")
        other_deps = [
            "accelerate",
            "scipy==1.13.1",
            "librosa==0.10.2",
            "huggingface-hub>=0.28.1",
            "munch==4.0.0",
            "einops==0.8.0",
            "descript-audio-codec==1.0.0",
            "pydub==0.25.1",
            "resemblyzer",
            "transformers==4.46.3",
            "soundfile==0.12.1",
            "modelscope==1.18.1",
            "funasr==1.1.5",
            "numpy==1.26.4",
            "pyyaml",
            "python-dotenv",
            "hydra-core==1.3.2",
        ]
        subprocess.run(
            ["uv", "pip", "install"] + other_deps,
            env=env,
            check=True,
        )

        SENTINEL.touch()
        print("[setup] Dependencies installed.")

    # 4. Re-execute inside the managed environment
    python = str(VENV_DIR / "bin" / "python")
    print("[setup] Restarting in managed environment…\n")
    os.execv(python, [python] + sys.argv)


if not _in_managed_venv():
    _bootstrap()
    sys.exit(0)  # unreachable after execv

# Apply source patches on every run so an already-cloned repo is kept up to date.
_patch_seed_vc()

# ── Real imports (only reached inside managed venv) ──────────────────────────
import argparse
import datetime

# Redirect all model/dataset caches into the project folder so nothing
# scatters across ~/. These must be set before importing torch/transformers.
_CACHE_DIR = str(SEED_VC_DIR / "checkpoints" / "hf_cache")
os.environ.setdefault("HF_HOME", _CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", _CACHE_DIR)
os.environ.setdefault("MODELSCOPE_CACHE", _CACHE_DIR)
os.environ.setdefault("FUNASR_CACHE", _CACHE_DIR)

# Enable CPU fallback for MPS ops not yet implemented in Metal
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import soundfile as sf
import numpy as np

# Add seed-vc source to path so its internal modules can be imported
sys.path.insert(0, str(SEED_VC_DIR))

from omegaconf import OmegaConf
from hydra.utils import instantiate


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot audio-to-audio voice conversion (Seed-VC v2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tuning guide:
  Intelligibility / timing problems?
    Lower --temperature (try 0.5–0.7) and raise --intelligibility (try 0.9–1.0).
    --top-p 0.7 also helps keep the AR model closer to the source content.
    For perfect temporal alignment, use --no-style (timbre-only, no AR pass).

  Voice not close enough to target?
    Raise --similarity (try 0.85–1.0). Use a cleaner, longer target sample.

  Output too fast/slow?
    Adjust --length-adjust (e.g. 0.9 to speed up, 1.1 to slow down).

Examples:
  uv run clone_voice.py speech.wav voice.wav
  uv run clone_voice.py speech.wav voice.wav -o result.wav
  uv run clone_voice.py speech.wav voice.wav -o ~/Desktop/
  uv run clone_voice.py speech.wav voice.wav --temperature 0.7 --intelligibility 0.9
  uv run clone_voice.py speech.wav voice.wav --no-style --similarity 0.85
  uv run clone_voice.py speech.wav voice.wav --diffusion-steps 50
        """,
    )
    parser.add_argument("source", type=Path, help="Source audio (content to preserve)")
    parser.add_argument("target", type=Path, help="Target voice sample — clean, single-speaker audio. Optimal 10–20 s; hard cap 25 s (longer is silently truncated)")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output file path or directory. If a directory (or omitted), the file is named "
             "<source>-converted-<timestamp>.wav inside it [default dir: ./output]",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=30,
        metavar="N",
        help="Quality vs speed: 10=fast, 30=default, 50=best [default: 30]",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        metavar="F",
        help="AR randomness 0.5–1.5; lower = more faithful to source [default: 0.7]",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        metavar="F",
        help="AR nucleus sampling 0.5–1.0; lower = more deterministic [default: 0.7]",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.7,
        metavar="F",
        help="Voice similarity to target 0.0–1.0 [default: 0.7]",
    )
    parser.add_argument(
        "--intelligibility",
        type=float,
        default=0.9,
        metavar="F",
        help="Speech clarity 0.0–1.0; raise if words sound garbled [default: 0.9]",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.5,
        metavar="F",
        help="Penalise repeated tokens 1.0–2.0 [default: 1.5]",
    )
    parser.add_argument(
        "--length-adjust",
        type=float,
        default=1.0,
        metavar="F",
        help="Output speed: <1.0=faster, >1.0=slower [default: 1.0]",
    )
    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Timbre-only mode: no AR pass — preserves exact timing and phrasing",
    )
    return parser.parse_args()


# ── Device selection ──────────────────────────────────────────────────────────
def select_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16  # MPS autocast requires bfloat16 or float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(device: torch.device, dtype: torch.dtype):
    print(f"[model] Loading Seed-VC v2  (device={device}, dtype={dtype})")
    print("[model] First run: downloading model weights from HuggingFace (~500 MB)…")

    cfg = OmegaConf.load(SEED_VC_DIR / "configs" / "v2" / "vc_wrapper.yaml")
    wrapper = instantiate(cfg)
    wrapper.load_checkpoints()  # auto-downloads to seed-vc/checkpoints/hf_cache/
    wrapper = wrapper.to(device)
    wrapper.eval()

    # Pre-allocate AR caches (required for convert_voice_with_streaming)
    wrapper.setup_ar_caches(
        max_batch_size=1,
        max_seq_len=4096,
        dtype=dtype,
        device=device,
    )

    print("[model] Ready.\n")
    return wrapper


# ── Conversion ────────────────────────────────────────────────────────────────
def run_conversion(
    wrapper,
    source: Path,
    target: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, np.ndarray]:
    convert_style = not args.no_style
    mode = "voice + style/accent transfer" if convert_style else "timbre only"
    print(f"[convert] Source : {source.name}")
    print(f"[convert] Target : {target.name}")
    print(f"[convert] Mode   : {mode}")
    print(f"[convert] Steps  : {args.diffusion_steps}  |  temp={args.temperature}"
          f"  |  top_p={args.top_p}  |  sim={args.similarity}"
          f"  |  intel={args.intelligibility}  |  rep_pen={args.repetition_penalty}")
    print("[convert] Running… (may take a minute)")

    # Always use convert_voice_with_streaming for both modes.
    #
    # convert_timbre is intentionally avoided: it processes the entire audio in
    # one pass with no chunking, overflows the DiT's 8192-token context window
    # on anything but very short clips, and passes inference_cfg_rate as a plain
    # float while solve_euler requires a two-element list.
    #
    # convert_voice_with_streaming:
    #   - chunks long audio automatically (max_context_window logic)
    #   - passes inference_cfg_rate=[intelligebility, similarity] (correct type)
    #   - convert_style=False → skips the AR pass (timbre-only, same as --no-style)
    #   - convert_style=True  → runs the AR pass (full accent/emotion transfer)
    #
    # Note: 'intelligebility' is a typo in the original Seed-VC parameter name.
    gen = wrapper.convert_voice_with_streaming(
        source_audio_path=str(source),
        target_audio_path=str(target),
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility,
        similarity_cfg_rate=args.similarity,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        convert_style=convert_style,
        anonymization_only=False,
        device=device,
        dtype=dtype,
        stream_output=True,
    )
    sr, audio = None, None
    for _chunk_bytes, full_audio in gen:
        if full_audio is not None:
            sr, audio = full_audio

    if audio is None:
        raise RuntimeError("Conversion produced no output — check your audio files.")

    return sr, audio


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Validate inputs
    for path, label in [(args.source, "source"), (args.target, "target")]:
        if not path.exists():
            sys.exit(f"Error: {label} file not found: {path}")

    # Resolve all paths to absolute before any directory changes
    args.source = args.source.resolve()
    args.target = args.target.resolve()
    if args.output is not None:
        args.output = args.output.resolve()

    # Seed-VC resolves some paths relative to its own directory
    os.chdir(SEED_VC_DIR)

    device, dtype = select_device()
    print(f"[device] {device}\n")

    wrapper = load_model(device, dtype)
    sr, audio = run_conversion(wrapper, args.source, args.target, args, device, dtype)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_stem = args.source.stem
    default_name = f"{source_stem}-converted-{timestamp}.wav"

    if args.output is None:
        out_path = Path("./output") / default_name
    elif args.output.suffix.lower() == ".wav":
        out_path = args.output          # explicit filename
    else:
        out_path = args.output / default_name   # treat as directory

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)

    print(f"\n[done] Saved → {out_path}")


if __name__ == "__main__":
    main()
