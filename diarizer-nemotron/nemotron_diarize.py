#!/usr/bin/env python3
"""Diarizzazione con nvidia/Nemotron-3-Diarization via transformers (ROCm).

Stessa interfaccia del diarizer pyannote: JSON [{start,end,speaker}] su stdout e
il WAV normalizzato (`audio.wav`, 16 kHz mono, loudnorm) nella directory --out,
cosi' meeting-transcribe puo' usarlo senza cambiare il resto.

`--speakers`, `--min-speakers`, `--max-speakers` e `--threshold` sono accettati
per compatibilita' con la riga di comando ma ignorati: Nemotron ha 8 canali e
stima da se' quanti parlanti ci sono. `--model` permette un checkpoint diverso.

Nota: non usa transformers.audio_utils.load_audio, che passa da torchcodec e in
questa immagine non carica libtorchcodec_image.so (FFmpeg non combaciante).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL = "nvidia/Nemotron-3-Diarization"
NORMALIZED = "audio.wav"


def normalize(source: str, target: Path) -> None:
    """One loudnorm pass over the whole file: a far-field participant otherwise
    reaches the model tens of dB down (same recipe as the pyannote image)."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", source,
         "-af", "loudnorm=I=-18:TP=-2:LRA=11",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default=".")
    parser.add_argument("--speakers", type=int, default=-1)
    parser.add_argument("--min-speakers", type=int, default=-1)
    parser.add_argument("--max-speakers", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"nemotron: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}",
          file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    normalized = out / NORMALIZED
    normalize(args.input, normalized)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForAudioFrameClassification.from_pretrained(args.model)
    model.to(device).eval()

    sample_rate = processor.feature_extractor.sampling_rate
    audio, rate = sf.read(str(normalized), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        raise SystemExit(f"normalized audio is {rate} Hz, expected {sample_rate}")

    inputs = processor(audio, sampling_rate=rate).to(device, dtype=model.dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits          # (1, frames, 8), one frame per 10 ms
    print(f"logits: {tuple(logits.shape)}", file=sys.stderr)
    segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
    payload = [{"start": round(float(seg["Start"]), 3),
                "end": round(float(seg["End"]), 3),
                "speaker": int(seg["Speaker"])} for seg in segments]
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
