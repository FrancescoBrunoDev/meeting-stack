#!/usr/bin/env python3
"""Speaker diarization with pyannote.audio on the Strix Halo GPU (ROCm).

Same JSON output as ../diarizer (sherpa): a list of {start, end, speaker} and
the normalized 16 kHz mono WAV in --out. The pyannote pipeline estimates the
number of speakers by itself; --speakers pins it when the count is known.
Falls back to the CPU when ROCm sees no device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline

MODEL = "pyannote/speaker-diarization-community-1"
NORMALIZED = "audio.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default=".")
    parser.add_argument("--speakers", type=int, default=-1)
    parser.add_argument("--min-speakers", type=int, default=-1)
    parser.add_argument("--max-speakers", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pyannote: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
          file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    normalized = out / NORMALIZED
    # One pass of loudness normalization over the whole file: a far-field or
    # quiet participant otherwise reaches the ASR tens of dB down.
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", args.input,
         "-af", "loudnorm=I=-18:TP=-2:LRA=11",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(normalized)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    token = os.environ.get("HF_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(args.model, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(args.model, use_auth_token=token)
    pipeline.to(device)

    kwargs = {}
    if args.speakers and args.speakers > 0:
        kwargs["num_speakers"] = args.speakers
    elif args.min_speakers > 0 or args.max_speakers > 0:
        if args.min_speakers > 0:
            kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers > 0:
            kwargs["max_speakers"] = args.max_speakers
    data, sample_rate = sf.read(str(normalized), dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    annotation = getattr(diarization, "speaker_diarization", diarization)
    segments = [{"start": round(segment.start, 3), "end": round(segment.end, 3),
                 "speaker": int(label.split("_")[-1])}
                for segment, _, label in annotation.itertracks(yield_label=True)]
    json.dump(segments, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
