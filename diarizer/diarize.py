#!/usr/bin/env python3
"""Speaker diarization for one audio file, as a JSON list of segments.

Runs inside the diarizer container. It decodes any format ffmpeg understands
to 16 kHz mono PCM, diarizes with sherpa-onnx (pyannote segmentation + a
3D-Speaker embedding), writes the normalized `audio.wav` into --out so the
caller can cut segments without an ffmpeg of its own, and prints the segments
as JSON on stdout. Progress and diagnostics go to stderr, so stdout is only
the JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

SEGMENTATION = "/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDING = "/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
NORMALIZED = "audio.wav"


def decode(source: Path, target: Path) -> None:
    # One pass of loudness normalization over the whole file: a far-field or
    # quiet participant otherwise reaches the ASR tens of dB down.
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(source),
         "-af", "loudnorm=I=-18:TP=-2:LRA=11",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def samples(path: Path) -> "np.ndarray":
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit("diarizer: the decoder did not produce 16 kHz mono 16-bit PCM")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def diarize(audio: "np.ndarray", speakers: int, threshold: float,
            segmentation: str = SEGMENTATION, embedding: str = EMBEDDING, threads: int = 4):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation, window_shift_ratio=0.1),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=speakers, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise SystemExit("diarizer: a model file is missing from the image")
    return sherpa_onnx.OfflineSpeakerDiarization(config).process(audio).sort_by_start_time()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diarize one audio file to JSON segments.")
    parser.add_argument("input", help="audio file, any format ffmpeg reads")
    parser.add_argument("--out", default=".", help="directory for the normalized audio.wav")
    parser.add_argument("--speakers", type=int, default=-1,
                        help="known number of speakers; -1 lets clustering decide (default)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="clustering threshold when --speakers is -1; lower finds more speakers")
    parser.add_argument("--segmentation", default=SEGMENTATION, help="pyannote ONNX model (testing)")
    parser.add_argument("--embedding", default=EMBEDDING, help="speaker-embedding ONNX model (testing)")
    parser.add_argument("--threads", type=int, default=4, help="ONNX threads (default 4)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    normalized = out / NORMALIZED
    decode(Path(args.input), normalized)
    result = diarize(samples(normalized), args.speakers, args.threshold,
                     args.segmentation, args.embedding, args.threads)
    json.dump([{"start": round(r.start, 3), "end": round(r.end, 3), "speaker": r.speaker}
               for r in result], sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
