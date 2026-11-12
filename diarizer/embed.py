#!/usr/bin/env python3
"""One speaker embedding for one WAV on stdin, as a JSON vector on stdout.

This is the disposable half of the diarizer: not "who is speaking" but "is this
still the same voice". The live path cuts a line when the voice changes, which a
pause cannot tell you when two people talk without one, and it needs an answer in
the time one clip takes — so no clustering, no segmentation model, just the
3D-Speaker embedding the image already carries.

The WAV comes on stdin because the caller runs inside another container: its
/tmp is not the host's, and a path would be a trap.
"""

from __future__ import annotations

import io
import json
import sys
import wave

import numpy as np
import sherpa_onnx

EMBEDDING = "/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


def extractor() -> "sherpa_onnx.SpeakerEmbeddingExtractor":
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMBEDDING, num_threads=4))


def vector(engine, payload: bytes) -> list[float]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError("16 kHz mono 16-bit PCM only")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
    stream = engine.create_stream()
    stream.accept_waveform(16000, audio / 32768.0)
    stream.input_finished()
    if not engine.is_ready(stream):
        raise ValueError("too little audio")
    return [round(value, 6) for value in engine.compute(stream)]


def serve(engine) -> None:
    """One process, many clips: 4-byte big-endian length, then the WAV; one JSON out.

    Starting a container per clip costs 0.40 s of pure startup, which in a live
    session is the difference between keeping up and falling behind.
    """
    while True:
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            return
        size = int.from_bytes(header, "big")
        payload = sys.stdin.buffer.read(size)
        if len(payload) < size:
            return
        try:
            answer: dict = {"vector": vector(engine, payload)}
        except (ValueError, EOFError, wave.Error) as error:
            answer = {"error": str(error)}
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()


def main() -> None:
    if "--loop" in sys.argv:
        serve(extractor())
        return
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        payload = open(sys.argv[1], "rb").read()
    else:
        payload = sys.stdin.buffer.read()
    json.dump(vector(extractor(), payload), sys.stdout)


if __name__ == "__main__":
    main()
