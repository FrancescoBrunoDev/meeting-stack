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

import json
import sys
import wave

import numpy as np
import sherpa_onnx

EMBEDDING = "/models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


def main() -> None:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        source = open(sys.argv[1], "rb")
    else:
        source = sys.stdin.buffer
    with wave.open(source, "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise SystemExit("embed: 16 kHz mono 16-bit PCM only")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMBEDDING, num_threads=4))
    stream = extractor.create_stream()
    stream.accept_waveform(16000, audio / 32768.0)
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise SystemExit("embed: too little audio")
    json.dump([round(value, 6) for value in extractor.compute(stream)], sys.stdout)


if __name__ == "__main__":
    main()
