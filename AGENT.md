# Working on meeting-stack

Diarized transcripts of long meetings on the Strix Halo box (`ubuntu`, user
`francesco`). Read `README.md` for the user-facing description; this file is what
an agent needs before touching anything.

## Layout

```
meeting-transcribe        CLI: one audio file -> PREFIX.md + PREFIX.json
meeting-job               service: REST POST /jobs + MCP POST /mcp, one worker
meeting-eval              score a transcript against a reference
lockfile.py               flock wrapper; the PATH is the contract, not the code
diarizer/                 sherpa (ONNX, CPU)
diarizer-pyannote-rocm/   pyannote.audio on the iGPU
diarizer-nemotron/        NVIDIA Nemotron-3-Diarization via transformers (default)
systemd/meeting-job.service
test_meeting.py
```

There is no runner, no model-server launcher and no image pipeline here: those
live in the archived `halogen-runner` repo and are out of scope.

## The machine's moving parts

- ASR: Lemonade `http://127.0.0.1:8731`, OpenAI-compatible, `Whisper-Large-v3-Turbo`,
  `response_format=verbose_json`. Resident model; do not unload it.
- Diarizers: one `podman run --rm` container per job, GPU via `/dev/kfd` + `/dev/dri`.
- Service: user unit `meeting-job.service` on port 8099 (`podman-restart` is not
  involved; the unit is plain systemd).
- GPU lock: `/run/user/1000/halogen-gpu.lock`, shared with image generation.
  Never rename it in one place only — two different paths means two different
  locks and the protection silently stops working.

## Rules

- Do not touch the router aliases on `:8080` or any Hermes profile config.
- Do not evict resident models to make room; check what is loaded first
  (`curl -s localhost:8731/api/v1/health`).
- Never report a number you did not measure. The README's figures (1 m 38 s,
  119 turns, 8 speakers, 0-2290 s of 2293 s) come from a real 38:13 German
  meeting; anything else must be measured or labelled as an estimate.
- A change to `--asr-mode full`, the fusion, or the word rebuilding needs a test
  in `test_meeting.py` and a run on a real recording before it ships.

## Gotchas that already cost time

- The whisper server wraps lines at 32 characters and Lemonade turns that wrap
  into a **space**, so words arrive split ("Insel" -> "In sel"). The `text` field
  is unusable; rebuild from `words[]` (a piece without a leading space continues
  the previous word) and glue across segment boundaries (`starts_mid_word`).
- `transformers.audio_utils.load_audio` does not work in the nemotron image
  (torchcodec cannot load its FFmpeg library): read the WAV with `soundfile`.
- The nemotron image needs `librosa` on top of `transformers>=5.18`.
- `halogen-diarizer-nemotron` is built `FROM halogen-diarizer-pyannote-rocm`;
  removing that base image breaks rebuilds.
- The service runs without a token on this box (empty `MEETING_TOKEN`), by
  decision: it listens on `0.0.0.0:8099`, so anyone on the LAN or the tailnet can
  start a job.

## After changing the service or the unit

```sh
systemctl --user daemon-reload
systemctl --user restart meeting-job.service
curl -s localhost:8099/health
```

The unit passes the ASR endpoint, model, diarizer and mode explicitly, so the
running configuration is visible in `systemctl --user cat meeting-job.service`.
