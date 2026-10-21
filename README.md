# meeting-stack

Diarized transcripts of long meetings, on the Strix Halo box.

Two tools, a web dashboard and three diarizer images:

- **`meeting-transcribe`** — one audio file in, a speaker-labelled transcript out
  (`PREFIX.md` + `PREFIX.json`).
- **`meeting-job`** — the same work as a service: `POST /jobs` for uploads and an
  **MCP** endpoint (`POST /mcp`) with `transcribe_meeting`, `meeting_status` and
  `meeting_result`.
- **`meeting-eval`** — score a transcript against a reference before changing the
  pipeline.
- **`dashboard.html`** — the web UI `meeting-job` serves at `/` (and `/ui`):
  upload with a progress bar, live job list, and the transcript in a dialog next
  to an audio player — the sentence being spoken is highlighted and followed
  while it plays, and clicking a line seeks there. Built from `dashboard/`, one
  self-contained file, no CDN at runtime.

## How it works

```
audio ──> diarizer container ──> audio.wav (16 kHz mono, loudnorm)
                                     │
                                     ├──> one ASR pass over the whole file (verbose_json, with timestamps)
                                     │
                                     └──> every ASR segment takes the speaker with the largest overlap
                                                    │
                                                    └──> turns ──> markdown + json
```

Nothing is cut. The ASR owns the words and the diarizer only labels them, so a
diarizer edge that is half a second off moves a word instead of losing audio.
That replaced an older path that cut the meeting into 150 s clips for the ASR:
there a wrong edge cost audio, and an output-token cap dropped tails silently (a
240 s clip of dense speech lost 170 of 575 words with `status=200`).

One thing the ASR gets wrong on its own: on audio it cannot decode (music,
crosstalk, a phone held to a speaker) Whisper latches onto one sentence and
repeats it every second for minutes, with confident scores — a 38-minute meeting
got 134 s of `Ich denke, das ist doch nicht da.` that way, and the endpoint
ignores `condition_on_previous_text`, `vad_filter` and `no_speech_threshold`.
`collapse_repetitions()` therefore drops a run of identical segments once it is
physically impossible: four or more of them, covering eight seconds or more, of a
sentence at least four words long. "Ja" and "tschüss" repeated quickly survive.

Measured on a 38:13 German meeting (5.439 words): **1 m 38 s end to end** —
normalization, one diarization pass, one ASR pass, fusion — 119 turns, 8 speakers,
transcript covering 0 s to 2290 s of 2293 s.

## Requirements

- an OpenAI-compatible ASR that answers `response_format=verbose_json`
  (default: Lemonade on `http://127.0.0.1:8731` serving `Whisper-Large-v3-Turbo`)
- rootless **podman** with `/dev/kfd` and `/dev/dri` for the diarizer
- a built diarizer image (see below)

## Usage

```sh
./meeting-transcribe meeting.m4a                      # Whisper + Nemotron, one pass
./meeting-transcribe meeting.wav --language de --speakers 4
./meeting-transcribe meeting.wav --names "1:Anna,2:Bernd"   # names also bias the ASR
./meeting-transcribe meeting.wav --diarizer pyannote-rocm   # the other diarizer
```

The first run downloads the Nemotron checkpoint (~380 MB) into the Hugging Face
cache; the container mounts it, so later runs are offline.

## Diarizers

| `--diarizer` | what it is | notes |
|---|---|---|
| `nemotron` (default) | NVIDIA Nemotron-3-Diarization, 99 M params, 8 speakers, OpenMDW 1.1, via transformers in `diarizer-nemotron/` | no HF token; estimates the speaker count; 4.5x faster than pyannote here |
| `pyannote-rocm` | pyannote.audio 4.x on the iGPU (ROCm) | gated models, needs an HF token; honours `--min-speakers`/`--max-speakers` |
| `sherpa` | ONNX on the CPU, no GPU | over-splits on real meetings, wants `--speakers N`; embedding trained on Chinese/English |

Measured on 10 minutes of far-field German: pyannote labelled 44.2% of the audio
and left 37.6% of ASR segments without a speaker; Nemotron labelled 75.9% and
left 2.6%. On an independent German benchmark pyannote Community-1 scores 20.22%
DER against 7.3% for Nemotron — but the real difference here was recall: a
conservative VAD was dropping half the speech.

Because Nemotron has exactly eight output channels, a meeting with more than
eight voices will misattribute or lose some.

Build the images once:

```sh
podman build -t localhost/halogen-diarizer-nemotron diarizer-nemotron/
podman build -t localhost/halogen-diarizer-pyannote-rocm diarizer-pyannote-rocm/
podman build -t localhost/halogen-diarizer diarizer/            # only for --diarizer sherpa
```

`halogen-diarizer-nemotron` is built `FROM halogen-diarizer-pyannote-rocm`: that
11.6 GB image is its base, so do not delete it if you want to rebuild.

## Service: REST + MCP

The service and the dashboard run in one container. The diarizer containers are
*not* nested: the service starts them through the host's podman socket, so they
still get `/dev/kfd`, the diarizer images and the HF cache already on the host.

```sh
podman build -t localhost/meeting-job .     # Containerfile in this repo
systemctl --user enable --now podman.socket
install -m 644 systemd/meeting-job.service ~/.config/systemd/user/
install -d -m 700 ~/.config/meeting-stack
printf 'MEETING_TOKEN=%s\n' "$(openssl rand -hex 24)" > ~/.config/meeting-stack/meeting-job.env
chmod 600 ~/.config/meeting-stack/meeting-job.env
systemctl --user daemon-reload && systemctl --user enable --now meeting-job.service
```

With `MEETING_TOKEN` empty the service starts **without authentication**; every
route but `/health` then accepts anyone who can reach the port.

```sh
curl --data-binary @meeting.m4a \
     "http://box:8099/jobs?filename=meeting.m4a&language=de" -H "Authorization: Bearer $TOKEN"
curl -H "Authorization: Bearer $TOKEN" http://box:8099/jobs/<id>          # status + result
curl -H "Authorization: Bearer $TOKEN" http://box:8099/jobs/<id>/audio    # the recording
```

`GET /jobs/<id>/audio` answers byte ranges (206), so a player can seek; the
dashboard passes the token in the query (`?token=…`) because an `<audio src>`
cannot send headers, and only that route accepts it there. The recording is kept
when the service runs with `--keep-audio` (the systemd unit does) — the files
pile up in `inbox/` and nothing prunes them, so delete old ones yourself.

MCP (`POST /mcp`, stateless JSON-RPC 2.0 — `initialize`, `tools/list`,
`tools/call`, `ping`):

| tool | arguments |
|---|---|
| `transcribe_meeting` | `audio_base64`, `filename`, `speakers`, `min_speakers`, `max_speakers`, `language`, `hotwords`, `names`, `threshold`, `diarizer`, `model`, `asr_mode` |
| `meeting_status` | `id` |
| `meeting_result` | `id`, `format` (`markdown` or `json`) |

One job runs at a time, because the diarizer loads the GPU. `meeting-job` holds
an exclusive lock for the whole job at `/run/user/1000/halogen-gpu.lock`; any
other GPU workload on the box that respects the same path will not overlap with
a transcription. The path is the contract, not this code: `lockfile.py` is a 50
line `flock` wrapper, and image generation on the same box keeps an identical
copy.

## Dashboard

`GET /` serves `dashboard.html`. It is a React + Tailwind + shadcn/ui app, but
the repo ships only the built file — `meeting-job` has no static-asset route:

```sh
cd dashboard && npm install      # once
npm run build                    # -> ../dashboard.html (one inlined file)
```

Source lives in `dashboard/src`; the components in `dashboard/src/components/ui`
are the unmodified shadcn registry files (new-york, neutral theme). The bearer
token goes in the field at the bottom of the upload card and stays in
`localStorage`. Opening a finished job shows the recording in a native player
above the transcript; the line being spoken is highlighted and scrolled into
view, and clicking any line seeks to its timestamp.

## Flags worth knowing

- `--asr-mode full` (default) — one request for the whole file.
  `--asr-mode segments` is the older per-clip path, kept for ASR endpoints that
  return text without timestamps.
- `--merge-gap 2.0` — merge same-speaker ASR segments across gaps this long.
- `--model`, `--asr` — point the pipeline at another model or endpoint.

The legacy gufo path (`--asr http://127.0.0.1:8097 --model qwen3-asr-1.7b
--diarizer pyannote-rocm --asr-mode segments`) is documented but **not available
on this box any more**: the gufo container, its image and the Qwen3-ASR weights
were removed. Recreate them only if you really need that engine.

## Tests

```sh
python3 -m unittest test_meeting -v
```

They cover the pure functions — rebuilding words split by the server's 32
character wrap, the segment/speaker fusion, the turn merge — and assert that the
defaults stay on the verbose_json path.
