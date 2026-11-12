# meeting-stack

Diarized transcripts of long meetings, on the Strix Halo box.

Two tools, a service, a web page and three diarizer images:

- **`meeting-transcribe`** — one audio file in, a speaker-labelled transcript out
  (`PREFIX.md` + `PREFIX.json`).
- **`meeting-job`** — the same work as a service: `POST /jobs` for uploads and an
  **MCP** endpoint (`POST /mcp`) with `transcribe_meeting`, `meeting_status` and
  `meeting_result`.
- **`meeting-eval`** — score a transcript against a reference before changing the
  pipeline.
- **`GET /live`** — stream audio from the browser and read the translation as it
  arrives (see [Live](#live-audio-in-a-translated-line-out)).
- **`dashboard.html`** — the web UI `meeting-job` serves at `/` (and `/ui`):
  upload with a progress bar, live job list, and the transcript in a dialog next
  to an audio player — the sentence being spoken is highlighted and followed
  while it plays, and clicking a line seeks there. Built from `dashboard/`, one
  self-contained file, no CDN at runtime.

## How it works

```
audio ──> diarizer container ──> audio.wav (16 kHz mono, loudnorm)
                                     │
                                     ├──> one ASR request per turn (--asr-mode segments)
                                     │
                                     └──> the turn already carries its speaker
                                                    │
                                                    └──> turns ──> markdown + json
```

The box runs the per-turn path: gufo on `:8097` serving `qwen3-asr-1.7b`, which
answers `json` with text and nothing else — its `verbose_json` carries an empty
`segments` array, so the one-pass path cannot use it. Each diarizer turn is cut
out, padded into the silence beside it (`--turn-pad`-style padding of
`TURN_PAD_S`), transcribed, and kept with its speaker. Cutting at the diarizer's
edges costs audio when an edge is wrong, and a per-clip `max_tokens` cap can drop
a tail silently (a 240 s clip of dense speech lost 170 of 575 words with
`status=200`) — which is why the one-pass path exists and stays the default of
the CLI itself.

The one-pass path (`--asr-mode full`, the CLI default) sends the whole file to
an endpoint that answers `verbose_json` with real segment timestamps — Lemonade
on `:8731` with `Whisper-Large-v3-Turbo`. Nothing is cut: the ASR owns the words
and the diarizer only labels them, so a diarizer edge that is half a second off
moves a word instead of losing audio.

One thing Whisper gets wrong on its own: on audio it cannot decode (music,
crosstalk, a phone held to a speaker) it latches onto one sentence and
repeats it every second for minutes, with confident scores — a 38-minute meeting
got 134 s of `Ich denke, das ist doch nicht da.` that way, and the endpoint
ignores `condition_on_previous_text`, `vad_filter` and `no_speech_threshold`.
`collapse_repetitions()` therefore drops a run of identical segments once it is
physically impossible: four or more of them, covering eight seconds or more, of a
sentence at least four words long. "Ja" and "tschüss" repeated quickly survive.
It runs on the one-pass path only; Qwen3-ASR does not loop.

Measured on a 38:13 German meeting (5.439 words, `--diarizer nemotron`):

- one-pass Whisper, Lemonade: **1 m 38 s end to end** — normalization, one
diarization pass, one ASR pass, fusion — 119 turns, 8 speakers, transcript
covering 0 s to 2290 s of 2293 s.
- per-turn Qwen3-ASR, gufo with `--language de`: **4 m 02 s**, 229 turns. The
`--language` flag is not optional here: with `auto`, 25 of 226 turns of that same
German meeting came back in Chinese and Dutch. There is no hand-corrected
reference on this box, so the two transcripts only give their disagreement with
each other: **33.8% of words** (35.4% without `--language de`). That is not an
accuracy verdict for either model.

## Requirements

- an OpenAI-compatible ASR: gufo on `http://127.0.0.1:8097` serving
  `qwen3-asr-1.7b` for the per-turn path (see [AGENT.md](AGENT.md) for the
  container), or Lemonade on `http://127.0.0.1:8731` serving
  `Whisper-Large-v3-Turbo` for `--asr-mode full`
- rootless **podman** with `/dev/kfd` and `/dev/dri` for the diarizer
- a built diarizer image (see below)

## Usage

```sh
./meeting-transcribe meeting.m4a --asr http://127.0.0.1:8097 \
    --model qwen3-asr-1.7b --asr-mode segments --language de
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
when the service runs with `--keep-audio` (the systemd unit does) and deleted
after `--retain-audio-days` (default 30; `0` keeps it forever). Audio of a job
that is still queued or running is never touched, and the job list itself is
saved in `jobs.json` under `--dir`: restarting the service keeps the list and
the finished transcripts, and marks a job that was mid-run as an error with its
audio still there, so the dashboard's retry can run it again.

### Two tracks, one transcript

`zoom-audio-capture` records a call as two files, the system output (everybody
else) and the microphone (you). POST them both and they become **one job**:

```sh
curl --data-binary @zoom-20261020-205058-system-16k.mp3 \
     "http://box:8099/jobs?filename=zoom-20261020-205058-system-16k.mp3" -H "Authorization: Bearer $TOKEN"
curl --data-binary @zoom-20261020-205058-mic-16k.mp3 \
     "http://box:8099/jobs?filename=zoom-20261020-205058-mic-16k.mp3" -H "Authorization: Bearer $TOKEN"
# -> the same job id, twice
```

The recording key and the track come out of the file name
(`zoom-<stamp>-{system,mic}-16k.mp3`), so the recorder needs to know nothing
about pairing; `?pair=<key>&track=main|second` does the same explicitly, which
is what the dashboard sends. A job with one track stays `waiting` for the other
until `--pair-wait` seconds pass (default 900), then runs with what arrived.

### Announced recordings (`POST /hook`)

The audio above is still the way in — `POST /hook` does not carry audio. It is
the recorder saying *this recording is complete, and here is where it lives*:

```sh
curl -d '{"id":"2026-10-20/12-35-09","day":"2026-10-20","recording":"12-35-09",
          "remote_dir":"https://dav/BSB/Registrazioni/2026-10-20/12-35-09/",
          "files":["audio/zoom-20261020-123509-system-16k.mp3"]}' http://box:8099/hook
# -> {"ok":true,"pair":"zoom-20261020-123509","status":"missing_audio",...}
```

`GET /hooks` lists every announcement joined with the job its uploads produced.
`status` is the job status, or `missing_audio` when the announcement has no
upload behind it — the state that otherwise stays invisible until `--pair-wait`
gives up. Announcements are idempotent (the same body twice is a 200 and bumps
`count`) and stored in `hooks.json` under `--dir`, so they survive a restart
while the job list does not.

The two tracks are worth keeping apart. The system track is diarized as usual
(`Speaker 1…N` are the remote participants); a microphone track that carries
only the recorder's room is *one* speaker, so it is never clustered, only
*labelled* `Me` (`--second-name`; `--diarize-second` when several people share
the room). With `--asr-mode full` it is sent whole in one pass and never
diarized. With `segments` (gufo, which returns no timestamps) the diarizer runs
on it too, for the turn boundaries — without them there is nothing to place on
the timeline. If it finds more than one voice, its labels are kept and matched
to the first track's speakers instead of calling the whole room `Me`.

The two timelines are interleaved, and crosstalk stays two lines instead of one
garbled one, because the channels already separated it. When both tracks are
the same room recorded twice (a microphone that also hears the loudspeakers),
the second track's turns already covered by the first are dropped instead of
repeated; only speech the system track does not have (the room kept recording,
or a local voice the call did not carry) is added. `--second-offset` shifts the
microphone track if its file starts later than the system one.

Running the worker by hand means the same thing:

```sh
./meeting-transcribe system.mp3 --second mic.mp3 --output out/meeting
```

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

## Live: audio in, a translated line out

The same service also answers a live stream at `GET /live` (the page) and
`POST /live/<session>/audio` (16 kHz mono PCM16, raw body). The page captures the
microphone, or the tab's audio, and shows two columns: what was said and its
translation.

```sh
# the page posts one chunk every half second; here the same thing from a file
ffmpeg -v error -i meeting.wav -f s16le -ar 16000 -ac 1 - | split -b 16000 - /tmp/live-
for part in /tmp/live-*; do
  curl -s --data-binary @$part \
    "http://box:8099/live/demo/audio?language=de&to=it" -H "Authorization: Bearer $TOKEN"
  sleep 0.5
done
curl -H "Authorization: Bearer $TOKEN" "http://box:8099/live/demo?since=0"
curl -X POST -H "Authorization: Bearer $TOKEN" http://box:8099/live/demo/close
```

This is not a streaming ASR. A pause of 0.3 s ends an utterance, the utterance
goes to the ASR endpoint as one clip. The ASR punctuates every clip as if it
were a sentence, so its full stops are ignored: the line ends where the speaker
paused for real, 0.8 s (`--live-silence`), which is what keeps "…gibt es mehr
Gewalt." and "und Fehlverhalten als bislang bekannt." in one line instead of two
translated fragments. The line is then translated by an LLM (named in the
prompt: a bare ISO code is read as an English word) and, if the model answers
nothing at all, retried once without the context and then reported instead of
sitting on "…".
Semi-batch on purpose: a whole utterance is what keeps the words Whisper needs,
and one is cheap. Measured on this box with a looped 2.8 s Italian clip: 0.24 s
to transcribe 10 s through `Whisper-Large-v3-Turbo`, 0.78 s through
gufo/Qwen3-ASR, 0.59 s to translate a line with `Qwen3.8-27B-GGUF` (0.45 s of
generation at 35.7 tok/s, the rest prompt), and on the page the source line
appears at 3.4 s, the translation **starts 0.3 s later and completes 0.6 s
later** — the tokens are streamed into the row, so nothing waits for the whole
sentence to be translated. A line that never finds a pause is translated anyway
after 500 characters or 10 s. A speaker who does not pause for 20 s is cut anyway,
at the quietest frame.

While a stretch of speech has not reached a pause yet, the page says how many
seconds are held back (`2.8 s in attesa di una pausa`), then `trascrivo…` and
`traduco…`, and a level bar moves with the microphone: a transcript that stays
empty for ten seconds is supposed to look busy, not broken.

A line also ends when **the voice changes**, which is what a pause cannot tell you
when two people talk without one. That is a disposable diarization: one 28 MB
3D-Speaker embedding per clip, no clustering, one `podman run` (0.40 s measured,
and it runs beside the ASR request, so it costs no latency), compared with the
mean embedding of the line being built. Measured on the box with different texts
per voice: the same voice scores 0.888-0.954, different voices 0.144-0.408, so
the threshold is 0.6 (`--live-speaker`). Verified end to end: two voices separated
by 0.35 s of silence (too short for the line rule) come out as two lines, the same
voice with the same gap stays one. The limit is a change of voice *inside* one
clip, which no per-clip embedding can see.

```sh
podman build -t localhost/halogen-diarizer diarizer/   # the image the embedding runs in
```

The two lines before the current one go back to the model **as source and
translation pairs** (`Wir müssen… -> Dobbiamo…`), so the wording of the line
continues the wording already on screen. `--live-glossary "Candido,BSB,Zoom"`
adds names and terms the translation must leave untouched.

What it is not:

- **No speaker labels.** Live attribution would need online clustering; the
  batch transcript is still the labelled truth, so a live session is a preview.
- **One session at a time.** A live session holds the same
  `/run/user/1000/halogen-gpu.lock` for its whole life, so a job waits for it and
  vice versa (a second session gets a 409).
- **The microphone needs a secure context.** `getUserMedia` is refused on plain
  `http://` to another host: open the page as `http://127.0.0.1:8099/live` on the
  box, or behind https. Tab audio (`getDisplayMedia`) needs the same.
- **System audio depends on the engine, not the OS.** Chrome never gives audio
  from a *window*: share a **tab** (only the browser's own sound) or, from Chrome
  141 on macOS, the **entire screen** with "Condividi l'audio di sistema". Firefox
  — and Zen, which is Firefox — has **no audio at all** in `getDisplayMedia`
  (bugzilla 1541425, still open), so the share dialog is a dead end there. The
  path that always works is a virtual input device picked as the microphone:
  BlackHole 2ch on macOS (route the output through it, a Multi-Output Device keeps
  it audible), a PipeWire/Pulse loopback on Linux. Untick "filtri voce" for it:
  noise suppression eats the quiet half of the other person's voice. The input
  list only fills once the page has the microphone permission.

### Feeding it without a browser

`live-feed` is the same API as a script: ffmpeg reads the input (a file, or a
device) and the tool posts half-second chunks, printing each line with its
translation as it lands. That is the way in on macOS with Zen or Firefox, which
have no audio in `getDisplayMedia`, and for a call in a native app:

```sh
# macOS, the system audio of a Zoom/Teams call through BlackHole
./live-feed --url http://box:8099 --token "$TOKEN" --language de --to it \
    --input "avfoundation::BlackHole 2ch"

# Linux, the monitor of the speakers (pactl list short sources | grep monitor)
./live-feed --language de --to it --input "pulse:alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

# a recording, replayed at realtime speed into the live path
./live-feed --language de --to it --input meeting.m4a
```

The endpoint contract, if you write your own client: `POST /live/<id>/audio`
with raw **16 kHz mono PCM16** (half a second at a time is plenty, `?language=de&to=it`
on the first one), `GET /live/<id>?since=<cursor>` until `done` is true, then
`POST /live/<id>/close`. The token goes in the `Authorization` header; the first
chunk creates the session and takes the GPU lock (409 if a job has it). Every
row carries `i` (its index), `src`, `mt`, `done`, and `error` when it did not
work; `state` and `buffered` describe what the worker is doing meanwhile.

Flags: `--translate-model Qwen3.8-27B-GGUF` turns translation on (empty serves
the transcript alone), `--translate-url` (default Lemonade on `127.0.0.1:8731`)
is where the LLM lives, `--live-idle 120` closes a session that stopped sending
audio, `--live-glossary`, `--live-silence` (0.5 splits more, 1.2 merges more),
`--live-embedder` (empty = lines end on pauses only) and `--live-speaker`.
ASR comes from `--asr`/`--model` like the jobs do, so
gufo works too but wants an explicit `?language=`. Stopping frees the GPU lock as
soon as the last utterance is transcribed, not on the next housekeeping tick, so
a second session can start immediately.

## Flags worth knowing

- `--asr-mode full` (default) — one request for the whole file.
  `--asr-mode segments` is the older per-clip path, kept for ASR endpoints that
  return text without timestamps.
- `--merge-gap 2.0` — merge same-speaker ASR segments across gaps this long.
- `--second FILE` — the microphone track of a two-track recording; see above.
  `--second-name` (default `Me`), `--diarize-second`, `--second-offset`.
- `--diarization FILE` — take the final speakers from an external diarization
  (`{"diarization": [{start, end, speaker}]}` or a plain list), e.g. a pyannote
  precision-3 run; each turn gets the reference speaker it overlaps most. Also
  available as a job option (`diarization=<path>`), and the path must be visible
  inside the service container (i.e. under the mounted cache dir).
- `--pair-wait 900` (on `meeting-job`) — how long a one-track job waits for the
  other track before running anyway.
- `--retain-audio-days 30` (on `meeting-job`) — delete kept uploads older than
  this; `0` keeps them forever. Only done and failed jobs are candidates.
- `--model`, `--asr` — point the pipeline at another model or endpoint.

The service runs the gufo/Qwen3-ASR path (`--asr http://127.0.0.1:8097 --model
qwen3-asr-1.7b --diarizer nemotron --asr-mode segments`). The endpoint is a
`gufo serve asr` container with a 4.4 GB BF16 checkpoint; both the launch command
and the rebuild steps are in [AGENT.md](AGENT.md). Point the unit back at
`http://127.0.0.1:8731 --model Whisper-Large-v3-Turbo --asr-mode full` to return
to the one-pass path.

## Tests

```sh
python3 -m unittest test_meeting -v
```

They cover the pure functions — rebuilding words split by the server's 32
character wrap, the segment/speaker fusion, the turn merge — and assert that the
defaults stay on the verbose_json path.
