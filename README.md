# meeting-stack

Diarized transcripts of long meetings. The inference (Unsloth Studio, audio.cpp,
Mistral, Lemonade) lives on a GPU box; the service is one plain container that
deploys anywhere that can reach it.

Two tools, a service and a web page.

- **`meeting-transcribe`** — one audio file in, a speaker-labelled transcript out
  (`PREFIX.md` + `PREFIX.json`).
- **`meeting-job`** — the same work as a service: `POST /jobs` for uploads and an
  **MCP** endpoint (`POST /mcp`) with `transcribe_meeting`, `meeting_status` and
  `meeting_result`.
- **`meeting-eval`** — score a transcript against a reference before changing the
  pipeline.
- **`GET /live`** — a second view of the same dashboard: stream audio from the
  browser and read the translation as it arrives (see
  [Live](#live-audio-in-a-translated-line-out)).
- **`dashboard.html`** — the web UI `meeting-job` serves at `/` (and `/ui`):
  upload with a progress bar, live job list, and the transcript in a dialog next
  to an audio player — Text (paragraphs per speaker) or Segments (one timestamped
  line each), the sentence being spoken is highlighted and followed while it
  plays, and clicking a line seeks there. Built from `dashboard/`, one
  self-contained file, no CDN at runtime.

## How it works

```
audio ──> Unsloth Studio transcribe workflow (unsloth.*, default)
       │     raw upload -> /api/inference/audio/transcribe/source
       │     NDJSON: segments with words, speaker labels, timestamps
       │     the studio decodes the file: no ffmpeg, no --asr-dir, up to 4 h
       ├──> audio.cpp (moss-transcribe-diarize), 16 kHz mono WAV from ffmpeg
       └──> Mistral (voxtral-*, diarize=true)
                          │
                          └──> turns ──> markdown + json
```

No inference runs in this stack. The service deployment's batch path is Unsloth
Studio's transcribe workflow, the same one its own `/audio` page runs: the
original file is uploaded raw (the studio decodes it, so no normalization and no
shared directory), then a transcription named by the uploaded input answers as
NDJSON with timestamped segments carrying the speaker labels. It takes up to 4
hours (the studio's `max_seconds`). That is why meeting-stack does not use the
OpenAI-compatible `/v1/audio/transcriptions` route: it stops at 30 minutes, and
`verbose_json` drops the speakers anyway.

The direct local path is one request to
[audio.cpp](https://github.com/ggml-org/audio.cpp) serving
`moss-transcribe-diarize`: the model answers the words *and* the speaker labels
*and* the timestamps, so nothing is cut, no diarizer runs, and no edge that is
half a second off can move a word. The cloud path is one request to Mistral's
`/v1/audio/transcriptions` with `diarize=true` (auto-selected for `voxtral-*`
models); its 65,536-token context window is ~34 minutes, so longer recordings go
in `MISTRAL_MAX_S` chunks with the timestamps shifted back.

Measured on a 38:13 German meeting: 429 turns in 25 minutes of wall time on the
audio.cpp path, 30365 prompt tokens. The `max_tokens` cap has to be generous
(65536) and it must travel under `"options"` — at 24000 the model stops with
`reached max_tokens before EOS` and the transcript is *lost*, not truncated.

The HTTP API (batch jobs, live sessions, MCP) is written up in
[API.md](API.md), which is the file to hand to anything that has to talk to this
service.

## Requirements

- [Unsloth Studio](https://github.com/unslothai/unsloth) on
  `http://ubuntu:8888`, serving `MOSS-Transcribe-Diarize`, with its API key in
  `UNSLOTH_API_KEY` (default batch path)
- [audio.cpp](https://github.com/ggml-org/audio.cpp) on `http://ubuntu:8090`
  for the live transcriber and the direct `moss-transcribe-diarize` fallback;
  that fallback needs a directory it can read the normalized WAV from
  (`--asr-dir`): it takes a path, not an upload
- `MISTRAL_API_KEY` for the cloud path (batch `mistral.voxtral-*`, the cloud
  translators, live realtime)
- **podman or docker** for the service container, **ffmpeg** for the
  normalization on the audio.cpp/Mistral paths

## Usage

```sh
./meeting-transcribe meeting.m4a \
    --model unsloth.MOSS-Transcribe-Diarize          # Unsloth Studio (UNSLOTH_API_KEY)
./meeting-transcribe meeting.m4a \
    --model moss-transcribe-diarize --asr-dir ~/audio
./meeting-transcribe meeting.wav --names "1:Anna,2:Bernd"   # names also bias Mistral
./meeting-transcribe meeting.m4a --model mistral.voxtral-mini-transcribe-3    # goes to Mistral
```

`--asr-dir` is where the 16 kHz mono WAV is written so audio.cpp can open it;
`--asr-remote-dir` says how that same directory looks to the server when the two
differ (audio.cpp running in its own container: host `~/audio`, server
`/opt/audiocpp`).

## Service: REST + MCP

The service and the dashboard run in one container - a plain one now: no
podman socket, no device mounts, no nested containers. All inference is reached
over HTTP (audio.cpp, Mistral, Lemonade for the translations, the Bifrost
gateway).

The deployment is `docker-compose.yml`: it builds the image and runs it with
the config in `.env`.

```sh
cp .env.example .env        # fill MEETING_TOKEN, MISTRAL_API_KEY, UNSLOTH_API_KEY, UBUNTU_HOST
docker compose up -d --build        # or: podman-compose up -d --build
```

A plain `podman run` / `docker run` needs the same flags by hand: `--env-file
.env`, the `meeting-data:/data` and `meeting-db:/db` volumes, `-p 8099:8099`,
and the `command` from the compose file.

### Docker Compose / Coolify

`docker-compose.yml` deploys the same image on any docker/podman host — e.g. a
Coolify server in the same LAN as this box. Every deploy choice is an
environment variable: `MEETING_TOKEN`, `MISTRAL_API_KEY`, `UNSLOTH_API_KEY` and
`UBUNTU_HOST` (the LAN address of this box; audio.cpp `:8090`, Unsloth Studio
`:8888`, Lemonade `:8731` and Bifrost `:8080` all hang off it) must be set, the
rest — `MEETING_PORT`, `AUDIOCPP/LEMONADE/BIFROST_PORT`,
`BATCH/LIVE/TRANSLATE_MODEL` — carries a default in the file. Batch defaults to
`unsloth.MOSS-Transcribe-Diarize`, which needs only the studio's key. A bare
audio.cpp model from another machine travels as a multipart upload, and the
server **segfaults past ~20 minutes of recording** (measured: 11 min whole, 43
min crash), so the pipeline refuses the bigger ones with an actionable error.
A shared filesystem (`--asr-dir`) restores the path mode for any length. The
cache dir and the SQLite ledger are volumes, so jobs, transcripts and the file
history survive redeploys. WebDAV is optional: copy `.env.example` to `.env` and
fill `WEBDAV_URL`, `WEBDAV_USER` and `WEBDAV_PASSWORD` (see [Archive](#archive-webdav--sqlite));
without it the ledger still records every upload, no copy is made.

That is the live deployment today: Coolify on `raspberrypi2:8099`, with
`tailscale serve --bg --https=443 8099` over it — so the same service answers
at `https://raspberrypi2.tail4050b0.ts.net/`, which is also the easiest secure
context for the live page's `getUserMedia`. Without `--bg` the serve runs in
the foreground and a `Ctrl+C` removes it. If the port ever changes, redo the
serve with the new one.

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
by default (`--no-keep-audio` deletes) and pruned after `--retain-audio-days`
(default 30; `0` keeps it forever). Audio of a job
that is still queued or running is never touched, and the job list itself is
saved in `jobs.json` under `--dir`: restarting the service keeps the list and
the finished transcripts, and marks a job that was mid-run as an error with its
audio still there, so the dashboard's retry can run it again.

### Archive: WebDAV + SQLite

Every upload is a row in a small SQLite ledger (`archive.db` under `--dir`),
written the moment the file lands: job id, track, original name, size, SHA-256,
received time, processing status, and the WebDAV URL once copied. When
`WEBDAV_URL` is set, the file is PUT to
`WEBDAV_URL/WEBDAV_DIR/<job>/<track>-<name>` as the job ends; MKCOL creates the
folders. A WebDAV failure is recorded on the row and does not fail the job: the
local copy stays (even with `--no-keep-audio`), and `POST /jobs/<id>/retry`
re-runs the job and retries the archive.

| variable / flag | meaning |
|---|---|
| `WEBDAV_URL` / `--webdav-url` | the WebDAV collection, e.g. `https://dav/BSB/Registrazioni` |
| `WEBDAV_USER`, `WEBDAV_PASSWORD` | Basic auth (`--webdav-user`, `--webdav-password`) |
| `WEBDAV_DIR` / `--webdav-dir` | folder under the root (default `meeting-stack`) |
| `ARCHIVE_DB` / `--archive-db` | ledger path (default `<dir>/archive.db`) |

`GET /files` lists the ledger, newest first. The credentials come from `.env`
(compose `env_file`/interpolation, or `--env-file .env` for a plain
`podman run` / `docker run`) and the ledger lives on the `meeting-db` volume.

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

Both tracks are diarized by the ASR itself. The system track's
`Speaker 1…N` are the remote participants; a microphone track that carries
only the recorder's room comes back as *one* speaker and is labelled `Me`
(`--second-name`). A mic track that holds several voices keeps its own labels,
shifted after the system ones — unless both tracks recorded the same room, in
which case the shared clock matches each mic voice to the first track's speaker
(see below).

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
| `transcribe_meeting` | `audio_base64`, `filename`, `language`, `hotwords`, `names`, `model` |
| `meeting_status` | `id` |
| `meeting_result` | `id`, `format` (`markdown` or `json`) |

One job runs at a time (a single worker thread, in process); the ASR server
on the other end serializes its own GPU however it likes.

## Dashboard

`GET /` serves `dashboard.html`. It is a React + Tailwind + shadcn/ui app, but
the repo ships only the built file — `meeting-job` has no static-asset route:

```sh
cd dashboard && npm install      # once
npm run build                    # -> ../dashboard.html (one inlined file)
```

The same bundle answers `GET /live` with the live view (`dashboard/src/Live.tsx`):
`App` picks it from the path, the header links the two, and a full page load
between them keeps the state honest. The live page used to be a hand-written
`live.html` — it now uses the same components, so there is one design system and
one build.

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
Semi-batch on purpose: a whole utterance is what keeps the words the ASR
needs, and one is cheap. Measured on this box: 0.59 s to translate a line with
`Qwen3.8-27B-GGUF` (0.45 s of
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

The transcript shows the last three lines — the newest full, the ones above it
stepping back (opacities 1, 0.86, 0.72) — and **Expand** opens the whole thing,
scrollable with a snap and following the newest line unless you have scrolled up
to read.

The two lines before the current one go back to the model **as source and
translation pairs** (`Wir müssen… -> Dobbiamo…`), so the wording of the line
continues the wording already on screen. `--live-glossary "Candido,BSB,Zoom"`
adds names and terms the translation must leave untouched.

What it is not:

- **No speaker labels.** Neither the local nor the cloud live ASR runs the
  online clustering that would attribute the lines; the batch transcript is
  still the labelled truth, so a live session is a preview.
- **The microphone needs a secure context.** `getUserMedia` is refused on plain
  `http://` to another host: open the page as `http://127.0.0.1:8099/live` on the
  box, or behind https — on the tailnet, `tailscale serve --bg --https=443 8099`
  gives the page one for free. Tab audio (`getDisplayMedia`) needs the same.
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
chunk creates the session. Every
row carries `i` (its index), `src`, `mt`, `done`, and `error` when it did not
work; `state` and `buffered` describe what the worker is doing meanwhile.

The page remembers what you chose — spoken language, target, input device, voice
filters, and whether the transcript was expanded — in `localStorage` under
`live-*`, so the next meeting starts where the last one ended. A remembered
device that has since been unplugged is dropped rather than failing to open.
Flags for the live path: `--live-asr`/`--live-model` point it at a faster
endpoint than the jobs use — a clip every half second makes the fixed cost of a
request the budget — and `--live-silence` is the pause that ends a line
(0.8 s; clips are cut at 0.6 s).

`--translate-model Qwen3.8-27B-GGUF` turns translation on (empty serves
the transcript alone), `--translate-url` (default Lemonade on `127.0.0.1:8731`)
is where the LLM lives. The studio is **the** model gateway: its saved
connections carry the cloud keys (Mistral, OpenCode Go) and the Lemonade
models, so `unsloth.<id>` reaches its own and the studio-named Lemonade
models (`external::<connection>::<model>`), and `mistral.<id>`/`opencode.<id>`
ride the studio's connections as `external::` ids too — one URL and one key
(`UNSLOTH_URL`, default `http://ubuntu:8888`, `UNSLOTH_API_KEY`). Measured
live, its translation-specialised `unsloth.Hy-MT2-7B-GGUF` answers a line in
~0.3 s where the 27B needs ~0.6 s+. The direct paths (Lemonade at
`--translate-url`, Mistral's API, `--mt-gateway`) survive only for a
deployment with no studio at all.
`--live-idle 120` closes a session that stopped sending
audio, `--live-glossary`, `--live-silence` (0.5 splits more, 1.2 merges more).
ASR comes from `--live-asr`/`--live-model` (audio.cpp's streaming endpoint with
streaming, on by default, or one clip per utterance with `--no-live-stream`).

## Flags worth knowing

- A model audio.cpp serves (the `meeting-transcribe` default, no key needed) — one request to
  [audio.cpp](https://github.com/ggml-org/audio.cpp) with
  `MOSS-Transcribe-Diarize`, which returns the speaker labels and the
  timestamps itself. It needs `--asr-dir`, the directory audio.cpp reads the
  normalized WAV from (a path, not an upload — `--asr-remote-dir` when the two
  processes see it differently), and `ffmpeg`. One request covers a whole
  meeting: a 38-minute German recording came back as 429 turns in 25 minutes of
  wall time, with 30365 prompt tokens. The `max_tokens` cap has to be generous
  (65536) and it must travel under `"options"` — at 24000 the model stops with
  `reached max_tokens before EOS` and the transcript is lost, and at the top
  level of the request the server ignores it without a word.
- A `voxtral-*` model — one request to Mistral's `/v1/audio/transcriptions`
  with `diarize=true`; recordings longer than 34 minutes go in chunks. Needs
  `MISTRAL_API_KEY`. `mistral.voxtral-*` is the prefixed id the API and the
  dashboard speak.
- An `unsloth.*` model (e.g. `unsloth.MOSS-Transcribe-Diarize`) — Unsloth
  Studio's transcribe workflow on `$UNSLOTH_URL` (default
  `http://ubuntu:8888`) with `$UNSLOTH_API_KEY`: the original file travels as
  one raw upload (the studio decodes it — no normalization, no shared
  directory) and the workflow accepts up to 4 hours, where the OpenAI
  `/v1/audio/transcriptions` route stops at 30 minutes. Measured: 43 minutes of
  audio in ~15 minutes of wall time, speakers included. `--job-timeout` (4 h)
  is the service-side cap for one job.
- `--second FILE` — the microphone track of a two-track recording; see above.
  `--second-name` (default `Me`), `--second-offset`.
- `--pair-wait 900` (on `meeting-job`) — how long a one-track job waits for the
  other track before running anyway.
- `--retain-audio-days 30` (on `meeting-job`) — delete kept uploads older than
  this; `0` keeps them forever. Only done and failed jobs are candidates.
- `--model`, `--asr` — point the pipeline at another model or endpoint.

The compose deployment runs the Unsloth Studio path: `--model
unsloth.MOSS-Transcribe-Diarize` with `UNSLOTH_API_KEY`, uploads kept, live
streaming from audio.cpp on `http://ubuntu:8090`, translation through Lemonade
and the Bifrost gateway.

## Tests

```sh
python3 -m unittest test_meeting -v
```

They cover the pure functions — the audio.cpp, Unsloth Studio and Mistral
answer parsing, the turn merge and dedupe, the two-track matching, the live line
building — and assert that the standalone defaults stay on the keyless audio.cpp
path while `unsloth.*` is the studio's one-upload NDJSON workflow.
