# Eco-Remote

Diarized transcripts of long meetings. The inference (Unsloth Studio, audio.cpp,
Mistral) lives on a GPU box; the service is one plain container that deploys
anywhere that can reach it.

The worker, the evaluator, the service and a web page.

- **`meeting-transcribe`** — one audio file in, a speaker-labelled transcript out
  (`PREFIX.md` + `PREFIX.json`).
- **`meeting-job`** — the same work as a service: `POST /jobs` for uploads and an
  **MCP** endpoint (`POST /mcp`) with `transcribe_meeting`, `meeting_status` and
  `meeting_result`.
- **`meeting-eval`** — score a transcript against a reference before changing the
  pipeline.
- **`dashboard.html`** — the web UI `meeting-job` serves at `/` (and `/ui`):
  upload with a progress bar, the job list, and the transcript in a dialog next
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
hours (the studio's `max_seconds`). That is why Eco-Remote does not use the
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

The HTTP API (batch jobs and MCP) is written up in
[API.md](API.md), which is the file to hand to anything that has to talk to this
service.

## Requirements

- [Unsloth Studio](https://github.com/unslothai/unsloth) on
  `http://ubuntu:8888`, serving `MOSS-Transcribe-Diarize`, with its API key in
  `UNSLOTH_API_KEY` (default batch path)
- [audio.cpp](https://github.com/ggml-org/audio.cpp) on `http://ubuntu:8090`
  for the direct `moss-transcribe-diarize` fallback; it needs a directory it can
  read the normalized WAV from (`--asr-dir`): it takes a path, not an upload
- `MISTRAL_API_KEY` for the cloud path (batch `mistral.voxtral-*`)
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
over HTTP (audio.cpp, Unsloth Studio, Mistral).

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
`UBUNTU_HOST` (the LAN address of this box; audio.cpp `:8090` and Unsloth
Studio `:8888` hang off it) must be set, the rest — `MEETING_PORT`,
`AUDIOCPP_PORT`, `BATCH_MODEL` — carries a default in the file. Batch defaults to
`unsloth.MOSS-Transcribe-Diarize`, which needs only the studio's key. A bare
audio.cpp model from another machine travels as a multipart upload, and the
server **segfaults past ~20 minutes of recording** (measured: 11 min whole, 43
min crash), so the pipeline refuses the bigger ones with an actionable error.
A shared filesystem (`--asr-dir`) restores the path mode for any length. The
cache dir and the SQLite ledger are volumes, so jobs, transcripts and the file
history survive redeploys. WebDAV is optional: copy `.env.example` to `.env` and
fill `WEBDAV_URL`, `WEBDAV_USER` and `WEBDAV_PASSWORD` (see [Archive](#archive-webdav--sqlite));
without it the ledger still records every upload, no copy is made.

That is the deployment today: Coolify on `raspberrypi2:8099`, with
`tailscale serve --bg --https=443 8099` over it — so the same service answers
at `https://raspberrypi2.tail4050b0.ts.net/`. Without `--bg` the serve runs in
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
written the moment the file lands: job id, role (`audio` or transcript),
original name, size, SHA-256, received time, processing status, and the WebDAV
URL once copied. When `WEBDAV_URL` is set, the job folder
`WEBDAV_DIR/<job_id>/` is created and filled as the job ends: the audio as
`audio-<name>`, plus `transcript.md` and `transcript.json`. A WebDAV failure is
recorded on the row and does not fail the job: the local copy stays (even with
`--no-keep-audio`), and `POST /jobs/<id>/retry` re-runs the job and retries the
archive.

| variable / flag | meaning |
|---|---|
| `WEBDAV_URL` / `--webdav-url` | the WebDAV collection, e.g. `https://dav/BSB/Registrazioni` |
| `WEBDAV_USER`, `WEBDAV_PASSWORD` | Basic auth (`--webdav-user`, `--webdav-password`) |
| `WEBDAV_DIR` / `--webdav-dir` | folder under the root (default `meeting-stack`) |
| `ARCHIVE_DB` / `--archive-db` | ledger path (default `<dir>/archive.db`) |

`GET /archive` probes the WebDAV collection (cached for 30 s); `GET /files`
lists the ledger, newest first. The dashboard shows the same:
the `WebDAV ok/off/error` badge in the jobs header, a per-job Archive column
(`saved`/`failed`) and the job folder link in the transcript dialog. The
credentials come from `.env` (compose `env_file`/interpolation, or
`--env-file .env` for a plain `podman run` / `docker run`) and the ledger lives
on the `meeting-db` volume.

### One mixed recording

The recorder sends a single file with both sides already mixed (the system
output and the microphone). Upload it once:

```sh
curl --data-binary @zoom-20261020-205058-mix.mp3 \
     "http://box:8099/jobs?filename=zoom-20261020-205058-mix.mp3" -H "Authorization: Bearer $TOKEN"
```

The ASR diarizes and transcribes it in one pass: no second input, no pairing,
no `waiting` state. The dashboard's upload field takes one file too. The model is
not an upload option: it is the service's `--model` (compose `BATCH_MODEL`).

### Announced recordings (`POST /hook`)

The audio above is still the way in — `POST /hook` does not carry audio. It is
the recorder saying *this recording is complete, and here is where it lives*:

```sh
curl -d '{"id":"2026-10-20/12-35-09","day":"2026-10-20","recording":"12-35-09",
          "remote_dir":"https://dav/BSB/Registrazioni/2026-10-20/12-35-09/",
          "files":["audio/zoom-20261020-123509-mix.mp3"]}' http://box:8099/hook
# -> {"ok":true,"pair":"zoom-20261020-123509","status":"missing_audio",...}
```

`GET /hooks` lists every announcement joined with the job whose upload has the
same file name. `status` is the job status, or `missing_audio` when the
announcement has no upload behind it. Announcements are idempotent (the same
body twice is a 200 and bumps `count`) and stored in `hooks.json` under `--dir`,
so they survive a restart while the job list does not.

MCP (`POST /mcp`, stateless JSON-RPC 2.0 — `initialize`, `tools/list`,
`tools/call`, `ping`):

| tool | arguments |
|---|---|
| `transcribe_meeting` | `audio_base64`, `filename`, `language`, `hotwords`, `names` |
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

Source lives in `dashboard/src`; the components in `dashboard/src/components/ui`
are the unmodified shadcn registry files (new-york, neutral theme). The bearer
token goes in the field at the bottom of the upload card and stays in
`localStorage`. Opening a finished job shows the recording in a native player
above the transcript; the line being spoken is highlighted and scrolled into
view, and clicking any line seeks to its timestamp.

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
  `MISTRAL_API_KEY`. `mistral.voxtral-*` is the prefixed id the CLI accepts.
- An `unsloth.*` model (e.g. `unsloth.MOSS-Transcribe-Diarize`) — Unsloth
  Studio's transcribe workflow on `$UNSLOTH_URL` (default
  `http://ubuntu:8888`) with `$UNSLOTH_API_KEY`: the original file travels as
  one raw upload (the studio decodes it — no normalization, no shared
  directory) and the workflow accepts up to 4 hours, where the OpenAI
  `/v1/audio/transcriptions` route stops at 30 minutes. Measured: 43 minutes of
  audio in ~15 minutes of wall time, speakers included. `--job-timeout` (4 h)
  is the service-side cap for one job.
- `--retain-audio-days 30` (on `meeting-job`) — delete kept uploads older than
  this; `0` keeps them forever. Only done and failed jobs are candidates.
- `--model`, `--asr` — point the pipeline at another model or endpoint.

The compose deployment runs the Unsloth Studio path: `--model
unsloth.MOSS-Transcribe-Diarize` with `UNSLOTH_API_KEY`, uploads kept.

## Tests

```sh
python3 -m unittest test_meeting -v
```

They cover the pure functions — the audio.cpp, Unsloth Studio and Mistral
answer parsing, the paragraph merge, the archive ledger and WebDAV paths —
and assert that the standalone defaults stay on the keyless
audio.cpp path while `unsloth.*` is the studio's one-upload NDJSON workflow.
