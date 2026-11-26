# Working on Eco-Remote

Diarized transcripts of long meetings on the Strix Halo box (`ubuntu`, user
`francesco`). Read `README.md` for the user-facing description; this file is what
an agent needs before touching anything.

## Layout

```
Containerfile             the service + dashboard image (python3 + ffmpeg)
meeting-transcribe        CLI: one audio file -> PREFIX.md + PREFIX.json
meeting-job               service: REST POST /jobs + MCP POST /mcp, one worker
meeting-eval              score a transcript against a reference
dashboard/src/            React + shadcn/ui source of the web UI
dashboard.html            built, inlined single file — what :8099 serves
test_meeting.py
```

There is no runner, no model-server launcher and no image pipeline here: those
live in the archived `halogen-runner` repo and are out of scope.

## The machine's moving parts

No inference is managed here: no model-server launcher, no image pipeline, no
GPU lock. Everything is an HTTP endpoint the service calls.

- ASR, batch (default): Unsloth Studio `http://ubuntu:8888` with
  `audio-cpp/audio.cpp-gguf/MOSS-Transcribe-Diarize-GGUF` (the service id is
  `unsloth.MOSS-Transcribe-Diarize`), key in `$UNSLOTH_API_KEY`. The studio's
  transcribe workflow, the same path its `/audio` page uses: upload the file to
  `/v1/audio/inputs`, then POST `/api/inference/audio/transcribe/source` with
  `speakers: true` and read NDJSON until `complete`. Up to 4 hours
  (`max_seconds` from `/api/inference/audio/stt/capabilities`); the
  OpenAI-compatible `/v1/audio/transcriptions` route stops at 30 minutes. The
  studio decodes the file itself, so no ffmpeg, no normalized WAV and no
  `--asr-dir`.
- ASR, direct audio.cpp: `http://ubuntu:8090` with `moss-transcribe-diarize`
  (one request for the whole meeting; the model answers words, speakers and
  timestamps). It takes a path (not an upload) and 16 kHz mono WAV, hence
  `--asr-dir`/`--asr-remote-dir` and ffmpeg in the image.
  `max_tokens` must be large (65536) AND under `"options"`: at the top level the
  server ignores it and uses 5120, and a tight cap ends in `reached max_tokens
  before EOS`, which loses the transcript instead of truncating it. 38 minutes
  measured 25 minutes of wall time.
- ASR, cloud: Mistral `/v1/audio/transcriptions` with `diarize=true`
  (the `voxtral-*` model id routes there); key in `$MISTRAL_API_KEY`.
- Service: one container from `docker-compose.yml` (docker or podman-compose)
  on port 8099 - no podman socket, no device mounts.

## The dashboard

`dashboard.html` at the repo root is a build artifact: edit `dashboard/src`, then
`cd dashboard && npm run build` (writes `../dashboard.html`). `npm run dev` serves a
hot-reloading copy on `:5173` and proxies `/jobs`, `/health` and `/mcp` to the
service on `:8099`. The `dashboard/src/components/ui`
files are verbatim shadcn registry components; do not hand-edit them, re-fetch
from `https://ui.shadcn.com/r/styles/new-york-v4/<name>.json` instead.

## Rules

- Do not touch the router aliases on `:8080` or any Hermes profile config.
- Do not evict resident models to make room; check what is loaded first
  (`curl -s localhost:8731/api/v1/health`).
- Never report a number you did not measure. The README's figures (1 m 38 s,
  119 turns, 8 speakers, 0-2290 s of 2293 s) come from a real 38:13 German
  meeting; anything else must be measured or labelled as an estimate.
- A change to the answer parsing, the turn merging or the archive layout needs
  a test in `test_meeting.py` and a run on a real recording before it ships.

## Runtime: service in a plain container

`meeting-job` and the dashboard run in `localhost/meeting-job`, host network,
and call out over HTTP only. Consequences that bite:

- The normalized WAV must sit where audio.cpp can open it (`--asr-dir`). A
  temp dir in `/tmp` does not reach the ASR host: the workdir lives under the
  output dir. The compose service does not mount it because the deployed batch
  model is the studio's; a deployment that picks a bare audio.cpp model needs
  `-v ~/audio:/asr-share` and the two `--asr-dir` flags back.
- Input: one file with both sides already mixed. There is no pairing and no
  second input; `submit()` is content-addressed and queues immediately. Do not
  reintroduce a track dictionary: `Job.tracks` is kept as a one-entry
  `{"main": path}` only for the old persisted jobs.
- The batch model is not a per-job option: `POST /jobs` rejects `model`, MCP
  does not expose it, and `OPTION_FLAGS` has no model. It comes from `--model`
  (compose `BATCH_MODEL`); the dashboard has no model picker.
- Whisper-era gotchas of the removed one-pass path (words split by a 32
  character wrap, repetition loops) do not apply any more: the words come from
  one answer of MOSS or Mistral, parsed by `parse_diarized()` /
  `transcribe_mistral()`.
- Uploaded audio is kept (the default) so the dashboard can play it. The
  uploads are pruned after `--retain-audio-days` (default 30; `0` disables),
  hourly and at startup; a queued or running job's audio is never pruned.
  `GET /jobs/<id>/audio` serves them with ranges (206) and accepts the
  bearer token in the query string, because `<audio src>` cannot send headers —
  only that route checks the query. When the local file is gone (purged, or
  `--no-keep-audio`) and WebDAV holds the copy, the route streams it from there
  with the `Range` forwarded, and the job summary's `audio` flag stays true
  (`Archive.audio_name` + `job_state["audio"]`, hoisted into `Jobs._view`).
- Archive: every upload is a row in `<dir>/archive.db` (`Archive`, stdlib
  `sqlite3`), written in `submit()` before the queue sees it. When `WEBDAV_URL`
  is set, `_run`'s `finally` PUTs the audio (`audio-<name>`) and both
  transcripts into `<WEBDAV_URL>/<WEBDAV_DIR>/<job>/` **before**
  `--no-keep-audio` deletes the local copy; a WebDAV failure lands in
  `archive_error` and never fails the job. `GET /files` is the read side. The
  compose credentials come from `.env` (`WEBDAV_*`, gitignored; `.env.example`
  is committed). The DB is `/data/archive.db` on the `meeting-db` volume.
- Transcript dialog (`dashboard/src/App.tsx`): the finished job opens beside its
  audio with a **Text / Segments** switch, the shape Unsloth Studio's audio page
  uses. Text joins consecutive turns by the same speaker into a paragraph;
  Segments keeps one timestamped row each. Both seek the player at their start,
  and the active row/paragraph is highlighted and followed while playing.

## Gotchas that already cost time

- Send `language=de` from the recorder or the dashboard when you know it: with
  `language=auto` short, noisy turns can come back in whatever language the
  model thinks it heard (25 of 226 turns of the 38:13 German meeting came back
  in Chinese or Dutch on the old Qwen3 path; keep the habit for Mistral).
- The token comes from `MEETING_TOKEN` (compose passes it from `.env`). When it
  is empty the service runs without auth: it listens on `0.0.0.0:8099`, so anyone
  on the LAN or the tailnet can start a job.

## Endpoints this stack calls

- Unsloth Studio on `http://ubuntu:8888` - `/v1/audio/inputs` +
  `/api/inference/audio/transcribe/source` for batch diarization. Key in
  `$UNSLOTH_API_KEY` (compose passes it from `.env`).
- audio.cpp on `http://ubuntu:8090` - the direct batch fallback. Its build/run
  lives in that project, not here. `/v1/models/load` reloads
  `moss-transcribe-diarize` when the server unloaded it (the service does this
  itself on a 404).
- Mistral - key in `$MISTRAL_API_KEY` (compose passes it from `.env`).

## After changing the service or the image

```sh
docker compose up -d --build        # or: podman-compose up -d --build
curl -s localhost:8099/health
```

The compose `command` passes only what differs from the defaults (the directory
and the model choice): the running configuration is the `command` in
`docker-compose.yml`, the rest is the default contract tested in
`test_meeting.py`.
