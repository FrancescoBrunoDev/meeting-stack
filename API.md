# Eco-Remote API

Everything is one HTTP service: `meeting-job` on `http://<host>:8099`. You send a
finished recording and it answers with a diarized transcript.

The service itself runs no inference: Unsloth Studio on `:8888` answers the
default batch transcription, audio.cpp on `:8090` is the direct fallback, Mistral
the cloud one. Jobs run one at a time in a worker thread.

## Auth

`MEETING_TOKEN` empty: no auth at all, anyone who reaches the port can use it.
When it is set, every route except `GET /health` wants
`Authorization: Bearer <token>`. The playback route (`GET /jobs/<id>/audio`)
also accepts `?token=<token>`, because `<audio src>` cannot send headers.

## Batch

### Submit

```sh
curl -X POST "http://host:8099/jobs?filename=meeting.m4a&language=de" \
     -H 'content-type: application/octet-stream' --data-binary @meeting.m4a
```

The body **is** the audio: any format ffmpeg reads. `202` returns the job
summary (`id`, `status`, ...). The id is a hash of the bytes, so re-uploading the
same file returns the existing job instead of transcribing it twice; the options
of that second request replace the stored ones (so a corrected `language` takes
effect at the next `retry`), while the filename stays the one the archive holds.

`Content-Type` is not looked at. A `multipart/form-data` body is a `400`: no audio
format starts with `--`, and storing the envelope would leave the client with a
job over garbage after it deleted its own copy. An upload over `--max-upload`
(512 MiB) is answered `413` after the body has been read out, so the client
always sees it. `filename` is worth sending even though it is optional: it is the
only format hint (`<hash>.bin` without it), the `audio-<name>` archive name and
the `/hook` join key.

Query parameters are per-job options (all optional):

| name | meaning |
|---|---|
| `filename` | original name: the format hint, the `audio-<name>` archive name, and the join key for `/hook` |
| `language` | force the spoken language, e.g. `de` |
| `names` | `1:Anna,2:Bernd` |
| `hotwords` | names and jargon to bias the ASR |
| `max_tokens` | the audiocpp output cap |

The model is not a request option: the service uses its `--model` (compose
`BATCH_MODEL`). A `model` query parameter is a `400`.

The body is one already-mixed recording; there is no second input.

### Read

```sh
curl http://host:8099/jobs/<id>          # full job: status, progress, result
curl http://host:8099/jobs               # every job, summaries only
curl http://host:8099/jobs/<id>/audio    # the kept upload, range requests work
curl -X POST http://host:8099/jobs/<id>/retry
```

`status` is `queued`, `running`, `done`, `error` or `cancelled`; `progress` is
a human line, not a percentage. When `done`, the job carries `json` and `markdown`:

```json
{"source": "...", "model": "moss-transcribe-diarize", "language": "de",
 "duration": 2293.3,
 "turns": [{"speaker": 1, "start": 0.26, "end": 6.83, "text": "..."}]}
```

`duration` is the length of the recording, not how long the job took. Turns are
sorted; `speaker` is an integer, and `name` appears when `names` was given. Turns
can overlap on purpose: two people talking at once are two lines.

A turn is **a paragraph, not the model's segment**. MOSS closes a segment every
~5 seconds whatever is being said, so consecutive segments of the same speaker
with a small gap and a sentence still unfinished are joined, up to 400
characters. Do not assume one turn per utterance, and do not assume a turn
boundary is a pause.

### Stop, restart, delete

```sh
curl -X POST http://host:8099/jobs/<id>/stop      # stop it
curl -X POST http://host:8099/jobs/<id>/retry     # run it again
curl -X DELETE http://host:8099/jobs/<id>         # forget it
```

`stop` takes a job out of the queue, or kills the worker running it, and leaves
it `cancelled` — not `error`, because nothing went wrong. The upload is kept, so
`retry` can start it again; that also works for a `done`, `error` or `cancelled`
job. A job that is queued or running is returned unchanged by `retry`, so a
double click cannot queue it twice. `DELETE` removes the job, its transcript **and its
uploaded copy** — the file you sent is yours and is not touched. All three answer
`404` for an unknown id.

### MCP

`POST /mcp` speaks JSON-RPC 2.0 with three tools: `transcribe_meeting`
(`audio_base64` + `filename`/`language`/`hotwords`/`names`; no model),
`meeting_status` (`id`), `meeting_result` (`id`, `format: markdown|json`).

## Other routes

| route | what |
|---|---|
| `GET /health` | `{"status":"ok","jobs":N,"queue":N,"archive":bool}`, no auth |
| `GET /files` | the archive ledger, newest first: audio and transcript rows |
| `GET /archive` | cached WebDAV probe: `configured`, `reachable`, `url`, `error` |
| `GET /hooks` | announcements, joined with the job whose upload has their file name |
| `GET /`, `/ui` | the dashboard |
| `POST /hook` | the recorder announces a finished recording; idempotent |

`POST /hook` takes `{"day":"YYYY-MM-DD","recording":"HH-MM-SS","files":["..."],
"remote_dir":"...","language":"..."}`. It carries no audio and never starts a
job; it only records that the recorder finished a recording. `GET /hooks` joins
the announced file names to the uploaded job's `filename` and answers
`missing_audio` when no upload has arrived. The join is the base name, compared
exactly: a recorder that re-encodes before uploading announces the name it
actually sent (`zoom-20261020-123509-mix-16k.mp3`), not the master's. The
response keeps the legacy `pair` key: it is the recording key
(`zoom-YYYYMMDD-HHMMSS`), not a second track.

`GET /files` rows carry `job_id`, `track`, `filename`, `size`, `sha256`,
`received_at`, `status` (`received`, `running`, `queued`, `done`, `error`,
`cancelled`), `error`, `archived_at`, `webdav_url` and `archive_error`. The
audio row has `track=audio`; a finished job also has `transcript.md` and
`transcript.json`. The ledger is SQLite at `ARCHIVE_DB` (default
`<dir>/archive.db`); when `WEBDAV_URL` is set the job folder
`<WEBDAV_DIR>/<job_id>/` is created and the audio (`audio-<name>`) and both
transcripts are PUT there as the job ends. An archive failure is recorded on the
row without failing the transcription.

## Not part of this service

The ASR is a separate process the service calls: Unsloth Studio on `:8888`
(MOSS-Transcribe-Diarize for batch), audio.cpp on `:8090` (the same MOSS on the
direct batch fallback) or Mistral (cloud transcription). Nothing in this API
exposes them directly.
