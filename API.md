# meeting-stack API

Everything is one HTTP service: `meeting-job` on `http://<host>:8099`. It does two
different jobs, and they behave differently:

- **batch** — you send a finished recording, it answers with a diarized
  transcript. Slow, exact, speaker-labelled. This is the record.
- **live** — you stream microphone audio as it happens, it answers with a growing
  transcript and its translation. Fast, approximate, **no speaker labels**.

They share one GPU and one lock: a batch job and a live session never run at the
same time. A live session holds the lock for its whole life, so a job submitted
during one waits (and a second live session gets `409`).

## Auth

`MEETING_TOKEN` empty (the current state): no auth at all, anyone who reaches the
port can use it. When it is set, every route except `GET /health` wants
`Authorization: Bearer <token>`. The two audio routes (`GET /jobs/<id>/audio`,
`/live/...`) also accept `?token=<token>`, because `<audio src>` and an open
browser tab cannot send headers.

## Batch

### Submit

```sh
curl -X POST "http://host:8099/jobs?filename=meeting.m4a&language=de" \
     -H 'content-type: audio/octet-stream' --data-binary @meeting.m4a
```

The body **is** the audio: any format ffmpeg reads. `202` returns the job
summary (`id`, `status`, ...). The id is a hash of the bytes, so re-uploading the
same file returns the existing job instead of transcribing it twice.

Query parameters are per-job options (all optional):

| name | meaning |
|---|---|
| `filename` | original name, only used for the format hint |
| `language` | force the spoken language, e.g. `de` |
| `speakers` | known speaker count; omit to let the model decide |
| `names` | `1:Anna,2:Bernd` |
| `hotwords` | names and jargon to bias the ASR |
| `merge_gap`, `max_segment`, `max_tokens`, `threshold` | numbers |
| `second_name`, `second_offset` | the microphone track of a two-track recording |
| `diarize_second` | label several voices in the second track |

Two tracks of the same recording are paired by `filename`: upload both with the
same name and one job covers both.

### Read

```sh
curl http://host:8099/jobs/<id>          # full job: status, progress, result
curl http://host:8099/jobs               # every job, summaries only
curl http://host:8099/jobs/<id>/audio    # the kept upload, range requests work
curl -X POST http://host:8099/jobs/<id>/retry
```

`status` is `running`, `done` or `error`; `progress` is a human line, not a
percentage. When `done`, the job carries `json` and `markdown`:

```json
{"source": "...", "model": "moss-transcribe-diarize", "language": "de",
 "duration": 2293.3,
 "turns": [{"speaker": 1, "start": 0.26, "end": 6.83, "text": "..."}]}
```

`duration` is the length of the recording, not how long the job took. Turns are
sorted; `speaker` is an integer, and `name` appears when `names` was given. Two
tracks can overlap on purpose (two people talking at once are two lines).

A turn is **a paragraph, not the model's segment**. MOSS closes a segment every
~5 seconds whatever is being said, so consecutive segments of the same speaker
with a small gap and a sentence still unfinished are joined, up to 400
characters. Do not assume one turn per utterance, and do not assume a turn
boundary is a pause.

### MCP

`POST /mcp` speaks JSON-RPC 2.0 with three tools: `transcribe_meeting`
(`audio_base64` + the same options), `meeting_status` (`id`), `meeting_result`
(`id`, `format: markdown|json`).

## Live

A session is created by its first audio chunk; the id is yours to choose.

```sh
# 1. audio: raw PCM, signed 16-bit little-endian, mono, 16 kHz
curl -X POST "http://host:8099/live/<id>/audio?language=de&to=it" \
     -H 'content-type: application/octet-stream' --data-binary @chunk.pcm

# 2. poll: rows and what the worker is doing
curl "http://host:8099/live/<id>?since=0"

# 3. end it (the last line is still translated after this)
curl -X POST http://host:8099/live/<id>/close
```

`?language=` is what the speaker talks (a hint), `?to=` is what to translate
into, `?mt_model=` overrides the translator for this session (omit for the
service's `--translate-model`). Chunks of any size: the page sends 0.5 s.

The poll answers `mt_model`: the translator the session is really using, so a
client that asked for one can see whether it got it.

`GET /live/models` lists what the box can translate with, asked of Lemonade
(`{"models": [{"id", "recipe"}], "default": "..."}`, chat models only), so a
client can offer the choice without hardcoding names.

The `id` there is what Lemonade lists, which for a model registered by hand is
**not** what its backend serves: those answer to `user.<id>`. The service tries
the listed name and, on a `404 model_not_found`, once more with the prefix, so
either spelling works. A streamed request reports that 404 inside a 200 stream,
which is why the check is in the stream reader and not only around the call.

The poll answers:

```json
{"cursor": 2, "open": true, "state": "asr", "buffered": 0.4,
 "language": "de", "to": "it", "done": false,
 "rows": [{"i": 0, "src": "...", "mt": "...", "done": true},
          {"i": 1, "src": "...", "mt": "",    "done": false}]}
```

- `rows[].src` is the transcript, `mt` the translation. **A row with `done:false`
  is still growing**: poll again and it will be longer.
- `cursor` is the number of rows that are finished. Read from `since=cursor`.
- `state` is `idle`, `asr` (transcribing) or `mt` (translating); `buffered` is
  how many seconds of audio are held waiting for a pause. The page uses both to
  show what it is doing.
- `open:false` means the session is closed: no new audio, but the rows are
  readable for a while (`--live-idle`, 120 s) so a late reader gets the tail.
- `409` when the GPU is busy with a batch job, `404` for an unknown session.

### How a line is built

This matters to any client that renders it:

- Audio is transcribed **while it arrives** (audio.cpp streaming endpoint), and
  the words land in the row as the model emits them — roughly every 3 s of
  speech. There is no word-by-word stream beyond that: it is the model's pace.
- A line **ends when the speaker pauses at least 1.5 s *and* the sentence is
  over** (the text ends with `.`, `!`, `?` or `…`). A pause in the middle of a
  sentence does not split it, however long it is.
- A line is also split at a full stop when it grows past 140 characters, so a
  paragraph stays two or three lines on screen.
- The translation follows the line: a partial one every 60 characters while it
  grows, and the final one when the line closes. **The final one replaces the
  partial**, so `mt` can change as a whole; the words that did not change stay
  byte-identical.
- `src` never contains the model's markers: `[0.00]`, `[S01]` and `Speaker 0:`
  are stripped, as are `[Noise]`/`[Laughter]` tags. A delta that only repeats
  what the line already says is dropped (the model loops on long audio).

### What live is not

- **No speaker labels.** Live attribution would need online clustering; the
  batch transcript is the labelled truth. A live session is a preview.
- **Not a record.** The streaming model invents words when the audio is noisy or
  after a loop. Use batch for anything you keep.
- One session at a time per GPU.

## Other routes

| route | what |
|---|---|
| `GET /health` | `{"status":"ok","jobs":N,"queue":N}`, no auth |
| `GET /`, `/ui`, `/live` | the dashboard (one bundle, the path picks the view) |
| `POST /hook` | the recorder announces a finished recording; idempotent |

## Not part of this service

The ASR, the translator and the diarizer are separate processes the service
calls: audio.cpp on `:8090` (MOSS for batch, VibeVoice 7B for live) and Lemonade
on `:8731` (the translator). Nothing in this API exposes them directly.
