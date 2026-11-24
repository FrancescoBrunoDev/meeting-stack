# Working on meeting-stack

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
dashboard/src/Live.tsx    the live view: capture, transcript, translation (shadcn
                          components, same bundle — GET /live serves dashboard.html
                          and App picks the view from the path). The rows are a fixed
                          window on the tail (`h-[calc(100svh-3.25rem)]` column,
                          `flex-1` card, `mt-auto` rows) with a per-row opacity
                          gradient and `snap-y`; a container mask was tried first and
                          faded nothing until the rows overflowed it.
live-feed                 the same live API as a CLI: ffmpeg input (file or device) -> lines
systemd/meeting-job.service
test_meeting.py
```

There is no runner, no model-server launcher and no image pipeline here: those
live in the archived `halogen-runner` repo and are out of scope.

## The machine's moving parts

No inference is managed here: no model-server launcher, no image pipeline, no
GPU lock. Everything is an HTTP endpoint the service calls.

- ASR, batch and live: audio.cpp `http://ubuntu:8090` with
  `moss-transcribe-diarize` (one request for the whole meeting; the model answers
  words, speakers and timestamps). It takes a path (not an upload) and 16 kHz
  mono WAV, hence `--asr-dir`/`--asr-remote-dir` and ffmpeg in the image.
  `max_tokens` must be large (65536) AND under `"options"`: at the top level the
  server ignores it and uses 5120, and a tight cap ends in `reached max_tokens
  before EOS`, which loses the transcript instead of truncating it. 38 minutes
  measured 25 minutes of wall time. Live: the streaming endpoint
  (`--live-stream`, `vibevoice_asr_streaming_7b_q8_0`) or one clip per
  utterance without it.
- ASR, cloud: Mistral `/v1/audio/transcriptions` with `diarize=true`
  (the `voxtral-*` model id routes there); key in
  `$MISTRAL_API_KEY`. The realtime models are the live ones.
- Translators: Lemonade `http://127.0.0.1:8731` (`--translate-model`), Mistral's
  chat models (`mistral.<id>`) and the Bifrost gateway `http://127.0.0.1:8080`
  (`--mt-gateway`, its `opencode.<id>` models).
- Service: user unit `meeting-job.service` on port 8099 - a plain
  `podman run`, no podman socket, no device mounts.

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
- A change to the answer parsing, the turn merging or the two-track matching
  needs a test in `test_meeting.py` and a run on a real recording before it
  ships.

## Runtime: service in a plain container

`meeting-job` and the dashboard run in `localhost/meeting-job`, host network,
and call out over HTTP only. Consequences that bite:

- The normalized WAV must sit where audio.cpp can open it (`--asr-dir`; the unit
  mounts `~/audio:/asr-share`). A temp dir in `/tmp` does not reach the ASR
  host: the per-track workdir lives under the output dir.
- Two-tracks: `meeting-job` derives `pair`/`track` from the recorder's file names
  (`zoom-<stamp>-{system,mic}-16k.mp3`), parks the job in `waiting` until both
  tracks arrive (`--pair-wait`, default 900 s), then runs the worker once with
  `--second`. Both tracks are diarized by the ASR itself. When both are the
  same room recorded twice (a mic that also hears the loudspeakers),
  `merge_tracks()` drops the second track's turns the first already covers, and
  `match_speakers()` maps each mic voice onto the first track's speaker by time
  overlap. `dedupe_turns()` then drops a repeated cut of the same utterance:
  that is why the final speaker count can be lower than the ASR's. A silent mic
  track is tolerated (`lenient`), not a failed job; do not "fix" that as a
  missing diarization step.
- Whisper-era gotchas of the removed one-pass path (words split by a 32
  character wrap, repetition loops) do not apply any more: the words come from
  one answer of MOSS or Mistral, parsed by `parse_diarized()` /
  `transcribe_mistral()`.
- `--keep-audio` is on in the unit so the dashboard can play a recording. The
  uploads are pruned after `--retain-audio-days` (default 30; `0` disables),
  hourly and at startup; a queued, waiting or running job's audio is never
  pruned. `GET /jobs/<id>/audio` serves them with ranges (206) and accepts the
  bearer token in the query string, because `<audio src>` cannot send headers —
  only that route checks the query.
- Live (`/live`, `POST /live/<id>/audio`, `dashboard/src/Live.tsx`): the browser resamples to
  16 kHz mono PCM16 (`new AudioContext({sampleRate: 16000})` — Safari gives 48 kHz
  and the page refuses, it would pitch everything) and posts half a second at a
  time. The server accumulates, cuts on the first pause of 0.45 s
  (`vad_cut`/`frame_rms`, ~20 lines of stdlib), sends the utterance as one WAV to
  `--asr`, and translates it with `--translate-model` on `--translate-url`, last
  two source lines as context and `chat_template_kwargs.enable_thinking=false`
  (without that the thinking model spends `max_tokens` on reasoning and returns
  an empty `content`). The row is appended **before** translating and the answer
  is read as SSE (`stream: true`), each partial written into the row: the page
  follows the cursor and rewrites the cells of the row it names, which is why
  `view()` sends `i` with every row — plus `state` (`idle`/`asr`/`mt`) and
  `buffered` (seconds of audio held back waiting for a pause), because ten silent
  seconds of speech read as a broken program — and why **the cursor stops at the
  first row still being translated**: a cursor that moved past an in-flight row left it on
  "…" for ever, because the worker translates a row that is not the last one.
  The marker is the row's own `done`, **not** `mt`: with streaming `mt` is
  non-empty from the first token, and keying on it abandoned rows half way
  ("aveva affid", "Anche lei ne").
  Context is the previous `LIVE_CONTEXT` (2) lines as `source -> translation` pairs, and `--live-glossary` pins names. The
  row is a **line, not a clip**: clips are cut tightly (`LIVE_CUT`, 0.3 s) so the
  ASR sees short requests and two speakers rarely share one; `_hear` appends each
  clip's text to `_pending` and the line closes on a **longer pause**
  (`--live-silence`, 0.8 s — `feed` queues a `CLOSE` marker once the silence
  outlasts it, so the ASR is not involved). The ASR punctuation is ignored on
  purpose — every clip ends with a full stop, and trusting it split "…mehr sexualisierte Gewalt." /
  "und Fehlverhalten als bislang bekannt." into two rows, each translated as its
  own sentence. A line with no pause is translated after `LIVE_PENDING_CHARS`
  (220, and then cut at a full stop) or `LIVE_PENDING_HOLD` (10 s), and on close.
  No speaker labels live: the ASR never names anybody inline.
  Measured on the box (2.8 s Italian clip, looped): 0.59 s per line with
  Qwen3.8-27B-GGUF (17 tokens at 35.7 tok/s); on the page the source line shows
  at 3.4 s, the translation starts 0.3 s later and completes 0.6 s later.
  `--live-idle` (120 s) closes an abandoned session, which stays readable so
  the last rows can be collected.
  A speaker who does not pause for 20 s is cut at the quietest frame. Live
  capture needs a secure context: `http://127.0.0.1:8099/live` or https, never
  plain http to another host.

## Gotchas that already cost time

- Send `language=de` from the recorder or the dashboard when you know it: with
  `language=auto` short, noisy turns can come back in whatever language the
  model thinks it heard (25 of 226 turns of the 38:13 German meeting came back
  in Chinese or Dutch on the old Qwen3 path; keep the habit for Mistral).
- Mistral diarizes each track on its own, so a mic track that hears the same
  room again comes back with its own `speaker_1`, ... - that is what
  `match_speakers()` is for; do not merge the two tracks' ids by a blanket shift.
- The service runs without a token on this box (empty `MEETING_TOKEN`), by
  decision: it listens on `0.0.0.0:8099`, so anyone on the LAN or the tailnet can
  start a job.

## Endpoints this stack calls

- audio.cpp on `http://ubuntu:8090` - its build/run lives in that project, not
  here. `/v1/models/load` reloads `moss-transcribe-diarize` when the server
  unloaded it (the service does this itself on a 404).
- Mistral - key in `$MISTRAL_API_KEY` (`~/.config/meeting-job.env` via the unit).
- Lemonade on `:8731` and Bifrost on `:8080` - translators; do not evict their
  resident models.

## After changing the service or the unit

```sh
systemctl --user daemon-reload
systemctl --user restart meeting-job.service
curl -s localhost:8099/health
```

The unit passes only what differs from the defaults (the directory, the
asr-dir share, the model choices, the gateway): the running configuration is
visible in `systemctl --user cat meeting-job.service`, the rest is the
default contract tested in `test_meeting.py`.
