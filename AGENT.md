# Working on meeting-stack

Diarized transcripts of long meetings on the Strix Halo box (`ubuntu`, user
`francesco`). Read `README.md` for the user-facing description; this file is what
an agent needs before touching anything.

## Layout

```
Containerfile             the service + dashboard image (python3 + podman-remote)
meeting-transcribe        CLI: one audio file -> PREFIX.md + PREFIX.json
meeting-job               service: REST POST /jobs + MCP POST /mcp, one worker
meeting-eval              score a transcript against a reference
lockfile.py               flock wrapper; the PATH is the contract, not the code
diarizer/                 sherpa (ONNX, CPU)
diarizer-pyannote-rocm/   pyannote.audio on the iGPU
diarizer-nemotron/        NVIDIA Nemotron-3-Diarization via transformers (default)
dashboard/src/            React + shadcn/ui source of the web UI
dashboard.html            built, inlined single file — what :8099 serves
live.html                 the live page at GET /live: capture, transcript, translation
systemd/meeting-job.service
test_meeting.py
```

There is no runner, no model-server launcher and no image pipeline here: those
live in the archived `halogen-runner` repo and are out of scope.

## The machine's moving parts

- ASR: gufo `http://127.0.0.1:8097`, `qwen3-asr-1.7b`, `response_format=json`,
  `--asr-mode segments`. Resident model (~4.8 GB RSS, ~15.9 GB of the
  iGPU), do not unload it.
- ASR, one-pass alternative: Lemonade `http://127.0.0.1:8731`, OpenAI-compatible,
  `Whisper-Large-v3-Turbo`, `response_format=verbose_json`. Resident model; do not
  unload it.
- Diarizers: one `podman run --rm` container per job, GPU via `/dev/kfd` + `/dev/dri`.
- Service: user unit `meeting-job.service` on port 8099 (`podman-restart` is not
  involved; the unit is plain systemd).
- GPU lock: `/run/user/1000/halogen-gpu.lock`, shared with image generation.
  Never rename it in one place only — two different paths means two different
  locks and the protection silently stops working.

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
- A change to `--asr-mode full`, the fusion, or the word rebuilding needs a test
  in `test_meeting.py` and a run on a real recording before it ships.

## Runtime: service in a container, diarizers on the host

`meeting-job` and the dashboard run in `localhost/meeting-job`; each job's
`podman run` goes through the host's rootless podman socket
(`CONTAINER_HOST=unix:///run/user/1000/podman/podman.sock`). Consequences that
bite:

- Every path handed to `podman run` (`-v source:/in`, `-v workdir:/out`, the HF
  cache) is a **host** path. The service container mounts the cache dirs at the
  same absolute path, and `HOME` is the host's, so `Path.home()` matches. A temp
  dir in `/tmp` breaks: the host cannot see the container's `/tmp` — the
  diarizer workdir therefore lives under the output dir.
- A remote podman rejects `--group-add keep-groups`; `podman_is_remote()` in
  `meeting-transcribe` skips the flag in that case. It is only needed on the
  host path, and `/dev/kfd` + `/dev/dri` are 0666 on this box.
- Whisper loops: on undecodable audio it emits the same sentence every second for
  minutes (134 s in one real meeting) with *high* logprobs, because Lemonade's
  endpoint ignores `condition_on_previous_text`, `vad_filter` and
  `no_speech_threshold`. `collapse_repetitions()` in `meeting-transcribe` is the
  only guard; it runs on the `--asr-mode full` path only, and it drops, never
  keeps, a suspected run.
- Two-tracks: `meeting-job` derives `pair`/`track` from the recorder's file names
  (`zoom-<stamp>-{system,mic}-16k.mp3`), parks the job in `waiting` until both
  tracks arrive (`--pair-wait`, default 900 s), then runs the worker once with
  `--second`. The microphone track is one speaker by construction, but it is
  not sent in one pass unless `--asr-mode full` (`mic_is_diarized()`): gufo has
  no verbose_json, so a whole-track request there returns `segments: []` and
  the job dies with "ASR returned no segments". With `segments` the diarizer
  provides the turn boundaries; its labels are kept (and matched onto the first
  track's speakers) when it found more than one voice, otherwise
  `second_track_turns(..., diarized=False)` collapses every turn to `Me`. Do not
  "fix" that as a missing diarization step; a silent mic track is tolerated
  (`lenient`), not a failed job. When both tracks are the same room recorded
  twice (a mic that also hears the loudspeakers), `merge_tracks()` drops the
  second track's turns the first already covers instead of duplicating them.
  `dedupe_turns()` then drops a repeated cut of the same utterance (two
  overlapping speaker ids, same words): that is why the final speaker count can
  be lower than the diarizer's. `--diarization FILE` replaces the guessed
  speakers with an external reference (e.g. pyannote precision-3) by time
  overlap; the service accepts the same path as the `diarization` job option.
- `podman.socket` must be enabled (`systemctl --user enable --now podman.socket`)
  or every job fails with a connection error.
- Job state is in `jobs.json` under `--dir` (atomic replace, restored at
  startup): the list, options, status and the paths of the uploads survive a
  restart, and a finished job's transcript is read back from `out/<id>.json` and
  `.md`. A job that was `running` when the service died becomes an `error` with
  its audio kept, so "retry" works; a job that was `queued` is queued again; a
  `waiting` job is still picked up by `--pair-wait`. Announcements (`hooks.json`)
  use the same atomic write.
- `POST /hook` carries no audio; it is `POST /jobs` that does. The hook exists so
  an announced-but-never-uploaded recording shows up as `missing_audio` in
  `GET /hooks` instead of being guessed after `--pair-wait`. Both are idempotent
  by recording key (`zoom-<stamp>`), which is how a retry or a recorder restart
  sweep is absorbed.
- `--keep-audio` is on in the unit so the dashboard can play a recording. The
  uploads are pruned after `--retain-audio-days` (default 30; `0` disables),
  hourly and at startup; a queued, waiting or running job's audio is never
  pruned. `GET /jobs/<id>/audio` serves them with ranges (206) and accepts the
  bearer token in the query string, because `<audio src>` cannot send headers —
  only that route checks the query.
- Live (`/live`, `POST /live/<id>/audio`, `live.html`): the browser resamples to
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
  row is a **line, not a clip**: `_hear` appends each clip's text to `_pending`
  and the line closes only when a **pause** ended the clip (`vad_cut` returns
  `by_pause`; a forced cut at `LIVE_MAX` does not close it). The ASR punctuation
  is ignored on purpose — gufo and Whisper put a full stop at the end of every
  clip, and trusting it split "…mehr sexualisierte Gewalt." / "und Fehlverhalten
  als bislang bekannt." into two rows, each translated as its own sentence. The
  pause length is `--live-silence` (0.8 s; 0.45 split a broadcast into fragments,
  0.9 stopped cutting at all on speech that breathes for 0.8 s). A line with no
  pause is translated after `LIVE_PENDING_CHARS` (500) or `LIVE_PENDING_HOLD`
  (10 s), and on close. Past `LIVE_PENDING_CHARS` (220) the line is additionally
  cut at the first full stop after half that — the only place the ASR's
  punctuation is trusted — so four sentences of broadcast German do not become one
  row whose translation takes four seconds. Two traps already paid for: a bare ISO code in the prompt
  is read as an English word ("…code is it"), so `LANG_NAMES` passes the name;
  and an empty answer (possible on a fragment) is retried once without the
  context, then written into the row as an error, never left as a silent "…".
  No speaker labels live.
  `Lock(--lock)` for its whole life — **its own instance**, because `flock` on one
  open file description does not exclude itself and reusing `Jobs.lock` would let
  a session and a diarizer overlap. So jobs wait for it and a second session gets
  a 409; `--live-idle` (120 s) closes an abandoned one. `LiveSession._work` calls
  `Live.settled()` when it drains and that releases the lock at once — doing it in
  the reaper made a restart right after "Ferma" wait up to 5 s for a 409 — while
  the session stays readable for `--live-idle` so the last rows can be collected.
  Measured on the box (2.8 s Italian clip, looped): 0.24 s per 10 s clip through
  Whisper-Large-v3-Turbo, 0.78 s through gufo; 0.59 s per line with
  Qwen3.8-27B-GGUF (17 tokens at 35.7 tok/s); on the page the source line shows at
  3.4 s, the translation starts 0.3 s later and completes 0.6 s later.
  A speaker who does not pause for 20 s is cut at the quietest frame. Live
  capture needs a secure context: `http://127.0.0.1:8099/live` or https, never
  plain http to another host.

## Gotchas that already cost time

- gufo's `verbose_json` returns an **empty** `segments` array, so `--asr-mode
  full` dies with "ASR returned no segments". `segments` mode is the only path
  against gufo. Its WAV loader also rejects recordings over 30 minutes, which
  the per-turn clips never approach.
- Send `language=de` from the recorder or the dashboard. With `language=auto`
  Qwen3-ASR answers short, noisy turns in whatever language it thinks it heard:
  25 of 226 turns of the 38:13 German meeting came back in Chinese or Dutch.
  With `--language de` there were 0. A 1 s clip can still come back as a
  translation of what it guessed (`Let me try.` -> `Lass mich mal.`).
- Qwen3-ASR sometimes glues two words (`wahrscheinlichSie`). It is a tokenizer
  artifact of the `json` path, where there is no `words[]` to rebuild from; the
  `rejoin()`/`unwrap()` machinery only applies to `verbose_json`.

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

## The gufo ASR server

One container, started once, restarted by `podman-restart.service` (enabled):

```sh
cd ~/Github/gufo
podman run -d --name gufo-asr --restart=unless-stopped --network host \
  --device /dev/kfd --device /dev/dri \
  -v ~/Github/gufo/install:/opt/gufo:ro -v ~/Github/gufo/models:/models:ro \
  localhost/gufo-ci:hip /opt/gufo/bin/gufo serve asr \
  --model /models/Qwen3-ASR-1.7B --port 8097
curl -s localhost:8097/v1/audio/transcriptions -F file=@clip.wav \
  -F model=qwen3-asr-1.7b -F language=de
```

Rebuild the image, the binary and the weights (all three live in the gufo
checkout, `install/` and `models/` are gitignored):

```sh
podman build -t localhost/gufo-ci:hip -f tools/ci/Containerfile.ci-hip .
podman run --rm -v "$PWD":/src -w /src localhost/gufo-ci:hip bash -lc \
  'cmake --preset release && cmake --build --preset release --parallel 16 \
   && cmake --install build/release --prefix /src/install'
hf download Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --local-dir models/Qwen3-ASR-1.7B
podman restart gufo-asr
```

The build runs in the container (`/src`), never on the host: there is no ROCm
HIP toolchain installed here.

## After changing the service or the unit

```sh
systemctl --user daemon-reload
systemctl --user restart meeting-job.service
curl -s localhost:8099/health
```

The unit passes the ASR endpoint, model, diarizer and mode explicitly, so the
running configuration is visible in `systemctl --user cat meeting-job.service`.
