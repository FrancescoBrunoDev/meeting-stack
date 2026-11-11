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
