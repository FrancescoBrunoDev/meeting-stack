#!/usr/bin/env python3
"""Tests for the parts of meeting-transcribe and meeting-job that decide the
transcript.

The ASRs (audio.cpp, Mistral) are external, so what is tested here is
everything that can be wrong without them: parsing their answers, merging
paragraphs and turns, and the defaults that decide which path runs.
"""
import base64
import contextlib
import hashlib
import io
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wav_bytes(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)
    return buffer.getvalue()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


meeting = load_module("meeting_transcribe", HERE / "meeting-transcribe")

meeting_job = load_module("meeting_job", HERE / "meeting-job")


class WordRebuilding(unittest.TestCase):
    """The server's fixed-width line wrap; unwrap() is the readable fallback."""

    def test_unwrap_only_collapses_whitespace(self):
        # A break inside a word cannot be undone safely here: joining
        # "Satz\nzweiter" would produce "Satzzweiter".
        self.assertEqual(meeting.unwrap("die In\nsel verlassen"), "die In sel verlassen")

class AnnouncedRecordings(unittest.TestCase):
    """POST /hook: a recording is announced once its tracks are on WebDAV."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.hooks = meeting_job.Hooks(Path(self.dir.name) / "hooks.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_folder_names_become_the_pair_key_the_uploads_use(self):
        self.assertEqual(meeting_job.recording_pair("2026-10-20", "12-35-09"),
                         "zoom-20261020-123509")
        self.assertIsNone(meeting_job.recording_pair("2026-10-20", ""))
        self.assertIsNone(meeting_job.recording_pair("20-10-2026", "12-35-09"))

    def test_announcement_survives_a_restart(self):
        entry, error = self.hooks.announce({"day": "2026-10-20", "recording": "12-35-09",
                                            "remote_dir": "https://dav/x/2026-10-20/12-35-09/",
                                            "files": ["audio/zoom-20261020-123509-system-16k.mp3"]})
        self.assertIsNone(error)
        self.assertEqual(entry["pair"], "zoom-20261020-123509")
        self.assertEqual(len(entry["files"]), 1)
        again = meeting_job.Hooks(Path(self.dir.name) / "hooks.json")
        self.assertEqual(again.get("zoom-20261020-123509")["first_seen"],
                         entry["first_seen"])

    def test_the_announced_language_is_kept(self):
        entry, _ = self.hooks.announce({"day": "2026-10-20", "recording": "12-35-09",
                                        "language": "it"})
        self.assertEqual(entry["language"], "it")
        self.assertEqual(self.hooks.get("zoom-20261020-123509")["language"], "it")

    def test_a_retry_is_the_same_announcement(self):
        body = {"day": "2026-10-20", "recording": "12-35-09", "files": ["audio/a.mp3"]}
        first, _ = self.hooks.announce(body)
        second, _ = self.hooks.announce(body)
        self.assertEqual(first["pair"], second["pair"])
        self.assertEqual(second["count"], 2)
        self.assertEqual(len(self.hooks.list()), 1)

    def test_a_bad_body_is_refused(self):
        self.assertEqual(self.hooks.announce({"day": "nope"})[1],
                         "expected day=YYYY-MM-DD and recording=HH-MM-SS")
        self.assertEqual(len(self.hooks.list()), 0)

    def test_an_announcement_joins_the_upload_by_file_name(self):
        entry, _ = self.hooks.announce({
            "day": "2026-10-20", "recording": "12-35-09",
            "files": ["audio/zoom-20261020-123509-mix.mp3"]})
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as jobs_dir:
            jobs = meeting_job.Jobs(HERE / "meeting-transcribe", Path(jobs_dir))
            try:
                job = jobs.submit(b"mixed", {}, "zoom-20261020-123509-mix.mp3")
                view = meeting_job.hook_view(self.hooks, jobs, entry["pair"])[0]
                self.assertEqual(view["job"], job["id"])
                self.assertNotEqual(view["status"], "missing_audio")
            finally:
                jobs.stop()

    def test_announced_but_never_uploaded_is_visible(self):
        self.hooks.announce({"day": "2026-10-20", "recording": "12-35-09"})
        with tempfile.TemporaryDirectory() as jobs_dir:
            jobs = meeting_job.Jobs(HERE / "meeting-transcribe", Path(jobs_dir))
            try:
                views = meeting_job.hook_view(self.hooks, jobs)
                self.assertEqual([view["status"] for view in views], ["missing_audio"])
            finally:
                jobs.stop()


class Defaults(unittest.TestCase):
    """The defaults are the pipeline's contract: audio.cpp, diarizing itself."""

    def test_asr_endpoint_is_audio_cpp(self):
        self.assertEqual(meeting.DEFAULT_ASR, "http://ubuntu:8090")

    def test_model_is_moss(self):
        self.assertEqual(meeting.DEFAULT_ASR_MODEL, "moss-transcribe-diarize")

    def test_audio_cpp_answers_are_split_on_their_own_markers(self):
        turns = meeting.parse_diarized(
            "[1.74][S01] Hallo zusammen.[2.61][3.06][S02] Hallo.[3.51][11.10][S02] Morgen.[12.01]")
        self.assertEqual([(turn["speaker"], turn["start"], turn["end"], turn["text"])
                          for turn in turns],
                         [(1, 1.74, 2.61, "Hallo zusammen."), (2, 3.06, 3.51, "Hallo."),
                          (2, 11.1, 12.01, "Morgen.")])

    def test_an_empty_audio_cpp_answer_is_no_turns(self):
        self.assertEqual(meeting.parse_diarized(""), [])

    def test_the_batch_model_is_not_a_per_job_option(self):
        self.assertNotIn("model", {key for key, _ in meeting_job.OPTION_FLAGS})

    def test_the_model_names_its_provider(self):
        self.assertTrue(meeting.is_mistral_asr("voxtral-mini-transcribe-3"))
        self.assertTrue(meeting.is_mistral_asr("mistral.voxtral-small-transcribe-3"))
        self.assertFalse(meeting.is_mistral_asr("moss-transcribe-diarize"))


class RecordingLength(unittest.TestCase):
    def test_the_length_comes_from_the_file_not_the_last_word(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(b"\0\0" * 16000 * 3)
            self.assertEqual(meeting.wav_seconds(path), 3.0)

    def test_something_that_is_not_audio_has_no_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.wav"
            path.write_bytes(b"not a wav")
            self.assertIsNone(meeting.wav_seconds(path))


class MergeParagraphs(unittest.TestCase):
    """The model draws a line every five seconds; most of them are not breaks."""

    def test_a_sentence_cut_in_the_middle_is_one_paragraph(self):
        turns = [{"speaker": 1, "start": 7.8, "end": 13.2, "text": "perché è proprio"},
                 {"speaker": 1, "start": 13.2, "end": 18.5, "text": "istintivo."}]
        merged = meeting.merge_paragraphs(turns)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "perché è proprio istintivo.")
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (7.8, 18.5))

    def test_a_finished_sentence_stays_its_own_paragraph(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 3.0, "text": "Non capisco."},
                 {"speaker": 1, "start": 3.0, "end": 6.0, "text": "Sta funzionando?"}]
        self.assertEqual(len(meeting.merge_paragraphs(turns)), 2)

    def test_a_change_of_speaker_is_never_joined(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 3.0, "text": "perché è proprio"},
                 {"speaker": 2, "start": 3.0, "end": 6.0, "text": "istintivo."}]
        merged = meeting.merge_paragraphs(turns)
        self.assertEqual([turn["speaker"] for turn in merged], [1, 2])

    def test_a_silence_is_a_break(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 3.0, "text": "perché è proprio"},
                 {"speaker": 1, "start": 5.0, "end": 8.0, "text": "istintivo."}]
        self.assertEqual(len(meeting.merge_paragraphs(turns)), 2)

    def test_a_monologue_is_cut_at_the_length_cap(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 3.0, "text": "a" * 300},
                 {"speaker": 1, "start": 3.0, "end": 6.0, "text": "b" * 200}]
        merged = meeting.merge_paragraphs(turns)
        self.assertEqual(len(merged), 2)
        self.assertTrue(all(len(turn["text"]) <= meeting.PARAGRAPH_CHARS for turn in merged))

    def test_overlapping_turns_are_not_joined(self):
        # Two people can be talking at the same moment.
        turns = [{"speaker": 1, "start": 0.0, "end": 5.0, "text": "perché è proprio"},
                 {"speaker": 1, "start": 2.0, "end": 6.0, "text": "istintivo."}]
        self.assertEqual(len(meeting.merge_paragraphs(turns)), 2)


class SingleFile(unittest.TestCase):
    """The recorder sends one mixed file; there is no second input to wait for."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.jobs = meeting_job.Jobs(Path("/bin/true"), Path(self.dir.name))
        self.addCleanup(self.jobs.stop)

    def test_an_upload_is_queued_immediately(self):
        job = self.jobs.submit(b"mixed", {}, "meeting.mp3")
        self.assertNotEqual(job["status"], "waiting")

    def test_the_same_bytes_are_one_job(self):
        first = self.jobs.submit(b"mixed", {}, "meeting.mp3")
        again = self.jobs.submit(b"mixed", {}, "meeting.mp3")
        self.assertEqual(first["id"], again["id"])

    def test_two_recordings_are_two_jobs(self):
        first = self.jobs.submit(b"one", {}, "meeting.mp3")
        second = self.jobs.submit(b"two", {}, "meeting.mp3")
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(list((Path(self.dir.name) / "inbox").iterdir())), 2)


class AudioCppUploads(unittest.TestCase):
    """The upload path, used when audio.cpp cannot read a shared directory."""

    def test_audio_cpp_upload_when_no_shared_dir(self):
        """No --asr-dir (the service and audio.cpp on different machines):
        the WAV travels as one multipart request, options included."""
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"text": "[0.10][S01] Hallo.[1.20]"}).encode()

        def fake_open(req, timeout=0):
            captured["url"] = req.full_url
            captured["body"] = req.data.decode("utf-8", "replace")
            return FakeResponse()

        orig = meeting.urllib.request.urlopen
        meeting.urllib.request.urlopen = fake_open
        orig_norm, orig_wav = meeting.normalize_wav, meeting.wav_seconds
        meeting.normalize_wav = lambda source, out: out.write_bytes(b"RIFF" + b"\0" * 100)
        meeting.wav_seconds = lambda path: 5.0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "in.wav"
                src.write_bytes(b"x")
                args = types.SimpleNamespace(asr="http://192.168.1.5:8090",
                                             model="moss-transcribe-diarize",
                                             asr_dir=None, asr_remote_dir=None,
                                             asr_timeout=10, max_tokens=512)
                (Path(tmp) / "work").mkdir()
                turns = meeting.audiocpp_turns(args, src, Path(tmp) / "work", lenient=False)
        finally:
            meeting.urllib.request.urlopen = orig
            meeting.normalize_wav, meeting.wav_seconds = orig_norm, orig_wav
        self.assertEqual(captured["url"], "http://192.168.1.5:8090/v1/audio/transcriptions")
        self.assertIn('name="model"\r\r\nmoss-transcribe-diarize'.replace("\r\r\n", "\r\n\r\n"),
                      captured["body"])
        self.assertIn("max_tokens", captured["body"])
        self.assertEqual(turns, [{"speaker": 1, "start": 0.1, "end": 1.2, "text": "Hallo."}])

    def test_audio_cpp_upload_refuses_what_would_kill_the_server(self):
        """A too-big multipart does not fail the job: it SEGFAULTS the shared
        audio.cpp (measured at 80 MB). The upload path must refuse first."""
        orig_norm, orig_wav = meeting.normalize_wav, meeting.wav_seconds
        meeting.normalize_wav = lambda source, out: out.write_bytes(b"\0" * 100)
        meeting.wav_seconds = lambda path: 60.0
        orig_limit = meeting.AUDIOCPP_UPLOAD_MAX_BYTES
        meeting.AUDIOCPP_UPLOAD_MAX_BYTES = 10
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "in.wav"
                src.write_bytes(b"x")
                args = types.SimpleNamespace(asr="http://192.168.1.5:8090",
                                             model="moss-transcribe-diarize",
                                             asr_dir=None, asr_remote_dir=None,
                                             asr_timeout=10, max_tokens=512)
                (Path(tmp) / "work").mkdir()
                with self.assertRaises(SystemExit) as raised:
                    meeting.audiocpp_turns(args, src, Path(tmp) / "work", lenient=False)
                self.assertIn("voxtral", str(raised.exception))
        finally:
            meeting.normalize_wav, meeting.wav_seconds = orig_norm, orig_wav
            meeting.AUDIOCPP_UPLOAD_MAX_BYTES = orig_limit


class JobSummaryFields(unittest.TestCase):
    """The list endpoint strips the transcript, so it must keep what the table shows."""

    def test_the_summary_keeps_length_tracks_and_whether_a_retry_can_run(self):
        job = meeting_job.Job("abc", {})
        summary = meeting_job.Handler._summary(job.snapshot())
        self.assertIn("duration", summary)
        self.assertIn("tracks", summary)
        self.assertIn("audio", summary)  # the retry button is only offered with audio
        self.assertIn("archive", summary)  # the WebDAV badge reads it
        self.assertNotIn("json", summary)


class Retry(unittest.TestCase):
    """A failed job keeps its upload, so it can be run again without re-uploading."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        self.jobs = meeting_job.Jobs(Path("/bin/true"), root,                                      )
        # Not submit(): that queues the job and the worker races the test.
        audio = root / "inbox" / "meeting.wav"
        audio.write_bytes(b"audio")
        self.job = meeting_job.Job("0123456789abcdef", {})
        self.job.tracks = {"main": audio}
        self.jobs._jobs[self.job.id] = self.job

    def test_a_failed_job_is_queued_again_without_its_error(self):
        self.job.status, self.job.error, self.job.finished = "error", "ASR unreachable", 1.0
        again = self.jobs.retry(self.job.id)
        self.assertIsNone(again["error"])
        self.assertIn(again["status"], ("queued", "running", "done"))

    def test_a_running_job_is_left_alone(self):
        self.job.status = "running"
        self.assertEqual(self.jobs.retry(self.job.id)["status"], "running")

    def test_a_job_whose_audio_is_gone_is_not_run_again(self):
        self.job.status, self.job.error = "error", "boom"
        for path in self.job.tracks.values():
            path.unlink()
        again = self.jobs.retry(self.job.id)
        self.assertEqual(again["status"], "error")
        self.assertEqual(again["error"], "boom")

    def test_an_unknown_job_is_none(self):
        self.assertIsNone(self.jobs.retry("nope"))


class Stopping(unittest.TestCase):
    """Stop, restart and delete: the three things a queue needs to be usable."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.worker = self.root / "fake-worker"
        self.worker.write_text("#!/bin/sh\nsleep 30\n")
        self.worker.chmod(0o755)
        self.jobs = meeting_job.Jobs(self.worker, self.root, 
                                     )
        self.addCleanup(self.jobs.stop)

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def submit(self):
        return self.jobs.submit(b"audio", {}, "meeting.wav")["id"]

    def test_a_running_job_is_killed_and_stays_cancelled(self):
        identity = self.submit()
        self.assertTrue(self.wait_for(lambda: self.jobs._current is not None),
                        "the worker never started")
        self.assertEqual(self.jobs.cancel(identity)["status"], "cancelled")
        self.assertTrue(self.wait_for(lambda: self.jobs._current_job is None))
        job = self.jobs.get(identity)
        self.assertEqual(job["status"], "cancelled", "a killed worker is not a failure")
        self.assertIsNone(job["error"])

    def test_a_cancelled_job_in_the_queue_never_runs(self):
        audio = self.root / "inbox" / "meeting.wav"
        audio.write_bytes(b"audio")
        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks, job.status = {"main": audio}, "cancelled"
        self.jobs._jobs[job.id] = job
        self.jobs._queue.put(job)
        time.sleep(0.4)
        self.assertIsNone(job.started, "the worker picked up a cancelled job")
        self.assertEqual(job.status, "cancelled")

    def test_a_finished_job_is_not_cancelled_any_more(self):
        audio = self.root / "inbox" / "meeting.wav"
        audio.write_bytes(b"audio")
        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks, job.status = {"main": audio}, "done"
        self.jobs._jobs[job.id] = job
        self.assertEqual(self.jobs.cancel(job.id)["status"], "done")

    def test_a_cancelled_job_can_be_run_again(self):
        audio = self.root / "inbox" / "meeting.wav"
        audio.write_bytes(b"audio")
        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks, job.status = {"main": audio}, "cancelled"
        self.jobs._jobs[job.id] = job
        self.assertIn(self.jobs.retry(job.id)["status"], ("queued", "running", "done"))

    def test_delete_takes_the_transcript_and_the_upload_with_it(self):
        audio = self.root / "inbox" / "meeting.wav"
        audio.write_bytes(b"audio")
        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks, job.status = {"main": audio}, "error"
        self.jobs._jobs[job.id] = job
        transcript = self.jobs.out / f"{job.id}.md"
        transcript.write_text("# a transcript")
        self.assertIsNotNone(self.jobs.delete(job.id))
        self.assertIsNone(self.jobs.get(job.id))
        self.assertFalse(transcript.exists())
        self.assertFalse(audio.exists())
        self.assertFalse(any(self.jobs.out.glob(f"{job.id}.*")))

    def test_an_unknown_job_cannot_be_stopped_or_deleted(self):
        self.assertIsNone(self.jobs.cancel("nope"))
        self.assertIsNone(self.jobs.delete("nope"))


class JobRoutes(unittest.TestCase):
    """The HTTP layer itself: a decorator on the wrong method breaks every route."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.jobs = meeting_job.Jobs(Path("/bin/true"), self.root,
                                      )
        self.addCleanup(self.jobs.stop)
        self.server = meeting_job.Server(
            ("127.0.0.1", 0), self.jobs,
            meeting_job.Hooks(self.root / "hooks.json"), "", 10 * 1024 * 1024)
        self.addCleanup(self.server.server_close)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def call(self, path, method="GET", body=b""):
        request = urllib.request.Request(self.base + path, data=body or None, method=method)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def add_job(self, status):
        audio = self.root / "inbox" / "meeting.wav"
        audio.write_bytes(b"audio")
        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks, job.status = {"main": audio}, status
        self.jobs._jobs[job.id] = job
        return job

    def test_the_list_and_one_job_answer(self):
        self.add_job("done")
        self.assertEqual(len(self.call("/jobs")["jobs"]), 1)
        self.assertEqual(self.call("/jobs/0123456789abcdef")["status"], "done")

    def test_a_job_can_be_stopped_over_http(self):
        job = self.add_job("waiting")
        self.assertEqual(self.call(f"/jobs/{job.id}/stop", "POST")["status"], "cancelled")

    def test_a_job_can_be_deleted_over_http(self):
        job = self.add_job("error")
        self.assertEqual(self.call(f"/jobs/{job.id}", "DELETE")["status"], "error")
        self.assertIsNone(self.jobs.get(job.id))

    def test_an_unknown_job_is_404(self):
        for path, method in (("/jobs/nope/stop", "POST"), ("/jobs/nope", "DELETE")):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.call(path, method)
            self.assertEqual(caught.exception.code, 404)

    def test_a_model_query_parameter_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.call("/jobs?model=mistral.voxtral-small-transcribe-3", "POST", b"audio")
        self.assertEqual(caught.exception.code, 400)
        self.assertIn(b"fixed", caught.exception.read())
        self.assertEqual(self.jobs.list(), [])

    def test_mcp_does_not_offer_or_accept_a_model(self):
        tool = next(tool for tool in meeting_job.tools()
                    if tool["name"] == "transcribe_meeting")
        self.assertNotIn("model", tool["inputSchema"]["properties"])
        with self.assertRaises(meeting_job.ToolError):
            meeting_job.call_tool(
                self.jobs, "transcribe_meeting",
                {"audio_base64": base64.b64encode(b"audio").decode(),
                 "model": "mistral.voxtral-small-transcribe-3"})


class Persistence(unittest.TestCase):
    """A restart empties the queue; the list, the transcript and retry must survive."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def start(self, **kwargs):
        jobs = meeting_job.Jobs(Path("/bin/true"), self.root, 
                                **kwargs)
        self.addCleanup(jobs.stop)
        return jobs

    def test_a_finished_job_and_its_transcript_survive_a_restart(self):
        jobs = self.start()
        audio = self.root / "inbox" / "kept.wav"
        audio.write_bytes(b"audio")
        job = meeting_job.Job("0123456789abcdef", {"language": "de"})
        job.status, job.finished = "done", 1.0
        job.tracks = {"main": audio}
        Path(f"{self.root / 'out' / job.id}.json").write_text('{"turns": []}')
        Path(f"{self.root / 'out' / job.id}.md").write_text("# transcript\n")
        jobs._jobs[job.id] = job
        jobs._persist()

        restored = self.start().get(job.id)
        self.assertEqual(restored["status"], "done")
        self.assertEqual(restored["options"], {"language": "de"})
        self.assertEqual(restored["markdown"], "# transcript\n")
        self.assertTrue(restored["audio"])

    def test_a_job_cut_off_by_the_restart_is_an_error_with_the_audio_kept(self):
        jobs = self.start()
        audio = self.root / "inbox" / "kept.wav"
        audio.write_bytes(b"audio")
        job = meeting_job.Job("fedcba9876543210", {})
        job.status, job.started = "running", 1.0
        job.tracks = {"main": audio}
        jobs._jobs[job.id] = job
        jobs._persist()

        restored = self.start().get(job.id)
        self.assertEqual(restored["status"], "error")
        self.assertIn("restarted", restored["error"])
        self.assertTrue(audio.is_file())  # so the dashboard's retry can run it


class Retention(unittest.TestCase):
    """Uploads are kept for playback and retries, not forever."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def start(self, **kwargs):
        jobs = meeting_job.Jobs(Path("/bin/true"), self.root, 
                                **kwargs)
        self.addCleanup(jobs.stop)
        return jobs

    def test_audio_past_the_retention_is_deleted_unless_a_job_still_needs_it(self):
        jobs = self.start(retain_days=30)
        inbox = self.root / "inbox"
        past = time.time() - 40 * 86400
        for name in ("old.wav", "needed.wav"):
            path = inbox / name
            path.write_bytes(b"x")
            os.utime(path, (past, past))
        fresh = inbox / "fresh.wav"
        fresh.write_bytes(b"x")

        job = meeting_job.Job("0123456789abcdef", {})
        job.status, job.tracks = "running", {"main": inbox / "needed.wav"}
        jobs._jobs[job.id] = job

        self.assertEqual(jobs.prune(), 1)
        self.assertFalse((inbox / "old.wav").exists())
        self.assertTrue((inbox / "needed.wav").is_file())
        self.assertTrue(fresh.is_file())

    def test_zero_days_keeps_everything(self):
        jobs = self.start(retain_days=0)
        path = self.root / "inbox" / "old.wav"
        path.write_bytes(b"x")
        os.utime(path, (time.time() - 400 * 86400,) * 2)
        self.assertEqual(jobs.prune(), 0)
        self.assertTrue(path.is_file())


class Archiving(unittest.TestCase):
    """Every upload is a row in the SQLite ledger and, when configured, a WebDAV copy."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def test_a_received_file_is_recorded_and_archived(self):
        sent = []

        class StubWebDAV:
            def put(self, name, data, folder=None):
                sent.append({"name": name, "data": data, "folder": folder})
                return f"https://dav.example/meeting-stack/{folder}/{name}"

            def folder_url(self, folder):
                return f"https://dav.example/meeting-stack/{folder}"

        archive = meeting_job.Archive(self.root / "archive.db", StubWebDAV())
        audio = b"RIFF a meeting"
        archive.received("0123456789abcdef", "audio", "zoom-1-system-16k.mp3", audio)
        row = archive.rows()[0]
        self.assertEqual(row["status"], "received")
        self.assertEqual(row["size"], len(audio))
        self.assertEqual(row["sha256"], hashlib.sha256(audio).hexdigest())

        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks = {"main": self.root / "inbox" / "meeting.mp3"}
        job.tracks["main"].parent.mkdir()
        job.tracks["main"].write_bytes(audio)
        job.status, job.finished = "done", 1.0
        job.markdown = "# meeting\n"
        job.json = {"turns": [{"speaker": 1, "start": 0.0, "end": 1.0, "text": "hi"}]}
        archive.finish(job)

        row = next(row for row in archive.rows() if row["track"] == "audio")
        self.assertEqual(row["status"], "done")
        self.assertIsNotNone(row["archived_at"])
        self.assertEqual(row["webdav_url"],
                         "https://dav.example/meeting-stack/0123456789abcdef/"
                         "audio-zoom-1-system-16k.mp3")
        self.assertIn({"name": "audio-zoom-1-system-16k.mp3", "data": audio,
                       "folder": "0123456789abcdef"}, sent)
        self.assertEqual([item["name"] for item in sent],
                         ["audio-zoom-1-system-16k.mp3", "transcript.md",
                          "transcript.json"])
        self.assertEqual(len(archive.rows()), 3)  # audio + transcript.md + .json
        self.assertTrue(archive.archived(job.id))

    def test_a_webdav_failure_is_kept_on_the_row_not_raised(self):
        class BrokenWebDAV:
            def put(self, name, data, folder=None):
                raise urllib.error.URLError("dav is down")

        archive = meeting_job.Archive(self.root / "archive.db", BrokenWebDAV())
        audio = self.root / "inbox" / "meeting.mp3"
        audio.parent.mkdir()
        audio.write_bytes(b"audio")
        archive.received("0123456789abcdef", "audio", "meeting.mp3", b"audio")
        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks, job.status = {"main": audio}, "done"
        with io.StringIO() as output:  # the archive prints its failure
            with contextlib.redirect_stdout(output):
                archive.finish(job)
        row = next(row for row in archive.rows() if row["track"] == "audio")
        self.assertEqual(row["status"], "done")
        self.assertIsNone(row["archived_at"])
        self.assertIn("dav is down", row["archive_error"])
        self.assertTrue(audio.is_file())
        self.assertFalse(archive.archived(job.id))

    def test_webdav_makes_the_folders_then_puts_the_quoted_name(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_open(request, timeout=0):
            calls.append((request.get_method(), request.full_url,
                          request.headers.get("Authorization")))
            if request.get_method() == "MKCOL":
                # 405 = the collection already exists, which is not an error.
                raise urllib.error.HTTPError(request.full_url, 405, "exists", {}, None)
            return Response()

        original = meeting_job.urllib.request.urlopen
        meeting_job.urllib.request.urlopen = fake_open
        try:
            dav = meeting_job.WebDAV("https://dav.example/root", "user", "pass",
                                     "meetings")
            url = dav.put("main-a b.mp3", b"audio", folder="job1")
            # force is what the status probe uses: it must hit the wire even
            # though the folder was created a moment ago.
            self.assertEqual(dav.ensure(force=True),
                             "https://dav.example/root/meetings")
        finally:
            meeting_job.urllib.request.urlopen = original
        self.assertEqual([call[0] for call in calls],
                         ["MKCOL", "MKCOL", "PUT", "MKCOL"])
        self.assertEqual(calls[0][1], "https://dav.example/root/meetings")
        self.assertEqual(calls[1][1], "https://dav.example/root/meetings/job1")
        self.assertEqual(calls[2][1],
                         "https://dav.example/root/meetings/job1/main-a%20b.mp3")
        self.assertEqual(url, calls[2][1])
        expected = "Basic " + meeting_job.base64.b64encode(b"user:pass").decode()
        self.assertEqual(calls[2][2], expected)

    def test_without_webdav_the_ledger_still_records_and_keeps(self):
        archive = meeting_job.Archive(self.root / "archive.db")
        archive.received("0123456789abcdef", "audio", "meeting.mp3", b"audio")
        self.assertEqual(archive.rows()[0]["status"], "received")
        self.assertTrue(archive.archived("0123456789abcdef"))
        self.assertEqual(archive.job_state("0123456789abcdef")["state"], "off")

    def test_the_job_state_walks_from_pending_to_archived(self):
        class StubWebDAV:
            def put(self, name, data, folder=None):
                return f"https://dav.example/meeting-stack/{folder}/{name}"

            def ensure(self, timeout=None, force=False):
                return "https://dav.example/meeting-stack"

            def target(self):
                return "https://dav.example/meeting-stack"

            def folder_url(self, folder):
                return f"https://dav.example/meeting-stack/{folder}"

        archive = meeting_job.Archive(self.root / "archive.db", StubWebDAV())
        identity = "0123456789abcdef"
        self.assertIsNone(archive.job_state(identity))
        archive.received(identity, "audio", "meeting.mp3", b"audio")
        self.assertEqual(archive.job_state(identity)["state"], "pending")

        audio = self.root / "inbox" / "meeting.mp3"
        audio.parent.mkdir()
        audio.write_bytes(b"audio")
        job = meeting_job.Job(identity, {})
        job.tracks, job.status = {"main": audio}, "done"
        archive.finish(job)
        state = archive.job_state(identity)
        self.assertEqual(state["state"], "archived")
        self.assertEqual(state["url"],
                         "https://dav.example/meeting-stack/0123456789abcdef/"
                         "audio-meeting.mp3")
        self.assertEqual(state["folder"],
                         "https://dav.example/meeting-stack/0123456789abcdef")

    def test_the_probe_cache_skips_a_second_round_trip(self):
        class CountingWebDAV:
            calls = 0

            def ensure(self, timeout=None, force=False):
                self.calls += 1
                return "https://dav.example/meeting-stack"

            def target(self):
                return "https://dav.example/meeting-stack"

        dav = CountingWebDAV()
        archive = meeting_job.Archive(self.root / "cache.db", dav)
        archive.check()
        archive.check()
        self.assertEqual(dav.calls, 1)
        archive.check(max_age=0)
        self.assertEqual(dav.calls, 2)

    def test_the_probe_reports_configured_and_unreachable(self):
        class OkWebDAV:
            def ensure(self, timeout=None, force=False):
                return "https://dav.example/meeting-stack"

            def target(self):
                return "https://dav.example/meeting-stack"

        class BrokenWebDAV:
            def ensure(self, timeout=None, force=False):
                raise urllib.error.URLError("dav is down")

            def target(self):
                return "https://dav.example/meeting-stack"

        off = meeting_job.Archive(self.root / "off.db")
        self.assertEqual(off.check()["configured"], False)
        ok = meeting_job.Archive(self.root / "ok.db", OkWebDAV())
        self.assertEqual(ok.check(), {"configured": True, "reachable": True,
                                      "error": None,
                                      "url": "https://dav.example/meeting-stack"})
        broken = meeting_job.Archive(self.root / "broken.db", BrokenWebDAV())
        answer = broken.check()
        self.assertTrue(answer["configured"])
        self.assertFalse(answer["reachable"])
        self.assertIn("dav is down", answer["error"])

    def test_submit_records_the_upload_before_the_worker_runs(self):
        jobs = meeting_job.Jobs(Path("/bin/true"), self.root)
        self.addCleanup(jobs.stop)
        job = jobs.submit(b"audio", {}, "meeting.mp3")
        rows = jobs.archive.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job_id"], job["id"])
        self.assertEqual(rows[0]["filename"], "meeting.mp3")
        self.assertEqual(rows[0]["track"], "audio")
        self.assertEqual(job["archive"]["state"], "off")

class ByteRange(unittest.TestCase):
    """`GET /jobs/<id>/audio` must answer 206 to a player that seeks."""

    def setUp(self):
        self.range = meeting_job.byte_range

    def test_no_header_means_whole_file(self):
        self.assertIsNone(self.range(None, 100))
        self.assertIsNone(self.range("bytes=0-", 0))

    def test_open_ended_range(self):
        self.assertEqual(self.range("bytes=10-", 100), (10, 99))

    def test_end_is_clamped_to_the_file(self):
        self.assertEqual(self.range("bytes=90-500", 100), (90, 99))

    def test_suffix_range_is_the_tail(self):
        self.assertEqual(self.range("bytes=-20", 100), (80, 99))

    def test_garbage_and_unsatisfiable_fall_back_to_the_whole_file(self):
        self.assertIsNone(self.range("items=0-1", 100))
        self.assertIsNone(self.range("bytes=100-", 100))
        self.assertIsNone(self.range("bytes=a-b", 100))


class BatchProviders(unittest.TestCase):
    """The batch providers: Mistral uploads and the Unsloth workflow."""

    def test_mistral_batch_turns_keep_speakers_and_context_bias(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "segments": [
                        {"text": " Hallo Andreas.", "start": 1.2, "end": 2.4,
                         "speaker_id": "speaker_1"},
                        {"text": " Guten Morgen.", "start": 2.8, "end": 3.9,
                         "speaker_id": "speaker_2"},
                    ]
                }).encode()

        def fake_open(req, timeout=0):
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization")
            captured["body"] = req.data.decode("utf-8", "replace")
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            wav.write_bytes(wav_bytes(b"\x00\x00" * 1600))
            original = meeting.urllib.request.urlopen
            meeting.urllib.request.urlopen = fake_open
            try:
                turns = meeting.transcribe_mistral(
                    "https://api.mistral.ai/v1/audio/transcriptions", wav,
                    "mistral.voxtral-mini-transcribe-3", "German",
                    "Andreas, Kita", "secret-key", 10)
            finally:
                meeting.urllib.request.urlopen = original
        self.assertEqual(captured["auth"], "Bearer secret-key")
        self.assertIn('name="model"\r\n\r\nvoxtral-mini-transcribe-3', captured["body"])
        self.assertIn('name="diarize"\r\n\r\ntrue', captured["body"])
        self.assertIn('name="language"\r\n\r\nde', captured["body"])
        self.assertIn('name="context_bias"\r\n\r\nAndreas', captured["body"])
        self.assertIn('name="context_bias"\r\n\r\nKita', captured["body"])
        self.assertEqual(turns, [
            {"speaker": "speaker_1", "start": 1.2, "end": 2.4, "text": "Hallo Andreas."},
            {"speaker": "speaker_2", "start": 2.8, "end": 3.9, "text": "Guten Morgen."},
        ])

    def test_unsloth_asr_route_and_answer(self):
        # routing: the prefix is the provider
        self.assertTrue(meeting.is_unsloth_asr("unsloth.MOSS-Transcribe-Diarize"))
        self.assertFalse(meeting.is_unsloth_asr("moss-transcribe-diarize"))
        # the studio id: a short name rides the map, an id with slashes passes
        self.assertEqual(meeting.unsloth_studio_id("unsloth.MOSS-Transcribe-Diarize"),
                         "audio-cpp/audio.cpp-gguf/MOSS-Transcribe-Diarize-GGUF")
        self.assertEqual(meeting.unsloth_studio_id("unsloth.audio-cpp/x/y.z"),
                         "audio-cpp/x/y.z")
        # the complete event's segments become turns, in order, S02 kept
        answer = {"type": "complete", "segments": [
            {"start": 1.62, "end": 2.52, "text": "Hallo zusammen. ", "speaker": "S01"},
            {"start": 3.0, "end": None, "text": "", "speaker": "S02"},
            {"start": 3.1, "end": 4.0, "text": "Morgen.", "speaker": "S02"},
        ]}
        self.assertEqual(meeting.unsloth_answer_turns(answer), [
            {"speaker": "S01", "start": 1.62, "end": 2.52, "text": "Hallo zusammen."},
            {"speaker": "S02", "start": 3.1, "end": 4.0, "text": "Morgen."},
        ])
        self.assertEqual(meeting.number_speakers(meeting.unsloth_answer_turns(answer))[1]["speaker"], 2)

    def test_unsloth_batch_is_the_studios_transcribe_workflow(self):
        """One upload, then the workflow's NDJSON answer. This is the path the
        studio's own /audio page runs, and it takes up to 4 hours."""
        captured = []
        events = [
            {"type": "progress", "phase": "transcribing"},
            {"type": "complete", "segments": [
                {"start": 0.2, "end": 4.6, "text": " Ciao. ", "speaker": "S01"},
                {"start": 5.0, "end": 8.8, "text": "Bene.", "speaker": "S02"},
            ]},
        ]

        class FakeResponse:
            def __init__(self, payload=None, lines=None):
                self.payload, self.lines = payload, lines or []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(self.payload).encode()

            def __iter__(self):
                return iter(json.dumps(event).encode() for event in self.lines)

        def fake_open(req, timeout=0):
            captured.append({"url": req.full_url,
                             "auth": req.headers.get("Authorization"),
                             "body": req.data.decode("utf-8", "replace")})
            if "audio/inputs" in req.full_url:
                return FakeResponse({"id": "input_1"})
            return FakeResponse(lines=events)

        original = meeting.urllib.request.urlopen
        meeting.urllib.request.urlopen = fake_open
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "meeting.mp3"
                src.write_bytes(b"audio")
                turns = meeting.transcribe_unsloth(
                    "http://studio:8888", "secret", src,
                    "unsloth.MOSS-Transcribe-Diarize", "de", 10)
        finally:
            meeting.urllib.request.urlopen = original
        self.assertEqual(len(captured), 2)
        self.assertIn("/v1/audio/inputs?name=meeting.mp3", captured[0]["url"])
        self.assertEqual(captured[0]["auth"], "Bearer secret")
        self.assertEqual(captured[1]["url"],
                         "http://studio:8888/api/inference/audio/transcribe/source")
        body = json.loads(captured[1]["body"])
        self.assertEqual(body["source"], {"input_id": "input_1"})
        self.assertEqual(body["model"],
                         "audio-cpp/audio.cpp-gguf/MOSS-Transcribe-Diarize-GGUF")
        self.assertTrue(body["speakers"])
        self.assertEqual(body["language"], "de")
        self.assertEqual(turns, [
            {"speaker": "S01", "start": 0.2, "end": 4.6, "text": "Ciao."},
            {"speaker": "S02", "start": 5.0, "end": 8.8, "text": "Bene."},
        ])

    def test_mistral_turns_chunks_recordings_over_limit(self):
        calls = []
        orig_max = meeting.MISTRAL_MAX_S
        orig_norm = meeting.normalize_wav
        orig_tx = meeting.transcribe_mistral
        meeting.MISTRAL_MAX_S = 1.0
        meeting.normalize_wav = lambda src, dst: dst.write_bytes(
            wav_bytes(b"\x00\x00" * 32000))
        meeting.transcribe_mistral = lambda *a, **k: (
            calls.append(a),
            [{"speaker": "speaker_1", "start": 0.2, "end": 0.8, "text": "Teil."}])[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                args = types.SimpleNamespace(
                    model="voxtral-mini-transcribe-3", language="de",
                    asr_timeout=10, mistral_key="secret")
                turns = meeting.mistral_turns(
                    args, Path(tmp) / "in.wav", Path(tmp), "", lenient=False)
        finally:
            meeting.MISTRAL_MAX_S = orig_max
            meeting.normalize_wav = orig_norm
            meeting.transcribe_mistral = orig_tx
        self.assertEqual(len(calls), 2)
        # number_speakers() relabels Mistral's "speaker_1" to 1
        self.assertEqual(turns, [
            {"speaker": 1, "start": 0.2, "end": 0.8, "text": "Teil."},
            {"speaker": 1, "start": 1.2, "end": 1.8, "text": "Teil."},
        ])



if __name__ == "__main__":
    unittest.main()
