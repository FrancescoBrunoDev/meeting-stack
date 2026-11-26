#!/usr/bin/env python3
"""Tests for the parts of meeting-transcribe and meeting-job that decide the
transcript.

The ASRs (audio.cpp, Mistral) are external, so what is tested here is
everything that can be wrong without them: parsing their answers, merging and
deduplicating turns, matching two tracks, building the live lines, and the
defaults that decide which path runs.
"""
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

class Turns(unittest.TestCase):

    def test_an_utterance_cut_for_two_speakers_is_kept_once(self):
        turns = [{"speaker": 4, "start": 10.0, "end": 12.0,
                  "text": "Ja wegen der Zeitabrechnung, aber."},
                 {"speaker": 3, "start": 10.5, "end": 13.0,
                  "text": "wegen der Zeitabrechnung, aber ja."}]
        self.assertEqual(len(meeting.dedupe_turns(turns)), 1)

    def test_different_words_talking_at_once_stay_two_lines(self):
        turns = [{"speaker": 1, "start": 10.0, "end": 12.0, "text": "ich sage etwas"},
                 {"speaker": 2, "start": 10.5, "end": 13.0, "text": "und ich etwas anderes"}]
        self.assertEqual(len(meeting.dedupe_turns(turns)), 2)

    def test_a_shared_single_word_is_not_a_duplicate(self):
        turns = [{"speaker": 1, "start": 10.0, "end": 11.0, "text": "ja"},
                 {"speaker": 2, "start": 10.5, "end": 11.5, "text": "ja"}]
        self.assertEqual(len(meeting.dedupe_turns(turns)), 2)

    def test_the_same_words_far_apart_are_two_turns(self):
        turns = [{"speaker": 1, "start": 10.0, "end": 11.0, "text": "guten morgen zusammen"},
                 {"speaker": 1, "start": 30.0, "end": 31.0, "text": "guten morgen zusammen"}]
        self.assertEqual(len(meeting.dedupe_turns(turns)), 2)


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

    def test_a_long_pause_does_not_end_a_half_sentence(self):
        fake = types.SimpleNamespace(
            transcribe=lambda pcm, language, model="": "", translate=lambda *a, **k: "",
)
        session = meeting_job.LiveSession(fake, "de", "it")
        session._pending, session._pending_row = "die drei Tage sind", 0
        session.rows.append({"src": session._pending, "mt": "", "done": False})
        # ten seconds of silence: longer than any threshold the old rule had
        session._pcm = b"\x00\x00" * int(meeting_job.LIVE_RATE * 10)
        session._noise = meeting_job.LIVE_FLOOR
        session._close_on_pause()
        self.assertTrue(session._queue.empty(), "a half sentence must stay open")
        session._pending = "die drei Tage sind vorbei."
        session._close_on_pause()
        self.assertFalse(session._queue.empty(), "a full stop ends the line")

    def test_a_repeated_delta_is_not_appended_twice(self):
        line = ("Allora, quando una pausa viene fatta si inizia un nuovo paragrafo. "
                "E invece no, ecco che siamo inculati subito.")
        self.assertEqual(
            meeting_job.repeated_tail(line, "E invece no, ecco che siamo inculati subito."),
            "")
        self.assertEqual(
            meeting_job.repeated_tail(line, "inculati subito. Conosco questo paragrafo."),
            "Conosco questo paragrafo.")

    def test_an_empty_audio_cpp_answer_is_no_turns(self):
        self.assertEqual(meeting.parse_diarized(""), [])

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

    def test_overlapping_tracks_are_not_joined(self):
        # Two tracks can carry the same speaker at the same moment.
        turns = [{"speaker": 1, "start": 0.0, "end": 5.0, "text": "perché è proprio"},
                 {"speaker": 1, "start": 2.0, "end": 6.0, "text": "istintivo."}]
        self.assertEqual(len(meeting.merge_paragraphs(turns)), 2)


class TwoTracks(unittest.TestCase):
    """System track + microphone track: one conversation, two files."""

    def test_the_tracks_are_interleaved_by_time(self):
        first = [{"speaker": 1, "start": 0.0, "end": 4.0, "text": "hallo"},
                 {"speaker": 2, "start": 9.0, "end": 12.0, "text": "ja"}]
        second = [{"speaker": 3, "start": 5.0, "end": 8.0, "text": "guten tag"}]
        merged = meeting.merge_tracks(first, second)
        self.assertEqual([t["text"] for t in merged], ["hallo", "guten tag", "ja"])

    def test_an_offset_moves_the_second_track(self):
        second = [{"speaker": 3, "start": 5.0, "end": 8.0, "text": "guten tag"}]
        merged = meeting.merge_tracks([], second, offset=2.5)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (7.5, 10.5))

    def test_the_same_room_recorded_twice_is_not_repeated(self):
        # A microphone that also hears the loudspeakers: same speech, other
        # segmentation. Only the tail the system track does not have survives.
        first = [{"speaker": 1, "start": 0.0, "end": 40.0, "text": "hallo zusammen"}]
        second = [{"speaker": 5, "start": 0.1, "end": 39.9, "text": "hallo zusammen"},
                  {"speaker": 5, "start": 41.0, "end": 50.0, "text": "tschüss"}]
        merged = meeting.merge_tracks(first, second)
        self.assertEqual([t["text"] for t in merged], ["hallo zusammen", "tschüss"])

    def test_tracks_of_the_same_room_are_recognised_as_one(self):
        first = [{"speaker": 1, "start": 0.0, "end": 10.0, "text": "a"}]
        same = [{"speaker": 5, "start": 0.1, "end": 9.9, "text": "a"}]
        other = [{"speaker": 5, "start": 12.0, "end": 20.0, "text": "b"}]
        self.assertTrue(meeting.redundant(first, same))
        self.assertFalse(meeting.redundant(first, other))

    def test_crosstalk_on_complementary_tracks_is_kept(self):
        # Local and remote alternate: little shared speech, so nothing is dropped.
        first = [{"speaker": 1, "start": 0.0, "end": 10.0, "text": "a"},
                 {"speaker": 1, "start": 20.0, "end": 30.0, "text": "b"}]
        second = [{"speaker": 2, "start": 10.5, "end": 20.5, "text": "c"}]
        merged = meeting.merge_tracks(first, second)
        self.assertEqual([t["text"] for t in merged], ["a", "c", "b"])

    def test_a_repeated_short_turn_is_dropped_even_without_overlap(self):
        first = [{"speaker": 1, "start": 5.2, "end": 5.5, "text": "ja"},
                 {"speaker": 1, "start": 30.0, "end": 40.0, "text": "weiter"}]
        second = [{"speaker": 2, "start": 5.6, "end": 6.0, "text": "ja"},
                  {"speaker": 2, "start": 32.0, "end": 38.0, "text": "weiter"}]
        self.assertEqual([t["text"] for t in meeting.merge_tracks(first, second)], ["ja", "weiter"])

    def test_a_voice_of_the_second_track_gets_the_speaker_it_overlaps(self):
        first = [{"speaker": 1, "start": 0.0, "end": 10.0, "text": "a"},
                 {"speaker": 2, "start": 10.0, "end": 20.0, "text": "b"}]
        second = [{"speaker": 1, "start": 0.0, "end": 9.0, "text": "a"},
                  {"speaker": 2, "start": 10.0, "end": 19.0, "text": "b"},
                  {"speaker": 3, "start": 21.0, "end": 25.0, "text": "c"}]
        matched = meeting.match_speakers(first, second)
        self.assertEqual([t["speaker"] for t in matched], [1, 2, 3])

    def test_a_microphone_track_is_one_speaker_after_the_remote_ones(self):
        turns = [{"speaker": 0, "start": 1.0, "end": 2.0, "text": "a"},
                 {"speaker": 1, "start": 3.0, "end": 4.0, "text": "b"}]
        labelled = meeting.second_track_turns(turns, after=4, name="Me")
        self.assertEqual([t["speaker"] for t in labelled], [5, 5])
        self.assertEqual({t["name"] for t in labelled}, {"Me"})

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

    def test_a_diarized_second_track_numbsers_after_the_first(self):
        turns = [{"speaker": 0, "start": 1.0, "end": 2.0, "text": "a"},
                 {"speaker": 1, "start": 3.0, "end": 4.0, "text": "b"}]
        labelled = meeting.second_track_turns(turns, after=3, name="Room", diarized=True)
        self.assertEqual([t["speaker"] for t in labelled], [4, 5])

    def test_a_silent_mic_track_is_not_a_failed_job(self):
        args = types.SimpleNamespace(model="voxtral-mini-transcribe-3",
                                     language=None, asr_timeout=10, mistral_key="k")
        original = meeting.transcribe_mistral
        meeting.transcribe_mistral = lambda *a, **k: []
        original_norm = meeting.normalize_wav
        meeting.normalize_wav = lambda source, out: None
        try:
            with tempfile.TemporaryDirectory() as directory:
                turns = meeting.track_turns(args, Path(directory) / "mic.wav",
                                            Path(directory) / "work", "", lenient=True)
                self.assertEqual(turns, [])
        finally:
            meeting.transcribe_mistral = original
            meeting.normalize_wav = original_norm


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


class TrackPairing(unittest.TestCase):
    """The recorder uploads one file per track; they must become one job."""

    def test_the_recorders_file_names_pair_themselves(self):
        self.assertEqual(meeting_job.track_and_pair("zoom-20261020-205058-system-16k.mp3", {}),
                         ("main", "zoom-20261020-205058"))
        self.assertEqual(meeting_job.track_and_pair("zoom-20261020-205058-mic-16k.mp3", {}),
                         ("second", "zoom-20261020-205058"))

    def test_a_plain_upload_is_its_own_job(self):
        self.assertEqual(meeting_job.track_and_pair("meeting.m4a", {}), ("main", None))

    def test_query_parameters_win_over_the_file_name(self):
        self.assertEqual(meeting_job.track_and_pair("x.mp3", {"track": "mic", "pair": "abc"}),
                         ("second", "abc"))

    def setUp(self):
        # The worker thread may still be writing when the test ends.
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        self.jobs = meeting_job.Jobs(Path("/bin/true"), root,                                      pair_wait=10000)

    def test_both_tracks_land_on_one_job(self):
        first = self.jobs.submit(b"system", {}, "zoom-1-system-16k.mp3")
        self.assertEqual(first["status"], "waiting")
        self.assertEqual(self.jobs.queued(), 0)
        second = self.jobs.submit(b"mic", {}, "zoom-1-mic-16k.mp3")
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["tracks"], ["main", "second"])

    def test_the_upload_order_does_not_matter(self):
        first = self.jobs.submit(b"mic", {}, "zoom-2-mic-16k.mp3")
        self.assertEqual(first["status"], "waiting")
        second = self.jobs.submit(b"system", {}, "zoom-2-system-16k.mp3")
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["tracks"], ["main", "second"])

    def test_the_same_track_twice_is_one_job(self):
        first = self.jobs.submit(b"system", {}, "zoom-3-system-16k.mp3")
        again = self.jobs.submit(b"system", {}, "zoom-3-system-16k.mp3")
        self.assertEqual(again["id"], first["id"])
        self.assertEqual(again["tracks"], ["main"])


class Retry(unittest.TestCase):
    """A failed job keeps its upload, so it can be run again without re-uploading."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        self.jobs = meeting_job.Jobs(Path("/bin/true"), root,                                      pair_wait=10000)
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
                                     pair_wait=10000)
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
                                      pair_wait=10000)
        self.addCleanup(self.jobs.stop)
        self.server = meeting_job.Server(
            ("127.0.0.1", 0), self.jobs,
            meeting_job.Hooks(self.root / "hooks.json"), "", 10 * 1024 * 1024, None)
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


class Persistence(unittest.TestCase):
    """A restart empties the queue; the list, the transcript and retry must survive."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def start(self, **kwargs):
        jobs = meeting_job.Jobs(Path("/bin/true"), self.root, 
                                pair_wait=10000, **kwargs)
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

    def test_a_job_waiting_for_its_second_track_stays_waiting(self):
        jobs = self.start()
        snap = jobs.submit(b"mic", {}, "zoom-1-mic-16k.mp3")
        self.assertEqual(snap["status"], "waiting")
        self.assertEqual(self.start().get(snap["id"])["status"], "waiting")

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
                                pair_wait=10000, **kwargs)
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
        sent = {}

        class StubWebDAV:
            def put(self, name, data):
                sent["name"], sent["data"] = name, data
                return f"https://dav.example/meeting-stack/{name}"

        archive = meeting_job.Archive(self.root / "archive.db", StubWebDAV())
        audio = b"RIFF a meeting"
        archive.received("0123456789abcdef", "main", "zoom-1-system-16k.mp3", audio)
        row = archive.rows()[0]
        self.assertEqual(row["status"], "received")
        self.assertEqual(row["size"], len(audio))
        self.assertEqual(row["sha256"], hashlib.sha256(audio).hexdigest())

        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks = {"main": self.root / "inbox" / "meeting.mp3"}
        job.tracks["main"].parent.mkdir()
        job.tracks["main"].write_bytes(audio)
        job.status, job.finished = "done", 1.0
        archive.finish(job)

        row = archive.rows()[0]
        self.assertEqual(row["status"], "done")
        self.assertIsNotNone(row["archived_at"])
        self.assertEqual(row["webdav_url"],
                         "https://dav.example/meeting-stack/main-zoom-1-system-16k.mp3")
        self.assertEqual(sent,
                         {"name": "main-zoom-1-system-16k.mp3", "data": audio})
        self.assertTrue(archive.archived(job.id))

    def test_a_webdav_failure_is_kept_on_the_row_not_raised(self):
        class BrokenWebDAV:
            def put(self, name, data):
                raise urllib.error.URLError("dav is down")

        archive = meeting_job.Archive(self.root / "archive.db", BrokenWebDAV())
        audio = self.root / "inbox" / "meeting.mp3"
        audio.parent.mkdir()
        audio.write_bytes(b"audio")
        archive.received("0123456789abcdef", "main", "meeting.mp3", b"audio")
        job = meeting_job.Job("0123456789abcdef", {})
        job.tracks, job.status = {"main": audio}, "done"
        with io.StringIO() as output:  # the archive prints its failure
            with contextlib.redirect_stdout(output):
                archive.finish(job)
        row = archive.rows()[0]
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
            url = dav.put("main-a b.mp3", b"audio")
            # force is what the status probe uses: it must hit the wire even
            # though the folder was created a moment ago.
            self.assertEqual(dav.ensure(force=True),
                             "https://dav.example/root/meetings")
        finally:
            meeting_job.urllib.request.urlopen = original
        self.assertEqual([call[0] for call in calls], ["MKCOL", "PUT", "MKCOL"])
        self.assertEqual(calls[0][1], "https://dav.example/root/meetings")
        self.assertEqual(calls[1][1],
                         "https://dav.example/root/meetings/main-a%20b.mp3")
        self.assertEqual(url, calls[1][1])
        expected = "Basic " + meeting_job.base64.b64encode(b"user:pass").decode()
        self.assertEqual(calls[1][2], expected)

    def test_without_webdav_the_ledger_still_records_and_keeps(self):
        archive = meeting_job.Archive(self.root / "archive.db")
        archive.received("0123456789abcdef", "main", "meeting.mp3", b"audio")
        self.assertEqual(archive.rows()[0]["status"], "received")
        self.assertTrue(archive.archived("0123456789abcdef"))
        self.assertEqual(archive.job_state("0123456789abcdef")["state"], "off")

    def test_the_job_state_walks_from_pending_to_archived(self):
        class StubWebDAV:
            def put(self, name, data):
                return f"https://dav.example/meeting-stack/{name}"

            def ensure(self, timeout=None, force=False):
                return "https://dav.example/meeting-stack"

            def target(self):
                return "https://dav.example/meeting-stack"

        archive = meeting_job.Archive(self.root / "archive.db", StubWebDAV())
        identity = "0123456789abcdef"
        self.assertIsNone(archive.job_state(identity))
        archive.received(identity, "main", "meeting.mp3", b"audio")
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
                         "https://dav.example/meeting-stack/main-meeting.mp3")

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
        jobs = meeting_job.Jobs(Path("/bin/true"), self.root, pair_wait=10000)
        self.addCleanup(jobs.stop)
        job = jobs.submit(b"audio", {}, "meeting.mp3")
        rows = jobs.archive.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job_id"], job["id"])
        self.assertEqual(rows[0]["filename"], "meeting.mp3")
        self.assertEqual(rows[0]["track"], "main")
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


class LiveCutting(unittest.TestCase):
    """Live audio is cut into utterances where the speaker pauses."""

    def pcm(self, seconds, amplitude):
        return b"".join(amplitude.to_bytes(2, "little", signed=True)
                        for _ in range(int(meeting_job.LIVE_RATE * seconds)))

    def test_cut_lands_at_the_start_of_the_pause(self):
        cut = meeting_job.vad_cut(self.pcm(1.0, 4000) + self.pcm(1.0, 0), 200)
        self.assertIsNotNone(cut)
        at, speech = cut
        self.assertTrue(speech)
        self.assertAlmostEqual(at / meeting_job.LIVE_BYTES_PER_S, 1.0, delta=0.05)

    def test_a_short_breath_does_not_end_the_clip(self):
        # 0.2 s of silence is a gap between words, not a clip boundary: clips are
        # cut tightly because the line, not the clip, is what must stay whole
        self.assertIsNone(meeting_job.vad_cut(self.pcm(1.0, 4000) + self.pcm(0.2, 0), 200))

    def test_a_sentence_still_going_is_left_alone(self):
        self.assertIsNone(meeting_job.vad_cut(self.pcm(1.0, 4000), 200))

    def test_a_silence_is_not_an_utterance(self):
        self.assertIsNone(meeting_job.vad_cut(self.pcm(3.0, 0), 200))
        cut = meeting_job.vad_cut(self.pcm(meeting_job.LIVE_MAX + 1, 0), 200)
        self.assertIsNotNone(cut)
        self.assertFalse(cut[1])

    def test_a_stretch_without_a_pause_is_cut_before_the_limit(self):
        cut = meeting_job.vad_cut(self.pcm(meeting_job.LIVE_MAX + 5, 4000), 200)
        self.assertIsNotNone(cut)
        at, speech = cut
        self.assertTrue(speech)
        self.assertLessEqual(at / meeting_job.LIVE_BYTES_PER_S, meeting_job.LIVE_MAX)

    def test_the_cut_is_not_dragged_to_the_end_of_the_buffer(self):
        # speech, a pause, then someone already talking again: the line ends at
        # the pause, and the words after it belong to the next line
        cut = meeting_job.vad_cut(self.pcm(1.0, 4000) + self.pcm(1.0, 0) + self.pcm(1.0, 4000), 200)
        self.assertIsNotNone(cut)
        at, speech = cut
        self.assertTrue(speech)
        self.assertAlmostEqual(at / meeting_job.LIVE_BYTES_PER_S, 1.0, delta=0.05)

    def test_the_quiet_room_below_the_floor_still_counts_as_silence(self):
        self.assertIsNone(meeting_job.vad_cut(self.pcm(1.0, 100) + self.pcm(1.0, 100), 200))

    def test_wav_is_16k_mono_and_as_long_as_the_pcm(self):
        clip = self.pcm(1.5, 4000)
        with wave.open(io.BytesIO(meeting_job.wav_bytes(clip))) as handle:
            self.assertEqual((handle.getframerate(), handle.getnchannels(),
                              handle.getsampwidth()), (16000, 1, 2))
            self.assertEqual(handle.getnframes(), len(clip) // 2)


class LiveSession(unittest.TestCase):
    """A session asks the endpoints for one line and keeps the order."""

    def fake(self, rows, text="Guten Tag.", translated="Buongiorno"):
        def transcribe(pcm, language, model=""):
            rows.append(("asr", len(pcm)))
            return text

        def translate(src, target, context, on_token=None, model=""):
            rows.append(("mt", target, list(context)))
            if on_token is not None:
                on_token(translated[:4])
            return translated

        return types.SimpleNamespace(transcribe=transcribe, translate=translate)

    def speech(self):
        return (4000).to_bytes(2, "little", signed=True) * 16000

    def say(self, session, seconds=1.0):
        """One clip, then the pause that closes the line — what the browser sends."""
        speech = (4000).to_bytes(2, "little", signed=True) * int(16000 * seconds)
        session.feed(speech + b"\x00\x00" * 16000)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not session._pending:
            time.sleep(0.01)
        session.feed(b"\x00\x00" * 16000)          # silence past LIVE_LINE
        time.sleep(0.05)

    def drain(self, session, limit=5.0):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline and not session.done:
            time.sleep(0.02)
        self.assertTrue(session.done)

    def test_the_page_can_pick_the_translator(self):
        """The session carries the model the page asked for, all the way down."""
        chosen = []

        def translate(src, target, context, on_token=None, model=""):
            chosen.append(model)
            return "Buongiorno"

        live = types.SimpleNamespace(transcribe=lambda pcm, language, model="": "Guten Tag.",
                                     translate=translate)
        session = meeting_job.LiveSession(live, "de", "it", meeting_job.LIVE_LINE,
                                          "translategemma-4b-FLM")
        session.feed(self.speech() + b"\x00\x00" * 16000)
        session.close()
        self.drain(session)
        self.assertEqual(chosen, ["translategemma-4b-FLM"])

    def open_line(self, text):
        """A line being built, with nothing translated yet."""
        session = meeting_job.LiveSession(self.fake([]), "de", "it")
        session._pending, session._pending_row = text, 0
        session.rows.append({"src": text, "mt": "", "done": False})
        return session

    def test_a_short_line_is_translated_without_waiting_for_the_pause(self):
        """The worker used to sit idle while the page showed a line with nothing under it."""
        session = self.open_line("die drei Tage sind weg")      # 23 characters
        session._maybe_partial()
        self.assertFalse(session._queue.empty(),
                         "the first translation of a line must not wait for 60 characters")

    def test_a_line_too_short_to_translate_is_left_alone(self):
        session = self.open_line("die drei")                    # 8 characters
        session._maybe_partial()
        self.assertTrue(session._queue.empty())

    def test_a_second_partial_waits_for_the_line_to_grow(self):
        session = self.open_line("die drei Tage sind vorbei")   # 26 characters
        session._maybe_partial()
        session._partial_busy = False                            # the worker is free again
        session._maybe_partial()
        self.assertEqual(session._queue.qsize(), 1,
                         "the same text twice is a request nobody will read")

    def test_a_cut_clip_becomes_one_translated_row(self):
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it")
        session.feed(self.speech() + b"\x00\x00" * 16000)
        session.close()
        self.drain(session)
        self.assertEqual(session.rows, [{"src": "Guten Tag.", "mt": "Buongiorno",
                                         "done": True}])
        self.assertEqual([call[0] for call in calls], ["asr", "mt"])

    def test_the_row_is_there_before_the_translation_is_finished(self):
        # Streaming: the page has something to rewrite while the tokens arrive.
        seen = []

        def translate(src, target, context, on_token=None, model=""):
            on_token("Buon")
            seen.extend(session.view(0)["rows"])
            return "Buongiorno"

        fake = types.SimpleNamespace(transcribe=lambda pcm, language, model="": "Guten Tag.",
                                     translate=translate)
        session = meeting_job.LiveSession(fake, "de", "it")
        session.feed(self.speech() + b"\x00\x00" * 16000)
        session.close()
        self.drain(session)
        self.assertEqual(seen, [{"i": 0, "src": "Guten Tag.", "mt": "Buon",
                                "done": False}])
        self.assertEqual(session.rows[0]["mt"], "Buongiorno")

    def test_the_context_of_the_next_line_is_the_previous_pair(self):
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it")
        self.say(session)
        self.say(session)
        session.close()
        self.drain(session)
        translations = [call for call in calls if call[0] == "mt"]
        self.assertEqual(len(translations), 2)
        self.assertEqual(translations[0][2], [])
        self.assertEqual(translations[1][2], ["Guten Tag. -> Buongiorno"])

    def test_a_dead_asr_stream_is_opened_again(self):
        # audio.cpp unloads a model it thinks is idle after two minutes, which
        # kills the stream it is serving. The old session went on feeding the
        # dead socket: no rows for the rest of the call, no error anywhere.
        opened = []

        class DeadStream:
            def __init__(self, base, model, on_text):
                self.dead = False
                self.reason = ""
                opened.append(self)

            def feed(self, pcm):
                self.dead = True          # the model went away under us
                self.reason = "the service closed the stream"

            def close(self):
                pass

        original = meeting_job.StreamingASR
        meeting_job.StreamingASR = DeadStream
        try:
            fake = types.SimpleNamespace(
                transcribe=lambda pcm, language, model="": "", translate=lambda *a, **k: "",
                stream_asr=True, asr="http://127.0.0.1:8090", model="m")
            session = meeting_job.LiveSession(fake, "de", "it")
            session.feed(self.speech())
            session.feed(self.speech())
            self.assertEqual(len(opened), 2)
            self.assertEqual(session._stream_restarts, 1)
        finally:
            meeting_job.StreamingASR = original

    def test_a_half_sentence_is_translated_when_the_stream_stops(self):
        calls = []
        session = meeting_job.LiveSession(self.fake(calls, text="und dann"), "de", "it")
        session.feed(self.speech() + b"\x00\x00" * 16000)
        session.close()
        self.drain(session)
        self.assertIn("mt", [call[0] for call in calls])
        self.assertEqual(session.rows[0]["mt"], "Buongiorno")

    def test_the_prompt_names_the_language_instead_of_its_code(self):
        # "the language whose ISO 639-1 code is it" reads as the English pronoun:
        # the model answered in English. The name fixes it.
        for code, name in (("it", "Italian"), ("de", "German"), ("en", "English")):
            prompt = meeting_job.translate_prompt(code)
            self.assertIn(name, prompt)
            self.assertNotIn(f"code is {code}", prompt)

    def test_every_row_carries_its_index(self):
        session = meeting_job.LiveSession(self.fake([]), "de", "it")
        self.say(session)
        self.say(session)
        session.close()
        self.drain(session)
        self.assertEqual([row["i"] for row in session.view(0)["rows"]], [0, 1])
        self.assertEqual([row["i"] for row in session.view(1)["rows"]], [1])

    def test_a_failing_endpoint_is_a_row_not_a_dead_session(self):
        def transcribe(pcm, language, model=""):
            raise OSError("connection refused")

        session = meeting_job.LiveSession(
            types.SimpleNamespace(transcribe=transcribe, translate=lambda *a: "",
), "de", "it")
        session.feed(self.speech() + b"\x00\x00" * 16000)
        session.close()
        self.drain(session)
        self.assertIn("connection refused", session.rows[0]["error"])

class LiveLines(unittest.TestCase):
    """A line is a stretch of speech between two pauses, not one ASR clip."""

    def drain(self, session, limit=5.0):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline and not session.done:
            time.sleep(0.02)
        self.assertTrue(session.done)

    def fake(self, calls):
        return types.SimpleNamespace(
            transcribe=lambda pcm, language, model="": "",
            translate=lambda src, target, context, on_token=None, model="": (
                calls.append(src), "tradotto")[1])

    def test_a_clip_that_stops_for_a_breath_waits_for_the_rest(self):
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it")
        session._hear("Bei der Bundeswehr gibt es mehr Gewalt,", closed=False)
        self.assertEqual(calls, [], "half a sentence must not be translated yet")
        self.assertEqual(len(session.rows), 1)
        session._hear("und Fehlverhalten als bislang bekannt.", closed=True)
        self.assertEqual(len(session.rows), 1, "the two clips are one line")
        self.assertEqual(session.rows[0]["src"],
                         "Bei der Bundeswehr gibt es mehr Gewalt, und Fehlverhalten "
                         "als bislang bekannt.")
        self.assertEqual(calls, [session.rows[0]["src"]])

    def test_a_pause_closes_the_line_so_the_next_one_starts(self):
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it")
        session._hear("Heute im Studio.", closed=True)
        session._hear("Ein schönen guten Abend.", closed=True)
        self.assertEqual([row["src"] for row in session.rows],
                         ["Heute im Studio.", "Ein schönen guten Abend."])
        self.assertEqual(len(calls), 2)

    def test_the_cursor_stops_at_a_row_still_being_translated(self):
        # The page follows the cursor. If the cursor moves past a row the worker
        # is still translating, that row stays "…" for ever (that was the bug).
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it")
        session._hear("Heute im Studio.", closed=False)
        self.assertEqual(session.view(0)["cursor"], 0)
        session._hear("und dann noch etwas.", closed=True)
        view = session.view(0)
        self.assertEqual(view["cursor"], 1)
        self.assertEqual(view["rows"][0]["mt"], "tradotto")

    def test_an_empty_answer_is_retried_and_then_reported(self):
        calls = []
        answers = iter(["Heute im Studio.", "", ""])   # the first line translates

        def translate(src, target, context, on_token=None, model=""):
            calls.append(list(context))
            return next(answers)

        session = meeting_job.LiveSession(
            types.SimpleNamespace(transcribe=lambda pcm, language, model="": "", translate=translate,
), "de", "it")
        session._hear("Heute im Studio.", closed=True)
        session._hear("Ein schönen guten Abend.", closed=True)
        self.assertEqual(len(calls), 3, "one retry, without the context")
        self.assertEqual(calls[-1], [])
        self.assertEqual(session.rows[1]["error"], "the model returned no translation")

    def test_the_cursor_does_not_move_while_a_line_is_streaming(self):
        # `mt` holds the first token long before the line is finished: a cursor
        # that keys on it moves on, and the page keeps "aveva affid" for ever.
        cursors = []

        def translate(src, target, context, on_token=None, model=""):
            on_token("aveva")
            cursors.append(session.view(0)["cursor"])
            on_token("aveva affidato al suo agente pubblicitario Högel.")
            cursors.append(session.view(0)["cursor"])
            return "aveva affidato al suo agente pubblicitario Högel."

        session = meeting_job.LiveSession(
            types.SimpleNamespace(transcribe=lambda pcm, language, model="": "",
                                  translate=translate), "de", "it")
        session._hear("Ihr Werbeauftragte Högel in Auftrag gegeben hatte.", closed=True)
        self.assertEqual(cursors, [0, 0], "the row was abandoned mid-sentence")
        self.assertEqual(session.view(0)["cursor"], 1)
        self.assertEqual(session.rows[0]["mt"],
                         "aveva affidato al suo agente pubblicitario Högel.")

    def test_a_line_that_runs_on_is_cut_at_a_full_stop(self):
        long_de = ("Ihr Werbeauftragte Högel in Auftrag gegeben hatte. Demnach wird höchstens "
                   "jeder dritte Fall überhaupt gemeldet. Am häufigsten betroffen sind junge "
                   "Frauen und nicht heterosexuelle Männer. Verteidigungsminister Pistorius "
                   "sagte, für solche Vorfälle gebe es Null Toleranz.")
        calls = []
        fake = types.SimpleNamespace(
            transcribe=lambda pcm, language, model="": "",
            translate=lambda src, target, context, on_token=None, model="": (
                calls.append(src), "tradotto")[1])
        session = meeting_job.LiveSession(fake, "de", "it")
        session._hear(long_de, closed=False)     # no pause, and past 220 characters
        self.assertTrue(calls, "the long line was not translated at all")
        self.assertLess(len(calls[0]), len(long_de))
        self.assertTrue(session.rows[0]["done"])
        self.assertFalse(session.rows[-1]["done"], "the rest waits in its own line")

    def test_the_view_says_what_the_worker_is_doing(self):
        # "Nothing on screen" has to be distinguishable from "broken".
        session = meeting_job.LiveSession(self.fake([]), "de", "it")
        self.assertEqual(session.view(0)["state"], "idle")
        self.assertEqual(session.view(0)["buffered"], 0.0)
        session.feed(b"\x00\x00" * 12000)          # 0.75 s of audio, no pause yet
        self.assertEqual(session.view(0)["buffered"], 0.8)
        session._set_state("asr")
        self.assertEqual(session.view(0)["state"], "asr")
        self.assertFalse(session.view(0)["done"])
        session.close()
        self.drain(session)
        self.assertTrue(session.view(0)["done"])

    def test_the_model_markers_never_reach_the_page(self):
        self.assertEqual(
            meeting_job.clean_live_text("[0.00][S01] Das sind ja dann die drei Tage.[2.48]"),
            "Das sind ja dann die drei Tage.")
        self.assertEqual(
            meeting_job.clean_live_text(" \n Speaker 0:Das sind ja dann die drei Tage."),
            "Das sind ja dann die drei Tage.")
        self.assertEqual(
            meeting_job.clean_live_text("Okay. [Noise] Ja, natürlich [Laughter] weiter."),
            "Okay. Ja, natürlich weiter.")

    def test_silence_closes_the_line_without_waiting_for_the_asr(self):
        # Clips are cut at 0.45 s so the ASR sees short, cheap requests; the line
        # is closed by silence that keeps going, or by a change of voice.
        calls = []
        fake = types.SimpleNamespace(
            transcribe=lambda pcm, language, model="": "Guten Abend.",
            translate=lambda src, target, context, on_token=None, model="": (
                calls.append(src), "Buonasera.")[1],
)
        session = meeting_job.LiveSession(fake, "de", "it")
        speech = (4000).to_bytes(2, "little", signed=True) * 16000
        session.feed(speech + b"\x00\x00" * 16000)     # 1 s of speech, then silence
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not session._pending:
            time.sleep(0.01)
        session.feed(b"\x00\x00" * 16000)             # the pause keeps going
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not calls:
            time.sleep(0.02)
        self.assertEqual(calls, ["Guten Abend."], "the pause did not close the line")
        session.close()
        self.drain(session)

    def test_turn_end_closes_the_line_and_the_next_turn_starts_a_new_one(self):
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it",
                                          asr_model="voxtral-mini-transcribe-realtime-3")
        session._streamed("Guten Tag, wie geht es?", dedupe=False)
        session._turn_end()                 # the cloud ASR said the turn is over
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and session._pending:
            time.sleep(0.01)
        self.assertFalse(session._pending, "turn.end did not close the line")
        session._streamed("Danke, sehr gut.", dedupe=False)
        self.assertEqual(len(session.rows), 2)
        self.assertEqual(session.rows[0]["src"], "Guten Tag, wie geht es?")
        self.assertEqual(session.rows[1]["src"], "Danke, sehr gut.")
        session._queue.put(None)
        self.drain(session)
        self.assertTrue(session.rows[0]["done"])
        self.assertEqual(session.view(0)["asr_model"], "voxtral-mini-transcribe-realtime-3")

    def test_voxtral_pieces_do_not_split_a_word(self):
        # Measured on the realtime API: token pieces cut inside words, and a
        # piece without its own whitespace continues the word (' Hall','o').
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it",
                                          asr_model="voxtral-mini-transcribe-realtime-3")
        for piece in ("Guten", " Abend", ".", " Hall", "o", "."):
            session._streamed(piece, dedupe=False)
        self.assertEqual(session.rows[0]["src"], "Guten Abend. Hallo.")

    def test_whole_segment_streams_keep_the_separator(self):
        # audio.cpp sends whole segments with no relation to word starts:
        # they always take the space back.
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it",
                                          asr_model="vibevoice_asr_streaming_7b_q8_0")
        session._streamed("Hall", dedupe=True)
        session._streamed("o", dedupe=True)
        self.assertEqual(session.rows[0]["src"], "Hall o")

    def test_a_stale_turn_end_does_not_close_the_next_line(self):
        calls = []
        session = meeting_job.LiveSession(self.fake(calls), "de", "it",
                                          asr_model="voxtral-mini-transcribe-realtime-3")
        session._streamed("Erstes.", dedupe=False)
        session._turn_end()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and session._pending:
            time.sleep(0.01)                # the timer closed row 0
        session._streamed("Zweites.", dedupe=False)
        with session._lock:
            row, text = session._pending_row, session._pending
        session._close_turn(row - 1)       # the stale timer fires too late
        with session._lock:
            self.assertEqual(session._pending, text, "it closed the new line")
        session._queue.put(None)
        self.drain(session)

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
            wav.write_bytes(meeting_job.wav_bytes(b"\x00\x00" * 1600))
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

    def test_unsloth_model_id_rides_the_namespace(self):
        self.assertEqual(meeting_job.unsloth_model_id("unsloth.Hy-MT2-7B-GGUF"),
                         "unsloth/Hy-MT2-7B-GGUF")
        self.assertEqual(meeting_job.unsloth_model_id("unsloth.Qwen/Qwen3-4B-GGUF"),
                         "Qwen/Qwen3-4B-GGUF")

    def test_mistral_translator_omits_extra_fields_that_cause_422(self):
        sent = {}
        with tempfile.TemporaryDirectory() as tmp:
            live = meeting_job.Live(
                "http://127.0.0.1:8090", "m", "http://127.0.0.1:8731",
                "mistral.ministral-8b-latest")
            live._ask = lambda payload, on_token=None: (sent.update(payload), "Ciao")[1]
            result = live.translate("Hallo", "it", [], model="mistral.ministral-8b-latest")
        self.assertEqual(result, "Ciao")
        self.assertNotIn("reasoning_effort", sent)
        self.assertNotIn("chat_template_kwargs", sent)

    def test_mistral_turns_chunks_recordings_over_limit(self):
        calls = []
        orig_max = meeting.MISTRAL_MAX_S
        orig_norm = meeting.normalize_wav
        orig_tx = meeting.transcribe_mistral
        meeting.MISTRAL_MAX_S = 1.0
        meeting.normalize_wav = lambda src, dst: dst.write_bytes(
            meeting_job.wav_bytes(b"\x00\x00" * 32000))
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

    def test_mistral_second_track_is_matched_not_shifted(self):
        """Voxtral diarizes each track separately; the mic track that hears the
        same room must be matched to the system speakers, not become speaker 4."""
        import subprocess
        calls = []
        real_run = subprocess.run

        def fake_run(cmd, **kw):
            calls.append(cmd)
            class R: returncode = 0; stderr = ""
            return R()

        main_turns = [
            {"speaker": 1, "start": 0.0, "end": 25.0, "text": "Hallo zusammen."},
            {"speaker": 2, "start": 10.0, "end": 20.0, "text": "Guten Morgen."},
        ]
        mic_turns = [
            {"speaker": 1, "start": 0.5, "end": 24.0, "text": "Hallo zusammen."},  # same room
            {"speaker": 2, "start": 30.0, "end": 40.0, "text": "Nur ich hier."},   # mic only
        ]
        orig_mt = meeting.mistral_turns
        orig_norm = meeting.normalize_wav
        orig_wavs = meeting.wav_seconds
        state = {"n": 0}
        def fake_mistral(args, source, workdir, hotwords, lenient):
            state["n"] += 1
            return [dict(t) for t in (main_turns if state["n"] == 1 else mic_turns)]
        meeting.mistral_turns = fake_mistral
        meeting.normalize_wav = lambda src, dst: None
        meeting.wav_seconds = lambda p: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                args = types.SimpleNamespace(
                    model="voxtral-mini-transcribe-3",
                    language="de", diarize_second=False, second_name="Me",
                    second_offset=0.0, merge_gap=2.0, output=Path(tmp) / "out",
                    diarization=None, names=None, hotwords="")
                # replicate the main() second-track branch
                labelled = meeting.number_speakers(main_turns)
                second = [dict(t) for t in mic_turns]
                after = max(t["speaker"] for t in labelled)
                voices = len({t["speaker"] for t in second})
                if meeting.redundant(labelled, second, args.second_offset):
                    second = meeting.match_speakers(labelled, second, args.second_offset)
                elif voices > 1:
                    second = meeting.second_track_turns(second, after, args.second_name, diarized=True)
                else:
                    second = meeting.second_track_turns(second, after, args.second_name)
        finally:
            meeting.mistral_turns = orig_mt
            meeting.normalize_wav = orig_norm
            meeting.wav_seconds = orig_wavs
        speakers = sorted({t["speaker"] for t in second})
        self.assertEqual(speakers, [1, 3],
                         "mic voice heard on the system track must map to it, "
                         "a mic-only voice gets a new number — not a blanket shift to 3,4")
        by_speaker = {t["speaker"]: t["text"] for t in second}
        self.assertEqual(by_speaker[1], "Hallo zusammen.")
        self.assertEqual(by_speaker[3], "Nur ich hier.")

    def test_live_session_forwards_per_session_glossary(self):
        seen = []

        def translate(src, target, context, on_token=None, model="", glossary=""):
            seen.append(glossary)
            return "Ciao Andreas."

        live = types.SimpleNamespace(
            transcribe=lambda pcm, language, model="": "",
            translate=translate,
            glossary="Kita")
        session = meeting_job.LiveSession(live, "de", "it", glossary="Andreas")
        self.assertEqual(session.glossary, "Kita, Andreas")
        session._hear("Hallo Andreas.", closed=True)
        session.close()
        self.drain(session)
        self.assertEqual(seen, ["Kita, Andreas"])


if __name__ == "__main__":
    unittest.main()
