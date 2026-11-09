#!/usr/bin/env python3
"""Tests for the parts of meeting-transcribe that decide the transcript.

The pipeline talks to two external services (an ASR endpoint and a diarizer
container), so what is tested here is everything that can be wrong without them:
rebuilding words the server split, fusing ASR segments with speaker labels,
merging turns, and the defaults that decide which path runs.
"""
import importlib.util
import sys
import tempfile
import unittest
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


meeting = load_module("meeting_transcribe", HERE / "meeting-transcribe")

sys.path.insert(0, str(HERE))  # meeting-job imports lockfile from next to itself
meeting_job = load_module("meeting_job", HERE / "meeting-job")


class WordRebuilding(unittest.TestCase):
    """The server's `text` is broken; `words[]` keeps the truth."""

    def test_continuation_pieces_are_joined(self):
        words = [{"word": " Für"}, {"word": " die"}, {"word": " In"}, {"word": "sel"}]
        self.assertEqual(meeting.rejoin(words), " Für die Insel")

    def test_subword_split_in_three(self):
        words = [{"word": " Hong"}, {"word": "k"}, {"word": "ong"}]
        self.assertEqual(meeting.rejoin(words), " Hongkong")

    def test_starts_mid_word_flags_a_continuation(self):
        self.assertTrue(meeting.starts_mid_word([{"word": "sel"}]))
        self.assertFalse(meeting.starts_mid_word([{"word": " verl"}]))

    def test_unwrap_only_collapses_whitespace(self):
        # Without words[] a break inside a word cannot be undone safely: joining
        # "Satz\nzweiter" would produce "Satzzweiter". rejoin() is the real fix.
        self.assertEqual(meeting.unwrap("die In\nsel verlassen"), "die In sel verlassen")

    def test_unwrap_turns_other_newlines_into_spaces(self):
        self.assertEqual(meeting.unwrap("erster Satz\nzweiter Satz"), "erster Satz zweiter Satz")

    def test_empty_word_list(self):
        self.assertEqual(meeting.rejoin([]), "")
        self.assertFalse(meeting.starts_mid_word([]))


class Fusion(unittest.TestCase):
    """The ASR owns the words; the diarizer only labels them."""

    def setUp(self):
        self.diar = [
            {"start": 0.0, "end": 10.0, "speaker": 7},
            {"start": 10.0, "end": 20.0, "speaker": 3},
        ]

    def test_largest_overlap_wins(self):
        asr = [{"start": 1.0, "end": 2.0, "text": "a"},
               {"start": 9.5, "end": 12.0, "text": "b"}]      # 2.0 s of speaker 3, 0.5 s of 7
        fused = meeting.fuse_segments(asr, self.diar)
        self.assertEqual([f["speaker"] for f in fused], [7, 3])

    def test_a_gap_inherits_the_previous_speaker(self):
        asr = [{"start": 1.0, "end": 2.0, "text": "a"},
               {"start": 30.0, "end": 31.0, "text": "b"}]     # past the diarizer's end
        fused = meeting.fuse_segments(asr, self.diar)
        self.assertEqual(fused[1]["speaker"], 7)
        self.assertEqual(fused[1]["text"], "b")               # no text lost

    def test_first_segment_takes_the_nearest_speaker(self):
        asr = [{"start": 25.0, "end": 26.0, "text": "a"}]     # before/after everything
        fused = meeting.fuse_segments(asr, self.diar)
        self.assertEqual(fused[0]["speaker"], 3)

    def test_no_diarizer_is_not_a_crash(self):
        fused = meeting.fuse_segments([{"start": 0.0, "end": 1.0, "text": "a"}], [])
        self.assertEqual(fused[0]["speaker"], 0)
        self.assertEqual(fused[0]["text"], "a")


class Turns(unittest.TestCase):

    def test_same_speaker_merges_within_the_gap(self):
        segs = [{"speaker": 1, "start": 0.0, "end": 2.0, "text": "hallo"},
                {"speaker": 1, "start": 3.0, "end": 4.0, "text": "welt"},
                {"speaker": 2, "start": 5.0, "end": 6.0, "text": "ja"}]
        turns = meeting.build_turns_labeled(segs, gap=2.0)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["text"], "hallo welt")

    def test_a_word_cut_across_segments_is_glued_without_a_space(self):
        segs = [{"speaker": 1, "start": 0.0, "end": 2.0, "text": "die In", "joins": False},
                {"speaker": 1, "start": 2.0, "end": 3.0, "text": "sel", "joins": True}]
        turns = meeting.build_turns_labeled(segs, gap=0.5)
        self.assertEqual(turns[0]["text"], "die Insel")

    def test_a_speaker_change_starts_a_new_turn(self):
        segs = [{"speaker": 1, "start": 0.0, "end": 2.0, "text": "a"},
                {"speaker": 2, "start": 2.0, "end": 3.0, "text": "b"}]
        self.assertEqual(len(meeting.build_turns_labeled(segs, gap=5.0)), 2)


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
    """The defaults are the pipeline's contract: they must stay on verbose_json."""

    def test_asr_endpoint_is_the_verbose_json_one(self):
        self.assertIn("8731", meeting.DEFAULT_ASR)

    def test_model_is_whisper(self):
        self.assertEqual(meeting.DEFAULT_ASR_MODEL, "Whisper-Large-v3-Turbo")

    def test_diarizer_is_nemotron(self):
        self.assertEqual(meeting.DEFAULT_DIARIZER, "nemotron")

    def test_asr_mode_defaults_to_full(self):
        source = (HERE / "meeting-transcribe").read_text(encoding="utf-8")
        self.assertIn('"--asr-mode", default="full"', source)


if __name__ == "__main__":
    unittest.main()


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

    def test_a_microphone_track_is_one_speaker_after_the_remote_ones(self):
        turns = [{"speaker": 0, "start": 1.0, "end": 2.0, "text": "a"},
                 {"speaker": 1, "start": 3.0, "end": 4.0, "text": "b"}]
        labelled = meeting.second_track_turns(turns, after=4, name="Me")
        self.assertEqual([t["speaker"] for t in labelled], [5, 5])
        self.assertEqual({t["name"] for t in labelled}, {"Me"})

    def test_a_diarized_second_track_numbsers_after_the_first(self):
        turns = [{"speaker": 0, "start": 1.0, "end": 2.0, "text": "a"},
                 {"speaker": 1, "start": 3.0, "end": 4.0, "text": "b"}]
        labelled = meeting.second_track_turns(turns, after=3, name="Room", diarized=True)
        self.assertEqual([t["speaker"] for t in labelled], [4, 5])


class JobSummaryFields(unittest.TestCase):
    """The list endpoint strips the transcript, so it must keep what the table shows."""

    def test_the_summary_keeps_length_tracks_and_whether_a_retry_can_run(self):
        job = meeting_job.Job("abc", {})
        summary = meeting_job.Handler._summary(job.snapshot())
        self.assertIn("duration", summary)
        self.assertIn("tracks", summary)
        self.assertIn("audio", summary)  # the retry button is only offered with audio
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
        self.jobs = meeting_job.Jobs(Path("/bin/true"), root, lock_path=root / "lock",
                                     pair_wait=10000)

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
        self.jobs = meeting_job.Jobs(Path("/bin/true"), root, lock_path=root / "lock",
                                     pair_wait=10000)
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


class RepetitionLoop(unittest.TestCase):
    """Whisper re-emits one sentence every second on audio it cannot decode."""

    @staticmethod
    def segment(start, end, text):
        return {"speaker": 1, "start": start, "end": end, "text": text}

    def test_a_long_run_of_the_same_sentence_is_dropped(self):
        loop = [self.segment(100 + i, 101 + i, "Ich denke, das ist doch nicht da.")
                for i in range(20)]
        others = [self.segment(0, 3, "Guten Morgen."), self.segment(200, 203, "Tschüss.")]
        kept = meeting.collapse_repetitions(others[:1] + loop + others[1:])
        self.assertEqual([s["text"] for s in kept], ["Guten Morgen.", "Tschüss."])

    def test_a_short_loop_is_dropped_too(self):
        loop = [self.segment(50 + i, 51 + i, "Ich habe noch einen Foto") for i in range(9)]
        self.assertEqual(meeting.collapse_repetitions(loop), [])

    def test_quick_repeats_of_a_short_word_are_kept(self):
        goodbye = [self.segment(10 + i, 11 + i, "Tschüss.") for i in range(5)]
        self.assertEqual(len(meeting.collapse_repetitions(goodbye)), 5)

    def test_a_few_repeats_of_a_sentence_are_kept(self):
        twice = [self.segment(10, 12, "Das ist auch nicht da."),
                 self.segment(12, 14, "Das ist auch nicht da."),
                 self.segment(14, 17, "Und dann weiter.")]
        self.assertEqual(len(meeting.collapse_repetitions(twice)), 3)

    def test_punctuation_and_case_do_not_hide_a_loop(self):
        loop = [self.segment(2 * i, 2 * i + 1.5, f"ja, das ist ja auch nicht da{i and '.' or '!'}")
                for i in range(6)]
        self.assertEqual(meeting.collapse_repetitions(loop), [])


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
