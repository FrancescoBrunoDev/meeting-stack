#!/usr/bin/env python3
"""Tests for the parts of meeting-transcribe that decide the transcript.

The pipeline talks to two external services (an ASR endpoint and a diarizer
container), so what is tested here is everything that can be wrong without them:
rebuilding words the server split, fusing ASR segments with speaker labels,
merging turns, and the defaults that decide which path runs.
"""
import io
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
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
    """The defaults are the pipeline's contract: they must stay on verbose_json."""

    def test_asr_endpoint_is_the_verbose_json_one(self):
        self.assertIn("8731", meeting.DEFAULT_ASR)

    def test_model_is_whisper(self):
        self.assertEqual(meeting.DEFAULT_ASR_MODEL, "Whisper-Large-v3-Turbo")

    def test_diarizer_is_nemotron(self):
        self.assertEqual(meeting.DEFAULT_DIARIZER, "nemotron")

    def test_audio_cpp_answers_are_split_on_their_own_markers(self):
        turns = meeting.parse_diarized(
            "[1.74][S01] Hallo zusammen.[2.61][3.06][S02] Hallo.[3.51][11.10][S02] Morgen.[12.01]")
        self.assertEqual([(turn["speaker"], turn["start"], turn["end"], turn["text"])
                          for turn in turns],
                         [(1, 1.74, 2.61, "Hallo zusammen."), (2, 3.06, 3.51, "Hallo."),
                          (2, 11.1, 12.01, "Morgen.")])

    def test_an_empty_audio_cpp_answer_is_no_turns(self):
        self.assertEqual(meeting.parse_diarized(""), [])

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

    def test_padding_never_shrinks_an_overlapping_turn(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 5.0, "text": "a"},
                 {"speaker": 2, "start": 4.0, "end": 8.0, "text": "b"}]
        padded = meeting.pad_turns(turns, 0.3)
        self.assertTrue(all(t["end"] >= t["start"] for t in padded))

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

    def test_the_mic_track_is_diarized_when_the_asr_has_no_verbose_json(self):
        # gufo answers text only, so the per-turn path needs the diarizer's edges.
        self.assertTrue(meeting.mic_is_diarized("segments", False))
        self.assertFalse(meeting.mic_is_diarized("full", False))
        self.assertTrue(meeting.mic_is_diarized("full", True))

    def test_a_silent_mic_track_is_not_a_failed_job(self):
        args = types.SimpleNamespace(
            diarizer="nemotron", speakers=-1, threshold=0.5, diarizer_timeout=10,
            min_speakers=-1, max_speakers=-1, asr_mode="segments", model="m",
            language=None, asr_timeout=10, max_segment=150.0, merge_gap=2.0)
        original = meeting.diarize
        meeting.diarize = lambda *a, **k: []
        try:
            with tempfile.TemporaryDirectory() as directory:
                turns = meeting.track_turns(args, "http://x", Path(directory) / "mic.mp3",
                                            Path(directory) / "work", "", lenient=True)
                self.assertEqual(turns, [])
        finally:
            meeting.diarize = original


class ExternalDiarization(unittest.TestCase):
    """A perfect reference (pyannote precision-3) must replace the guessed labels."""

    def write(self, payload) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "ref.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_pyannote_file_and_a_plain_list_are_both_read(self):
        segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        self.assertEqual(meeting.load_diarization(self.write({"diarization": segments})), segments)
        self.assertEqual(meeting.load_diarization(self.write(segments)), segments)

    def test_every_turn_takes_the_reference_speaker_it_overlaps(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 4.0, "text": "a"},
                 {"speaker": 4, "start": 4.0, "end": 8.0, "text": "b"},
                 {"speaker": 2, "start": 8.0, "end": 9.0, "text": "c", "name": "Me"}]
        reference = [{"start": 0.0, "end": 3.5, "speaker": "SPEAKER_01"},
                     {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_02"},
                     {"start": 8.2, "end": 9.0, "speaker": "SPEAKER_00"}]
        relabelled = meeting.relabel_turns(turns, reference)
        self.assertEqual([t["speaker"] for t in relabelled], [1, 2, 3])
        self.assertTrue(all(t.get("name") is None for t in relabelled))

    def test_a_turn_in_a_reference_gap_keeps_the_previous_speaker(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 2.0, "text": "a"},
                 {"speaker": 2, "start": 3.0, "end": 4.0, "text": "b"}]
        reference = [{"start": 0.0, "end": 2.0, "speaker": "S0"},
                     {"start": 5.0, "end": 6.0, "speaker": "S1"}]
        relabelled = meeting.relabel_turns(turns, reference)
        self.assertEqual([t["speaker"] for t in relabelled], [1, 1])

    def test_an_empty_reference_changes_nothing(self):
        turns = [{"speaker": 1, "start": 0.0, "end": 2.0, "text": "a"}]
        self.assertEqual(meeting.relabel_turns(turns, []), turns)


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


class Persistence(unittest.TestCase):
    """A restart empties the queue; the list, the transcript and retry must survive."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def start(self, **kwargs):
        jobs = meeting_job.Jobs(Path("/bin/true"), self.root, lock_path=self.root / "lock",
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
        jobs = meeting_job.Jobs(Path("/bin/true"), self.root, lock_path=self.root / "lock",
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
        def transcribe(pcm, language):
            rows.append(("asr", len(pcm)))
            return text

        def translate(src, target, context, on_token=None):
            rows.append(("mt", target, list(context)))
            if on_token is not None:
                on_token(translated[:4])
            return translated

        return types.SimpleNamespace(transcribe=transcribe, translate=translate,
                                     settled=lambda session: None,
                                     embedder="", speaker=meeting_job.LIVE_SPEAKER)

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

        def translate(src, target, context, on_token=None):
            on_token("Buon")
            seen.extend(session.view(0)["rows"])
            return "Buongiorno"

        fake = types.SimpleNamespace(transcribe=lambda pcm, language: "Guten Tag.",
                                     translate=translate, settled=lambda session: None,
                                     embedder="", speaker=meeting_job.LIVE_SPEAKER)
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
        def transcribe(pcm, language):
            raise OSError("connection refused")

        session = meeting_job.LiveSession(
            types.SimpleNamespace(transcribe=transcribe, translate=lambda *a: "",
                                  settled=lambda session: None,
                                     embedder="", speaker=meeting_job.LIVE_SPEAKER), "de", "it")
        session.feed(self.speech() + b"\x00\x00" * 16000)
        session.close()
        self.drain(session)
        self.assertIn("connection refused", session.rows[0]["error"])

    def test_a_second_session_waits_for_the_lock(self):
        with tempfile.TemporaryDirectory() as root:
            lock = Path(root) / "gpu.lock"
            holder = meeting_job.Lock(lock)
            self.assertTrue(holder.acquire(0))
            live = meeting_job.Live("http://127.0.0.1:9", "m", "http://127.0.0.1:9", "", lock)
            self.assertIsNone(live.feed("a", b"\x00\x00" * 1000, "de", "it"))
            holder.release()
            session = live.feed("a", b"\x00\x00" * 1000, "de", "it")
            self.assertIsNotNone(session)
            session.close()

    def test_stopping_frees_the_gpu_for_the_next_session(self):
        # The lock is released when the worker drains, not on the next reaper
        # tick: otherwise "Ferma" and "start again" five seconds later is a 409.
        with tempfile.TemporaryDirectory() as root:
            live = meeting_job.Live("http://127.0.0.1:9", "m", "http://127.0.0.1:9", "",
                                    Path(root) / "gpu.lock")
            first = live.feed("a", b"\x00\x00" * 1000, "de", "it")
            self.assertIsNotNone(first)
            first.close()
            self.drain(first)
            second = live.feed("b", b"\x00\x00" * 1000, "de", "it")
            self.assertIsNotNone(second, "the GPU lock was still held after the drain")
            second.close()

class LiveLines(unittest.TestCase):
    """A line is a stretch of speech between two pauses, not one ASR clip."""

    def drain(self, session, limit=5.0):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline and not session.done:
            time.sleep(0.02)
        self.assertTrue(session.done)

    def fake(self, calls):
        return types.SimpleNamespace(
            transcribe=lambda pcm, language: "",
            translate=lambda src, target, context, on_token=None: (
                calls.append(src), "tradotto")[1],
            settled=lambda session: None,
                                     embedder="", speaker=meeting_job.LIVE_SPEAKER)

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

        def translate(src, target, context, on_token=None):
            calls.append(list(context))
            return next(answers)

        session = meeting_job.LiveSession(
            types.SimpleNamespace(transcribe=lambda pcm, language: "", translate=translate,
                                  settled=lambda session: None,
                                     embedder="", speaker=meeting_job.LIVE_SPEAKER), "de", "it")
        session._hear("Heute im Studio.", closed=True)
        session._hear("Ein schönen guten Abend.", closed=True)
        self.assertEqual(len(calls), 3, "one retry, without the context")
        self.assertEqual(calls[-1], [])
        self.assertEqual(session.rows[1]["error"], "the model returned no translation")

    def test_the_cursor_does_not_move_while_a_line_is_streaming(self):
        # `mt` holds the first token long before the line is finished: a cursor
        # that keys on it moves on, and the page keeps "aveva affid" for ever.
        cursors = []

        def translate(src, target, context, on_token=None):
            on_token("aveva")
            cursors.append(session.view(0)["cursor"])
            on_token("aveva affidato al suo agente pubblicitario Högel.")
            cursors.append(session.view(0)["cursor"])
            return "aveva affidato al suo agente pubblicitario Högel."

        session = meeting_job.LiveSession(
            types.SimpleNamespace(transcribe=lambda pcm, language: "",
                                  translate=translate, settled=lambda session: None,
                                     embedder="", speaker=meeting_job.LIVE_SPEAKER), "de", "it")
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
            transcribe=lambda pcm, language: "", settled=lambda session: None,
            translate=lambda src, target, context, on_token=None: (
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

    def test_a_change_of_voice_ends_the_line_without_a_pause(self):
        # Two people talking without pausing: only the voice tells them apart.
        calls = []
        voices = iter([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        clips = iter(["Guten Abend, ich begrüsse Sie.", "Bei der Bundeswehr gibt es mehr Gewalt."])
        fake = types.SimpleNamespace(
            transcribe=lambda pcm, language: next(clips),
            translate=lambda src, target, context, on_token=None: (
                calls.append(src), "tradotto")[1],
            embed=lambda pcm: next(voices),
            settled=lambda session: None, embedder="img", speaker=meeting_job.LIVE_SPEAKER)
        session = meeting_job.LiveSession(fake, "de", "it")
        speech = (4000).to_bytes(2, "little", signed=True) * 16000
        session._queue.put((speech, False))       # nobody pauses
        session._queue.put((speech, False))
        session._queue.put(None)
        self.drain(session)
        self.assertEqual([row["src"] for row in session.rows],
                         ["Guten Abend, ich begrüsse Sie.",
                          "Bei der Bundeswehr gibt es mehr Gewalt."])
        self.assertEqual(len(calls), 2, "the first line was closed by the voice change")

    def test_the_same_voice_keeps_one_line(self):
        calls = []
        fake = types.SimpleNamespace(
            transcribe=lambda pcm, language: "und das ist wichtig.",
            translate=lambda src, target, context, on_token=None: (
                calls.append(src), "tradotto")[1],
            embed=lambda pcm: [1.0, 0.0, 0.0],
            settled=lambda session: None, embedder="img", speaker=meeting_job.LIVE_SPEAKER)
        session = meeting_job.LiveSession(fake, "de", "it")
        speech = (4000).to_bytes(2, "little", signed=True) * 16000
        for _ in range(2):
            session._queue.put((speech, False))
        session._queue.put(None)
        self.drain(session)
        self.assertEqual(len(session.rows), 1)
        self.assertEqual(calls, ["und das ist wichtig. und das ist wichtig."])

    def test_silence_closes_the_line_without_waiting_for_the_asr(self):
        # Clips are cut at 0.45 s so the ASR sees short, cheap requests; the line
        # is closed by silence that keeps going, or by a change of voice.
        calls = []
        fake = types.SimpleNamespace(
            transcribe=lambda pcm, language: "Guten Abend.",
            translate=lambda src, target, context, on_token=None: (
                calls.append(src), "Buonasera.")[1],
            settled=lambda session: None, embedder="", speaker=meeting_job.LIVE_SPEAKER)
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
