#!/usr/bin/env python3
"""Tests for the parts of meeting-transcribe that decide the transcript.

The pipeline talks to two external services (an ASR endpoint and a diarizer
container), so what is tested here is everything that can be wrong without them:
rebuilding words the server split, fusing ASR segments with speaker labels,
merging turns, and the defaults that decide which path runs.
"""
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


meeting = load_module("meeting_transcribe", HERE / "meeting-transcribe")


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
