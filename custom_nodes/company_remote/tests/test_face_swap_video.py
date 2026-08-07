from __future__ import annotations

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import server


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from custom_nodes.company_remote.face_swap_video import (
    DEFAULT_SCENE_MAPPING,
    _assign_identity_faces,
    _encode_frames,
    _mux_original_audio,
    _parse_scene_mapping,
    _rule_for_frame,
)


class FaceSwapVideoTests(unittest.TestCase):
    def test_default_mapping_covers_known_video_frames(self) -> None:
        rules = _parse_scene_mapping(DEFAULT_SCENE_MAPPING)
        self.assertEqual(len(rules), 6)
        self.assertEqual(_rule_for_frame(rules, 0).characters, {"A": 0, "B": 1})
        self.assertEqual(_rule_for_frame(rules, 21).characters, {"C": 0})
        self.assertEqual(_rule_for_frame(rules, 483).characters, {"C": 0})

    def test_mapping_rejects_overlapping_ranges(self) -> None:
        value = json.dumps(
            [
                {"start_frame": 0, "end_frame": 10, "characters": {"A": 0}},
                {"start_frame": 10, "end_frame": 20, "characters": {"B": 0}},
            ]
        )
        with self.assertRaisesRegex(ValueError, "重叠"):
            _parse_scene_mapping(value)

    def test_unmapped_frame_is_left_unchanged(self) -> None:
        rules = _parse_scene_mapping(
            json.dumps([{"start_frame": 2, "end_frame": 5, "characters": {"A": 0}}])
        )
        self.assertIsNone(_rule_for_frame(rules, 1))

    def test_identity_assignment_handles_single_remaining_face(self) -> None:
        face_b = SimpleNamespace(normed_embedding=np.array([0.0, 1.0], dtype=np.float32))
        assignments = _assign_identity_faces(
            [face_b],
            ["A", "B"],
            {
                "A": np.array([1.0, 0.0], dtype=np.float32),
                "B": np.array([0.0, 1.0], dtype=np.float32),
            },
            threshold=0.25,
        )
        self.assertEqual(list(assignments), ["B"])
        self.assertIs(assignments["B"], face_b)

    def test_encode_frames_builds_monotonic_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.mp4"
            output_path = Path(directory) / "output.mp4"
            with av.open(str(source_path), mode="w") as output:
                stream = output.add_stream("libx264", rate=24)
                stream.width = 64
                stream.height = 64
                stream.pix_fmt = "yuv420p"
                for index in range(19):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((64, 64, 3), index, dtype=np.uint8),
                        format="rgb24",
                    )
                    # Deliberately use a source time base that does not match the
                    # encoder, reproducing the real smoke-test input conditions.
                    frame.pts = index * 512
                    frame.time_base = Fraction(1, 12288)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)

            stats = _encode_frames(
                str(source_path),
                output_path,
                lambda _index, frame: (frame, ["A"]),
            )

            with av.open(str(output_path), mode="r") as result:
                frames = list(result.decode(video=0))
            self.assertEqual(stats["frames_processed"], 19)
            self.assertEqual(stats["frames_with_swaps"], 19)
            self.assertEqual(len(frames), 19)
            self.assertEqual([frame.pts for frame in frames], sorted({frame.pts for frame in frames}))

    def test_audio_mux_does_not_truncate_video_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            silent_path = Path(directory) / "silent.mp4"
            source_path = Path(directory) / "short_audio.mp4"
            output_path = Path(directory) / "muxed.mp4"

            with av.open(str(silent_path), mode="w") as output:
                stream = output.add_stream("libx264", rate=24)
                stream.width = 64
                stream.height = 64
                stream.pix_fmt = "yuv420p"
                for index in range(19):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((64, 64, 3), index, dtype=np.uint8),
                        format="rgb24",
                    )
                    frame.pts = index
                    frame.time_base = Fraction(1, 24)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)

            with av.open(str(source_path), mode="w") as output:
                stream = output.add_stream("aac", rate=48_000)
                stream.layout = "stereo"
                for index in range(20):
                    frame = av.AudioFrame(format="fltp", layout="stereo", samples=1024)
                    frame.sample_rate = 48_000
                    frame.pts = index * 1024
                    frame.time_base = Fraction(1, 48_000)
                    frame.planes[0].update(bytes(frame.planes[0].buffer_size))
                    frame.planes[1].update(bytes(frame.planes[1].buffer_size))
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)

            _mux_original_audio(silent_path, str(source_path), output_path)

            with av.open(str(output_path), mode="r") as result:
                frames = list(result.decode(video=0))
                audio_streams = len(result.streams.audio)
            self.assertEqual(len(frames), 19)
            self.assertEqual(audio_streams, 1)


if __name__ == "__main__":
    unittest.main()
