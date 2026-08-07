from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import av
import torch
import server


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from custom_nodes.company_remote import three_person_seedance_video as subject


class ThreePersonSeedanceVideoTests(unittest.TestCase):
    def test_default_segments_cover_the_484_frame_sample(self) -> None:
        segments = subject._parse_segments(subject.DEFAULT_SEGMENTS, frame_count=484)
        self.assertEqual([(item.start_frame, item.end_frame) for item in segments], [(0, 195), (196, 483)])

    def test_segments_must_be_contiguous_and_cover_every_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "连续开始"):
            subject._parse_segments('[{"start_frame": 0, "end_frame": 20}, {"start_frame": 22, "end_frame": 39}]', frame_count=40)
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            subject._parse_segments('[{"start_frame": 0, "end_frame": 38}]', frame_count=40)

    def test_request_duration_rounds_up_and_stays_in_seedance_range(self) -> None:
        first = subject._task_for_segment(subject.SeedanceSegment(1, 0, 195), fps=11616 / 493)
        second = subject._task_for_segment(subject.SeedanceSegment(2, 196, 483), fps=11616 / 493)
        self.assertEqual(first["request_duration"], 9)
        self.assertEqual(second["request_duration"], 13)
        self.assertAlmostEqual(first["source_duration"] + second["source_duration"], 20.541666, places=5)

    def test_too_short_segment_is_rejected_before_remote_submission(self) -> None:
        with self.assertRaisesRegex(ValueError, "4-15"):
            subject._task_for_segment(subject.SeedanceSegment(1, 0, 20), fps=24.0)

    def test_asset_upload_video_is_constant_24_fps(self) -> None:
        source_path = Path(__file__).resolve().parents[3] / "input" / "three_person_face_swap_source.mp4"
        source_video = subject.InputImpl.VideoFromFile(str(source_path))
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "asset_upload.mp4"
            upload_video, report = subject._prepare_asset_upload_video(source_video, output_path)

            self.assertEqual(report["fps"], 24.0)
            self.assertGreaterEqual(report["average_fps"], 23.8)
            self.assertLessEqual(report["average_fps"], 60.0)
            self.assertTrue(output_path.is_file())
            with av.open(upload_video.get_stream_source(), mode="r") as container:
                self.assertEqual(float(container.streams.video[0].base_rate), 24.0)

    def test_two_segment_pipeline_normalizes_merges_and_restores_audio_without_remote_request(self) -> None:
        source_path = Path(__file__).resolve().parents[3] / "input" / "three_person_face_swap_source.mp4"
        video = subject.InputImpl.VideoFromFile(str(source_path))
        image = torch.ones((1, 320, 320, 3), dtype=torch.float32)
        image_assets = iter(
            [
                ("asset-A", '{"asset_id":"asset-A"}', True),
                ("asset-B", '{"asset_id":"asset-B"}', True),
                ("asset-C", '{"asset_id":"asset-C"}', True),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            registered_video_fps = []

            def generate_from_registered_segment(**kwargs):
                source_asset_id = str(kwargs["source_video_asset_id"])
                segment_number = int(source_asset_id.rsplit("-", 1)[-1])
                request_files = sorted(output_root.glob("company_remote/three_person_seedance/*/requests/request_*.mp4"))
                request_path = next(path for path in request_files if path.stem.endswith(f"{segment_number:04d}"))
                report = json.dumps({"status": "completed", "task_id": f"task-{segment_number}"})
                callback = kwargs.get("submitted_callback")
                if callback:
                    callback(f"task-{segment_number}")
                return subject.InputImpl.VideoFromFile(str(request_path)), str(request_path), f"task-{segment_number}", report

            video_asset_counter = iter(range(1, 3))

            def register_video_asset(video, **_kwargs):
                with av.open(video.get_stream_source(), mode="r") as container:
                    stream = container.streams.video[0]
                    registered_video_fps.append(
                        {
                            "nominal": float(stream.base_rate),
                            "average": float(stream.average_rate),
                        }
                    )
                return f"asset-video-{next(video_asset_counter)}", '{"asset_type":"Video"}', False

            with (
                mock.patch.object(subject.folder_paths, "get_output_directory", return_value=str(output_root)),
                mock.patch.object(subject, "create_seedance_image_asset", side_effect=lambda *_args, **_kwargs: next(image_assets)),
                mock.patch.object(
                    subject,
                    "create_seedance_video_asset",
                    side_effect=register_video_asset,
                ),
                mock.patch.object(subject, "generate_three_person_asset_video", side_effect=generate_from_registered_segment),
            ):
                _result_video, final_path, report_text = subject.generate_three_person_seedance_video(
                    video,
                    image,
                    image,
                    image,
                    segments_json=subject.DEFAULT_SEGMENTS,
                    prompt="test",
                    model="doubao-seedance-2-0-fast-260128",
                    resolution="720p",
                    ratio="adaptive",
                    reuse_cached_assets=True,
                    force_rerun_segments=False,
                    watermark=False,
                    seed=0,
                )

            report = json.loads(report_text)
            self.assertEqual(report["status"], "success")
            self.assertEqual([item["request_duration"] for item in report["segments"]], [9, 13])
            self.assertEqual([item["nominal"] for item in registered_video_fps], [24.0, 24.0])
            self.assertTrue(all(23.8 <= item["average"] <= 60.0 for item in registered_video_fps))
            self.assertEqual([item["asset_upload"]["fps"] for item in report["segments"]], [24.0, 24.0])
            with av.open(final_path, mode="r") as container:
                self.assertTrue(container.streams.audio)
                stream = container.streams.video[0]
                self.assertEqual(stream.frames, 484)
                self.assertAlmostEqual(float(stream.duration * stream.time_base), 20.541666, places=3)

    def test_replacement_workflow_uses_seedance_node_and_disables_paid_run_by_default(self) -> None:
        workflow_path = (
            Path(__file__).resolve().parents[3]
            / "user"
            / "default"
            / "workflows"
            / "三人物ABC_按镜头流式视频换脸_CPU_20秒样片.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node["type"] for node in workflow["nodes"]}
        self.assertIn("CompanyThreePersonSeedanceVideo", node_types)
        self.assertNotIn("CompanyThreePersonFaceSwapVideo", node_types)
        seedance = next(node for node in workflow["nodes"] if node["type"] == "CompanyThreePersonSeedanceVideo")
        saver = next(node for node in workflow["nodes"] if node["type"] == "SaveVideo")
        self.assertEqual(seedance["mode"], 4)
        self.assertEqual(saver["mode"], 4)
        self.assertEqual(seedance["widgets_values"][2], "doubao-seedance-2-0-fast-260128")
        self.assertEqual(seedance["widgets_values"][3], "720p")


if __name__ == "__main__":
    unittest.main()
