from __future__ import annotations

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import av
import server
import torch


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from custom_nodes.company_remote import nodes
from custom_nodes.company_remote import client
from custom_nodes.company_remote.config_store import RemoteMediaConfig
from custom_nodes.company_remote import three_person_wan27_video as full_video


class Wan27ThreeImageDirectTests(unittest.TestCase):
    def test_base64_image_delivery_does_not_call_tos(self) -> None:
        config = RemoteMediaConfig(
            name="aliyun_dashscope_video_direct",
            base_url="https://dashscope.aliyuncs.com",
            media_delivery="base64",
            tos_enabled=False,
        )
        image = np.zeros((1, 240, 240, 3), dtype=np.float32)
        with mock.patch.object(client, "_upload_media_to_tos") as upload:
            url = client._image_to_url(image, config, role="reference_image_1")

        self.assertTrue(url.startswith("data:image/png;base64,"))
        upload.assert_not_called()

    def test_direct_node_forces_base64_without_mutating_saved_config(self) -> None:
        config = SimpleNamespace(tos_enabled=True, media_delivery="tos_presigned")
        with (
            mock.patch.object(nodes, "_load_provider_config", return_value=config) as load_config,
            mock.patch.object(nodes, "generate_dashscope_video", return_value=(object(), "/tmp/result.mp4")) as generate,
        ):
            result = nodes.CompanyWan27ThreeImageDirectVideo.execute(
                object(), object(), object(), "test prompt", "wan2.7-r2v-2026-06-12", "720P", "16:9", 10
            )

        self.assertEqual(result[1], "/tmp/result.mp4")
        load_config.assert_called_once_with("aliyun_dashscope_video_direct")
        submitted_config = generate.call_args.args[0]
        kwargs = generate.call_args.kwargs
        self.assertFalse(submitted_config.tos_enabled)
        self.assertEqual(submitted_config.media_delivery, "base64")
        self.assertEqual(len(kwargs["reference_images"]), 3)
        self.assertEqual(kwargs["reference_videos"], [])
        self.assertTrue(config.tos_enabled)
        self.assertEqual(config.media_delivery, "tos_presigned")

    def test_video_edit_node_uses_temp_oss_header_and_three_base64_images(self) -> None:
        config = SimpleNamespace(
            tos_enabled=True,
            media_delivery="tos_presigned",
            extra_headers={"X-DashScope-Async": "enable"},
        )
        with (
            mock.patch.object(nodes, "_load_provider_config", return_value=config) as load_config,
            mock.patch.object(nodes, "generate_dashscope_video", return_value=(object(), "/tmp/edited.mp4")) as generate,
        ):
            result = nodes.CompanyWan27ThreePersonVideoEdit.execute(
                object(), object(), object(), object(), "replace A/B/C", "wan2.7-videoedit", "720P"
            )

        self.assertEqual(result[1], "/tmp/edited.mp4")
        load_config.assert_called_once_with("aliyun_dashscope_video_direct")
        submitted_config = generate.call_args.args[0]
        kwargs = generate.call_args.kwargs
        self.assertFalse(submitted_config.tos_enabled)
        self.assertEqual(submitted_config.media_delivery, "base64")
        self.assertEqual(submitted_config.extra_headers["X-DashScope-OssResourceResolve"], "enable")
        self.assertEqual(kwargs["operation"], "dashscope_video_edit")
        self.assertEqual(kwargs["model"], "wan2.7-videoedit")
        self.assertEqual(len(kwargs["reference_images"]), 3)
        self.assertEqual(kwargs["duration"], 0)
        self.assertEqual(kwargs["audio_setting"], "origin")
        self.assertTrue(config.tos_enabled)

    def test_dashscope_temp_oss_upload_uses_model_bound_policy(self) -> None:
        config = RemoteMediaConfig(
            name="aliyun_dashscope_video_direct",
            base_url="https://dashscope.aliyuncs.com",
            api_key="secret",
            timeout_seconds=30,
        )
        policy_response = mock.Mock(
            ok=True,
            json=lambda: {
                "data": {
                    "upload_dir": "dashscope-instant/account/day/token",
                    "upload_host": "https://dashscope-file.example.com",
                    "oss_access_key_id": "ak",
                    "signature": "signature",
                    "policy": "policy",
                    "x_oss_object_acl": "private",
                    "x_oss_forbid_overwrite": "true",
                }
            },
        )
        upload_response = mock.Mock(status_code=200)
        with mock.patch.object(client, "_request_raw", side_effect=[policy_response, upload_response]) as request:
            url, key = client._upload_media_to_dashscope_temporary_oss(
                config,
                content=b"video",
                mime="video/mp4",
                extension=".mp4",
                role="edit_video",
                model="wan2.7-videoedit",
            )

        self.assertEqual(url, f"oss://{key}")
        policy_call, upload_call = request.call_args_list
        self.assertEqual(policy_call.kwargs["params"]["model"], "wan2.7-videoedit")
        self.assertEqual(upload_call.args[:2], ("POST", "https://dashscope-file.example.com"))
        self.assertIn("file", upload_call.kwargs["files"])

    def test_workflow_is_ready_to_run_and_uses_video_edit_node(self) -> None:
        path = Path(__file__).resolve().parents[3] / "user" / "default" / "workflows" / "Wan2.7_三人物参考图直传_无TOS无火山_10秒验证.json"
        workflow = json.loads(path.read_text(encoding="utf-8"))
        source = next(node for node in workflow["nodes"] if node["type"] == "LoadVideo")
        direct = next(node for node in workflow["nodes"] if node["type"] == "CompanyWan27ThreePersonVideoEdit")
        saver = next(node for node in workflow["nodes"] if node["type"] == "SaveVideo")
        self.assertEqual(source["widgets_values"][0], "three_person_asset_gateway_test_10s_24fps.mp4")
        self.assertEqual(direct["mode"], 0)
        self.assertEqual(saver["mode"], 0)
        self.assertEqual(direct["widgets_values"][1], "wan2.7-videoedit")
        self.assertEqual(direct["widgets_values"][2:5], ["720P", 0, "origin"])
        self.assertEqual(len([node for node in workflow["nodes"] if node["type"] == "LoadImage"]), 3)

    def test_full_video_segments_cover_all_484_frames_within_wan_limit(self) -> None:
        fps = 11616 / 493
        segments = full_video._parse_segments(full_video.DEFAULT_WAN27_SEGMENTS, frame_count=484, fps=fps)

        self.assertEqual(
            [(segment.start_frame, segment.end_frame) for segment in segments],
            [(0, 195), (196, 373), (374, 483)],
        )
        durations = [(segment.end_frame - segment.start_frame + 1) / fps for segment in segments]
        self.assertTrue(all(2 <= duration <= 10 for duration in durations))
        self.assertAlmostEqual(sum(durations), 20.541666, places=5)

    def test_full_video_pipeline_merges_484_frames_and_restores_audio_without_remote_request(self) -> None:
        source_path = Path(__file__).resolve().parents[3] / "input" / "three_person_face_swap_source.mp4"
        video = full_video.InputImpl.VideoFromFile(str(source_path))
        image = torch.ones((1, 320, 320, 3), dtype=torch.float32)
        submitted = []

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)

            def fake_generate(video_input, *_args, seed, **_kwargs):
                path = Path(video_input.get_stream_source())
                with av.open(str(path), mode="r") as container:
                    stream = container.streams.video[0]
                    submitted.append(
                        {
                            "duration": float(stream.duration * stream.time_base),
                            "fps": float(stream.base_rate),
                            "seed": seed,
                        }
                    )
                return full_video.InputImpl.VideoFromFile(str(path)), str(path)

            with (
                mock.patch.object(full_video.folder_paths, "get_output_directory", return_value=str(output_root)),
                mock.patch.object(full_video, "_generate_wan27_segment", side_effect=fake_generate),
            ):
                _result_video, final_path, report_text = full_video.generate_three_person_wan27_full_video(
                    video,
                    image,
                    image,
                    image,
                    segments_json=full_video.DEFAULT_WAN27_SEGMENTS,
                    prompt="replace A/B/C",
                    resolution="720P",
                    reuse_completed_segments=True,
                    force_rerun_segments=False,
                    prompt_extend=True,
                    watermark=False,
                    seed=17,
                    negative_prompt="no identity swap",
                )

            report = json.loads(report_text)
            self.assertEqual(report["status"], "success")
            self.assertEqual(len(submitted), 3)
            self.assertEqual([item["seed"] for item in submitted], [17, 17, 17])
            self.assertTrue(all(2 <= item["duration"] <= 10 for item in submitted))
            self.assertEqual([item["fps"] for item in submitted], [24.0, 24.0, 24.0])
            with av.open(final_path, mode="r") as container:
                self.assertTrue(container.streams.audio)
                stream = container.streams.video[0]
                self.assertEqual(stream.frames, 484)
                self.assertEqual(Fraction(stream.average_rate), Fraction(11616, 493))
                self.assertAlmostEqual(float(stream.duration * stream.time_base), 20.541666, places=3)

    def test_visible_split_and_merge_pipeline_keeps_484_frames_and_original_audio(self) -> None:
        source_path = Path(__file__).resolve().parents[3] / "input" / "three_person_face_swap_source.mp4"
        source_video = full_video.InputImpl.VideoFromFile(str(source_path))

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(full_video.folder_paths, "get_output_directory", return_value=directory):
                segment_1, segment_2, segment_3, split_report_text = (
                    full_video.split_three_person_wan27_full_video(
                        source_video,
                        segments_json=full_video.DEFAULT_WAN27_SEGMENTS,
                    )
                )
                _video, final_path, merge_report_text = full_video.merge_three_person_wan27_segments(
                    source_video,
                    segment_1,
                    segment_2,
                    segment_3,
                    segments_json=full_video.DEFAULT_WAN27_SEGMENTS,
                )

            split_report = json.loads(split_report_text)
            merge_report = json.loads(merge_report_text)
            self.assertEqual(split_report["stage"], "split_only_no_remote_request")
            self.assertEqual(len(split_report["segments"]), 3)
            self.assertEqual(merge_report["stage"], "normalize_merge_restore_audio")
            with av.open(final_path, mode="r") as container:
                self.assertTrue(container.streams.audio)
                self.assertEqual(container.streams.video[0].frames, 484)

    def test_full_workflow_exposes_three_paid_branches_and_merge(self) -> None:
        path = (
            Path(__file__).resolve().parents[3]
            / "user"
            / "default"
            / "workflows"
            / "Wan2.7_三人物参考图直传_无TOS无火山_20秒完整处理.json"
        )
        workflow = json.loads(path.read_text(encoding="utf-8"))
        source = next(node for node in workflow["nodes"] if node["type"] == "LoadVideo")
        split = next(node for node in workflow["nodes"] if node["type"] == "CompanyWan27SplitThreeSegments")
        processors = [node for node in workflow["nodes"] if node["type"] == "CompanyWan27ThreePersonVideoEdit"]
        merge = next(node for node in workflow["nodes"] if node["type"] == "CompanyWan27MergeThreeSegments")
        savers = [node for node in workflow["nodes"] if node["type"] == "SaveVideo"]

        self.assertEqual(source["widgets_values"][0], "three_person_face_swap_source.mp4")
        self.assertIn('"end_frame": 483', split["widgets_values"][0])
        self.assertEqual(len(processors), 3)
        self.assertTrue(all(processor["mode"] == 0 for processor in processors))
        self.assertTrue(all(processor["widgets_values"][1:5] == ["wan2.7-videoedit", "720P", 0, "origin"] for processor in processors))
        self.assertEqual(len(savers), 4)
        self.assertEqual(merge["mode"], 0)
        self.assertFalse(any(node["type"] == "CompanyWan27ThreePersonFullVideo" for node in workflow["nodes"]))
        self.assertEqual(len(workflow["nodes"]), 13)
        self.assertEqual(len(workflow["links"]), 21)


if __name__ == "__main__":
    unittest.main()
