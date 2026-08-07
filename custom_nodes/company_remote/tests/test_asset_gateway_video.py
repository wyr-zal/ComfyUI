from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import server


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from custom_nodes.company_remote import asset_gateway_video


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        base_url="http://asset-gateway.test",
        timeout_seconds=30,
        max_poll_attempts=3,
        poll_interval_seconds=1,
        auth_header="Authorization",
        auth_prefix="Bearer",
        extra_headers={},
        get_api_key=lambda: "test-key",
    )


class AssetGatewayVideoTests(unittest.TestCase):
    def test_workflow_preserves_widget_slots_for_linked_asset_strings(self) -> None:
        workflow_path = (
            Path(__file__).resolve().parents[3]
            / "user"
            / "default"
            / "workflows"
            / "三人物ABC_火山资源ID_Seedance转换_10秒验证.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node = next(
            item
            for item in workflow["nodes"]
            if item["type"] == "CompanySeedanceAssetGatewayThreePersonVideo"
        )
        self.assertEqual(node["widgets_values"][:4], ["", "", "", ""])
        self.assertEqual(node["widgets_values"][5:9], [
            "doubao-seedance-2-0-fast-260128",
            "480p",
            "adaptive",
            10,
        ])

    def test_payload_uses_asset_uris_in_metadata_content(self) -> None:
        payload = asset_gateway_video.build_three_person_video_payload(
            source_video_asset_id="asset-source",
            character_a_asset_id="asset-A",
            character_b_asset_id="asset-B",
            character_c_asset_id="asset-C",
            prompt="保留源视频，替换三个人物。",
            model="doubao-seedance-2-0-fast-260128",
            resolution="480p",
            ratio="adaptive",
            duration=10,
            generate_audio=False,
            watermark=False,
            seed=7,
        )

        self.assertEqual(payload["seconds"], "10")
        self.assertEqual(payload["metadata"]["resolution"], "480p")
        content = payload["metadata"]["content"]
        self.assertEqual(content[0]["video_url"]["url"], "asset://asset-source")
        self.assertEqual(
            [item["image_url"]["url"] for item in content[1:]],
            ["asset://asset-A", "asset://asset-B", "asset://asset-C"],
        )
        self.assertEqual([item["role"] for item in content], [
            "reference_video",
            "reference_image",
            "reference_image",
            "reference_image",
        ])

    def test_payload_rejects_invalid_asset_id_and_duration(self) -> None:
        values = {
            "source_video_asset_id": "asset-source",
            "character_a_asset_id": "asset-A",
            "character_b_asset_id": "asset-B",
            "character_c_asset_id": "asset-C",
            "prompt": "test",
            "model": "doubao-seedance-2-0-fast-260128",
            "resolution": "480p",
            "ratio": "adaptive",
            "duration": 10,
            "generate_audio": False,
            "watermark": False,
            "seed": 0,
        }
        with self.assertRaisesRegex(Exception, "asset-"):
            asset_gateway_video.build_three_person_video_payload(
                **{**values, "character_b_asset_id": "invalid"}
            )
        with self.assertRaisesRegex(Exception, "4 到 15"):
            asset_gateway_video.build_three_person_video_payload(**{**values, "duration": 16})

    def test_resume_task_skips_submit(self) -> None:
        result_video = object()
        with (
            mock.patch.object(asset_gateway_video, "get_config", return_value=_config()),
            mock.patch.object(asset_gateway_video, "_json_request") as request,
            mock.patch.object(
                asset_gateway_video,
                "_poll_task",
                return_value=("https://result.test/video.mp4", {"status": "completed"}, 2),
            ),
            mock.patch.object(asset_gateway_video, "_download_video", return_value="/tmp/video.mp4"),
            mock.patch.object(asset_gateway_video.InputImpl, "VideoFromFile", return_value=result_video),
        ):
            result = asset_gateway_video.generate_three_person_asset_video(
                source_video_asset_id="asset-source",
                character_a_asset_id="asset-A",
                character_b_asset_id="asset-B",
                character_c_asset_id="asset-C",
                prompt="test",
                model="doubao-seedance-2-0-fast-260128",
                resolution="480p",
                ratio="adaptive",
                duration=10,
                generate_audio=False,
                watermark=False,
                seed=0,
                resume_task_id="task-existing",
            )

        self.assertIs(result[0], result_video)
        self.assertEqual(result[2], "task-existing")
        self.assertTrue(json.loads(result[3])["resumed"])
        request.assert_not_called()

    def test_resume_task_still_validates_asset_ids(self) -> None:
        with mock.patch.object(asset_gateway_video, "get_config", return_value=_config()):
            with self.assertRaisesRegex(Exception, "asset-"):
                asset_gateway_video.generate_three_person_asset_video(
                    source_video_asset_id="invalid",
                    character_a_asset_id="asset-A",
                    character_b_asset_id="asset-B",
                    character_c_asset_id="asset-C",
                    prompt="test",
                    model="doubao-seedance-2-0-fast-260128",
                    resolution="480p",
                    ratio="adaptive",
                    duration=10,
                    generate_audio=False,
                    watermark=False,
                    seed=0,
                    resume_task_id="task-existing",
                )


if __name__ == "__main__":
    unittest.main()
