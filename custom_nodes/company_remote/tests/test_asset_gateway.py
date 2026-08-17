from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import server


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from custom_nodes.company_remote import asset_gateway


def _config(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        base_url="http://asset-gateway.test",
        submit_path="/v1/assets",
        timeout_seconds=30,
        auth_header="Authorization",
        auth_prefix="Bearer",
        extra_headers={},
        get_api_key=lambda: "test-key",
    )


class AssetGatewayTests(unittest.TestCase):
    def test_registers_image_through_tos_without_exposing_signed_url(self) -> None:
        image = torch.from_numpy(np.full((1, 320, 400, 3), 0.5, dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(asset_gateway.folder_paths, "get_user_directory", return_value=directory),
                mock.patch.object(asset_gateway, "get_config", side_effect=[_config("gateway"), _config("tos")]),
                mock.patch.object(
                    asset_gateway,
                    "_upload_media_to_tos",
                    return_value=("https://tos.test/person.png?secret=hidden", "assets/person.png"),
                ) as upload,
                mock.patch.object(
                    asset_gateway,
                    "_register_asset",
                    return_value=("asset-test-123", {"asset_id": "asset-test-123"}),
                ) as register,
                mock.patch.object(
                    asset_gateway,
                    "_wait_for_asset_active",
                    return_value=({"Status": "Active"}, 2),
                ) as wait_for_active,
            ):
                asset_id, report_text, reused = asset_gateway.create_seedance_image_asset(
                    image,
                    character_label="人物 A",
                )

            report = json.loads(report_text)
            self.assertEqual(asset_id, "asset-test-123")
            self.assertFalse(reused)
            self.assertEqual(report["image"]["width"], 400)
            self.assertNotIn("secret=hidden", report_text)
            upload.assert_called_once()
            register.assert_called_once_with(
                mock.ANY,
                image_url="https://tos.test/person.png?secret=hidden",
                asset_type="Image",
            )
            wait_for_active.assert_called_once_with(mock.ANY, "asset-test-123")
            self.assertEqual(report["asset_status"], "Active")
            self.assertEqual(report["status_poll_attempts"], 2)

    def test_reuses_cached_asset_for_identical_image(self) -> None:
        image = torch.from_numpy(np.full((1, 320, 400, 3), 0.25, dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(asset_gateway.folder_paths, "get_user_directory", return_value=directory),
                mock.patch.object(asset_gateway, "get_config", side_effect=lambda name: _config(name)),
                mock.patch.object(
                    asset_gateway,
                    "_upload_media_to_tos",
                    return_value=("https://tos.test/person.png?secret=hidden", "assets/person.png"),
                ) as upload,
                mock.patch.object(
                    asset_gateway,
                    "_register_asset",
                    return_value=("asset-cached-456", {"asset_id": "asset-cached-456"}),
                ) as register,
                mock.patch.object(
                    asset_gateway,
                    "_wait_for_asset_active",
                    return_value=({"Status": "Active"}, 1),
                ) as wait_for_active,
            ):
                first = asset_gateway.create_seedance_image_asset(image, character_label="人物 A")
                second = asset_gateway.create_seedance_image_asset(image, character_label="人物 B")

            self.assertFalse(first[2])
            self.assertTrue(second[2])
            self.assertEqual(second[0], "asset-cached-456")
            self.assertEqual(upload.call_count, 1)
            self.assertEqual(register.call_count, 1)
            self.assertEqual(wait_for_active.call_count, 2)

    def test_person_publish_reuses_hash_and_never_returns_signed_tos_url(self) -> None:
        image = torch.from_numpy(np.full((1, 320, 400, 3), 0.75, dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(asset_gateway.folder_paths, "get_user_directory", return_value=directory),
                mock.patch.object(asset_gateway, "get_config", side_effect=lambda name: _config(name)),
                mock.patch.object(
                    asset_gateway,
                    "_upload_media_to_tos",
                    return_value=("https://tos.test/person.png?signature=hidden", "assets/person.png"),
                ) as upload,
                mock.patch.object(
                    asset_gateway,
                    "_register_asset",
                    return_value=("asset-person-123", {"asset_id": "asset-person-123"}),
                ) as register,
                mock.patch.object(
                    asset_gateway,
                    "_wait_for_asset_active",
                    return_value=({"Status": "Active"}, 1),
                ) as wait_for_active,
            ):
                first = asset_gateway.publish_seedance_person_image(image, character_label="第 1 段人物 A")
                second = asset_gateway.publish_seedance_person_image(image, character_label="第 2 段人物 A")

            self.assertEqual(first["tos"], {"status": "uploaded", "object_key": "assets/person.png"})
            self.assertEqual(first["asset_library"]["asset_id"], "asset-person-123")
            self.assertEqual(second["tos"]["status"], "reused")
            self.assertTrue(second["asset_library"]["cache_reused"])
            self.assertNotIn("signature=hidden", json.dumps(first, ensure_ascii=False))
            self.assertEqual(upload.call_count, 1)
            self.assertEqual(register.call_count, 1)
            self.assertEqual(wait_for_active.call_count, 2)

    def test_person_publish_keeps_tos_result_when_asset_library_is_unavailable(self) -> None:
        image = torch.from_numpy(np.full((1, 320, 400, 3), 0.6, dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(asset_gateway.folder_paths, "get_user_directory", return_value=directory),
                mock.patch.object(asset_gateway, "get_config", side_effect=lambda name: _config(name)),
                mock.patch.object(
                    asset_gateway,
                    "_upload_media_to_tos",
                    return_value=("https://tos.test/person.png?signature=hidden", "assets/person.png"),
                ) as upload,
                mock.patch.object(
                    asset_gateway,
                    "_register_asset",
                    side_effect=asset_gateway.CompanyRemoteAPIError("素材库暂时不可用"),
                ),
            ):
                report = asset_gateway.publish_seedance_person_image(image, character_label="第 1 段人物 B")

            upload.assert_called_once()
            self.assertEqual(report["tos"]["status"], "uploaded")
            self.assertEqual(report["asset_library"]["status"], "warning")
            self.assertIn("素材库暂时不可用", report["asset_library"]["error"])

    def test_person_publish_propagates_tos_failure(self) -> None:
        image = torch.from_numpy(np.full((1, 320, 400, 3), 0.4, dtype=np.float32))
        with (
            mock.patch.object(asset_gateway, "get_config", side_effect=lambda name: _config(name)),
            mock.patch.object(
                asset_gateway,
                "_upload_media_to_tos",
                side_effect=asset_gateway.CompanyRemoteAPIError("TOS 写入失败"),
            ),
            mock.patch.object(asset_gateway, "_register_asset") as register,
        ):
            with self.assertRaisesRegex(asset_gateway.CompanyRemoteAPIError, "TOS 写入失败"):
                asset_gateway.publish_seedance_person_image(image, character_label="第 1 段人物 C")

        register.assert_not_called()

    def test_rejects_too_small_image_before_upload(self) -> None:
        image = torch.zeros((1, 299, 400, 3), dtype=torch.float32)
        with mock.patch.object(asset_gateway, "get_config", side_effect=lambda name: _config(name)):
            with self.assertRaisesRegex(Exception, "300"):
                asset_gateway.create_seedance_image_asset(image, character_label="人物 C")

    def test_abc_registration_runs_in_parallel_and_preserves_output_order(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_create(_image, *, character_label, reuse_cached):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            suffix = character_label[-1]
            report = json.dumps({"character": character_label, "asset_id": f"asset-{suffix}"}, ensure_ascii=False)
            return f"asset-{suffix}", report, False

        with mock.patch.object(asset_gateway, "create_seedance_image_asset", side_effect=fake_create):
            result = asset_gateway.create_seedance_abc_assets(object(), object(), object())

        self.assertEqual(result[:3], ("asset-A", "asset-B", "asset-C"))
        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(json.loads(result[3])["parallel_workers"], 3)

    def test_registers_video_through_tos_and_reuses_cache(self) -> None:
        source_info = {"kind": "file", "basename": "source.mp4"}
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(asset_gateway.folder_paths, "get_user_directory", return_value=directory),
                mock.patch.object(asset_gateway, "get_config", side_effect=lambda name: _config(name)),
                mock.patch.object(
                    asset_gateway,
                    "_video_to_bytes",
                    return_value=(b"fake-mp4", "video/mp4", ".mp4", source_info),
                ),
                mock.patch.object(
                    asset_gateway,
                    "_upload_media_to_tos",
                    return_value=("https://tos.test/source.mp4?secret=hidden", "assets/source.mp4"),
                ) as upload,
                mock.patch.object(
                    asset_gateway,
                    "_register_asset",
                    return_value=("asset-video-123", {"asset_id": "asset-video-123"}),
                ) as register,
                mock.patch.object(
                    asset_gateway,
                    "_wait_for_asset_active",
                    return_value=({"Status": "Active"}, 1),
                ) as wait_for_active,
            ):
                first = asset_gateway.create_seedance_video_asset(object())
                second = asset_gateway.create_seedance_video_asset(object())

        self.assertEqual(first[0], "asset-video-123")
        self.assertFalse(first[2])
        self.assertTrue(second[2])
        self.assertNotIn("secret=hidden", first[1])
        upload.assert_called_once()
        register.assert_called_once_with(
            mock.ANY,
            image_url="https://tos.test/source.mp4?secret=hidden",
            asset_type="Video",
        )
        self.assertEqual(wait_for_active.call_count, 2)

    def test_waits_until_asset_is_active(self) -> None:
        config = _config("gateway")
        config.max_poll_attempts = 3
        config.poll_interval_seconds = 0.1
        with (
            mock.patch.object(
                asset_gateway,
                "_get_asset",
                side_effect=[{"Status": "Processing"}, {"Status": "Active"}],
            ),
            mock.patch.object(asset_gateway.time, "sleep") as sleep,
        ):
            payload, attempts = asset_gateway._wait_for_asset_active(config, "asset-test")

        self.assertEqual(payload["Status"], "Active")
        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(0.1)

    def test_failed_asset_reports_provider_error(self) -> None:
        config = _config("gateway")
        config.max_poll_attempts = 1
        config.poll_interval_seconds = 0.1
        with mock.patch.object(
            asset_gateway,
            "_get_asset",
            return_value={
                "Status": "Failed",
                "Error": {"Code": "PolicyViolation", "Message": "copyright restrictions"},
            },
        ):
            with self.assertRaisesRegex(Exception, "PolicyViolation.*copyright restrictions"):
                asset_gateway._wait_for_asset_active(config, "asset-failed")

    def test_failed_cached_asset_is_evicted(self) -> None:
        image = torch.from_numpy(np.full((1, 320, 400, 3), 0.75, dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(asset_gateway.folder_paths, "get_user_directory", return_value=directory),
                mock.patch.object(asset_gateway, "get_config", side_effect=lambda name: _config(name)),
                mock.patch.object(
                    asset_gateway,
                    "_upload_media_to_tos",
                    return_value=("https://tos.test/person.png", "assets/person.png"),
                ),
                mock.patch.object(
                    asset_gateway,
                    "_register_asset",
                    return_value=("asset-failed", {"asset_id": "asset-failed"}),
                ),
                mock.patch.object(
                    asset_gateway,
                    "_wait_for_asset_active",
                    side_effect=[({"Status": "Active"}, 1), RuntimeError("asset failed")],
                ),
            ):
                asset_gateway.create_seedance_image_asset(image, character_label="人物 A")
                with self.assertRaisesRegex(RuntimeError, "asset failed"):
                    asset_gateway.create_seedance_image_asset(image, character_label="人物 A")
                cache = asset_gateway._load_cache()

        self.assertEqual(cache["assets"], {})

    def test_video_and_abc_registration_preserves_output_order(self) -> None:
        reports = {
            "video": json.dumps({"asset_type": "Video", "asset_id": "asset-source"}),
            "A": json.dumps({"asset_type": "Image", "asset_id": "asset-A"}),
            "B": json.dumps({"asset_type": "Image", "asset_id": "asset-B"}),
            "C": json.dumps({"asset_type": "Image", "asset_id": "asset-C"}),
        }

        def fake_video(*_args, **_kwargs):
            return "asset-source", reports["video"], False

        def fake_image(_image, *, character_label, reuse_cached):
            suffix = character_label[-1]
            return f"asset-{suffix}", reports[suffix], False

        with (
            mock.patch.object(asset_gateway, "create_seedance_video_asset", side_effect=fake_video),
            mock.patch.object(asset_gateway, "create_seedance_image_asset", side_effect=fake_image),
        ):
            result = asset_gateway.create_seedance_video_abc_assets(object(), object(), object(), object())

        self.assertEqual(result[:4], ("asset-source", "asset-A", "asset-B", "asset-C"))
        self.assertEqual(json.loads(result[4])["parallel_workers"], 4)


if __name__ == "__main__":
    unittest.main()
