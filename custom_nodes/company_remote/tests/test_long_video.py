from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import av
import numpy as np
import torch
from PIL import Image

import server


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from custom_nodes.company_remote import long_video
from custom_nodes.company_remote import client as company_client
from custom_nodes.company_remote import nodes as company_nodes


class LongVideoTests(unittest.TestCase):
    def test_atomic_json_write_retries_transient_windows_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            replace_calls = 0
            original_replace = os.replace

            def transiently_locked_replace(source, destination):
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 1:
                    raise PermissionError(5, "Access is denied", str(destination))
                return original_replace(source, destination)

            with (
                mock.patch.object(long_video.os, "replace", side_effect=transiently_locked_replace),
                mock.patch.object(long_video.time, "sleep") as sleep,
            ):
                long_video._atomic_write_json(path, {"status": "ready"})

            self.assertEqual(replace_calls, 2)
            sleep.assert_called_once_with(long_video._JSON_WRITE_REPLACE_DELAY_SECONDS)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "ready"})
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_openai_text_retries_an_empty_success_response(self) -> None:
        empty = {
            "id": "chatcmpl-empty",
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
        }
        success = {
            "id": "chatcmpl-success",
            "choices": [{"finish_reason": "stop", "message": {"content": "有效结果"}}],
        }
        with (
            mock.patch.object(company_client, "_submit", side_effect=[empty, success]) as submit,
            mock.patch.object(company_client.time, "sleep"),
        ):
            result = company_client._submit_openai_text(
                object(),
                {"messages": []},
                model="gpt-5.6-terra",
                max_attempts=3,
            )
        self.assertEqual(result, "有效结果")
        self.assertEqual(submit.call_count, 2)

    def test_auto_asset_failure_summary_groups_real_root_causes_and_stage_state(self) -> None:
        members = []
        for index in range(1, 15):
            if index == 4:
                message = (
                    'Remote request failed with HTTP 502: {"error":{"type":"service_unavailable_error",'
                    '"code":"server_is_overloaded","message":"Our servers are currently overloaded. Please try again later."}}'
                )
            else:
                message = (
                    'Remote request failed with HTTP 503: {"error":{"message":"auth_unavailable: '
                    'no auth available (providers=codex, model=gpt-5.6-terra)",'
                    '"type":"server_error","code":"internal_server_error"}}'
                )
            members.append(
                {
                    "index": index,
                    "auto_asset_status": "analysis_failed",
                    "auto_asset_analysis_attempts": 3,
                    "auto_asset_errors": [{"kind": "analysis", "message": message, "attempts": 3}],
                }
            )
        tasks = [
            {"index": 1, "logical_segments": [1, 2], "reference_package_status": "blocked_by_asset_failure"},
            {"index": 2, "logical_segments": [3, 4], "reference_package_status": "blocked_by_asset_failure"},
            {"index": 3, "logical_segments": [5, 6], "reference_package_status": "blocked_by_asset_failure"},
            {"index": 4, "logical_segments": [7, 8, 9, 10], "reference_package_status": "blocked_by_asset_failure"},
            {"index": 5, "logical_segments": [11, 12, 13, 14], "reference_package_status": "blocked_by_asset_failure"},
        ]
        job = cast(
            long_video.LongVideoJob,
            SimpleNamespace(manifest={"logical_member_tasks": members, "tasks": tasks}),
        )

        summary = long_video._auto_asset_failure_summary(job, request_tasks=tasks)
        categories = {item["category"]: item for item in summary["categories"]}
        formatted = long_video._format_auto_asset_failure_summary(summary)

        self.assertEqual(summary["failed_shot_count"], 14)
        self.assertEqual(summary["total_shot_count"], 14)
        self.assertEqual(summary["blocked_group_indexes"], [1, 2, 3, 4, 5])
        self.assertEqual(categories["auth_unavailable"]["failed_shot_count"], 13)
        self.assertEqual(categories["auth_unavailable"]["attempts"], 3)
        self.assertEqual(categories["auth_unavailable"]["representative_error"]["http_status"], 503)
        self.assertEqual(categories["auth_unavailable"]["representative_error"]["provider_code"], "internal_server_error")
        self.assertEqual(categories["server_is_overloaded"]["failed_shot_count"], 1)
        self.assertEqual(categories["server_is_overloaded"]["shot_indexes"], [4])
        self.assertFalse(summary["stage_state"]["image_generation"]["started"])
        self.assertFalse(summary["stage_state"]["seedance"]["started"])
        self.assertIn("14/14 个镜头不可用", formatted)
        self.assertIn("远端服务无可用认证：13 个镜头", formatted)
        self.assertIn("HTTP 502 / server_is_overloaded", formatted)
        self.assertIn("图片生成未启动", formatted)
        self.assertIn("Seedance 尚未启动", formatted)

    def test_auto_asset_failure_summary_deduplicates_shots_and_sanitizes_secrets(self) -> None:
        unsafe = (
            'HTTP 503 Authorization: Bearer secret-token api_key=sk-supersecret '
            'url=https://example.com/result.png?X-Amz-Signature=secret '
            'path=C:\\private\\source.png data:image/png;base64,AAAAAA '
            '{"error":{"code":"server_is_overloaded","message":"busy"}}'
        )
        member = {
            "index": 1,
            "auto_asset_status": "degraded",
            "auto_asset_errors": [
                {"kind": "person", "error_kind": "generation_failed", "message": unsafe, "attempts": 2},
                {"kind": "scene", "error_kind": "generation_failed", "message": unsafe, "attempts": 3},
            ],
            "auto_asset_warnings": [{"kind": "person_asset_library", "message": "仅素材库 warning"}],
        }
        task = {"index": 1, "logical_segments": [1], "reference_package_status": "blocked_by_asset_failure"}
        job = cast(
            long_video.LongVideoJob,
            SimpleNamespace(manifest={"logical_member_tasks": [member], "tasks": [task]}),
        )

        summary = long_video._auto_asset_failure_summary(job, request_tasks=[task])
        encoded = json.dumps(summary, ensure_ascii=False)
        category = summary["categories"][0]

        self.assertEqual(summary["failed_shot_count"], 1)
        self.assertEqual(category["failed_shot_count"], 1)
        self.assertEqual(category["error_record_count"], 2)
        self.assertEqual(category["attempts"], 3)
        self.assertTrue(summary["stage_state"]["image_generation"]["started"])
        self.assertNotIn("secret-token", encoded)
        self.assertNotIn("sk-supersecret", encoded)
        self.assertNotIn("X-Amz-Signature", encoded)
        self.assertNotIn("C:\\private", encoded)
        self.assertNotIn("AAAAAA", encoded)
        self.assertNotIn("素材库 warning", encoded)

    def test_auto_asset_failure_summary_ignores_ready_member_warnings(self) -> None:
        member = {
            "index": 1,
            "auto_asset_status": "ready",
            "auto_asset_errors": [],
            "auto_asset_warnings": [{"kind": "person_asset_library", "message": "素材库暂时不可用"}],
        }
        task = {"index": 1, "logical_segments": [1], "reference_package_status": "ready", "attempts": 0}
        job = cast(
            long_video.LongVideoJob,
            SimpleNamespace(manifest={"logical_member_tasks": [member], "tasks": [task]}),
        )

        summary = long_video._auto_asset_failure_summary(job, request_tasks=[task])

        self.assertEqual(summary["failed_shot_count"], 0)
        self.assertEqual(summary["categories"], [])

    def test_fixed_column_image_preview_saves_persistent_output_images(self) -> None:
        images = torch.stack(
            [torch.full((40, 80, 3), index / 6.0) for index in range(6)],
            dim=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(company_nodes.folder_paths, "get_output_directory", return_value=directory):
                result = company_nodes.CompanyFixedColumnImagePreview.execute(images, columns="2", gap=8)

            descriptors = result.ui["fixed_grid_images"]
            self.assertIs(result[0], images)
            self.assertEqual(len(descriptors), 6)
            self.assertEqual(result.ui["fixed_grid_columns"], (2,))
            self.assertEqual(result.ui["fixed_grid_gap"], (8,))
            expected_subfolder = Path("company_remote") / "fixed_column_previews"
            self.assertTrue(all(item["type"] == "output" for item in descriptors))
            self.assertTrue(all(Path(item["subfolder"]) == expected_subfolder for item in descriptors))
            self.assertTrue(all((Path(directory) / item["subfolder"] / item["filename"]).is_file() for item in descriptors))

    def test_fixed_column_preview_restores_images_from_node_properties(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "web/fixed_column_image_preview.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('const PROPERTY_NAME = "last_fixed_grid_images"', script)
        self.assertIn("nodeType.prototype.onConfigure", script)
        self.assertIn("persistImages(this, images)", script)
        self.assertIn("info?.properties?.[PROPERTY_NAME]", script)

    def test_auto_asset_progress_writes_file_and_updates_progress_bar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"index": 3, "start": 1.25, "duration": 2.5, "auto_asset_status": "building"}
            job = SimpleNamespace(
                job_dir=root,
                manifest_path=root / "manifest.json",
                manifest={
                    "job_id": "job-progress",
                    "status": "planned",
                    "tasks": [task, {"index": 4}],
                },
            )
            progress_bar = mock.Mock()
            with mock.patch.object(long_video, "_send_auto_asset_progress_event") as send_event:
                payload = long_video._emit_auto_asset_progress(
                    job,
                    progress_bar,
                    value=0.5,
                    total=2,
                    phase="analysis_started",
                    message="第 3 段正在分析人物和背景。",
                    task=task,
                    extra={"people_count": 1, "error": r"C:\\private\\source.mp4 读取失败"},
                )

            saved = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["job_id"], "job-progress")
            self.assertEqual(saved["task_index"], 3)
            self.assertEqual(saved["auto_asset_status_counts"]["building"], 1)
            self.assertEqual(saved["auto_asset_status_counts"]["planned"], 1)
            self.assertEqual(saved["extra"]["people_count"], 1)
            self.assertEqual(job.manifest["auto_asset_progress"]["phase"], "analysis_started")
            progress_bar.update_absolute.assert_called_once_with(0.5, 2)
            self.assertEqual(payload["progress_path"], str(root / "progress.json"))
            sent = send_event.call_args.args[0]
            self.assertNotIn("manifest", sent)
            self.assertNotIn("progress_path", sent)
            self.assertNotIn(r"C:\\private", json.dumps(sent, ensure_ascii=False))
            self.assertIn("[本机路径]", json.dumps(sent, ensure_ascii=False))

    def test_auto_asset_progress_frontend_restores_from_node_properties(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "web/auto_asset_progress.js"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('const EVENT_NAME = "company_remote.auto_asset_progress"', script)
        self.assertIn('const PROPERTY_NAME = "last_auto_asset_progress"', script)
        self.assertIn("api.addEventListener(EVENT_NAME", script)
        self.assertIn("nodeType.prototype.onConfigure", script)
        self.assertIn("payloadFromFinalReport", script)
        self.assertIn("CompanyLongVideoPipelineAssetVideoGenerator", script)
        self.assertIn("当前分镜源帧", script)
        self.assertIn("素材库已入库", script)

    def test_auto_asset_preview_uses_output_descriptors_without_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            shot_root = output_root / "company_remote" / "shot_0001"
            source_start = shot_root / "source_start.png"
            source_end = shot_root / "source_end.png"
            person = shot_root / "people" / "P1.png"
            scene = shot_root / "scenes" / "scene_start.png"
            for path, shade in ((source_start, 0.1), (source_end, 0.2), (person, 0.4), (scene, 0.6)):
                long_video._save_image_tensor(torch.full((1, 32, 48, 3), shade), path)

            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(output_root)):
                preview = long_video._auto_asset_preview_payload(
                    {"index": 1},
                    source_frames={"source_start": str(source_start), "source_end": str(source_end)},
                    people=[
                        {
                            "slot": "P1",
                            "path": str(person),
                            "publication": {
                                "tos": {"status": "uploaded", "object_key": "safe/key.png"},
                                "asset_library": {"status": "active", "asset_id": "asset-p1"},
                            },
                        }
                    ],
                    scenes=[{"role": "scene_start", "path": str(scene)}],
                )

            serialized = json.dumps(preview, ensure_ascii=False)
            self.assertNotIn(str(output_root), serialized)
            self.assertEqual(preview["source_start"]["type"], "output")
            self.assertEqual(preview["source_end"]["filename"], "source_end.png")
            self.assertEqual(preview["converted"][0]["asset_library_status"], "active")
            self.assertEqual(preview["converted"][1]["kind"], "scene")
            self.assertEqual(preview["converted"][1]["asset_library_status"], "not_registered")

    def test_person_master_quality_gate_rejects_unconverted_western_master(self) -> None:
        quality = long_video._normalize_person_master_quality(
            {
                "target_style_score": 0.03,
                "source_style_residue": 0.96,
                "subject_contract_score": 0.98,
                "subject_count_match": True,
            },
            visual_style=long_video.AUTO_ASSET_STYLE_WESTERN,
        )

        self.assertEqual(quality["verdict"], "retry")
        self.assertTrue(any("目标风格强度不足" in item for item in quality["reasons"]))
        self.assertTrue(any("原样残留过多" in item for item in quality["reasons"]))

    def test_person_master_quality_gate_approves_strong_western_master(self) -> None:
        quality = long_video._normalize_person_master_quality(
            {
                "target_style_score": 0.95,
                "source_style_residue": 0.03,
                "subject_contract_score": 0.93,
                "subject_count_match": True,
            },
            visual_style=long_video.AUTO_ASSET_STYLE_WESTERN,
        )

        self.assertEqual(quality["verdict"], "approved")
        self.assertEqual(quality["reasons"], [])

    def test_person_master_quality_gate_preserves_nonhuman_subjects(self) -> None:
        quality = long_video._normalize_person_master_quality(
            {
                "target_style_score": 0.00,
                "source_style_residue": 1.00,
                "subject_contract_score": 0.96,
                "subject_count_match": True,
                "subject_kind": "animal",
            },
            visual_style=long_video.AUTO_ASSET_STYLE_WESTERN,
        )

        self.assertEqual(quality["verdict"], "approved")
        self.assertEqual(quality["subject_kind"], "animal")

    def test_person_master_generation_rejects_unqualified_asset_before_cache_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = torch.ones((3, 400, 400, 3))
            person = {
                "slot": "P1",
                "appearance": "年轻女性，黑色盘发，浅色立领上衣",
                "first_bbox": [0.05, 0.05, 0.95, 0.95],
                "last_bbox": [0.05, 0.05, 0.95, 0.95],
                "identity_key": "heroine",
                "reuse_confidence": 1.0,
            }
            with (
                mock.patch.object(
                    long_video,
                    "_generate_auto_asset_image",
                    return_value=torch.full((1, 256, 256, 3), 0.5),
                ),
                mock.patch.object(
                    long_video,
                    "_evaluate_person_master_quality",
                    return_value={
                        "verdict": "retry",
                        "reasons": ["人物母版目标风格强度不足（0.03）"],
                    },
                ) as quality_check,
            ):
                entry, error = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=long_video._empty_auto_asset_cache(),
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="AI-Zero-Token",
                    reuse_threshold=0.92,
                    visual_style=long_video.AUTO_ASSET_STYLE_WESTERN,
                    validate_person_master=True,
                )

        self.assertIsNone(entry)
        self.assertEqual(error["error_kind"], "quality_gate_failed")
        self.assertIn("目标风格强度不足", error["message"])
        self.assertEqual(error["quality"]["verdict"], "retry")
        self.assertFalse((root / "shot_0001" / "people" / "P1.png").exists())
        quality_check.assert_called_once()

    def test_integrated_frame_quality_gate_rejects_weak_conversion(self) -> None:
        quality = long_video._normalize_integrated_frame_quality(
            {
                "target_style_score": 0.62,
                "source_style_residue": 0.31,
                "composition_preservation": 0.94,
                "person_count_match": True,
                "person_identity_score": 0.96,
                "scene_identity_score": 0.95,
                "verdict": "approved",
            },
            require_person_identity=True,
        )
        self.assertEqual(quality["verdict"], "retry")
        self.assertTrue(any("目标风格强度不足" in item for item in quality["reasons"]))
        self.assertTrue(any("原风格残留过多" in item for item in quality["reasons"]))

    def test_integrated_frame_quality_gate_approves_strong_consistent_conversion(self) -> None:
        quality = long_video._normalize_integrated_frame_quality(
            {
                "target_style_score": 0.96,
                "source_style_residue": 0.04,
                "composition_preservation": 0.93,
                "person_count_match": True,
                "person_identity_score": 0.94,
                "scene_identity_score": 0.96,
            },
            require_person_identity=True,
        )
        self.assertEqual(quality["verdict"], "approved")
        self.assertEqual(quality["reasons"], [])

    def test_replacement_quality_accepts_changed_environment_when_spatial_contract_holds(self) -> None:
        quality = long_video._normalize_integrated_frame_quality(
            {
                "target_style_score": 0.95,
                "source_style_residue": 0.03,
                "composition_preservation": 0.91,
                "person_count_match": True,
                "person_identity_score": 0.94,
                "scene_replacement_score": 0.87,
            },
            require_person_identity=True,
        )

        self.assertEqual(quality["verdict"], "approved")
        self.assertEqual(quality["scene_replacement_score"], 0.87)
        self.assertNotIn("scene_identity_score", quality)

    def test_small_person_crop_is_not_used_as_identity_master(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            small = Path(directory) / "small.png"
            large = Path(directory) / "large.png"
            Image.new("RGB", (33, 71), "red").save(small)
            Image.new("RGB", (160, 180), "blue").save(large)
            small_result = long_video._person_master_resolution([str(small)])
            large_result = long_video._person_master_resolution([str(large)])
        self.assertFalse(small_result["eligible_for_identity_master"])
        self.assertTrue(large_result["eligible_for_identity_master"])

    def test_integrated_cache_keeps_the_strongest_global_style_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weak = root / "weak.png"
            strong = root / "strong.png"
            weaker = root / "weaker.png"
            for path, color in ((weak, "red"), (strong, "blue"), (weaker, "green")):
                Image.new("RGB", (32, 32), color).save(path)
            cache = long_video._empty_auto_asset_cache()
            cache["scenes"]["by_id"]["place"] = {
                "place_id": "place",
                "versions": {"view": {"version_id": "view", "view_id": "view"}},
            }
            for path, score in ((weak, 0.88), (strong, 0.97), (weaker, 0.91)):
                long_video._record_integrated_frame_cache(
                    cache,
                    {
                        "path": str(path),
                        "place_id": "place",
                        "view_id": "view",
                        "quality": {"verdict": "approved", "target_style_score": score},
                    },
                    has_people=True,
                )
        self.assertEqual(cache["scenes"]["style_master_path"], str(strong))
        self.assertEqual(cache["scenes"]["style_master_quality_score"], 0.97)

    def test_persistent_asset_library_merges_only_the_requested_style(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_user_directory = long_video.folder_paths.get_user_directory()
            original_output_directory = long_video.folder_paths.get_output_directory()
            original_configured_output_directory = long_video.folder_paths.output_directory
            output_root = root / "output"
            western_cache = output_root / "company_remote" / "manual_batch_series" / "western" / "job" / "asset_cache" / "index.json"
            anime_cache = output_root / "company_remote" / "long_video_jobs" / "anime" / "asset_cache" / "index.json"
            western_manifest = western_cache.parent.parent / "manifest.json"
            anime_manifest = anime_cache.parent.parent / "manifest.json"
            western = long_video._empty_auto_asset_cache()
            western["people"]["by_id"]["western-person"] = {
                "person_id": "western-person",
                "identity_keys": ["traveler"],
                "converted_path": str(root / "western.png"),
                "source_observations": [],
            }
            anime = long_video._empty_auto_asset_cache()
            anime["people"]["by_id"]["anime-person"] = {
                "person_id": "anime-person",
                "identity_keys": ["traveler"],
                "converted_path": str(root / "anime.png"),
                "source_observations": [],
            }
            for path, cache, manifest, style in (
                (western_cache, western, western_manifest, "western"),
                (anime_cache, anime, anime_manifest, "anime_2d"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cache), encoding="utf-8")
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(
                    json.dumps(
                        {
                            "auto_asset_options": {
                                "visual_style": style,
                                "prompt_version": long_video.AUTO_ASSET_PROMPT_VERSION,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            try:
                long_video.folder_paths.set_user_directory(str(root / "user"))
                long_video.folder_paths.set_output_directory(str(output_root))
                long_video.folder_paths.output_directory = str(output_root)
                library = long_video._load_auto_asset_library("western")
            finally:
                long_video.folder_paths.set_user_directory(original_user_directory)
                long_video.folder_paths.set_output_directory(original_output_directory)
                long_video.folder_paths.output_directory = original_configured_output_directory

        self.assertIn("western-person", library["people"]["by_id"])
        self.assertNotIn("anime-person", library["people"]["by_id"])

    def test_weak_cached_integrated_frame_is_regenerated_on_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            scene = root / "scene.png"
            cached = root / "cached.png"
            for path, color in ((source, "red"), (scene, "blue"), (cached, "green")):
                Image.new("RGB", (128, 128), color).save(path)
            task = {"auto_asset_analysis": {"people": [], "background": {}}}
            spec = {
                "role": "frame_start",
                "source_path": source,
                "scene": {
                    "path": str(scene),
                    "place_id": "place",
                    "view_id": "view",
                    "integrated_frame_path": str(cached),
                    "integrated_frame_quality": {"verdict": "approved", "target_style_score": 0.88},
                },
            }
            generated = torch.full((1, 128, 128, 3), 0.75)
            approved = {
                "verdict": "approved",
                "target_style_score": 0.97,
                "source_style_residue": 0.03,
                "composition_preservation": 0.95,
                "person_count_match": True,
                "person_identity_score": 1.0,
                "scene_identity_score": 0.96,
                "reasons": [],
            }
            with (
                mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated) as image_request,
                mock.patch.object(long_video, "_evaluate_integrated_frame_quality", return_value=approved),
            ):
                first, first_error = long_video._integrated_frame_attempt(
                    task=task,
                    spec=spec,
                    root=root,
                    attempt=1,
                    style_master_path=None,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="medium",
                    image_provider="WisArt",
                    visual_style="western",
                    style_prompt="",
                )
                second, second_error = long_video._integrated_frame_attempt(
                    task=task,
                    spec=spec,
                    root=root,
                    attempt=2,
                    style_master_path=None,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="medium",
                    image_provider="WisArt",
                    visual_style="western",
                    style_prompt="",
                    retry_reasons=["低于本组自适应风格下限"],
                )
        self.assertIsNone(first_error)
        self.assertTrue(first["reused_from_cache"])
        self.assertIsNone(second_error)
        self.assertFalse(second["reused_from_cache"])
        self.assertEqual(image_request.call_count, 1)

    def test_integrated_reference_limit_preserves_temporal_member_coverage(self) -> None:
        items = [
            {
                "member_index": member,
                "temporal_order": order,
                "identity": f"{member}-{role}",
            }
            for order, (member, role) in enumerate(
                (member, role) for member in range(1, 6) for role in ("start", "end")
            )
        ]
        selected, omitted = long_video._select_integrated_reference_items(items, package_limit=9)
        self.assertEqual(len(selected), 9)
        self.assertEqual(omitted, 1)
        self.assertEqual({item["member_index"] for item in selected}, {1, 2, 3, 4, 5})
        self.assertEqual([item["temporal_order"] for item in selected], sorted(item["temporal_order"] for item in selected))

    def test_person_tos_failure_blocks_the_current_auto_asset_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person.png"
            long_video._save_image_tensor(torch.full((1, 32, 48, 3), 0.5), path)
            entry = {"slot": "P1", "path": str(path)}
            with mock.patch.object(long_video, "publish_seedance_person_image", side_effect=RuntimeError("TOS 写入失败")):
                warning, error = long_video._publish_auto_person_asset(entry, task_index=1)

        self.assertIsNone(warning)
        self.assertIsNotNone(error)
        self.assertEqual(error["error_kind"], "tos_upload_failed")
        self.assertIn("TOS 写入失败", error["message"])
        self.assertEqual(entry["publication"]["tos"]["status"], "failed")

    def test_person_asset_library_failure_is_a_warning_not_a_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person.png"
            long_video._save_image_tensor(torch.full((1, 32, 48, 3), 0.5), path)
            entry = {"slot": "P1", "path": str(path)}
            publication = {
                "tos": {"status": "uploaded", "object_key": "safe/person.png"},
                "asset_library": {"status": "warning", "asset_id": "", "error": "素材库超时"},
            }
            with mock.patch.object(long_video, "publish_seedance_person_image", return_value=publication):
                warning, error = long_video._publish_auto_person_asset(entry, task_index=1)

        self.assertIsNone(error)
        self.assertEqual(warning["kind"], "person_asset_library")
        self.assertEqual(entry["publication"]["tos"]["status"], "uploaded")

    def test_hard_cut_detection_is_frame_accurate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard_cuts.mp4"
            frames = []
            for color in ((220, 30, 30), (30, 220, 30), (30, 30, 220)):
                frames.extend(np.full((48, 64, 3), color, dtype=np.uint8) for _ in range(10))
            self._make_video_from_frames(path, frames, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(path))
            with mock.patch("scenedetect.open_video", wraps=__import__("scenedetect").open_video) as open_video:
                plan, status, previews = long_video.detect_long_video_shots(
                    video=video,
                    mode="镜头优先（推荐）",
                    fixed_duration=10,
                    sensitivity="标准",
                    use_audio_silence=False,
                    auto_fallback=True,
                )
        self.assertEqual([item.frame for item in plan.boundaries], [10, 20])
        self.assertTrue(all(item.kind == "hard_cut" for item in plan.boundaries))
        self.assertEqual([round(item.duration, 3) for item in plan.shots], [1.0, 1.0, 1.0])
        self.assertEqual(tuple(previews.shape), (3, 48, 64, 3))
        self.assertEqual(json.loads(status)["effective_mode"], "shot_aware")
        self.assertEqual(open_video.call_count, 1)

    def test_scene_detection_uses_low_resolution_proxy_for_wide_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wide.mp4"
            frames = [np.full((96, 672, 3), index * 10, dtype=np.uint8) for index in range(20)]
            self._make_video_from_frames(source, frames, fps=10)
            proxy, metadata = long_video._scene_detection_source(str(source), root)
            with av.open(proxy, mode="r") as container:
                stream = container.streams.video[0]
                dimensions = (stream.width, stream.height)
                duration = float(container.duration / av.time_base)

        self.assertNotEqual(Path(proxy), source)
        self.assertTrue(metadata["proxy"])
        self.assertEqual(dimensions, (640, 92))
        self.assertAlmostEqual(duration, 2.0, delta=0.11)

    def test_shot_previews_are_downscaled_before_tensor_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wide_preview.mp4"
            frames = [np.full((96, 672, 3), index * 10, dtype=np.uint8) for index in range(20)]
            self._make_video_from_frames(source, frames, fps=10)
            previews = long_video._frames_at_times(str(source), [0.1, 1.1])

        self.assertEqual(tuple(previews.shape), (2, 73, 512, 3))

    def test_shot_inspector_previews_selected_shot_and_exports_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "hard_cuts.mp4"
            frames = []
            for color in ((220, 30, 30), (30, 220, 30), (30, 30, 220)):
                frames.extend(np.full((48, 64, 3), color, dtype=np.uint8) for _ in range(10))
            self._make_video_from_frames(path, frames, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(path))
            plan, _status, _previews = long_video.detect_long_video_shots(
                video=video,
                mode="镜头优先（推荐）",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(root / "output")):
                selected_video, selected_frames, boundary_frames, report, export_directory = (
                    long_video.inspect_long_video_shots(plan, shot_index=2, export_all_shots=True)
                )
            parsed = json.loads(report)
            exported = [Path(item) for item in parsed["exported_paths"]]
            self.assertAlmostEqual(float(selected_video.get_duration()), 1.0, delta=0.11)
            self.assertEqual(tuple(selected_frames.shape), (3, 48, 64, 3))
            self.assertEqual(tuple(boundary_frames.shape), (4, 48, 64, 3))
            self.assertEqual(parsed["selected_shot"]["index"], 2)
            self.assertEqual(len(parsed["boundary_preview_order"]), 2)
            self.assertEqual(len(exported), 3)
            self.assertTrue(all(path.is_file() for path in exported))
            self.assertEqual(Path(export_directory).name, "shots")

    def test_shot_inspector_rejects_invalid_index(self) -> None:
        plan = long_video.LongVideoShotPlan(
            video=object(),
            total_duration=1.0,
            fps=10.0,
            requested_mode="shot_aware",
            effective_mode="shot_aware",
            fixed_duration=10,
            sensitivity="标准",
            use_audio_silence=False,
            auto_fallback=True,
            detector="test",
            config={},
            boundaries=[],
            shots=[long_video.LogicalShot(1, 0.0, 1.0, 1.0, "video_start", "video_end")],
        )
        with self.assertRaisesRegex(ValueError, "1-1"):
            long_video.inspect_long_video_shots(plan, shot_index=2, export_all_shots=False)

    def test_continuity_range_selects_and_rebases_consecutive_shots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard_cuts.mp4"
            frames = []
            for color in (
                (220, 30, 30),
                (30, 220, 30),
                (30, 30, 220),
                (220, 220, 30),
                (220, 30, 220),
            ):
                frames.extend(np.full((48, 64, 3), color, dtype=np.uint8) for _ in range(10))
            self._make_video_from_frames(path, frames, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(path))
            plan, _status, _previews = long_video.detect_long_video_shots(
                video=video,
                mode="镜头优先（推荐）",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
            )
            selected, selected_video, report = long_video.select_continuous_shot_range(
                plan,
                start_shot=2,
                shot_count=4,
            )
            parsed = json.loads(report)
            self.assertEqual(parsed["selected_source_shots"], [2, 3, 4, 5])
            self.assertEqual([item.index for item in selected.shots], [1, 2, 3, 4])
            self.assertEqual([item.start for item in selected.shots], [0.0, 1.0, 2.0, 3.0])
            self.assertEqual([item.time for item in selected.boundaries], [1.0, 2.0, 3.0])
            self.assertAlmostEqual(selected.total_duration, 4.0, delta=0.11)
            self.assertAlmostEqual(float(selected_video.get_duration()), 4.0, delta=0.11)

            auto_selected, auto_selected_video, auto_report = long_video.select_continuous_shot_range(
                plan,
                start_shot=2,
                shot_count=0,
            )
            auto_parsed = json.loads(auto_report)
            self.assertEqual(auto_parsed["selected_source_shots"], [2, 3, 4, 5])
            self.assertEqual(auto_parsed["requested_shot_count"], 0)
            self.assertEqual(auto_parsed["selected_shot_count"], 4)
            self.assertTrue(auto_parsed["auto_all_remaining"])
            self.assertEqual([item.index for item in auto_selected.shots], [1, 2, 3, 4])
            self.assertAlmostEqual(auto_selected.total_duration, 4.0, delta=0.11)
            self.assertAlmostEqual(float(auto_selected_video.get_duration()), 4.0, delta=0.11)

    def test_length_range_selector_uses_whole_shots_for_minutes_and_percent(self) -> None:
        class FakeVideo:
            def __init__(self, duration: float):
                self.duration = duration

            def get_duration(self) -> float:
                return self.duration

            def as_trimmed(self, _start: float, duration: float, strict_duration: bool = True):
                return FakeVideo(duration)

        plan = long_video.LongVideoShotPlan(
            video=FakeVideo(600.0),
            total_duration=600.0,
            fps=30.0,
            requested_mode="shot_aware",
            effective_mode="shot_aware",
            fixed_duration=10,
            sensitivity="标准",
            use_audio_silence=False,
            auto_fallback=True,
            detector="test",
            config={},
            boundaries=[
                long_video.ShotBoundary(60.0, 1800, "hard_cut", "test", 1.0),
                long_video.ShotBoundary(120.0, 3600, "hard_cut", "test", 1.0),
                long_video.ShotBoundary(190.0, 5700, "hard_cut", "test", 1.0),
                long_video.ShotBoundary(240.0, 7200, "hard_cut", "test", 1.0),
            ],
            shots=[
                long_video.LogicalShot(1, 0.0, 60.0, 60.0, "video_start", "hard_cut"),
                long_video.LogicalShot(2, 60.0, 60.0, 120.0, "hard_cut", "hard_cut"),
                long_video.LogicalShot(3, 120.0, 70.0, 190.0, "hard_cut", "hard_cut"),
                long_video.LogicalShot(4, 190.0, 50.0, 240.0, "hard_cut", "hard_cut"),
                long_video.LogicalShot(5, 240.0, 360.0, 600.0, "hard_cut", "video_end"),
            ],
        )

        selected, selected_video, report = long_video.select_long_video_length_range(
            plan,
            start_shot=1,
            limit_mode="按分钟",
            limit_minutes=3.0,
            limit_percent=30.0,
            shot_count=0,
        )
        parsed = json.loads(report)
        self.assertEqual(parsed["selected_source_shots"], [1, 2, 3])
        self.assertEqual(parsed["length_range_selection"]["resolved_shot_count"], 3)
        self.assertEqual(parsed["length_range_selection"]["target_duration_seconds"], 180.0)
        self.assertEqual(parsed["length_range_selection"]["selected_duration_seconds"], 190.0)
        self.assertEqual(parsed["length_range_selection"]["over_target_seconds"], 10.0)
        self.assertAlmostEqual(selected.total_duration, 190.0)
        self.assertAlmostEqual(float(selected_video.get_duration()), 190.0)

        percent_selected, _percent_video, percent_report = long_video.select_long_video_length_range(
            plan,
            start_shot=1,
            limit_mode="按总长百分比",
            limit_minutes=3.0,
            limit_percent=30.0,
            shot_count=0,
        )
        percent_parsed = json.loads(percent_report)
        self.assertEqual([item.index for item in percent_selected.shots], [1, 2, 3])
        self.assertEqual(percent_parsed["length_range_selection"]["target_duration_seconds"], 180.0)

        all_selected, _all_video, all_report = long_video.select_long_video_length_range(
            plan,
            start_shot=2,
            limit_mode="全部剩余",
            limit_minutes=0.0,
            limit_percent=0.0,
            shot_count=0,
        )
        all_parsed = json.loads(all_report)
        self.assertEqual(all_parsed["selected_source_shots"], [2, 3, 4, 5])
        self.assertTrue(all_parsed["auto_all_remaining"])

        count_selected, _count_video, count_report = long_video.select_long_video_length_range(
            plan,
            start_shot=2,
            limit_mode="按镜头数量",
            limit_minutes=0.0,
            limit_percent=0.0,
            shot_count=2,
        )
        count_parsed = json.loads(count_report)
        self.assertEqual([item.index for item in count_selected.shots], [1, 2])
        self.assertEqual(count_parsed["selected_source_shots"], [2, 3])

    def test_continuity_range_rejects_selection_past_last_shot(self) -> None:
        plan = long_video.LongVideoShotPlan(
            video=object(),
            total_duration=2.0,
            fps=10.0,
            requested_mode="shot_aware",
            effective_mode="shot_aware",
            fixed_duration=10,
            sensitivity="标准",
            use_audio_silence=False,
            auto_fallback=True,
            detector="test",
            config={},
            boundaries=[],
            shots=[
                long_video.LogicalShot(1, 0.0, 1.0, 1.0, "video_start", "hard_cut"),
                long_video.LogicalShot(2, 1.0, 1.0, 2.0, "hard_cut", "video_end"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "只剩 1 个镜头"):
            long_video.select_continuous_shot_range(plan, start_shot=2, shot_count=2)

    def test_manual_batch_range_prefers_boundary_and_records_source_slices(self) -> None:
        class FakeVideo:
            def __init__(self, path: Path, duration: float):
                self.path = path
                self.duration = duration

            def get_stream_source(self):
                return str(self.path)

            def get_duration(self) -> float:
                return self.duration

            def as_trimmed(self, _start: float, duration: float, strict_duration: bool = True):
                return FakeVideo(self.path, duration)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            self._make_source_video(source, duration=70.0, fps=1)
            video = FakeVideo(source, 70.0)
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=70.0,
                fps=1.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[long_video.ShotBoundary(35.0, 35, "hard_cut", "test", 1.0)],
                shots=[
                    long_video.LogicalShot(1, 0.0, 70.0, 70.0, "video_start", "video_end"),
                ],
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                selected, _video, report, state_json = long_video.select_manual_batch_range(
                    plan,
                    action="新建系列",
                    series_id="manual_demo",
                    batch_minutes=0.5,
                    boundary_tolerance=10.0,
                )
            parsed = json.loads(report)
            state = json.loads(state_json)
            self.assertEqual(parsed["batch"]["source_start"], 0.0)
            self.assertEqual(parsed["batch"]["source_end"], 35.0)
            self.assertTrue(parsed["batch"]["virtual_slices"][0]["is_inside_shot_split"])
            self.assertEqual(parsed["batch"]["virtual_slices"][0]["parent_shot_index"], 1)
            self.assertEqual(selected.shots[0].start, 0.0)
            self.assertAlmostEqual(selected.total_duration, 35.0)
            self.assertEqual(state["contract"], long_video.MANUAL_BATCH_CONTRACT)
            self.assertTrue(Path(state["current_batch"]["state_path"]).is_file())

    def test_manual_batch_absolute_range_selects_exact_window(self) -> None:
        class FakeVideo:
            def __init__(self, path: Path, duration: float):
                self.path = path
                self.duration = duration

            def get_stream_source(self):
                return str(self.path)

            def get_duration(self) -> float:
                return self.duration

            def as_trimmed(self, _start: float, duration: float, strict_duration: bool = True):
                return FakeVideo(self.path, duration)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            self._make_source_video(source, duration=600.0, fps=1)
            video = FakeVideo(source, 600.0)
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=600.0,
                fps=1.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[
                    long_video.ShotBoundary(60.0, 60, "hard_cut", "test", 1.0),
                    long_video.ShotBoundary(120.0, 120, "hard_cut", "test", 1.0),
                    long_video.ShotBoundary(190.0, 190, "hard_cut", "test", 1.0),
                    long_video.ShotBoundary(240.0, 240, "hard_cut", "test", 1.0),
                ],
                shots=[
                    long_video.LogicalShot(1, 0.0, 60.0, 60.0, "video_start", "hard_cut"),
                    long_video.LogicalShot(2, 60.0, 60.0, 120.0, "hard_cut", "hard_cut"),
                    long_video.LogicalShot(3, 120.0, 70.0, 190.0, "hard_cut", "hard_cut"),
                    long_video.LogicalShot(4, 190.0, 50.0, 240.0, "hard_cut", "hard_cut"),
                    long_video.LogicalShot(5, 240.0, 360.0, 600.0, "hard_cut", "video_end"),
                ],
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                selected, _video, report, _state = long_video.select_manual_batch_range(
                    plan,
                    action="新建系列",
                    series_id="abs_demo",
                    batch_minutes=1.0,
                    boundary_tolerance=10.0,
                    start_second=30.0,
                    end_second=90.0,
                )
            parsed = json.loads(report)
            self.assertEqual(parsed["batch"]["source_start"], 30.0)
            self.assertEqual(parsed["batch"]["source_end"], 90.0)
            self.assertEqual(parsed["batch"]["source_shot_indices"], [1, 2])
            self.assertTrue(parsed["batch"]["absolute_range"]["exact_endpoints"])
            self.assertEqual(parsed["batch"]["absolute_range"]["requested_start_second"], 30.0)
            self.assertEqual(parsed["batch"]["absolute_range"]["requested_end_second"], 90.0)
            self.assertAlmostEqual(selected.total_duration, 60.0)

    def test_manual_batch_absolute_start_only_uses_batch_minutes(self) -> None:
        class FakeVideo:
            def __init__(self, path: Path, duration: float):
                self.path = path
                self.duration = duration

            def get_stream_source(self):
                return str(self.path)

            def get_duration(self) -> float:
                return self.duration

            def as_trimmed(self, _start: float, duration: float, strict_duration: bool = True):
                return FakeVideo(self.path, duration)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            self._make_source_video(source, duration=600.0, fps=1)
            video = FakeVideo(source, 600.0)
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=600.0,
                fps=1.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[
                    long_video.ShotBoundary(120.0, 120, "hard_cut", "test", 1.0),
                    long_video.ShotBoundary(190.0, 190, "hard_cut", "test", 1.0),
                    long_video.ShotBoundary(240.0, 240, "hard_cut", "test", 1.0),
                ],
                shots=[
                    long_video.LogicalShot(1, 0.0, 120.0, 120.0, "video_start", "hard_cut"),
                    long_video.LogicalShot(2, 120.0, 70.0, 190.0, "hard_cut", "hard_cut"),
                    long_video.LogicalShot(3, 190.0, 50.0, 240.0, "hard_cut", "hard_cut"),
                    long_video.LogicalShot(4, 240.0, 360.0, 600.0, "hard_cut", "video_end"),
                ],
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                selected, _video, report, _state = long_video.select_manual_batch_range(
                    plan,
                    action="新建系列",
                    series_id="abs_start",
                    batch_minutes=0.5,
                    boundary_tolerance=10.0,
                    start_second=120.0,
                    end_second=0.0,
                )
            parsed = json.loads(report)
            self.assertEqual(parsed["batch"]["source_start"], 120.0)
            self.assertEqual(parsed["batch"]["source_end"], 150.0)
            self.assertFalse(parsed["batch"]["absolute_range"]["exact_endpoints"])
            self.assertIsNone(parsed["batch"]["absolute_range"]["requested_end_second"])
            self.assertAlmostEqual(selected.total_duration, 30.0)

    def test_manual_batch_absolute_range_rejects_end_before_start(self) -> None:
        class FakeVideo:
            def __init__(self, path: Path, duration: float):
                self.path = path
                self.duration = duration

            def get_duration(self) -> float:
                return self.duration

        plan = long_video.LongVideoShotPlan(
            video=FakeVideo(Path("source.mp4"), 600.0),
            total_duration=600.0,
            fps=1.0,
            requested_mode="shot_aware",
            effective_mode="shot_aware",
            fixed_duration=10,
            sensitivity="标准",
            use_audio_silence=False,
            auto_fallback=True,
            detector="test",
            config={},
            boundaries=[],
            shots=[long_video.LogicalShot(1, 0.0, 600.0, 600.0, "video_start", "video_end")],
        )
        with self.assertRaises(ValueError):
            long_video.select_manual_batch_range(
                plan,
                action="新建系列",
                series_id="abs_bad",
                batch_minutes=1.0,
                boundary_tolerance=10.0,
                start_second=50.0,
                end_second=40.0,
            )

    def test_manual_batch_identity_mismatch_is_rejected(self) -> None:
        expected = {
            "path": "source.mp4",
            "size": 10,
            "mtime_ns": 1,
            "quick_hash": "old",
            "duration": 10.0,
            "file_duration": 10.0,
            "fps": 24.0,
            "frame_count": 240,
            "trim_start": 0.0,
            "trim_duration": 0.0,
        }
        changed = dict(expected, quick_hash="new")
        self.assertFalse(long_video._manual_batch_source_identity_matches(expected, changed))
        self.assertTrue(long_video._manual_batch_source_identity_matches(expected, dict(expected)))

    def test_manual_batch_commit_only_advances_cursor_and_rejects_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_video = root / "company_remote" / "manual_batch_series" / "demo" / "final.mp4"
            final_frame = root / "company_remote" / "manual_batch_series" / "demo" / "frame.png"
            final_video.parent.mkdir(parents=True)
            final_video.write_bytes(b"video")
            Image.new("RGB", (4, 4), "white").save(final_frame)
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                state = {
                    "contract": long_video.MANUAL_BATCH_CONTRACT,
                    "series_id": "demo",
                    "source_duration": 120.0,
                    "next_cursor": 0.0,
                    "completed_batches": [],
                    "current_batch": {
                        "batch_id": "B1_0_60000",
                        "batch_index": 1,
                        "attempt": 1,
                        "source_start": 0.0,
                        "source_end": 60.0,
                        "source_duration": 60.0,
                    },
                }
                invalid = {
                    "contract": long_video.MANUAL_BATCH_CONTRACT,
                    "series_id": "demo",
                    "batch_id": "B1_0_60000",
                    "batch_index": 1,
                    "attempt": 1,
                    "source_start": 0.0,
                    "source_end": 60.0,
                    "source_duration": 60.0,
                    "final_video": str(root / "missing.mp4"),
                    "final_frame": str(final_frame),
                }
                self.assertFalse(long_video._manual_batch_apply_commit(state, invalid))
                self.assertEqual(state["next_cursor"], 0.0)
                valid = dict(invalid, final_video=str(final_video), attempt=1)
                self.assertTrue(long_video._manual_batch_apply_commit(state, valid))
                self.assertEqual(state["next_cursor"], 60.0)
                self.assertEqual(state["last_committed_final_frame"], str(final_frame))

                malformed = dict(valid, attempt="not-a-number")
                self.assertFalse(long_video._manual_batch_apply_commit(state, malformed))
                self.assertEqual(state["next_cursor"], 60.0)

    def test_manual_batch_retry_replaces_same_batch_without_advancing_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_video = root / "company_remote" / "manual_batch_series" / "demo" / "first.mp4"
            first_frame = root / "company_remote" / "manual_batch_series" / "demo" / "first.png"
            retry_video = root / "company_remote" / "manual_batch_series" / "demo" / "retry.mp4"
            retry_frame = root / "company_remote" / "manual_batch_series" / "demo" / "retry.png"
            first_video.parent.mkdir(parents=True)
            first_video.write_bytes(b"first")
            retry_video.write_bytes(b"retry")
            Image.new("RGB", (4, 4), "white").save(first_frame)
            Image.new("RGB", (4, 4), "black").save(retry_frame)
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                state = {
                    "contract": long_video.MANUAL_BATCH_CONTRACT,
                    "series_id": "demo",
                    "source_duration": 120.0,
                    "next_cursor": 0.0,
                    "completed_batches": [],
                    "current_batch": {
                        "batch_id": "B1_0_60000",
                        "batch_index": 1,
                        "attempt": 1,
                        "source_start": 0.0,
                        "source_end": 60.0,
                        "source_duration": 60.0,
                    },
                }
                first = {
                    "contract": long_video.MANUAL_BATCH_CONTRACT,
                    "series_id": "demo",
                    "batch_id": "B1_0_60000",
                    "batch_index": 1,
                    "source_start": 0.0,
                    "source_end": 60.0,
                    "source_duration": 60.0,
                    "final_video": str(first_video),
                    "final_frame": str(first_frame),
                    "attempt": 1,
                }
                retry = dict(first, final_video=str(retry_video), final_frame=str(retry_frame), attempt=2)
                self.assertTrue(long_video._manual_batch_apply_commit(state, first))
                state["current_batch"]["attempt"] = 2
                self.assertTrue(long_video._manual_batch_apply_commit(state, retry))
                self.assertEqual(state["next_cursor"], 60.0)
                self.assertEqual(len(state["completed_batches"]), 1)
                self.assertEqual(state["completed_batches"][0]["attempt"], 2)
                self.assertEqual(len(state["completed_batches"][0]["attempt_history"]), 1)

    def test_manual_batch_rejects_incomplete_continue_and_unsafe_series_id(self) -> None:
        class FakeVideo:
            def __init__(self, path: Path, duration: float):
                self.path = path
                self.duration = duration

            def get_stream_source(self):
                return str(self.path)

            def get_duration(self) -> float:
                return self.duration

            def as_trimmed(self, _start: float, duration: float, strict_duration: bool = True):
                return FakeVideo(self.path, duration)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            self._make_source_video(source, duration=70.0, fps=1)
            plan = long_video.LongVideoShotPlan(
                video=FakeVideo(source, 70.0),
                total_duration=70.0,
                fps=1.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[],
                shots=[long_video.LogicalShot(1, 0.0, 70.0, 70.0, "video_start", "video_end")],
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                with self.assertRaisesRegex(ValueError, "系列 ID"):
                    long_video.select_manual_batch_range(
                        plan,
                        action="新建系列",
                        series_id="../escape",
                        batch_minutes=1.0,
                    )
                long_video.select_manual_batch_range(
                    plan,
                    action="新建系列",
                    series_id="continue_demo",
                    batch_minutes=1.0,
                )
                with self.assertRaisesRegex(ValueError, "当前批次还没有完成"):
                    long_video.select_manual_batch_range(
                        plan,
                        action="继续下一批",
                        series_id="continue_demo",
                        batch_minutes=1.0,
                    )

    def test_manual_batch_reference_role_is_cross_batch_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "asset.png"
            Image.new("RGB", (4, 4), "red").save(image_path)
            package = {"items": [{"role": "person_1", "path": str(image_path)}]}
            frame = torch.zeros((1, 4, 4, 3))
            adapter = long_video.get_video_engine_adapter("seedance")
            references, roles = long_video._references_from_auto_package(
                adapter,
                {"index": 1, "reference_package": package},
                previous_end_frame=frame,
                continuity_role="cross_batch_final_frame",
            )
            self.assertEqual(roles, ["person_1", "cross_batch_final_frame"])
            self.assertEqual(len(references), 2)

    def test_seedance_reference_package_uses_active_person_asset_uri(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "person.png"
            Image.new("RGB", (4, 4), "red").save(image_path)
            package = {
                "items": [
                    {"role": "person_1", "path": str(image_path), "asset_id": "asset-person"},
                    {"role": "scene_2", "path": str(image_path)},
                ]
            }
            references, roles = long_video._references_from_auto_package(
                long_video.get_video_engine_adapter("seedance"),
                {"index": 1, "reference_package": package},
                previous_end_frame=None,
            )

        self.assertEqual(roles, ["person_1", "scene_2"])
        self.assertIsInstance(references[0], company_client.SeedanceAssetReference)
        self.assertEqual(references[0].asset_id, "asset-person")
        self.assertIsInstance(references[1], torch.Tensor)

    def test_v3_packer_persists_failure_summary_for_all_group_members(self) -> None:
        member_one = {
            "index": 1,
            "auto_asset_status": "analysis_failed",
            "auto_asset_analysis_attempts": 3,
            "auto_asset_errors": [
                {
                    "kind": "analysis",
                    "attempts": 3,
                    "message": 'HTTP 503: {"error":{"code":"internal_server_error","message":"auth_unavailable: no auth available"}}',
                }
            ],
        }
        member_two = {
            "index": 2,
            "auto_asset_status": "degraded",
            "auto_asset_errors": [
                {"kind": "person", "error_kind": "tos_upload_failed", "message": "TOS 写入失败"}
            ],
        }
        task = {"index": 1, "logical_segments": [1, 2]}
        job = cast(
            long_video.LongVideoJob,
            SimpleNamespace(
                engine="seedance",
                manifest={
                    "auto_asset_options": {},
                    "logical_member_tasks": [member_one, member_two],
                    "tasks": [task],
                },
            ),
        )

        report = long_video._v3_pack_single_group_reference(
            job,
            task,
            {
                "members_by_index": {1: member_one, 2: member_two},
                "visual_style": "western",
                "send_source_video": False,
                "is_manual_batch": True,
                "cross_batch_frame": "",
                "package_version": "test",
            },
        )

        self.assertEqual(task["reference_package_status"], "blocked_by_asset_failure")
        self.assertEqual(task["status"], "blocked_by_asset_failure")
        self.assertEqual(task["reference_package_failure"]["failed_shot_indexes"], [1, 2])
        self.assertEqual(report["failure_summary"]["failed_shot_indexes"], [1, 2])
        self.assertEqual(
            {item["category"] for item in report["failure_summary"]["categories"]},
            {"auth_unavailable", "tos_upload_failed"},
        )

    def test_seedance_packer_prefers_active_people_and_scene_masters_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "person.png"
            scene = root / "scene.png"
            Image.new("RGB", (16, 16), "red").save(person)
            Image.new("RGB", (16, 16), "blue").save(scene)
            member = {
                "index": 1,
                "auto_asset_status": "ready",
                "auto_asset_analysis": {"story_action": "人物站在广场。", "people": [{"appearance": "成年人"}]},
                "auto_assets": {
                    "people": [
                        {
                            "person_id": "person-1",
                            "path": str(person),
                            "publication": {"asset_library": {"status": "active", "asset_id": "asset-person"}},
                        },
                        {
                            "person_id": "person-2",
                            "path": str(person),
                            "publication": {"asset_library": {"status": "warning", "asset_id": "asset-not-ready"}},
                        },
                    ],
                    "scenes": [{"place_id": "square", "version_id": "v1", "path": str(scene)}],
                    "integrated_frames": [
                        {
                            "role": "frame_start",
                            "path": str(scene),
                            "quality": {
                                "verdict": "approved",
                                "target_style_score": 0.96,
                                "composition_preservation": 0.92,
                            },
                        }
                    ],
                },
            }
            task = {"index": 1, "logical_segments": [1]}
            job = long_video.LongVideoJob(
                video=None,
                assets=long_video.LongVideoAssets(manifest={}, people={}, backgrounds={}),
                prompt="测试",
                engine="seedance",
                model="Seedance 2.0 Fast",
                segment_duration=10,
                ai_model="gpt-5.4",
                max_retries=0,
                resume=False,
                force_rerun=False,
                negative_prompt="",
                total_duration=4.0,
                source_path="",
                job_dir=root / "job",
                manifest_path=root / "job" / "manifest.json",
                manifest={"auto_asset_options": {}},
            )
            long_video._v3_pack_single_group_reference(
                job,
                task,
                {
                    "members_by_index": {1: member},
                    "visual_style": "western",
                    "send_source_video": False,
                    "is_manual_batch": False,
                    "cross_batch_frame": "",
                    "package_version": "test",
                },
            )

        items = task["reference_package"]["items"]
        self.assertEqual([item["role"] for item in items], ["person_1", "scene_1"])
        self.assertEqual(items[0]["asset_id"], "asset-person")
        self.assertEqual(items[0]["kind"], "person_asset")
        self.assertEqual(items[1]["kind"], "scene_master")
        self.assertEqual(task["reference_package"]["packed_asset_count"], 2)
        self.assertEqual(task["reference_package"]["asset_ids"], ["asset-person"])
        self.assertFalse(task["reference_package"]["uses_integrated_frame_references"])

    def test_seedance_packer_adds_approved_integrated_frames_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "person.png"
            scene = root / "scene.png"
            frame = root / "frame.png"
            Image.new("RGB", (16, 16), "red").save(person)
            Image.new("RGB", (16, 16), "blue").save(scene)
            Image.new("RGB", (16, 16), "green").save(frame)
            member = {
                "index": 1,
                "auto_asset_status": "ready",
                "auto_asset_analysis": {"story_action": "人物站在广场。", "people": []},
                "auto_assets": {
                    "people": [{"person_id": "person-1", "path": str(person)}],
                    "scenes": [{"place_id": "square", "version_id": "v1", "path": str(scene)}],
                    "integrated_frames": [
                        {
                            "role": "frame_start",
                            "path": str(frame),
                            "quality": {"verdict": "approved", "target_style_score": 0.96},
                        }
                    ],
                },
            }
            task = {"index": 1, "logical_segments": [1]}
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    engine="seedance",
                    prompt="测试",
                    job_dir=root / "job",
                    manifest={"auto_asset_options": {"use_integrated_frame_references": True}},
                ),
            )
            long_video._v3_pack_single_group_reference(
                job,
                task,
                {
                    "members_by_index": {1: member},
                    "visual_style": "western",
                    "send_source_video": False,
                    "is_manual_batch": False,
                    "cross_batch_frame": "",
                    "package_version": "test",
                },
            )

        self.assertEqual(
            [item["role"] for item in task["reference_package"]["items"]],
            ["person_1", "scene_1", "integrated_frame_1"],
        )
        self.assertTrue(task["reference_package"]["uses_integrated_frame_references"])

    def test_seedance_payload_sends_active_person_asset_uri_without_tos_upload(self) -> None:
        config = company_client.RemoteMediaConfig(
            name="seedance2-test",
            base_url="https://ark.cn-beijing.volces.com",
            submit_path="/api/v3/contents/generations/tasks",
            request_template={"content": [{"type": "text", "text": "{prompt}"}]},
            tos_enabled=True,
            tos_bucket="bucket",
            tos_endpoint="tos-cn-beijing.volces.com",
            tos_region="cn-beijing",
        )
        image = torch.zeros((1, 4, 4, 3))
        asset = company_client.SeedanceAssetReference("asset-person", role="person_1")
        with mock.patch.object(company_client, "_upload_media_to_tos", return_value=("https://tos.test/scene", "scene.png")) as upload:
            payload, media_debug = company_client._build_payload_with_debug(
                config,
                values={"operation": "seedance2_reference_video", "prompt": "test"},
                images={"reference_images": [asset, image]},
                videos={},
            )

        self.assertEqual(payload["content"][1]["image_url"]["url"], "asset://asset-person")
        self.assertEqual(media_debug[0]["delivery"], "asset_uri")
        self.assertEqual(media_debug[0]["source"]["asset_id"], "asset-person")
        upload.assert_called_once()

    def test_continuity_range_materializes_selected_source_for_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=4.0)
            video = long_video.InputImpl.VideoFromFile(str(source))
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=4.0,
                fps=10.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[long_video.ShotBoundary(2.0, 20, "hard_cut", "test", 1.0)],
                shots=[
                    long_video.LogicalShot(1, 0.0, 2.0, 2.0, "video_start", "hard_cut"),
                    long_video.LogicalShot(2, 2.0, 2.0, 4.0, "hard_cut", "video_end"),
                ],
            )
            selected, _selected_video, _report = long_video.select_continuous_shot_range(
                plan,
                start_shot=2,
                shot_count=1,
            )
            image = torch.ones((1, 16, 16, 3))
            assets = long_video.LongVideoAssets(
                manifest={"people": [{"id": "A"}], "backgrounds": [{"id": "BG01"}]},
                people={"A": image},
                backgrounds={"BG01": image},
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(root / "output")):
                job = long_video.plan_long_video_job(
                    video=selected.video,
                    assets=assets,
                    prompt="test",
                    engine="seedance",
                    model="Seedance 2.0 Fast",
                    segment_duration=10,
                    ai_model="gpt-5.4",
                    max_retries=0,
                    resume=False,
                    force_rerun=False,
                    negative_prompt="",
                    shot_plan=selected,
                )
            self.assertNotEqual(Path(job.source_path), source)
            self.assertTrue(Path(job.source_path).is_file())
            with av.open(job.source_path, mode="r") as container:
                persisted_duration = float(container.duration / av.time_base)
            self.assertAlmostEqual(persisted_duration, 2.0, delta=0.11)

    def test_video_source_path_streams_trim_without_tensor_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=5.0, fps=10)
            trimmed = long_video.InputImpl.VideoFromFile(str(source)).as_trimmed(
                1.25,
                2.0,
                strict_duration=True,
            )
            self.assertIsNotNone(trimmed)

            with mock.patch.object(trimmed, "save_to", side_effect=AssertionError("save_to must not be called")):
                materialized = Path(long_video._video_source_path(trimmed, root / "streamed"))

            with av.open(str(materialized), mode="r") as container:
                duration = float(container.duration / av.time_base)
                dimensions = (container.streams.video[0].width, container.streams.video[0].height)
                has_audio = bool(container.streams.audio)

        self.assertAlmostEqual(duration, 2.0, delta=0.11)
        self.assertEqual(dimensions, (64, 48))
        self.assertTrue(has_audio)

    def test_fade_detection_is_merged_without_duplicate_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fade.mp4"
            frames = [np.full((48, 64, 3), (200, 40, 40), dtype=np.uint8) for _ in range(10)]
            frames.extend(
                np.full((48, 64, 3), np.array((200, 40, 40)) * (9 - index) / 10, dtype=np.uint8)
                for index in range(10)
            )
            frames.extend(np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5))
            frames.extend(
                np.full((48, 64, 3), np.array((40, 40, 200)) * (index + 1) / 10, dtype=np.uint8)
                for index in range(10)
            )
            frames.extend(np.full((48, 64, 3), (40, 40, 200), dtype=np.uint8) for _ in range(10))
            self._make_video_from_frames(path, frames, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(path))
            plan, _status, _previews = long_video.detect_long_video_shots(
                video=video,
                mode="镜头优先（推荐）",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
            )
        self.assertEqual(len(plan.boundaries), 1)
        self.assertEqual(plan.boundaries[0].kind, "fade")
        self.assertIn("threshold", plan.boundaries[0].detector)

    def test_continuous_camera_motion_is_not_split_into_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_motion.mp4"
            gradient = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
            base = np.stack((gradient, np.flip(gradient, axis=1), gradient), axis=-1)
            frames = [np.roll(base, shift=index * 2, axis=1) for index in range(40)]
            self._make_video_from_frames(path, frames, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(path))
            plan, _status, _previews = long_video.detect_long_video_shots(
                video=video,
                mode="镜头优先（推荐）",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
            )
        self.assertEqual(plan.boundaries, [])
        self.assertEqual(len(plan.shots), 1)

    def test_detector_failure_falls_back_and_records_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallback.mp4"
            self._make_source_video(path, duration=3.0)
            video = long_video.InputImpl.VideoFromFile(str(path))
            with mock.patch.object(long_video, "_run_scene_detectors", side_effect=RuntimeError("synthetic failure")):
                plan, status, _previews = long_video.detect_long_video_shots(
                    video=video,
                    mode="镜头优先（推荐）",
                    fixed_duration=10,
                    sensitivity="标准",
                    use_audio_silence=False,
                    auto_fallback=True,
                )
        parsed = json.loads(status)
        self.assertEqual(plan.effective_mode, "fixed_fallback")
        self.assertIn("synthetic failure", parsed["fallback_reason"])

    def test_long_shot_is_split_near_low_motion_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long_shot.mp4"
            frames = [np.full((48, 64, 3), (80, 120, 160), dtype=np.uint8) for _ in range(220)]
            self._make_video_from_frames(path, frames, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(path))
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=22.0,
                fps=10.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[],
                shots=[long_video.LogicalShot(1, 0.0, 22.0, 22.0, "video_start", "video_end")],
            )
            requests, details = long_video.adapt_shot_plan_to_requests(plan, engine="wan", source_path=str(path))
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(item.request_duration <= 10 for item in requests))
        self.assertAlmostEqual(sum(item.output_duration for item in requests), 22.0, places=5)
        self.assertIn(requests[0].split_reason, {"low_motion", "low_motion_silence"})
        self.assertEqual(details["engine_max_request_duration"], 10.0)

    def test_short_shot_is_padded_for_request_and_trimmed_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "short.mp4"
            self._make_source_video(source, duration=0.8, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(source))
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=0.8,
                fps=10.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[],
                shots=[long_video.LogicalShot(1, 0.0, 0.8, 0.8, "video_start", "video_end")],
            )
            requests, _details = long_video.adapt_shot_plan_to_requests(plan, engine="wan", source_path=str(source))
            task = requests[0].to_dict()
            job = SimpleNamespace(video=video, source_path=str(source), job_dir=root / "job", force_rerun=True)
            padded = long_video._source_segment_for_task(job, task)
            self.assertAlmostEqual(float(padded.get_duration()), 2.0, delta=0.15)
            padded_path = Path(padded.get_stream_source())
            trimmed = root / "trimmed.mp4"
            long_video._normalize_segment(
                padded_path,
                trimmed,
                duration=task["duration"],
                width=64,
                height=48,
                fps=10,
                trim_offset=task["trim_offset"],
            )
            with av.open(str(trimmed), mode="r") as container:
                output_duration = float(container.duration / av.time_base)
        self.assertAlmostEqual(requests[0].request_duration, 2.0)
        self.assertAlmostEqual(requests[0].trim_offset, 0.6)
        self.assertAlmostEqual(output_duration, 0.8, delta=0.11)

    def test_unpadded_shot_is_materialized_before_remote_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=6.0, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(source))
            task = {
                "index": 2,
                "start": 1.5,
                "duration": 2.8,
                "source_start": 1.5,
                "source_duration": 2.8,
                "request_duration": 3.0,
                "padding_start": 0.0,
                "padding_end": 0.0,
            }
            job = SimpleNamespace(video=video, source_path=str(source), job_dir=root / "job", force_rerun=True)

            segment = long_video._source_segment_for_task(job, task)
            content, _mime, _extension, source_info = company_client._video_to_bytes(segment)
            uploaded_duration = company_client._probe_video_duration_seconds(
                content,
                source_label="materialized request",
            )
            segment_path = Path(segment.get_stream_source())
            segment_duration = float(segment.get_duration())

        self.assertNotEqual(segment_path, source)
        self.assertEqual(source_info["basename"], "request_0002.mp4")
        self.assertAlmostEqual(segment_duration, 2.8, delta=0.15)
        self.assertAlmostEqual(uploaded_duration, 2.8, delta=0.15)

    def test_video_upload_materializes_active_trim_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            self._make_source_video(source, duration=5.0, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(source)).as_trimmed(
                1.25,
                2.0,
                strict_duration=True,
            )
            self.assertIsNotNone(video)

            content, mime, extension, source_info = company_client._video_to_bytes(video)
            uploaded_duration = company_client._probe_video_duration_seconds(
                content,
                source_label="trimmed upload",
            )

        self.assertEqual(mime, "video/mp4")
        self.assertEqual(extension, ".mp4")
        self.assertTrue(source_info["trim_materialized"])
        self.assertAlmostEqual(source_info["trim_start_seconds"], 1.25, places=6)
        self.assertAlmostEqual(source_info["trim_duration_seconds"], 2.0, places=6)
        self.assertAlmostEqual(uploaded_duration, 2.0, delta=0.15)

    def test_seedance_short_shot_is_padded_to_four_seconds_and_trimmed_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "short.mp4"
            self._make_source_video(source, duration=1.8, fps=10)
            video = long_video.InputImpl.VideoFromFile(str(source))
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=1.8,
                fps=10.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[],
                shots=[long_video.LogicalShot(1, 0.0, 1.8, 1.8, "video_start", "video_end")],
            )
            requests, details = long_video.adapt_shot_plan_to_requests(plan, engine="seedance", source_path=str(source))
            with (
                mock.patch.object(long_video, "_get_config", return_value=object()),
                mock.patch.object(long_video, "generate_video", return_value=(video, "generated.mp4")) as generate_request,
            ):
                long_video.get_video_engine_adapter("seedance").generate(
                    source_segment=video,
                    references=[],
                    prompt="测试",
                    model="Seedance 2.0 Fast",
                    duration=requests[0].request_duration,
                    negative_prompt="",
                )
            task = requests[0].to_dict()
            job = SimpleNamespace(video=video, source_path=str(source), job_dir=root / "job", force_rerun=True)
            padded = long_video._source_segment_for_task(job, task)
            self.assertAlmostEqual(float(padded.get_duration()), 4.0, delta=0.15)
            padded_path = Path(padded.get_stream_source())
            trimmed = root / "trimmed.mp4"
            long_video._normalize_segment(
                padded_path,
                trimmed,
                duration=task["duration"],
                width=64,
                height=48,
                fps=10,
                trim_offset=task["trim_offset"],
            )
            with av.open(str(trimmed), mode="r") as container:
                output_duration = float(container.duration / av.time_base)
        self.assertAlmostEqual(requests[0].request_duration, 4.0)
        self.assertAlmostEqual(requests[0].trim_offset, 1.1)
        self.assertEqual(details["engine_min_request_duration"], 4.0)
        self.assertIsInstance(generate_request.call_args.kwargs["duration"], int)
        self.assertEqual(generate_request.call_args.kwargs["duration"], 4)
        self.assertAlmostEqual(output_duration, 1.8, delta=0.11)

    def test_seedance_can_use_anime_images_without_uploading_source_video(self) -> None:
        reference = torch.ones((1, 24, 32, 3))
        with (
            mock.patch.object(long_video, "_get_config", return_value=object()),
            mock.patch.object(long_video, "generate_video", return_value=(None, "generated.mp4")) as generate_request,
        ):
            long_video.get_video_engine_adapter("seedance").generate(
                source_segment=object(),
                references=[reference],
                prompt="全动漫视频",
                model="Seedance 2.0 Fast",
                duration=4,
                negative_prompt="真人",
                include_source_video=False,
            )

        kwargs = generate_request.call_args.kwargs
        self.assertEqual(len(kwargs["reference_images"]), 1)
        self.assertIs(kwargs["reference_images"][0], reference)
        self.assertIsNone(kwargs["reference_video"])
        self.assertEqual(kwargs["reference_videos"], [])

    def test_anime_asset_prompts_convert_people_and_complete_scene(self) -> None:
        person_prompt = long_video._person_asset_prompt(
            {"appearance": "短发、黑色夹克、成年人"},
            visual_style="anime_2d",
        )
        scene_prompt = long_video._scene_asset_prompt(
            "城市街道与咖啡店",
            position="开始",
            visual_style="anime_2d",
        )

        self.assertIn("二维动漫角色", person_prompt)
        self.assertIn("禁止保留照片纹理", person_prompt)
        self.assertIn("整个环境完整重绘", scene_prompt)
        self.assertIn("二维动漫背景", scene_prompt)
        self.assertIn("移除所有真人", scene_prompt)

    def test_western_asset_prompts_use_replacement_contract(self) -> None:
        person_prompt = long_video._person_asset_prompt(
            {"appearance": "短发、黑色夹克、成年人"},
            visual_style=long_video.AUTO_ASSET_STYLE_WESTERN,
        )
        scene_prompt = long_video._scene_asset_prompt(
            "城市街道与咖啡店",
            position="开始",
            visual_style=long_video.AUTO_ASSET_STYLE_WESTERN,
        )
        frame_prompt = long_video._integrated_frame_prompt(
            {"people": [], "background": {"first_description": "城市街道"}},
            role="frame_start",
            person_master_count=0,
            has_scene_master=True,
            has_style_master=False,
            visual_style=long_video.AUTO_ASSET_STYLE_WESTERN,
            style_prompt="",
        )

        self.assertIn("替换母版", person_prompt)
        self.assertIn("替换参考图中的原人物", person_prompt)
        self.assertIn("替换原建筑", scene_prompt)
        self.assertIn("整帧替换", frame_prompt)
        self.assertNotIn("整帧重绘", frame_prompt)

    def test_target_resource_presets_switch_all_prompt_layers(self) -> None:
        cases = [
            (
                company_nodes.TARGET_RESOURCE_WESTERN,
                long_video.AUTO_ASSET_STYLE_WESTERN,
                "欧美",
                "欧美",
            ),
            (
                company_nodes.TARGET_RESOURCE_PHOTOREAL,
                long_video.AUTO_ASSET_STYLE_PHOTOREAL,
                "真人影视",
                "真人影视实景",
            ),
            (
                company_nodes.TARGET_RESOURCE_ANIME,
                long_video.AUTO_ASSET_STYLE_ANIME,
                "二维动漫",
                "二维动漫背景",
            ),
            (
                company_nodes.TARGET_RESOURCE_CG_3D,
                long_video.AUTO_ASSET_STYLE_CG_3D,
                "3D 游戏 CG",
                "PBR 材质",
            ),
            (
                company_nodes.TARGET_RESOURCE_COMIC,
                long_video.AUTO_ASSET_STYLE_COMIC,
                "漫画插画",
                "漫画插画背景",
            ),
        ]
        for resource_type, visual_style, person_keyword, scene_keyword in cases:
            with self.subTest(resource_type=resource_type):
                normalized_type, resolved_style, prompt, negative = company_nodes._target_resource_settings(
                    resource_type,
                    company_nodes.ANIME_LONG_VIDEO_PROMPT,
                    company_nodes.ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
                )
                person_prompt = long_video._person_asset_prompt(
                    {"appearance": "短发、黑色外套"},
                    visual_style=resolved_style,
                    style_prompt=prompt,
                )
                scene_prompt = long_video._scene_asset_prompt(
                    "城市街道",
                    position="开始",
                    visual_style=resolved_style,
                    style_prompt=prompt,
                )
                video_prompt = long_video._build_auto_asset_video_prompt(
                    prompt,
                    {"story_action": "人物向前走", "people": [{"appearance": "短发"}]},
                    reference_roles=["P1", "scene_start"],
                    soft_continuity=False,
                    visual_style=resolved_style,
                    send_source_video=False,
                )

                self.assertEqual(normalized_type, resource_type)
                self.assertEqual(resolved_style, visual_style)
                self.assertIn(person_keyword, person_prompt)
                self.assertIn(scene_keyword, scene_prompt)
                self.assertIn(person_keyword, video_prompt)
                self.assertTrue(negative)

    def test_custom_resource_type_preserves_user_prompts(self) -> None:
        result = company_nodes._target_resource_settings(
            company_nodes.TARGET_RESOURCE_CUSTOM,
            "水彩剪纸风格，低饱和冷色",
            "霓虹色，照片纹理",
        )

        self.assertEqual(result[0], company_nodes.TARGET_RESOURCE_CUSTOM)
        self.assertEqual(result[1], long_video.AUTO_ASSET_STYLE_CUSTOM)
        self.assertEqual(result[2], "水彩剪纸风格，低饱和冷色")
        self.assertEqual(result[3], "霓虹色，照片纹理")

    def test_legacy_custom_prompt_is_not_overwritten_by_anime_default(self) -> None:
        result = company_nodes._target_resource_settings("", "旧工作流的自定义风格", "旧负面提示词")

        self.assertEqual(result[0], company_nodes.TARGET_RESOURCE_CUSTOM)
        self.assertEqual(result[1], long_video.AUTO_ASSET_STYLE_CUSTOM)
        self.assertEqual(result[2], "旧工作流的自定义风格")
        self.assertEqual(result[3], "旧负面提示词")

    def test_target_resource_widget_is_appended_for_legacy_workflow_compatibility(self) -> None:
        schema = company_nodes.CompanyLongVideoAnimeAssetPlanner.define_schema()
        input_ids = [item.id for item in schema.inputs]

        self.assertEqual(input_ids[-3:], ["image_provider", "negative_prompt", "target_resource_type"])
        target_input = schema.inputs[-1]
        self.assertTrue(target_input.optional)
        self.assertEqual(getattr(target_input, "default"), company_nodes.TARGET_RESOURCE_ANIME)

    def test_v3_planner_adds_audio_switch_without_changing_legacy_schema(self) -> None:
        legacy = company_nodes.CompanyLongVideoAnimeAssetPlanner.define_schema()
        v3 = company_nodes.CompanyLongVideoAnimeAssetPlannerV3.define_schema()

        self.assertNotIn("use_original_audio", [item.id for item in legacy.inputs])
        self.assertEqual(v3.inputs[-3].id, "use_original_audio")
        self.assertTrue(getattr(v3.inputs[-3], "optional"))
        self.assertFalse(getattr(v3.inputs[-3], "default"))
        self.assertEqual(v3.inputs[-2].id, "use_integrated_frame_references")
        self.assertTrue(getattr(v3.inputs[-2], "optional"))
        self.assertFalse(getattr(v3.inputs[-2], "default"))
        self.assertEqual(v3.inputs[-1].id, "identity_mapping_json")
        self.assertTrue(getattr(v3.inputs[-1], "optional"))
        self.assertEqual(getattr(v3.inputs[-1], "default"), "")
        required_ids = [item.id for item in v3.inputs if not getattr(item, "optional")]
        optional_ids = [item.id for item in v3.inputs if getattr(item, "optional")]
        self.assertEqual(required_ids[-1], "image_provider")
        self.assertEqual(
            optional_ids[-5:],
            [
                "negative_prompt",
                "target_resource_type",
                "use_original_audio",
                "use_integrated_frame_references",
                "identity_mapping_json",
            ],
        )
        self.assertEqual(v3.node_id, "CompanyLongVideoAnimeAssetPlannerV3")

    def test_manual_batch_workflow_widgets_follow_comfy_input_order(self) -> None:
        workflow_root = Path(__file__).resolve().parents[3] / "user/default/workflows"
        workflow_names = (
            "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅.json",
            "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_流水线.json",
        )
        expected_tail = [
            "negative_prompt",
            "target_resource_type",
            "use_original_audio",
            "use_integrated_frame_references",
            "identity_mapping_json",
        ]
        for workflow_name in workflow_names:
            workflow = json.loads((workflow_root / workflow_name).read_text(encoding="utf-8"))
            planner = next(node for node in workflow["nodes"] if node["id"] == 4)
            widget_names = [item["name"] for item in planner["inputs"] if item.get("widget")]
            widget_values = dict(zip(widget_names, planner["widgets_values"], strict=True))
            self.assertEqual(widget_names[-5:], expected_tail, workflow_name)
            self.assertEqual(widget_values["target_resource_type"], company_nodes.TARGET_RESOURCE_WESTERN)
            self.assertIsInstance(widget_values["use_original_audio"], bool)
            self.assertIsInstance(widget_values["use_integrated_frame_references"], bool)
            self.assertIsInstance(widget_values["identity_mapping_json"], str)

    def test_manual_batch_range_selector_workflow_exposes_absolute_range(self) -> None:
        workflow_root = Path(__file__).resolve().parents[3] / "user/default/workflows"
        workflow_name = "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_含入库素材库查看.json"
        workflow = json.loads((workflow_root / workflow_name).read_text(encoding="utf-8"))
        selector = next(node for node in workflow["nodes"] if node["id"] == 3)
        self.assertEqual(selector["type"], "CompanyLongVideoManualBatchRangeSelector")
        widget_names = [item["name"] for item in selector["inputs"] if item.get("widget")]
        self.assertEqual(widget_names[-2:], ["start_minute", "end_minute"])
        widget_values = dict(zip(widget_names, selector["widgets_values"], strict=True))
        self.assertEqual(widget_values["start_minute"], 0.5)
        self.assertEqual(widget_values["end_minute"], 1.0)

    def test_v3_planner_passes_processing_contract_and_audio_choice(self) -> None:
        planned_job = SimpleNamespace(manifest={"job_id": "v3-test"}, manifest_path=Path("manifest.json"))
        with mock.patch.object(company_nodes, "plan_long_video_auto_asset_job", return_value=planned_job) as planner:
            company_nodes.CompanyLongVideoAnimeAssetPlannerV3.execute(
                shot_plan=object(),
                prompt=company_nodes.ANIME_LONG_VIDEO_PROMPT,
                model="Seedance 2.0 Fast",
                target_resource_type=company_nodes.TARGET_RESOURCE_ANIME,
                use_original_audio=True,
            )

        self.assertEqual(planner.call_args.kwargs["processing_contract_version"], 3)
        self.assertTrue(planner.call_args.kwargs["use_original_audio"])
        self.assertFalse(planner.call_args.kwargs["use_integrated_frame_references"])

    def test_manual_batch_finalizer_reads_series_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_video = root / "batch.mp4"
            final_video.write_bytes(b"video")
            job = SimpleNamespace(
                manifest={
                    "processing_contract_version": company_nodes.MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
                    "status": "success",
                    "final": str(final_video),
                    "manual_batch": {"series_id": "series-1", "batch_id": "B1", "attempt": 1},
                }
            )
            state = {
                "contract": company_nodes.MANUAL_BATCH_CONTRACT,
                "series_id": "series-1",
                "current_batch": {"batch_id": "B1", "attempt": 1},
                "series_complete": False,
                "next_cursor": 30.0,
            }
            with mock.patch.object(company_nodes, "_manual_batch_read_state", return_value=(state, root / "state.json")):
                result = company_nodes.CompanyLongVideoManualBatchFinalizerV1.execute(job)

        report = json.loads(result[3])
        self.assertEqual(report["series_id"], "series-1")
        self.assertEqual(report["batch_id"], "B1")
        self.assertEqual(report["status"], "completed")

    def test_target_resource_frontend_updates_both_prompt_widgets(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "web/target_resource_prompt_presets.js"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn('const TYPE_WIDGET = "target_resource_type"', script)
        self.assertIn('const PROMPT_WIDGET = "prompt"', script)
        self.assertIn('const NEGATIVE_WIDGET = "negative_prompt"', script)
        self.assertIn('"欧美化资源"', script)
        self.assertIn("而不是只更换面孔", script)
        self.assertIn("彻底重构为真实可信", script)
        self.assertIn("applyPreset(node, value)", script)

    def test_planner_execute_enforces_selected_resource_preset(self) -> None:
        planned_job = SimpleNamespace(manifest={"job_id": "style-test"}, manifest_path=Path("manifest.json"))
        with mock.patch.object(company_nodes, "plan_long_video_auto_asset_job", return_value=planned_job) as planner:
            company_nodes.CompanyLongVideoAnimeAssetPlanner.execute(
                shot_plan=object(),
                prompt=company_nodes.ANIME_LONG_VIDEO_PROMPT,
                model="Seedance 2.0 Fast",
                negative_prompt="WisArt",
                target_resource_type=company_nodes.TARGET_RESOURCE_PHOTOREAL,
            )

        options = planner.call_args.kwargs
        self.assertEqual(options["target_resource_type"], company_nodes.TARGET_RESOURCE_PHOTOREAL)
        self.assertEqual(options["visual_style"], long_video.AUTO_ASSET_STYLE_PHOTOREAL)
        self.assertEqual(options["prompt"], company_nodes.PHOTOREAL_LONG_VIDEO_PROMPT)
        self.assertEqual(options["negative_prompt"], company_nodes.PHOTOREAL_LONG_VIDEO_NEGATIVE_PROMPT)
        self.assertFalse(options["send_source_video"])

    def test_auto_asset_v2_checks_and_reuses_same_person_even_when_analysis_confidence_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 24, 32, 3))
            person = {
                "slot": "P1",
                "appearance": "短发、蓝色夹克、成年人",
                "first_bbox": [0.1, 0.1, 0.8, 0.9],
                "last_bbox": [0.1, 0.1, 0.8, 0.9],
                "identity_key": "blue_jacket_person",
                "reuse_confidence": 0.05,
            }
            generated = torch.full((1, 24, 32, 3), 0.5)
            with mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated) as image_request:
                first, first_error = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                )
                self.assertIsNone(first_error)
                first["publication"] = {
                    "tos": {"status": "uploaded", "object_key": "safe/person.png"},
                    "asset_library": {"status": "active", "asset_id": "asset-reused-person"},
                }
                long_video._merge_auto_asset_cache_entry(cache, first)

                with mock.patch.object(
                    long_video,
                    "_verify_auto_asset_cache",
                    return_value={"decision": "same", "confidence": 0.99, "hard_mismatches": []},
                ) as verify_request:
                    second, second_error = long_video._create_auto_person_asset(
                        root=root / "shot_0002",
                        person=person,
                        frames=frames,
                        cache=cache,
                        analysis_model="gpt-5.4",
                        image_model="gpt-image-2",
                        image_quality="low",
                        image_provider="WisArt",
                        reuse_threshold=0.92,
                    )

        self.assertIsNone(second_error)
        self.assertTrue(second["reused_from_cache"])
        self.assertEqual(second["path"], first["path"])
        self.assertEqual(image_request.call_count, 1)
        self.assertEqual(verify_request.call_count, 1)
        self.assertTrue(second["source_observations"])
        self.assertEqual(long_video._active_person_asset_id(second), "asset-reused-person")

    def test_auto_asset_v2_hard_person_mismatch_creates_new_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 24, 32, 3))
            first_person = {
                "slot": "P1",
                "appearance": "短发、蓝色夹克、成年人",
                "first_bbox": [0.1, 0.1, 0.8, 0.9],
                "identity_key": "jacket_person",
                "reuse_confidence": 1.0,
            }
            second_person = {**first_person, "appearance": "长发、红色连衣裙、成年人"}
            generated = torch.full((1, 24, 32, 3), 0.5)
            with mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated) as image_request:
                first, _ = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=first_person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                )
                long_video._merge_auto_asset_cache_entry(cache, first)
                with mock.patch.object(
                    long_video,
                    "_verify_auto_asset_cache",
                    return_value={
                        "decision": "different",
                        "confidence": 0.2,
                        "hard_mismatches": ["服装颜色冲突"],
                    },
                ):
                    second, second_error = long_video._create_auto_person_asset(
                        root=root / "shot_0002",
                        person=second_person,
                        frames=frames,
                        cache=cache,
                        analysis_model="gpt-5.4",
                        image_model="gpt-image-2",
                        image_quality="low",
                        image_provider="WisArt",
                        reuse_threshold=0.92,
                    )

        self.assertIsNone(second_error)
        self.assertFalse(second["reused_from_cache"])
        self.assertNotEqual(second["person_id"], first["person_id"])
        self.assertEqual(image_request.call_count, 2)
        self.assertEqual(second["suspected_matches"], [])

    def test_auto_asset_v2_uncertain_person_match_is_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 24, 32, 3))
            person = {
                "slot": "P1",
                "appearance": "短发、深色外套、成年人",
                "first_bbox": [0.1, 0.1, 0.8, 0.9],
                "identity_key": "dark_coat_person",
                "reuse_confidence": 1.0,
            }
            generated = torch.full((1, 24, 32, 3), 0.5)
            with mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated):
                first, _ = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                )
                long_video._merge_auto_asset_cache_entry(cache, first)
                with mock.patch.object(
                    long_video,
                    "_verify_auto_asset_cache",
                    return_value={
                        "decision": "uncertain",
                        "confidence": 0.86,
                        "reasons": ["侧脸遮挡，服装相似但脸部信息不足"],
                    },
                ):
                    second, second_error = long_video._create_auto_person_asset(
                        root=root / "shot_0002",
                        person=person,
                        frames=frames,
                        cache=cache,
                        analysis_model="gpt-5.4",
                        image_model="gpt-image-2",
                        image_quality="low",
                        image_provider="WisArt",
                        reuse_threshold=0.92,
                    )

        self.assertIsNone(second_error)
        self.assertFalse(second["reused_from_cache"])
        self.assertEqual(second["suspected_matches"][0]["decision"], "uncertain")
        self.assertIn("侧脸遮挡", second["suspected_matches"][0]["reason"])

    def test_identity_mapping_loader_normalizes_and_rejects_invalid(self) -> None:
        empty = long_video._load_identity_mapping("")
        self.assertEqual(empty["global_people"], {})
        self.assertEqual(empty["shot_people"], {})

        mapping = long_video._load_identity_mapping(
            json.dumps(
                {
                    "expected_distinct_people": 2,
                    "global_people": {
                        "hero": {"asset_id": " asset-hero-1 ", "status": "Confirmed"},
                        "": {"asset_id": "asset-dropped"},
                    },
                    "shot_people": {"3:P1": "hero", "3:P2": "", "": "hero"},
                }
            )
        )
        self.assertEqual(mapping["expected_distinct_people"], 2)
        self.assertEqual(mapping["global_people"]["hero"]["asset_id"], "asset-hero-1")
        self.assertEqual(mapping["global_people"]["hero"]["status"], "confirmed")
        self.assertNotIn("", mapping["global_people"])
        self.assertEqual(mapping["shot_people"], {"3:P1": "hero"})

        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "mapping.json"
            mapping_path.write_text(json.dumps({"global_people": {"a": {"asset_id": "asset-1"}}}), encoding="utf-8")
            from_file = long_video._load_identity_mapping(str(mapping_path))
        self.assertEqual(from_file["global_people"]["a"]["asset_id"], "asset-1")

        with self.assertRaises(ValueError):
            long_video._load_identity_mapping("{ not json")
        with self.assertRaises(ValueError):
            long_video._load_identity_mapping("[1, 2]")

    def test_identity_mapping_applies_ignore_and_linked_states(self) -> None:
        def build_task() -> dict:
            return {
                "index": 3,
                "auto_asset_analysis": {
                    "people": [
                        {"slot": "P1", "appearance": "远景背影"},
                        {"slot": "P2", "appearance": "灰西装男子"},
                    ]
                },
            }

        mapping = long_video._load_identity_mapping(
            {
                "global_people": {"hero": {"asset_id": "asset-hero-1", "path": "", "status": "confirmed"}},
                "shot_people": {"3:P1": "ignore", "shot_0003:P2": "hero"},
            }
        )
        task = build_task()
        long_video._apply_identity_mapping_to_analysis(task, mapping)
        first, second = task["auto_asset_analysis"]["people"]
        self.assertEqual(first["identity_state"], "partial")
        self.assertEqual(first["global_person_id"], "")
        self.assertEqual(second["identity_state"], "linked")
        self.assertEqual(second["global_person_id"], "hero")
        self.assertEqual(second["mapped_asset_id"], "asset-hero-1")

        unknown = long_video._load_identity_mapping({"shot_people": {"3:P2": "missing"}})
        with self.assertRaises(ValueError):
            long_video._apply_identity_mapping_to_analysis(build_task(), unknown)

        pending = long_video._load_identity_mapping(
            {
                "global_people": {"hero": {"asset_id": "asset-hero-1", "status": "pending"}},
                "shot_people": {"3:P2": "hero"},
            }
        )
        with self.assertRaises(ValueError):
            long_video._apply_identity_mapping_to_analysis(build_task(), pending)

    def test_identity_gate_blocks_low_confidence_new_person_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 24, 32, 3))
            person = {
                "slot": "P1",
                "appearance": "远景中的模糊人物",
                "first_bbox": [0.1, 0.1, 0.8, 0.9],
                "identity_key": "distant_person",
                "reuse_confidence": 0.3,
            }
            with mock.patch.object(long_video, "_generate_auto_asset_image") as image_request:
                entry, error = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                    enforce_identity_gate=True,
                )

        self.assertIsNone(error)
        self.assertEqual(entry["identity_state"], "partial")
        self.assertEqual(entry["path"], "")
        self.assertEqual(entry["person_id"], "")
        self.assertEqual(image_request.call_count, 0)

    def test_identity_gate_requires_review_for_uncertain_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 400, 400, 3))
            person = {
                "slot": "P1",
                "appearance": "短发、深色外套、成年人",
                "first_bbox": [0.05, 0.05, 0.95, 0.95],
                "identity_key": "dark_coat_person",
                "reuse_confidence": 1.0,
            }
            generated = torch.full((1, 256, 256, 3), 0.5)
            with mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated):
                first, first_error = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                    enforce_identity_gate=True,
                )
                self.assertIsNone(first_error)
                self.assertEqual(first["identity_state"], "confirmed")
                long_video._merge_auto_asset_cache_entry(cache, first)
                with mock.patch.object(
                    long_video,
                    "_verify_auto_asset_cache",
                    return_value={
                        "decision": "uncertain",
                        "confidence": 0.8,
                        "matched_features": ["脸型与发型接近", "年龄与体型一致"],
                        "missing_fields": [],
                        "reasons": ["两人五官与造型高度相似，但仍不足以自动确认是否同一人。"],
                    },
                ):
                    second, second_error = long_video._create_auto_person_asset(
                        root=root / "shot_0002",
                        person=person,
                        frames=frames,
                        cache=cache,
                        analysis_model="gpt-5.4",
                        image_model="gpt-image-2",
                        image_quality="low",
                        image_provider="WisArt",
                        reuse_threshold=0.92,
                        enforce_identity_gate=True,
                    )

        self.assertIsNone(second)
        self.assertEqual(second_error["error_kind"], "identity_review_required")
        self.assertEqual(second_error["identity_state"], "unresolved")

    def test_uncertain_match_needs_review_only_for_usable_evidence(self) -> None:
        self.assertTrue(
            long_video._uncertain_match_needs_review(
                {"decision": "uncertain", "confidence": 0.82, "missing_fields": []}
            )
        )
        # 候选帧没有可比对的人物：不是真实歧义，不应拦截。
        self.assertFalse(
            long_video._uncertain_match_needs_review(
                {"decision": "uncertain", "confidence": 0.99, "missing_fields": ["候选帧未见人物"]}
            )
        )
        # 校验请求失败伪造的 uncertain（置信度 0）：不应拦截。
        self.assertFalse(
            long_video._uncertain_match_needs_review(
                {"decision": "uncertain", "confidence": 0.0, "reasons": ["Remote request failed"]}
            )
        )
        self.assertFalse(long_video._uncertain_match_needs_review({"decision": "same", "confidence": 0.99}))

    def test_identity_gate_ignores_uncertain_without_comparable_person(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 400, 400, 3))
            person = {
                "slot": "P1",
                "appearance": "年轻女性、发髻、浅色立领上衣",
                "first_bbox": [0.05, 0.05, 0.95, 0.95],
                "identity_key": "shot_person_1",
                "reuse_confidence": 1.0,
            }
            generated = torch.full((1, 256, 256, 3), 0.5)
            with mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated) as image_request:
                first, first_error = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                    enforce_identity_gate=True,
                )
                self.assertIsNone(first_error)
                long_video._merge_auto_asset_cache_entry(cache, first)
                with mock.patch.object(
                    long_video,
                    "_verify_auto_asset_cache",
                    return_value={
                        "decision": "uncertain",
                        "confidence": 0.99,
                        "missing_fields": ["候选帧仅见云海与建筑，未见可辨识人物"],
                        "reasons": ["候选原始画面没有人物可供身份比对。"],
                    },
                ):
                    second, second_error = long_video._create_auto_person_asset(
                        root=root / "shot_0002",
                        person=person,
                        frames=frames,
                        cache=cache,
                        analysis_model="gpt-5.4",
                        image_model="gpt-image-2",
                        image_quality="low",
                        image_provider="WisArt",
                        reuse_threshold=0.92,
                        enforce_identity_gate=True,
                    )

        self.assertIsNone(second_error)
        self.assertIsNotNone(second)
        self.assertEqual(second["identity_state"], "confirmed")
        self.assertNotEqual(second["path"], "")
        self.assertEqual(image_request.call_count, 2)
        self.assertTrue(second["suspected_matches"])
        self.assertFalse(second["suspected_matches"][0]["needs_review"])

    def test_identity_gate_ignores_uncertain_from_request_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 400, 400, 3))
            person = {
                "slot": "P1",
                "appearance": "年轻女性、发髻、浅色立领上衣",
                "first_bbox": [0.05, 0.05, 0.95, 0.95],
                "identity_key": "shot_person_1",
                "reuse_confidence": 1.0,
            }
            generated = torch.full((1, 256, 256, 3), 0.5)
            with mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated) as image_request:
                first, _ = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                    enforce_identity_gate=True,
                )
                long_video._merge_auto_asset_cache_entry(cache, first)
                with mock.patch.object(
                    long_video,
                    "_verify_auto_asset_cache",
                    return_value={
                        "decision": "uncertain",
                        "confidence": 0.0,
                        "missing_fields": [],
                        "reasons": ['Remote request failed with HTTP 500: {"error":{"code":"internal_server_error"}}'],
                    },
                ):
                    second, second_error = long_video._create_auto_person_asset(
                        root=root / "shot_0002",
                        person=person,
                        frames=frames,
                        cache=cache,
                        analysis_model="gpt-5.4",
                        image_model="gpt-image-2",
                        image_quality="low",
                        image_provider="WisArt",
                        reuse_threshold=0.92,
                        enforce_identity_gate=True,
                    )

        self.assertIsNone(second_error)
        self.assertIsNotNone(second)
        self.assertEqual(second["identity_state"], "confirmed")
        self.assertEqual(image_request.call_count, 2)

    def test_mapped_person_reuses_manual_asset_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frames = torch.ones((3, 24, 32, 3))
            person = {
                "slot": "P1",
                "appearance": "灰西装男子",
                "first_bbox": [0.1, 0.1, 0.8, 0.9],
                "identity_key": "suit_person",
                "reuse_confidence": 0.2,
                "identity_state": "linked",
                "global_person_id": "hero",
                "mapped_asset_id": "asset-hero-1",
                "mapped_asset_path": "",
            }
            with mock.patch.object(long_video, "_generate_auto_asset_image") as image_request:
                entry, error = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=frames,
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                    enforce_identity_gate=True,
                )

        self.assertIsNone(error)
        self.assertEqual(entry["identity_state"], "linked")
        self.assertEqual(entry["global_person_id"], "hero")
        self.assertTrue(entry["reused_from_cache"])
        self.assertEqual(long_video._active_person_asset_id(entry), "asset-hero-1")
        self.assertEqual(image_request.call_count, 0)

    def test_valid_auto_asset_entries_accept_active_asset_id_without_path(self) -> None:
        assets = {
            "people": [
                {
                    "slot": "P1",
                    "path": "",
                    "publication": {"asset_library": {"status": "active", "asset_id": "asset-hero-1"}},
                },
                {"slot": "P2", "path": ""},
            ]
        }
        entries = long_video._valid_auto_asset_entries(assets, "people", "slot")
        self.assertIn("P1", entries)
        self.assertNotIn("P2", entries)

    def test_identity_mapping_store_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                missing = long_video.load_identity_mapping_record("series-1")
                self.assertFalse(missing["exists"])

                saved = long_video.save_identity_mapping_record(
                    "series-1",
                    {
                        "global_people": {"hero": {"asset_id": "asset-hero-1"}},
                        "shot_people": {"1:P1": "hero"},
                    },
                )
                self.assertTrue(saved["exists"])
                self.assertTrue(Path(saved["path"]).is_file())

                loaded = long_video.load_identity_mapping_record("series-1")
                self.assertTrue(loaded["exists"])
                self.assertEqual(loaded["mapping"]["global_people"]["hero"]["asset_id"], "asset-hero-1")
                self.assertEqual(loaded["mapping"]["shot_people"], {"1:P1": "hero"})

                with self.assertRaises(ValueError):
                    long_video.save_identity_mapping_record("bad/series", {})

    def test_planner_v3_forwards_identity_mapping(self) -> None:
        schema = company_nodes.CompanyLongVideoAnimeAssetPlannerV3.define_schema()
        input_ids = [getattr(item, "id", "") for item in schema.inputs]
        self.assertIn("identity_mapping_json", input_ids)
        self.assertEqual(input_ids[-1], "identity_mapping_json")

        mapping_json = json.dumps({"global_people": {"hero": {"asset_id": "asset-hero-1"}}})
        planned_job = SimpleNamespace(manifest={"job_id": "map-test"}, manifest_path=Path("manifest.json"))
        with mock.patch.object(company_nodes, "plan_long_video_auto_asset_job", return_value=planned_job) as planner:
            company_nodes.CompanyLongVideoAnimeAssetPlannerV3.execute(
                shot_plan=object(),
                prompt=company_nodes.ANIME_LONG_VIDEO_PROMPT,
                model="Seedance 2.0 Fast",
                identity_mapping_json=mapping_json,
            )
        self.assertEqual(planner.call_args.kwargs["identity_mapping"], mapping_json)

    def test_asset_library_viewer_lists_registered_people_and_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            user_root = root / "user"
            (user_root / "default" / "company_remote").mkdir(parents=True, exist_ok=True)
            output_root.mkdir(parents=True, exist_ok=True)
            person_png = output_root / "person.png"
            scene_png = output_root / "scene.png"
            long_video._save_image_tensor(torch.full((1, 128, 96, 3), 0.6), person_png)
            long_video._save_image_tensor(torch.full((1, 128, 96, 3), 0.3), scene_png)

            cache = long_video._empty_auto_asset_cache()
            cache["people"]["by_id"]["shot_0004_P1"] = {
                "person_id": "shot_0004_P1",
                "appearance": "灰西装男子",
                "converted_path": str(person_png),
                "publication": {"asset_library": {"status": "active", "asset_id": "asset-hero-1"}},
                "source_observations": [],
            }
            cache["people"]["by_id"]["shot_0006_P1"] = {
                "person_id": "shot_0006_P1",
                "appearance": "本地未入库人物",
                "converted_path": str(person_png),
                "publication": {"asset_library": {"status": "skipped", "asset_id": ""}},
                "source_observations": [],
            }
            cache["scenes"]["by_id"]["place-1"] = {
                "place_id": "place-1",
                "description": "山门场景",
                "style_master_path": str(scene_png),
            }
            library = {
                "output_root": str(output_root.resolve()),
                "styles": {long_video.AUTO_ASSET_STYLE_WESTERN: {"cache": cache}},
            }
            library_path = user_root / "default" / "company_remote" / "long_video_asset_library.json"
            library_path.write_text(json.dumps(library, ensure_ascii=False), encoding="utf-8")

            with (
                mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(output_root)),
                mock.patch.object(long_video.folder_paths, "get_user_directory", return_value=str(user_root)),
            ):
                inventory = long_video.collect_registered_asset_inventory(long_video.AUTO_ASSET_STYLE_WESTERN)
                grid, report, summary = long_video.build_asset_library_view(long_video.AUTO_ASSET_STYLE_WESTERN, 3)
                # 全部样式聚合也应包含 western。
                all_inventory = long_video.collect_registered_asset_inventory("")

        self.assertEqual(inventory["summary"]["people"], 2)
        self.assertEqual(inventory["summary"]["scenes"], 1)
        self.assertEqual(inventory["summary"]["registered_volcano_assets"], 1)
        registered = [item for item in inventory["people"] if item["registered"]]
        self.assertEqual(registered[0]["asset_id"], "asset-hero-1")
        self.assertTrue(all(entry["has_image"] for entry in inventory["entries"]))
        self.assertEqual(all_inventory["summary"]["total"], 3)
        self.assertIsInstance(grid, torch.Tensor)
        self.assertEqual(grid.ndim, 4)
        self.assertIn("已注册火山素材 1 个", summary)
        parsed = json.loads(report)
        self.assertEqual(parsed["summary"]["total"], 3)

    def test_asset_library_viewer_empty_library_returns_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            output_root.mkdir(parents=True, exist_ok=True)
            (root / "user").mkdir(parents=True, exist_ok=True)
            with (
                mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(output_root)),
                mock.patch.object(long_video.folder_paths, "get_user_directory", return_value=str(root / "user")),
            ):
                grid, report, summary = long_video.build_asset_library_view(long_video.AUTO_ASSET_STYLE_WESTERN, 3)
        self.assertEqual(grid.ndim, 4)
        self.assertIn("入库资源共 0 项", summary)
        self.assertEqual(json.loads(report)["summary"]["total"], 0)

    def test_auto_asset_v2_legacy_cache_without_source_observation_does_not_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_path = root / "legacy.png"
            long_video._save_image_tensor(torch.full((1, 16, 16, 3), 0.4), legacy_path)
            cache = long_video._normalize_auto_asset_cache(
                {
                    "version": 1,
                    "people": {
                        "legacy_person": {
                            "path": str(legacy_path),
                            "appearance": "旧缓存人物",
                        }
                    },
                    "scenes": {},
                }
            )
            person = {
                "slot": "P1",
                "appearance": "旧缓存人物",
                "first_bbox": [0.1, 0.1, 0.8, 0.9],
                "identity_key": "legacy_person",
                "reuse_confidence": 1.0,
            }
            with (
                mock.patch.object(long_video, "_generate_auto_asset_image", return_value=torch.ones((1, 16, 16, 3))),
                mock.patch.object(long_video, "_verify_auto_asset_cache") as verify_request,
            ):
                entry, error = long_video._create_auto_person_asset(
                    root=root / "shot_0001",
                    person=person,
                    frames=torch.ones((3, 24, 32, 3)),
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                )

        self.assertIsNone(error)
        self.assertFalse(entry["reused_from_cache"])
        self.assertEqual(entry["suspected_matches"][0]["candidate_id"], "legacy:legacy_person")
        verify_request.assert_not_called()

    def test_auto_asset_v2_same_place_new_view_reuses_replacement_scene_master(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = long_video._empty_auto_asset_cache()
            frame = torch.ones((1, 24, 32, 3))
            frame_a = root / "shot_0001" / "source_start.png"
            frame_b = root / "shot_0002" / "source_start.png"
            long_video._save_image_tensor(frame, frame_a)
            long_video._save_image_tensor(frame, frame_b)
            generated = torch.full((1, 24, 32, 3), 0.5)
            with mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated) as image_request:
                first, _ = long_video._create_auto_scene_asset(
                    root=root / "shot_0001",
                    role="scene_start",
                    source_frame=frame,
                    source_frame_path=frame_a,
                    description="街角咖啡店白天正面机位",
                    scene_key="street_cafe",
                    cache=cache,
                    analysis_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="low",
                    image_provider="WisArt",
                    reuse_threshold=0.92,
                    allow_cache=True,
                )
                long_video._merge_auto_asset_cache_entry(cache, first)
                with mock.patch.object(
                    long_video,
                    "_verify_auto_asset_cache",
                    return_value={"decision": "same_place_new_version", "confidence": 0.95},
                ):
                    second, second_error = long_video._create_auto_scene_asset(
                        root=root / "shot_0002",
                        role="scene_start",
                        source_frame=frame,
                        source_frame_path=frame_b,
                        description="街角咖啡店夜晚侧面机位",
                        scene_key="street_cafe",
                        cache=cache,
                        analysis_model="gpt-5.4",
                        image_model="gpt-image-2",
                        image_quality="low",
                        image_provider="WisArt",
                        reuse_threshold=0.92,
                        allow_cache=True,
                    )

        self.assertIsNone(second_error)
        self.assertTrue(second["reused_from_cache"])
        self.assertEqual(second["reuse_scope"], "same_place")
        self.assertEqual(second["place_id"], first["place_id"])
        self.assertEqual(second["path"], first["path"])
        self.assertEqual(image_request.call_count, 1)

    def test_anime_segment_generation_skips_source_segment_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=4.0)
            reference_path = root / "anime_reference.png"
            long_video._save_image_tensor(torch.ones((1, 24, 32, 3)), reference_path)
            result_path = root / "job" / "segments" / "segment_0001.mp4"
            manifest_path = root / "job" / "manifest.json"
            task = {
                "index": 1,
                "start": 0.0,
                "duration": 4.0,
                "source_start": 0.0,
                "source_duration": 4.0,
                "request_duration": 4.0,
                "trim_offset": 0.0,
                "status": "matched",
                "result": str(result_path),
                "prompt": "全动漫视频",
                "reference_package_status": "ready",
                "reference_package": {
                    "items": [{"role": "person_P1", "path": str(reference_path)}],
                },
            }
            manifest = {
                "version": 4,
                "job_id": "anime-no-source-video",
                "asset_mode": "auto_shot_assets",
                "auto_asset_options": {"visual_style": "anime_2d", "send_source_video": False},
                "tasks": [task],
            }
            job = long_video.LongVideoJob(
                video=long_video.InputImpl.VideoFromFile(str(source)),
                assets=long_video.LongVideoAssets(manifest={}, people={}, backgrounds={}),
                prompt="全动漫视频",
                engine="seedance",
                model="Seedance 2.0 Fast",
                segment_duration=10,
                ai_model="gpt-5.6-terra",
                max_retries=0,
                resume=False,
                force_rerun=False,
                negative_prompt="真人",
                total_duration=4.0,
                source_path=str(source),
                job_dir=root / "job",
                manifest_path=manifest_path,
                manifest=manifest,
            )
            long_video._atomic_write_json(manifest_path, manifest)

            with (
                mock.patch.object(
                    long_video,
                    "_source_segment_for_task",
                    side_effect=AssertionError("真人原分镜不得被物化或上传"),
                ) as source_request,
                mock.patch.object(
                    long_video.VideoEngineAdapter,
                    "generate",
                    return_value=(None, str(source)),
                ) as generate_request,
            ):
                result = long_video.generate_long_video_segments(job)

        source_request.assert_not_called()
        self.assertIsNone(generate_request.call_args.kwargs["source_segment"])
        self.assertFalse(generate_request.call_args.kwargs["include_source_video"])
        self.assertFalse(result.manifest["tasks"][0]["source_video_sent"])
        self.assertEqual(result.manifest["tasks"][0]["status"], "success")

    def test_real_person_privacy_error_retries_without_source_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=4.0)
            reference_path = root / "western_reference.png"
            long_video._save_image_tensor(torch.ones((1, 24, 32, 3)), reference_path)
            result_path = root / "job" / "segments" / "segment_0001.mp4"
            manifest_path = root / "job" / "manifest.json"
            task = {
                "index": 1,
                "start": 0.0,
                "duration": 4.0,
                "source_start": 0.0,
                "source_duration": 4.0,
                "request_duration": 4.0,
                "trim_offset": 0.0,
                "status": "matched",
                "result": str(result_path),
                "prompt": "原视频定义演绎内容。",
                "auto_asset_analysis": {
                    "shot_stability": "stable",
                    "story_action": "人物沿街道向前行走。",
                    "people": [{"appearance": "欧美影视风格成年人"}],
                },
                "reference_package_status": "ready",
                "reference_package": {
                    "items": [{"role": "person_P1", "path": str(reference_path)}],
                },
            }
            manifest = {
                "version": 4,
                "job_id": "privacy-fallback",
                "asset_mode": "auto_shot_assets",
                "auto_asset_options": {"visual_style": "western", "send_source_video": True},
                "tasks": [task],
            }
            job = long_video.LongVideoJob(
                video=long_video.InputImpl.VideoFromFile(str(source)),
                assets=long_video.LongVideoAssets(manifest={}, people={}, backgrounds={}),
                prompt="欧美化重新演绎",
                engine="seedance",
                model="Seedance 2.0 Fast",
                segment_duration=10,
                ai_model="gpt-5.6-terra",
                max_retries=0,
                resume=False,
                force_rerun=False,
                negative_prompt="",
                total_duration=4.0,
                source_path=str(source),
                job_dir=root / "job",
                manifest_path=manifest_path,
                manifest=manifest,
            )
            long_video._atomic_write_json(manifest_path, manifest)
            privacy_error = RuntimeError(
                "InputImageSensitiveContentDetected.PrivacyInformation: "
                "The request failed because the input image may contain real person."
            )
            with mock.patch.object(
                long_video.VideoEngineAdapter,
                "generate",
                side_effect=[privacy_error, (None, str(source))],
            ) as generate_request:
                result = long_video.generate_long_video_segments(job)

        self.assertEqual(generate_request.call_count, 2)
        self.assertTrue(generate_request.call_args_list[0].kwargs["include_source_video"])
        self.assertFalse(generate_request.call_args_list[1].kwargs["include_source_video"])
        updated = result.manifest["tasks"][0]
        self.assertEqual(updated["status"], "success")
        self.assertFalse(updated["source_video_sent"])
        self.assertEqual(updated["privacy_fallback"]["reason"], "real_person_privacy_detected")
        self.assertIn("依据本段剧情分析文字", updated["prompt"])

    def test_v2_manifest_migrates_to_v3(self) -> None:
        migrated = long_video._migrate_job_manifest(
            {"version": 2, "segment_duration": 10, "tasks": [{"index": 1, "start": 0.0, "duration": 5.0}]}
        )
        self.assertEqual(migrated["version"], 3)
        self.assertEqual(migrated["migrated_from_version"], 2)
        self.assertEqual(migrated["tasks"][0]["source_duration"], 5.0)
        self.assertEqual(migrated["segmentation"]["effective_mode"], "legacy_fixed_v2")

    def test_planner_can_resume_matching_v2_signature_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source)
            video = long_video.InputImpl.VideoFromFile(str(source))
            image = torch.ones((1, 16, 16, 3))
            assets, _summary = long_video.load_long_video_assets(
                '{"people":[{"id":"A"}],"backgrounds":[{"id":"BG01"}]}',
                people={"A": image},
                backgrounds={"BG01": image},
            )
            legacy_values = {
                "engine": "seedance",
                "model": "Seedance 2.0 Fast",
                "segment_duration": 10,
                "prompt": "test",
                "negative_prompt": "",
                "ai_model": "gpt-5.4",
                "asset_images": long_video._asset_image_signatures(assets.people, assets.backgrounds),
            }
            legacy_signature = long_video._job_signature(str(source), assets.manifest, legacy_values)
            legacy_dir = root / long_video.JOB_ROOT_NAME / legacy_signature[:20]
            legacy_dir.mkdir(parents=True)
            legacy_manifest = legacy_dir / "manifest.json"
            legacy_manifest.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "engine": "seedance",
                        "segment_duration": 10,
                        "tasks": [
                            {
                                "index": 1,
                                "logical_segment": 1,
                                "start": 0.0,
                                "duration": 2.4,
                                "status": "success",
                                "attempts": 1,
                                "result": str(source),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(root)):
                job = long_video.plan_long_video_job(
                    video=video,
                    assets=assets,
                    prompt="test",
                    engine="seedance",
                    model="Seedance 2.0 Fast",
                    segment_duration=10,
                    ai_model="gpt-5.4",
                    max_retries=0,
                    resume=True,
                    force_rerun=False,
                    negative_prompt="",
                )
        self.assertEqual(job.manifest["version"], 3)
        self.assertEqual(job.manifest["tasks"][0]["status"], "success")
        self.assertEqual(job.manifest["resumed_from_manifest"], str(legacy_manifest))

    def test_silence_detection_returns_audio_pause_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "silent.mp4"
            self._make_source_video(source, duration=2.4)
            points = long_video._detect_silence_points(str(source))
        self.assertTrue(points)
        self.assertGreaterEqual(points[0], 0.0)
        self.assertLessEqual(points[0], 2.4)

    def test_segment_windows_for_20_seconds(self) -> None:
        self.assertEqual(long_video.build_segment_windows(20, 10), [(0.0, 10.0), (10.0, 10.0)])
        self.assertEqual(long_video.build_segment_windows(20, 15), [(0.0, 15.0), (15.0, 5.0)])

    def test_short_tail_is_moved_into_previous_window(self) -> None:
        self.assertEqual(long_video.build_segment_windows(16, 15), [(0.0, 14.0), (14.0, 2.0)])

    def test_wan_splits_fifteen_second_window(self) -> None:
        windows = long_video.build_segment_windows(20, 15)
        self.assertEqual(
            long_video.split_windows_for_engine(windows, "wan"),
            [(0.0, 10.0, 1), (10.0, 5.0, 1), (15.0, 5.0, 2)],
        )
        self.assertEqual(
            long_video.split_windows_for_engine(windows, "seedance"),
            [(0.0, 15.0, 1), (15.0, 5.0, 2)],
        )

    def test_wan_people_contact_sheet_reserves_reference_slots(self) -> None:
        people = [torch.zeros((1, 16, 16, 3)) + index / 4 for index in range(1, 4)]
        background = [torch.ones((1, 16, 16, 3))]
        previous = torch.zeros((1, 16, 16, 3))
        references, roles = long_video._reference_images_for_task(
            long_video.get_video_engine_adapter("wan"),
            selected_people=people,
            selected_backgrounds=background,
            previous_end_frame=previous,
        )
        self.assertEqual(roles, ["people_contact_sheet", "background", "previous_segment_end_frame"])
        self.assertEqual(len(references), 3)
        self.assertEqual(tuple(references[0].shape), (1, 768, 1536, 3))

    def test_asset_manifest_is_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=directory):
                image = torch.ones((1, 8, 8, 3))
                _, first_path = long_video.create_asset_manifest(
                    asset_name="demo",
                    mapping_json='{"mapping":"A -> A"}',
                    people={"A": image},
                    backgrounds={"BG01": image},
                )
                time.sleep(0.000001)
                _, second_path = long_video.create_asset_manifest(
                    asset_name="demo",
                    mapping_json='{"mapping":"A -> A"}',
                    people={"A": image},
                    backgrounds={"BG01": image},
                )
            self.assertNotEqual(first_path, second_path)
            first = json.loads(Path(first_path).read_text(encoding="utf-8"))
            self.assertTrue(first["people"][0]["styled"].startswith("company_remote/long_video_assets/demo/"))

    def test_asset_mapping_includes_people_and_backgrounds(self) -> None:
        response = json.dumps(
            {
                "people": {"A": {"source": "原人物 A", "identity": "人物特征"}},
                "backgrounds": {"BG01": {"source": "原背景 BG01", "description": "场景特征"}},
                "mapping": {
                    "people": {"A": "原人物 A -> 欧美化人物 A"},
                    "backgrounds": {"BG01": "原背景 BG01 -> 欧美化背景 BG01"},
                },
            },
            ensure_ascii=False,
        )
        with (
            mock.patch.object(long_video, "_get_config", return_value=object()),
            mock.patch.object(long_video, "generate_openai_image_prompt_text", return_value=response) as request,
        ):
            result = long_video.analyze_asset_mapping(
                people={"A": torch.ones((1, 24, 16, 3))},
                backgrounds={"BG01": torch.ones((1, 16, 32, 3))},
                model="gpt-5.4",
            )
        parsed = json.loads(result)
        self.assertIn("A", parsed["mapping"]["people"])
        self.assertIn("BG01", parsed["mapping"]["backgrounds"])
        self.assertEqual(tuple(request.call_args.kwargs["image"].shape), (2, 512, 512, 3))

    def test_ffmpeg_normalize_concat_and_original_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source)
            segment_a = root / "a.mp4"
            segment_b = root / "b.mp4"
            long_video._normalize_segment(source, segment_a, duration=1.2, width=64, height=48, fps=10)
            long_video._normalize_segment(source, segment_b, duration=1.2, width=64, height=48, fps=10)
            final = root / "final.mp4"
            long_video._concat_and_mux([segment_a, segment_b], str(source), final, 2.4, root)
            self.assertTrue(final.is_file())
            with av.open(str(final), mode="r") as container:
                self.assertTrue(container.streams.video)
                self.assertTrue(container.streams.audio)
                actual = float(container.duration / av.time_base)
            self.assertGreaterEqual(actual, 2.2)
            self.assertLessEqual(actual, 2.6)

    def test_visible_stage_pipeline_without_remote_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source)
            video = long_video.InputImpl.VideoFromFile(str(source))
            image = torch.ones((1, 16, 16, 3))
            assets, _ = long_video.load_long_video_assets(
                '{"people":[{"id":"A"}],"backgrounds":[{"id":"BG01"}]}',
                people={"A": image},
                backgrounds={"BG01": image},
            )
            fake_analysis = {
                "people": ["A"],
                "backgrounds": ["BG01"],
                "action_prompt": "保持原视频剧情与动作。",
                "analysis_source": "test",
            }
            with (
                mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(root / "output")),
                mock.patch.object(long_video, "_segment_analysis", return_value=fake_analysis),
                mock.patch.object(long_video.VideoEngineAdapter, "generate", return_value=(None, str(source))),
            ):
                job = long_video.plan_long_video_job(
                    video=video,
                    assets=assets,
                    prompt="测试",
                    engine="seedance",
                    model="Seedance 2.0 Fast",
                    segment_duration=10,
                    ai_model="gpt-5.4",
                    max_retries=0,
                    resume=True,
                    force_rerun=False,
                    negative_prompt="",
                )
                self.assertEqual(job.manifest["version"], 3)
                self.assertEqual(job.manifest["segmentation"]["effective_mode"], "fixed")
                long_video.analyze_long_video_job(job)
                long_video.match_long_video_references(job)
                long_video.generate_long_video_segments(job)
                _, summary, previews = long_video.collect_long_video_results(job)
                final_video, final_path, manifest_path, status = long_video.merge_long_video_job(job)

            self.assertIn('"completed": 1', summary)
            self.assertEqual(tuple(previews.shape), (1, 512, 512, 3))
            self.assertTrue(Path(final_path).is_file())
            self.assertTrue(Path(manifest_path).is_file())
            self.assertEqual(json.loads(status)["status"], "success")
            self.assertGreater(float(final_video.get_duration()), 2.0)

    def test_continuity_preview_returns_each_generated_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            segment_root = output_root / "company_remote/long_video_jobs/test/segments"
            segment_root.mkdir(parents=True)
            first = segment_root / "segment_0001.mp4"
            second = segment_root / "segment_0002.mp4"
            self._make_source_video(first, duration=2.0)
            self._make_source_video(second, duration=2.0)
            job = SimpleNamespace(
                manifest_path=output_root / "company_remote/long_video_jobs/test/manifest.json",
                manifest={
                    "job_id": "test",
                    "continuity": "soft_previous_end_frame",
                    "tasks": [
                        {
                            "index": 1,
                            "logical_segment": 1,
                            "source_start": 0.0,
                            "duration": 2.0,
                            "status": "success",
                            "result": str(first),
                            "reference_roles": ["person_1", "background"],
                        },
                        {
                            "index": 2,
                            "logical_segment": 2,
                            "source_start": 2.0,
                            "duration": 2.0,
                            "status": "success",
                            "result": str(second),
                            "reference_roles": ["person_1", "background", "previous_segment_end_frame"],
                        },
                    ],
                }
            )
            with mock.patch.object(company_nodes.folder_paths, "get_output_directory", return_value=str(output_root)):
                result = company_nodes.CompanyLongVideoContinuityPreview.execute(job)
        report = json.loads(result[1])
        self.assertEqual(len(result.ui["images"]), 2)
        self.assertFalse(report["segments"][0]["uses_previous_end_frame"])
        self.assertTrue(report["segments"][1]["uses_previous_end_frame"])
        self.assertEqual(tuple(result[2].shape), (2, 512, 512, 3))

    def test_visual_workflow_exposes_all_stages(self) -> None:
        workflow_path = Path(__file__).resolve().parents[3] / "user/default/workflows/视频欧美转绘_长视频_可视化分阶段转绘.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node["type"] for node in workflow["nodes"]}
        expected = {
            "CompanyLongVideoAssetLoader",
            "CompanyLongVideoSegmentPlanner",
            "CompanyLongVideoSegmentAnalyzer",
            "CompanyLongVideoReferenceMatcher",
            "CompanyLongVideoSegmentGenerator",
            "CompanyLongVideoResultCollector",
            "CompanyLongVideoFinalMerger",
        }
        self.assertTrue(expected.issubset(node_types))
        self.assertEqual(len(workflow["links"]), 9)

    def test_shot_aware_workflow_exposes_detection_and_duration_adaptation(self) -> None:
        workflow_path = Path(__file__).resolve().parents[3] / "user/default/workflows/视频欧美转绘_长视频_镜头感知分阶段转绘.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node["type"] for node in workflow["nodes"]}
        expected = {
            "CompanyLongVideoAssetLoader",
            "CompanyLongVideoShotDetector",
            "CompanyLongVideoDurationAdapter",
            "CompanyLongVideoSegmentAnalyzer",
            "CompanyLongVideoReferenceMatcher",
            "CompanyLongVideoSegmentGenerator",
            "CompanyLongVideoResultCollector",
            "CompanyLongVideoFinalMerger",
        }
        self.assertTrue(expected.issubset(node_types))
        self.assertEqual(len(workflow["links"]), 11)

    def test_shot_detection_test_workflow_contains_no_remote_generation(self) -> None:
        workflow_path = Path(__file__).resolve().parents[3] / "user/default/workflows/视频欧美转绘_长视频_分镜检测测试.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node["type"] for node in workflow["nodes"]}
        self.assertIn("CompanyLongVideoShotDetector", node_types)
        self.assertIn("CompanyLongVideoShotInspector", node_types)
        self.assertIn("CompanyFixedColumnImagePreview", node_types)
        self.assertNotIn("CompanyLongVideoDurationAdapter", node_types)
        self.assertNotIn("CompanyLongVideoSegmentGenerator", node_types)
        self.assertNotIn("CompanyLongVideoFinalMerger", node_types)
        self.assertEqual(len(workflow["nodes"]), 8)
        self.assertEqual(len(workflow["links"]), 7)
        fixed_previews = [node for node in workflow["nodes"] if node["type"] == "CompanyFixedColumnImagePreview"]
        self.assertEqual([node["widgets_values"][0] for node in fixed_previews], ["3", "2"])
        self.assertEqual([len(node["widgets_values"]) for node in fixed_previews], [2, 2])

    def test_continuity_generation_test_workflow_limits_remote_scope(self) -> None:
        workflow_path = Path(__file__).resolve().parents[3] / "user/default/workflows/视频欧美转绘_长视频_连续分镜生成测试.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node["type"] for node in workflow["nodes"]}
        expected = {
            "CompanyLongVideoShotDetector",
            "CompanyLongVideoContinuityRangeSelector",
            "CompanyLongVideoAssetLoader",
            "CompanyLongVideoDurationAdapter",
            "CompanyLongVideoSegmentAnalyzer",
            "CompanyLongVideoReferenceMatcher",
            "CompanyLongVideoSegmentGenerator",
            "CompanyLongVideoContinuityPreview",
            "CompanyLongVideoFinalMerger",
            "SaveVideo",
        }
        self.assertTrue(expected.issubset(node_types))
        self.assertEqual(len(workflow["nodes"]), 15)
        self.assertEqual(len(workflow["links"]), 14)
        selector = next(node for node in workflow["nodes"] if node["type"] == "CompanyLongVideoContinuityRangeSelector")
        self.assertEqual(selector["widgets_values"], [1, 0])
        self.assertTrue(workflow["extra"]["long_video"]["remote_requests"])

    def test_auto_asset_job_builds_persistent_per_shot_assets_and_wan_reference_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=4.8)
            video = long_video.InputImpl.VideoFromFile(str(source))
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=4.8,
                fps=10.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[long_video.ShotBoundary(2.4, 24, "hard_cut", "test", 1.0)],
                shots=[
                    long_video.LogicalShot(1, 0.0, 2.4, 2.4, "video_start", "hard_cut"),
                    long_video.LogicalShot(2, 2.4, 2.4, 4.8, "hard_cut", "video_end"),
                ],
            )
            analysis = {
                "shot_stability": "stable",
                "story_action": "人物在街头自然交谈。",
                "people": [
                    {
                        "slot": "P1",
                        "appearance": "短发、夹克、手持咖啡的成年人",
                        "first_bbox": [0.1, 0.1, 0.7, 0.95],
                        "last_bbox": [0.1, 0.1, 0.7, 0.95],
                        "persistence": "both",
                        "identity_key": "jacket_coffee_person",
                        "reuse_confidence": 1.0,
                    }
                ],
                "background": {
                    "first_description": "街角咖啡店外景",
                    "last_description": "街角咖啡店外景",
                    "scene_key": "street_cafe",
                    "same_scene_confidence": 1.0,
                },
            }
            generated = torch.full((1, 24, 32, 3), 0.5)
            analysis_calls = 0

            def analyze_request(_frames, *, model):
                nonlocal analysis_calls
                analysis_calls += 1
                if analysis_calls == 1:
                    raise RuntimeError("temporary upstream EOF")
                return analysis

            image_request = mock.Mock(return_value=generated)
            verify_request = mock.Mock(return_value={"decision": "same", "confidence": 0.99})
            person_publication = {
                "tos": {"status": "uploaded", "object_key": "safe/person.png"},
                "asset_library": {
                    "status": "warning",
                    "asset_id": "",
                    "cache_reused": False,
                    "error": "素材库轮询超时",
                },
            }
            with (
                mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(root / "output")),
                mock.patch.object(long_video, "_auto_asset_analysis", side_effect=analyze_request),
                mock.patch.object(long_video.time, "sleep") as analysis_sleep,
                mock.patch.object(long_video, "_generate_auto_asset_image", image_request),
                mock.patch.object(long_video, "_verify_auto_asset_cache", verify_request),
                mock.patch.object(
                    long_video,
                    "publish_seedance_person_image",
                    return_value=person_publication,
                ) as publish_person,
            ):
                job = long_video.plan_long_video_auto_asset_job(
                    shot_plan=plan,
                    prompt="测试自动欧美化",
                    engine="wan",
                    model="wan2.7-r2v-2026-06-12",
                    ai_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="medium",
                    image_provider="AI-Zero-Token",
                    reuse_threshold=0.92,
                    max_retries=1,
                    resume=False,
                    force_rerun=False,
                    force_rerun_assets=False,
                    negative_prompt="",
                )
                result, report, previews = long_video.build_long_video_auto_assets(job, image_concurrency=0)
                packed, package_report, package_previews = long_video.pack_long_video_auto_references(result)

            self.assertEqual(job.manifest["version"], 4)
            self.assertEqual(job.manifest["asset_mode"], "auto_shot_assets")
            self.assertEqual(job.manifest["auto_asset_options"]["image_provider"], "AI-Zero-Token")
            self.assertEqual(image_request.call_count, 2)
            self.assertGreaterEqual(verify_request.call_count, 1)
            self.assertTrue(
                all(call.kwargs["image_provider"] == "AI-Zero-Token" for call in image_request.call_args_list)
            )
            self.assertEqual(analysis_calls, 3)
            analysis_sleep.assert_called_once_with(1.0)
            self.assertEqual(publish_person.call_count, 2, "只有两段人物图应上传并登记，场景图不入库")
            self.assertIn('"image_concurrency": 0', report)
            self.assertEqual(tuple(previews.shape[1:]), (512, 512, 3))
            self.assertEqual(tuple(package_previews.shape[1:]), (512, 512, 3))
            self.assertIn('"asset_mode": "auto_shot_assets"', report)
            self.assertIn('"engine": "wan"', package_report)
            self.assertEqual(
                [task["auto_asset_analysis_attempts"] for task in packed.manifest["tasks"]],
                [2, 1],
            )
            for task in packed.manifest["tasks"]:
                self.assertEqual(task["auto_asset_status"], "ready")
                self.assertEqual(task["auto_asset_warnings"][0]["kind"], "person_asset_library")
                self.assertTrue(Path(task["auto_assets"]["source_frames"]["source_start"]).is_file())
                shot_manifest = json.loads(
                    (long_video._auto_asset_root(packed, task) / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(shot_manifest["analysis_attempts"], task["auto_asset_analysis_attempts"])
                self.assertEqual(task["reference_package_status"], "ready")
                self.assertLessEqual(len(task["reference_package"]["items"]), 3)
                self.assertEqual(
                    [item["role"] for item in task["reference_package"]["items"]],
                    ["people_contact_sheet", "scene_sequence"],
                )
            self.assertFalse(packed.manifest["tasks"][0]["auto_assets"]["people"][0]["reused_from_cache"])
            self.assertTrue(packed.manifest["tasks"][1]["auto_assets"]["people"][0]["reused_from_cache"])
            self.assertTrue(packed.manifest["tasks"][1]["auto_assets"]["scenes"][0]["reused_from_cache"])

    def test_auto_asset_resume_only_regenerates_missing_stage_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=2.4)
            job_dir = root / "job"
            asset_root = job_dir / "shot_assets" / "shot_0001"
            asset_root.mkdir(parents=True)
            source_frames = {}
            for name in ("source_start", "source_middle", "source_end"):
                path = asset_root / f"{name}.png"
                long_video._save_image_tensor(torch.full((1, 24, 32, 3), 0.25), path)
                source_frames[name] = str(path)
            person_path = asset_root / "people" / "P1.png"
            end_scene_path = asset_root / "scenes" / "scene_end.png"
            long_video._save_image_tensor(torch.full((1, 24, 32, 3), 0.4), person_path)
            long_video._save_image_tensor(torch.full((1, 24, 32, 3), 0.6), end_scene_path)
            person_mtime = person_path.stat().st_mtime_ns
            end_scene_mtime = end_scene_path.stat().st_mtime_ns
            analysis = {
                "shot_stability": "uncertain",
                "story_action": "人物进入场景。",
                "people": [
                    {
                        "slot": "P1",
                        "appearance": "短发夹克成年人",
                        "first_bbox": [0.1, 0.1, 0.8, 0.9],
                        "last_bbox": [0.1, 0.1, 0.8, 0.9],
                        "persistence": "both",
                        "identity_key": "short_hair_jacket",
                        "reuse_confidence": 1.0,
                    }
                ],
                "background": {
                    "first_description": "街道入口",
                    "last_description": "街道内部",
                    "scene_key": "street",
                    "same_scene_confidence": 0.0,
                },
            }
            task = {
                "index": 1,
                "start": 0.0,
                "duration": 2.4,
                "source_start": 0.0,
                "source_duration": 2.4,
                "status": "planned",
                "result": str(job_dir / "segments" / "segment_0001.mp4"),
                "auto_asset_analysis": analysis,
                "auto_asset_status": "degraded",
                "auto_asset_errors": [{"kind": "scene", "role": "scene_start", "message": "temporary"}],
                "auto_assets": {
                    "prompt_version": long_video.AUTO_ASSET_PROMPT_VERSION,
                    "source_frames": source_frames,
                    "people": [{"slot": "P1", "path": str(person_path)}],
                    "scenes": [{"role": "scene_end", "path": str(end_scene_path)}],
                },
            }
            manifest_path = job_dir / "manifest.json"
            manifest = {
                "version": 4,
                "job_id": "resume-test",
                "asset_mode": "auto_shot_assets",
                "auto_asset_options": {
                    "image_model": "gpt-image-2",
                    "image_quality": "medium",
                    "reuse_threshold": 0.92,
                    "force_rerun_assets": False,
                },
                "tasks": [task],
            }
            long_video._atomic_write_json(manifest_path, manifest)
            job = long_video.LongVideoJob(
                video=long_video.InputImpl.VideoFromFile(str(source)),
                assets=long_video.LongVideoAssets(manifest={}, people={}, backgrounds={}),
                prompt="测试",
                engine="wan",
                model="wan2.7-r2v-2026-06-12",
                segment_duration=10,
                ai_model="gpt-5.6-terra",
                max_retries=0,
                resume=True,
                force_rerun=False,
                negative_prompt="",
                total_duration=2.4,
                source_path=str(source),
                job_dir=job_dir,
                manifest_path=manifest_path,
                manifest=manifest,
            )
            generated = torch.full((1, 24, 32, 3), 0.8)
            with (
                mock.patch.object(long_video, "_auto_asset_analysis", side_effect=AssertionError("analysis must be reused")),
                mock.patch.object(long_video, "_generate_auto_asset_image", return_value=generated) as image_request,
                mock.patch.object(
                    long_video,
                    "publish_seedance_person_image",
                    return_value={
                        "tos": {"status": "uploaded", "object_key": "safe/person.png"},
                        "asset_library": {"status": "active", "asset_id": "asset-person"},
                    },
                ),
            ):
                result, report, _previews = long_video.build_long_video_auto_assets(job)
                packed, _package_report, _package_previews = long_video.pack_long_video_auto_references(result)

            updated = packed.manifest["tasks"][0]
            self.assertEqual(image_request.call_count, 1)
            self.assertEqual(updated["auto_asset_status"], "ready")
            self.assertEqual(updated["reference_package_status"], "ready")
            self.assertEqual(updated["auto_asset_errors"], [])
            self.assertEqual(person_path.stat().st_mtime_ns, person_mtime)
            self.assertEqual(end_scene_path.stat().st_mtime_ns, end_scene_mtime)
            self.assertTrue((asset_root / "scenes" / "scene_start.png").is_file())
            self.assertIn('"analysis_reused": true', report)

    def test_auto_asset_generation_retries_transient_failure(self) -> None:
        generated = {"path": "unused"}
        error = {"kind": "scene", "message": "temporary", "error_kind": "generation_failed"}
        factory = mock.Mock(side_effect=[(None, error), (generated, None)])
        with mock.patch.object(long_video.time, "sleep"):
            entry, final_error = long_video._retry_auto_asset_creation(factory, max_retries=1)
        self.assertIs(entry, generated)
        self.assertIsNone(final_error)
        self.assertEqual(entry["generation_attempts"], 2)
        self.assertEqual(factory.call_count, 2)

    def test_auto_asset_analysis_retries_with_short_incremental_backoff(self) -> None:
        analysis = {"story_action": "ready"}
        factory = mock.Mock(
            side_effect=[
                RuntimeError("temporary EOF"),
                RuntimeError("temporary timeout"),
                analysis,
            ]
        )
        on_retry = mock.Mock()
        with mock.patch.object(long_video.time, "sleep") as sleep:
            result, attempts = long_video._retry_auto_asset_analysis(
                factory,
                max_retries=2,
                on_retry=on_retry,
            )
        self.assertIs(result, analysis)
        self.assertEqual(attempts, 3)
        self.assertEqual(factory.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])
        self.assertEqual(on_retry.call_count, 2)

    def test_auto_asset_analysis_does_not_retry_nonrecoverable_http_400(self) -> None:
        factory = mock.Mock(
            side_effect=company_client.CompanyRemoteAPIError("Remote request failed with HTTP 400: invalid request")
        )
        on_retry = mock.Mock()
        with mock.patch.object(long_video.time, "sleep") as sleep:
            with self.assertRaisesRegex(company_client.CompanyRemoteAPIError, "HTTP 400"):
                long_video._retry_auto_asset_analysis(factory, max_retries=2, on_retry=on_retry)

        self.assertEqual(factory.call_count, 1)
        on_retry.assert_not_called()
        sleep.assert_not_called()

    def test_auto_asset_planner_reuses_assets_when_engine_minimum_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._make_source_video(source, duration=1.8)
            video = long_video.InputImpl.VideoFromFile(str(source))
            plan = long_video.LongVideoShotPlan(
                video=video,
                total_duration=1.8,
                fps=10.0,
                requested_mode="shot_aware",
                effective_mode="shot_aware",
                fixed_duration=10,
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="test",
                config={},
                boundaries=[],
                shots=[long_video.LogicalShot(1, 0.0, 1.8, 1.8, "video_start", "video_end")],
            )
            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(root / "output")):
                original = long_video.plan_long_video_auto_asset_job(
                    shot_plan=plan,
                    prompt="测试自动欧美化",
                    engine="seedance",
                    model="Seedance 2.0 Fast",
                    ai_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="medium",
                    reuse_threshold=0.92,
                    max_retries=0,
                    resume=False,
                    force_rerun=False,
                    force_rerun_assets=False,
                    negative_prompt="",
                )
                original_task = original.manifest["tasks"][0]
                old_root = original.job_dir / "shot_assets" / "shot_0001"
                old_frame = old_root / "source_start.png"
                long_video._save_image_tensor(torch.ones((1, 16, 16, 3)), old_frame)
                original_task.update(
                    {
                        "request_duration": 2.0,
                        "trim_offset": 0.1,
                        "padding_start": 0.1,
                        "padding_end": 0.1,
                        "auto_asset_analysis": {"story_action": "测试"},
                        "auto_assets": {"source_frames": {"source_start": str(old_frame)}, "people": [], "scenes": []},
                        "auto_asset_errors": [],
                        "auto_asset_status": "ready",
                    }
                )
                long_video._write_auto_asset_manifest(old_root, original_task)
                long_video._atomic_write_json(original.manifest_path, original.manifest)
                resumed_plan = long_video.LongVideoShotPlan(
                    **{**plan.__dict__, "fixed_duration": 15}
                )
                resumed = long_video.plan_long_video_auto_asset_job(
                    shot_plan=resumed_plan,
                    prompt="测试自动欧美化",
                    engine="seedance",
                    model="Seedance 2.0 Fast",
                    ai_model="gpt-5.4",
                    image_model="gpt-image-2",
                    image_quality="medium",
                    reuse_threshold=0.92,
                    max_retries=0,
                    resume=True,
                    force_rerun=False,
                    force_rerun_assets=False,
                    negative_prompt="",
                )
                resumed_task = resumed.manifest["tasks"][0]
                resumed_frame = Path(resumed_task["auto_assets"]["source_frames"]["source_start"])
                self.assertEqual(resumed.manifest["resumed_from_manifest"], str(original.manifest_path))
                self.assertEqual(resumed_task["auto_asset_status"], "ready")
                self.assertTrue(resumed_frame.is_file())
                self.assertNotEqual(resumed_frame.parent, old_root)
                self.assertTrue(str(resumed_frame).startswith(str(resumed.job_dir)))

    def test_manual_batch_retry_inherits_compatible_prior_attempt_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            old_job = output_root / "company_remote/manual_batch_series/series_a/batches/B1_0_4000/attempt_001/job"
            old_shot = old_job / "shot_assets/shot_0001"
            source_frame = old_shot / "source_start.png"
            long_video._save_image_tensor(torch.ones((1, 16, 16, 3)), source_frame)
            old_batch = {
                "series_id": "series_a",
                "batch_id": "B1_0_4000",
                "attempt": 1,
                "source_start": 0.0,
                "source_end": 4.0,
                "source_duration": 4.0,
            }
            task = {
                "index": 1,
                "start": 0.0,
                "duration": 4.0,
                "source_start": 0.0,
                "source_duration": 4.0,
                "auto_asset_status": "masters_ready",
                "auto_asset_analysis": {"people": [], "background": {}},
                "auto_assets": {"source_frames": {"source_start": str(source_frame)}, "people": [], "scenes": []},
                "auto_asset_errors": [],
            }
            long_video._write_auto_asset_manifest(old_shot, task)
            options = {
                "image_model": "gpt-image-2",
                "image_quality": "low",
                "image_provider": "WisArt",
                "reuse_threshold": 0.85,
                "prompt_version": long_video.AUTO_ASSET_PROMPT_VERSION,
                "max_people_per_shot": 3,
                "visual_style": "western",
                "send_source_video": False,
                "target_resource_type": "欧美化资源",
                "processing_contract_version": 4,
                "use_original_audio": True,
                "grouping_version": long_video.V3_GROUPING_VERSION,
                "reference_package_version": long_video.MANUAL_BATCH_REFERENCE_PACKAGE_VERSION,
                "manual_batch": old_batch,
            }
            previous = {
                "processing_contract_version": 4,
                "asset_mode": "auto_shot_assets",
                "manual_batch": old_batch,
                "settings": {
                    "engine": "seedance",
                    "model": "Seedance 2.0 Fast",
                    "prompt": "欧美化",
                    "negative_prompt": "",
                    "ai_model": "gpt-test",
                },
                "auto_asset_options": options,
                "segmentation": {"boundaries_hash": "boundary", "effective_mode": "shot_aware"},
            }
            long_video._atomic_write_json(old_job / "manifest.json", previous)
            retry_batch = {**old_batch, "attempt": 2, "action": "重试当前批"}
            values = {**previous["settings"], "auto_asset_options": {**options, "manual_batch": retry_batch}}

            with mock.patch.object(long_video.folder_paths, "get_output_directory", return_value=str(output_root)):
                retry = long_video._manual_batch_retry_manifest(retry_batch)

            self.assertIsNotNone(retry)
            candidate, candidate_path = retry
            self.assertTrue(
                long_video._manual_batch_retry_assets_compatible(
                    candidate,
                    values=values,
                    segmentation={"boundaries_hash": "boundary", "effective_mode": "shot_aware"},
                    processing_contract_version=4,
                )
            )
            new_job = output_root / "company_remote/manual_batch_series/series_a/batches/B1_0_4000/attempt_002/job"
            inherited = dict(task)
            long_video._inherit_manual_batch_retry_asset_root(
                old_job_dir=candidate_path.parent,
                new_job_dir=new_job,
                task=inherited,
            )
            inherited_frame = Path(inherited["auto_assets"]["source_frames"]["source_start"])
            self.assertTrue(inherited_frame.is_file())
            self.assertTrue(str(inherited_frame).startswith(str(new_job)))
            self.assertEqual(inherited["auto_asset_status"], "masters_ready")

    def test_manual_batch_retry_reopens_only_quality_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("source_start", "source_middle", "source_end"):
                long_video._save_image_tensor(torch.ones((1, 16, 16, 3)), root / f"{name}.png")
            scene = root / "scene_start.png"
            long_video._save_image_tensor(torch.ones((1, 16, 16, 3)), scene)
            task = {
                "index": 1,
                "auto_asset_status": "degraded",
                "auto_asset_analysis": {
                    "shot_stability": "stable",
                    "people": [],
                    "background": {"same_scene_confidence": 0.99},
                },
                "auto_assets": {
                    "source_frames": {
                        name: str(root / f"{name}.png")
                        for name in ("source_start", "source_middle", "source_end")
                    },
                    "people": [],
                    "scenes": [{"role": "scene_start", "path": str(scene)}],
                    "integrated_frames": [{"role": "frame_start", "path": str(root / "weak.png")}],
                },
                "auto_asset_errors": [
                    {"kind": "integrated_frame", "error_kind": "quality_gate_failed"}
                ],
            }
            long_video._write_auto_asset_manifest(root, task)

            reopened = long_video._reset_manual_batch_quality_retry_state(
                task,
                root,
                reuse_threshold=0.85,
            )

        self.assertTrue(reopened)
        self.assertEqual(task["auto_asset_status"], "masters_ready")
        self.assertEqual(task["auto_asset_errors"], [])
        self.assertEqual(task["auto_assets"]["integrated_frames"], [])

    def test_manual_batch_retry_does_not_reopen_tos_or_source_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "index": 1,
                "auto_asset_status": "degraded",
                "auto_asset_errors": [{"kind": "person", "error_kind": "tos_upload_failed"}],
                "auto_assets": {},
            }
            reopened = long_video._reset_manual_batch_quality_retry_state(
                task,
                root,
                reuse_threshold=0.85,
            )

        self.assertFalse(reopened)
        self.assertEqual(task["auto_asset_status"], "degraded")

    def test_auto_asset_workflows_are_new_and_keep_video_generation_out_of_test_copy(self) -> None:
        root = Path(__file__).resolve().parents[3] / "user/default/workflows"
        test_workflow = json.loads((root / "视频欧美转绘_按镜头自动资产测试.json").read_text(encoding="utf-8"))
        full_workflow = json.loads((root / "视频欧美转绘_长视频_按镜头自动资产转绘.json").read_text(encoding="utf-8"))
        anime_workflow = json.loads(
            (root / "人物视频动漫化_长视频_按镜头自动资产转绘_Seedance版.json").read_text(encoding="utf-8")
        )
        anime_v2_workflow = json.loads(
            (root / "人物视频动漫化_长视频_按镜头自动资产转绘_Seedance版_素材库复用v2.json").read_text(
                encoding="utf-8"
            )
        )
        anime_v2_limited = json.loads(
            (root / "人物视频多风格转绘_长视频_按镜头自动资产转绘_Seedance版_素材库复用v2_限时生成.json").read_text(
                encoding="utf-8"
            )
        )
        anime_v3_limited = json.loads(
            (root / "人物视频多风格转绘_长视频_按镜头自动资产转绘_Seedance版_素材库复用v3_短镜头合并_音频可选_限时生成.json").read_text(
                encoding="utf-8"
            )
        )
        test_types = {node["type"] for node in test_workflow["nodes"]}
        full_types = {node["type"] for node in full_workflow["nodes"]}
        auto_types = {
            "CompanyLongVideoAutoAssetPlanner",
            "CompanyLongVideoAutoAssetBuilder",
            "CompanyLongVideoAutoReferencePacker",
        }
        self.assertTrue(auto_types.issubset(test_types))
        self.assertTrue(auto_types.issubset(full_types))
        self.assertNotIn("CompanyLongVideoSegmentGenerator", test_types)
        self.assertIn("CompanyLongVideoSegmentGenerator", full_types)
        self.assertFalse(test_workflow["extra"]["long_video"]["remote_video_requests"])
        self.assertTrue(full_workflow["extra"]["long_video"]["remote_video_requests"])
        anime_planner = next(node for node in anime_workflow["nodes"] if node["id"] == 4)
        self.assertEqual(anime_planner["type"], "CompanyLongVideoAnimeAssetPlanner")
        self.assertEqual(anime_workflow["extra"]["long_video"]["visual_style"], "anime_2d")
        self.assertFalse(anime_workflow["extra"]["long_video"]["send_source_video"])
        self.assertIn("CompanyLongVideoSegmentGenerator", {node["type"] for node in anime_workflow["nodes"]})
        anime_v2_types = {node["type"] for node in anime_v2_workflow["nodes"]}
        self.assertTrue(
            {
                "CompanyLongVideoAnimeAssetPlanner",
                "CompanyLongVideoAutoAssetBuilder",
                "CompanyLongVideoAutoReferencePacker",
                "CompanyLongVideoSegmentGenerator",
            }.issubset(anime_v2_types)
        )
        old_selector = next(node for node in anime_v2_workflow["nodes"] if node["id"] == 3)
        limited_selector = next(node for node in anime_v2_limited["nodes"] if node["id"] == 3)
        self.assertEqual(old_selector["type"], "CompanyLongVideoContinuityRangeSelector")
        self.assertEqual(limited_selector["type"], "CompanyLongVideoLengthRangeSelector")
        self.assertEqual(limited_selector["widgets_values"], [1, "按分钟", 2.0, 30.0, 0])
        self.assertTrue(anime_v2_limited["extra"]["long_video"]["range_limit"]["enabled"])
        limited_planner = next(node for node in anime_v2_limited["nodes"] if node["id"] == 4)
        self.assertIn("target_resource_type", [item["name"] for item in limited_planner["inputs"]])
        self.assertIn("多风格", limited_planner["title"])
        self.assertEqual(limited_planner["widgets_values"][-2], company_nodes.ANIME_LONG_VIDEO_NEGATIVE_PROMPT)
        self.assertEqual(limited_planner["widgets_values"][-1], company_nodes.TARGET_RESOURCE_ANIME)
        v3_planner = next(node for node in anime_v3_limited["nodes"] if node["id"] == 4)
        self.assertEqual(v3_planner["type"], "CompanyLongVideoAnimeAssetPlannerV3")
        self.assertEqual(v3_planner["widgets_values"][-1], False)
        self.assertIn("use_original_audio", [item["name"] for item in v3_planner["inputs"]])
        self.assertEqual(anime_v3_limited["extra"]["long_video"]["processing_contract_version"], 3)
        self.assertEqual(
            next(node for node in anime_v3_limited["nodes"] if node["id"] == 9)["type"],
            "CompanyLongVideoSegmentGenerator",
        )

    def test_v3_groups_adjacent_short_shots_and_preserves_logical_members(self) -> None:
        shots = [
            long_video.LogicalShot(1, 0.0, 2.0, 2.0, "video_start", "hard_cut"),
            long_video.LogicalShot(2, 2.0, 2.0, 4.0, "hard_cut", "hard_cut"),
            long_video.LogicalShot(3, 4.0, 12.0, 16.0, "hard_cut", "video_end"),
        ]
        plan = long_video.LongVideoShotPlan(
            video=object(),
            total_duration=16.0,
            fps=10.0,
            requested_mode="fixed",
            effective_mode="fixed",
            fixed_duration=10,
            sensitivity="标准",
            use_audio_silence=False,
            auto_fallback=True,
            detector="test",
            config={},
            boundaries=[],
            shots=shots,
        )

        members, groups, details = long_video.build_v3_logical_members_and_request_groups(
            plan,
            engine="seedance",
            source_path="unused.mp4",
        )

        self.assertEqual([item.logical_shot for item in members], [1, 2, 3])
        self.assertEqual([item.logical_shots for item in groups], [(1, 2), (3,)])
        self.assertEqual([item.request_duration for item in groups], [4.0, 12.0])
        self.assertEqual(details["legacy_request_count"], 3)
        self.assertEqual(details["request_count"], 2)

    def test_v3_keeps_unmergeable_short_shot_and_pads_it_to_four_seconds(self) -> None:
        shots = [
            long_video.LogicalShot(1, 0.0, 15.0, 15.0, "video_start", "hard_cut"),
            long_video.LogicalShot(2, 15.0, 1.0, 16.0, "hard_cut", "hard_cut"),
            long_video.LogicalShot(3, 16.0, 15.0, 31.0, "hard_cut", "video_end"),
        ]
        plan = long_video.LongVideoShotPlan(
            video=object(),
            total_duration=31.0,
            fps=10.0,
            requested_mode="fixed",
            effective_mode="fixed",
            fixed_duration=10,
            sensitivity="标准",
            use_audio_silence=False,
            auto_fallback=True,
            detector="test",
            config={},
            boundaries=[],
            shots=shots,
        )

        _members, groups, _details = long_video.build_v3_logical_members_and_request_groups(
            plan,
            engine="seedance",
            source_path="unused.mp4",
        )

        self.assertEqual([item.logical_shots for item in groups], [(1,), (2,), (3,)])
        self.assertEqual(groups[1].request_duration, 4.0)
        self.assertEqual(groups[1].padding_end, 3.0)
        self.assertEqual(groups[1].split_reason, "short_shot_padding")

    def test_seedance_audio_flag_is_v3_only(self) -> None:
        reference = torch.ones((1, 24, 32, 3))
        with (
            mock.patch.object(long_video, "_get_config", return_value=object()),
            mock.patch.object(long_video, "generate_video", return_value=(None, "generated.mp4")) as request,
        ):
            adapter = long_video.get_video_engine_adapter("seedance")
            adapter.generate(
                source_segment=None,
                references=[reference],
                prompt="legacy",
                model="Seedance 2.0 Fast",
                duration=4,
                negative_prompt="",
            )
            legacy_extra = request.call_args.kwargs["extra_values"]
            adapter.generate(
                source_segment=None,
                references=[reference],
                prompt="v3",
                model="Seedance 2.0 Fast",
                duration=4,
                negative_prompt="",
                generate_audio=True,
            )
            v3_extra = request.call_args.kwargs["extra_values"]

        self.assertNotIn("generate_audio", legacy_extra)
        self.assertTrue(v3_extra["generate_audio"])

    def test_v3_packer_limits_master_references_and_uses_asset_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = []
            for index in range(1, 6):
                person = root / f"person_{index}.png"
                scene = root / f"scene_{index}.png"
                long_video._save_image_tensor(torch.full((1, 32, 32, 3), index / 12.0), person)
                long_video._save_image_tensor(torch.full((1, 32, 32, 3), (index + 6) / 12.0), scene)
                members.append(
                    {
                        "index": index,
                        "auto_asset_status": "ready",
                        "auto_asset_analysis": {"story_action": f"动作 {index}", "people": []},
                        "auto_assets": {
                            "people": [{"person_id": f"P{index}", "appearance": f"人物 {index}", "path": str(person)}],
                            "scenes": [{"place_id": f"L{index}", "version_id": "V1", "path": str(scene)}],
                        },
                    }
                )
            for member in members:
                person_path = member["auto_assets"]["people"][0]["path"]
                scene_path = member["auto_assets"]["scenes"][0]["path"]
                member["auto_assets"]["integrated_frames"] = [
                    {
                        "role": "frame_start",
                        "path": person_path,
                        "quality": {"verdict": "approved", "target_style_score": 0.95, "composition_preservation": 0.92},
                    },
                    {
                        "role": "frame_end",
                        "path": scene_path,
                        "quality": {"verdict": "approved", "target_style_score": 0.91, "composition_preservation": 0.90},
                    },
                ]
            task = {"index": 2, "logical_segment": 1, "logical_segments": [1, 2, 3, 4, 5]}
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    engine="seedance",
                    prompt="测试",
                    job_dir=root / "job",
                    manifest={
                        "processing_contract_version": 3,
                        "auto_asset_options": {"visual_style": "anime_2d", "send_source_video": False},
                        "logical_member_tasks": members,
                        "tasks": [task],
                    },
                ),
            )

            reports, _tasks = long_video._v3_pack_group_references(job)
            package = task["reference_package"]
            first_package_key = package["package_key"]
            first_resume_key = job.manifest["reference_package_resume_key"]
            sheet_paths = [Path(item["path"]) for item in package["items"] if item.get("path")]
            dimensions = []
            for path in sheet_paths:
                with Image.open(path) as image:
                    dimensions.append(image.size)
            task["status"] = "success"
            task["generated_reference_package_key"] = first_package_key
            long_video._v3_pack_group_references(job)
            self.assertEqual(task["status"], "success")
            members[0]["auto_asset_analysis"]["story_action"] = "修改后的动作"
            long_video._v3_pack_group_references(job)
            changed_package_key = task["reference_package"]["package_key"]
            changed_resume_key = job.manifest["reference_package_resume_key"]

        self.assertEqual(reports[0]["unique_assets"], 10)
        self.assertEqual(reports[0]["packed_assets"], 9)
        self.assertEqual(reports[0]["omitted_masters"], 1)
        self.assertEqual(reports[0]["omitted_integrated_frames"], 0)
        self.assertEqual(len(sheet_paths), 9)
        self.assertEqual(package["reserved_continuity_slots"], 0)
        self.assertTrue(package["package_key"])
        self.assertNotEqual(first_package_key, changed_package_key)
        self.assertNotEqual(first_resume_key, changed_resume_key)
        self.assertEqual(task["status"], "matched")
        self.assertEqual(task["resume_invalidated"], "reference_package_changed")
        self.assertNotIn("generated_reference_package_key", task)
        self.assertTrue(all(max(width / height, height / width) <= 2.5 for width, height in dimensions))
        self.assertFalse(package["reference_timeline"])
        self.assertIn("person_1", task["prompt"])
        self.assertIn("scene_1", task["prompt"])
        self.assertIn("本段按独立镜头生成", task["prompt"])

    def test_manual_batch_packs_active_people_then_scenes_for_a_new_logical_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = []
            for index in range(1, 4):
                person = root / f"person_{index}.png"
                long_video._save_image_tensor(torch.full((1, 32, 32, 3), index / 4.0), person)
                members.append(
                    {
                        "index": index,
                        "auto_asset_status": "ready",
                        "auto_asset_analysis": {"story_action": f"动作 {index}", "people": []},
                        "auto_assets": {
                            "people": [
                                {
                                    "person_id": f"P{index}",
                                    "appearance": f"人物 {index}",
                                    "path": str(person),
                                    "publication": {"asset_library": {"status": "active", "asset_id": f"asset-person-{index}"}},
                                }
                            ],
                            "scenes": [],
                        },
                    }
                )
            for index in range(1, 7):
                scene = root / f"scene_{index}.png"
                scene_image = torch.zeros((1, 32, 32, 3))
                scene_image[..., 0] = index / 8.0
                scene_image[..., 1] = 1.0
                long_video._save_image_tensor(scene_image, scene)
                members[index % 3]["auto_assets"]["scenes"].append(
                    {"place_id": f"L{index}", "version_id": "V1", "path": str(scene)}
                )
            for member in members:
                member["auto_assets"]["integrated_frames"] = [
                    {
                        "role": "frame_start",
                        "path": member["auto_assets"]["scenes"][0]["path"],
                        "quality": {"verdict": "approved", "target_style_score": 0.95, "composition_preservation": 0.92},
                    },
                    {
                        "role": "frame_end",
                        "path": member["auto_assets"]["scenes"][1]["path"],
                        "quality": {"verdict": "approved", "target_style_score": 0.93, "composition_preservation": 0.91},
                    },
                ]
            task = {
                "index": 2,
                "logical_segment": 2,
                "logical_segments": [1, 2, 3],
                "continuity_from_previous_group": False,
            }
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    engine="seedance",
                    prompt="测试",
                    job_dir=root / "job",
                    manifest={
                        "processing_contract_version": 4,
                        "auto_asset_options": {"visual_style": "anime_2d", "send_source_video": False},
                        "logical_member_tasks": members,
                        "tasks": [task],
                    },
                ),
            )

            long_video._v3_pack_single_group_reference(
                job,
                task,
                {
                    "members_by_index": {item["index"]: item for item in members},
                    "visual_style": "anime_2d",
                    "send_source_video": False,
                    "is_manual_batch": True,
                    "cross_batch_frame": "",
                    "package_version": "test",
                },
            )

        items = task["reference_package"]["items"]
        self.assertEqual(len(items), 9)
        self.assertEqual(task["reference_package"]["reserved_continuity_slots"], 0)
        self.assertFalse(task["uses_previous_end_frame"])
        self.assertEqual([item["role"] for item in items[:3]], ["person_1", "person_2", "person_3"])
        self.assertEqual([item["asset_id"] for item in items[:3]], ["asset-person-1", "asset-person-2", "asset-person-3"])
        self.assertTrue(all(item["kind"] == "person_asset" for item in items[:3]))
        self.assertTrue(all(item["kind"] == "scene_master" for item in items[3:]))

    def test_manual_batch_continuity_is_only_for_one_split_logical_shot(self) -> None:
        self.assertFalse(
            long_video._request_group_continues_same_logical_shot(
                {"logical_segments": [1, 2]},
                {"logical_segments": [3]},
            )
        )
        self.assertTrue(
            long_video._request_group_continues_same_logical_shot(
                {"logical_segments": [1]},
                {"logical_segments": [1]},
            )
        )
        self.assertFalse(long_video._manual_batch_starts_inside_logical_shot([{"starts_inside_shot_split": False}]))
        self.assertTrue(long_video._manual_batch_starts_inside_logical_shot([{"starts_inside_shot_split": True}]))
        self.assertFalse(
            long_video._manual_batch_starts_inside_logical_shot(
                [{"is_inside_shot_split": True, "starts_inside_shot_split": False}]
            )
        )

    def test_split_logical_shot_keeps_active_people_and_packs_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = []
            for index in range(1, 4):
                person = root / f"person_{index}.png"
                person_image = torch.zeros((1, 32, 32, 3))
                person_image[..., 0] = index / 4.0
                long_video._save_image_tensor(person_image, person)
                member = {
                    "index": index,
                    "auto_asset_status": "ready",
                    "auto_asset_analysis": {"story_action": f"动作 {index}", "people": []},
                    "auto_assets": {
                        "people": [
                            {
                                "person_id": f"P{index}",
                                "appearance": f"人物 {index}",
                                "path": str(person),
                                "publication": {
                                    "asset_library": {
                                        "status": "active",
                                        "asset_id": f"asset-person-{index}",
                                    }
                                },
                            }
                        ],
                        "scenes": [],
                    },
                }
                members.append(member)
            for index in range(1, 7):
                scene = root / f"scene_{index}.png"
                scene_image = torch.zeros((1, 32, 32, 3))
                scene_image[..., 1] = index / 8.0
                long_video._save_image_tensor(scene_image, scene)
                members[index % 3]["auto_assets"]["scenes"].append(
                    {"place_id": f"L{index}", "version_id": "V1", "path": str(scene)}
                )
            for member in members:
                member["auto_assets"]["integrated_frames"] = [
                    {
                        "role": "frame_start",
                        "path": member["auto_assets"]["scenes"][0]["path"],
                        "quality": {"verdict": "approved", "target_style_score": 0.95, "composition_preservation": 0.92},
                    },
                    {
                        "role": "frame_end",
                        "path": member["auto_assets"]["scenes"][1]["path"],
                        "quality": {"verdict": "approved", "target_style_score": 0.93, "composition_preservation": 0.91},
                    },
                ]
            task = {
                "index": 2,
                "logical_segment": 1,
                "logical_segments": [1, 2, 3],
                "continuity_from_previous_group": True,
            }
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    engine="seedance",
                    prompt="测试",
                    job_dir=root / "job",
                    manifest={
                        "processing_contract_version": 4,
                        "auto_asset_options": {"visual_style": "anime_2d", "send_source_video": False},
                        "logical_member_tasks": members,
                        "tasks": [task],
                    },
                ),
            )

            long_video._v3_pack_single_group_reference(
                job,
                task,
                {
                    "members_by_index": {item["index"]: item for item in members},
                    "visual_style": "anime_2d",
                    "send_source_video": False,
                    "is_manual_batch": True,
                    "cross_batch_frame": "",
                    "package_version": "test",
                },
            )

        items = task["reference_package"]["items"]
        self.assertEqual(len(items), 8)
        self.assertEqual(task["reference_package"]["reserved_continuity_slots"], 1)
        self.assertTrue(task["uses_previous_end_frame"])
        self.assertEqual([item["role"] for item in items[:3]], ["person_1", "person_2", "person_3"])
        self.assertEqual([item["asset_id"] for item in items[:3]], ["asset-person-1", "asset-person-2", "asset-person-3"])
        self.assertTrue(all(item["kind"] == "scene_master" for item in items[3:]))

    def test_v3_normalization_preserves_long_video_and_only_pads_short_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            long_source = root / "long.mp4"
            short_source = root / "short.mp4"
            long_output = root / "long_output.mp4"
            short_output = root / "short_output.mp4"
            self._make_source_video(long_source, duration=5.2, fps=10)
            self._make_video_from_frames(
                short_source,
                [np.full((48, 64, 3), index, dtype=np.uint8) for index in range(20)],
                fps=10,
            )

            long_report = long_video._normalize_segment_v3(
                long_source,
                long_output,
                request_duration=4.0,
                width=64,
                height=48,
                fps=10,
                keep_generated_audio=True,
            )
            short_report = long_video._normalize_segment_v3(
                short_source,
                short_output,
                request_duration=4.0,
                width=64,
                height=48,
                fps=10,
                keep_generated_audio=False,
            )

        self.assertGreaterEqual(long_report["output_duration"], 5.1)
        self.assertEqual(long_report["video_padding"], 0.0)
        self.assertTrue(long_report["has_audio"])
        self.assertGreaterEqual(short_report["output_duration"], 3.9)
        self.assertAlmostEqual(short_report["video_padding"], 2.0, delta=0.11)
        self.assertFalse(short_report["has_audio"])

    def test_v3_concat_supports_generated_and_original_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            generated = root / "generated.mp4"
            silent = root / "silent.mp4"
            generated_final = root / "generated_final.mp4"
            original_final = root / "original_final.mp4"
            self._make_source_video(source, duration=2.4, fps=10)
            generated_media = long_video._normalize_segment_v3(
                source,
                generated,
                request_duration=4.0,
                width=64,
                height=48,
                fps=10,
                keep_generated_audio=True,
            )
            silent_media = long_video._normalize_segment_v3(
                source,
                silent,
                request_duration=4.0,
                width=64,
                height=48,
                fps=10,
                keep_generated_audio=False,
            )
            generated_report = long_video._concat_v3_segments(
                [{"index": 1, "source_start": 0.0, "source_duration": 2.4, "result": str(generated), "media": generated_media}],
                str(source),
                generated_final,
                root,
                use_original_audio=False,
            )
            original_report = long_video._concat_v3_segments(
                [{"index": 1, "source_start": 0.0, "source_duration": 2.4, "result": str(silent), "media": silent_media}],
                str(source),
                original_final,
                root,
                use_original_audio=True,
            )

        self.assertFalse(generated_report["use_original_audio"])
        self.assertTrue(original_report["use_original_audio"])
        self.assertGreaterEqual(generated_report["output_duration"], 3.9)
        self.assertGreaterEqual(original_report["output_duration"], 3.9)

    def test_segment_generator_surfaces_all_blocked_group_root_causes_before_seedance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = [
                {
                    "index": 1,
                    "auto_asset_status": "analysis_failed",
                    "auto_asset_analysis_attempts": 3,
                    "auto_asset_errors": [
                        {
                            "kind": "analysis",
                            "attempts": 3,
                            "message": 'HTTP 503: {"error":{"code":"internal_server_error","message":"auth_unavailable: no auth available"}}',
                        }
                    ],
                },
                {
                    "index": 2,
                    "auto_asset_status": "degraded",
                    "auto_asset_errors": [
                        {
                            "kind": "integrated_frame",
                            "error_kind": "quality_gate_failed",
                            "attempts": 2,
                            "message": "整帧转绘未通过质量验收",
                        }
                    ],
                },
            ]
            tasks = [
                {"index": 1, "logical_segments": [1], "reference_package_status": "blocked_by_asset_failure"},
                {"index": 2, "logical_segments": [2], "reference_package_status": "blocked_by_asset_failure"},
            ]
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    manifest_path=root / "manifest.json",
                    manifest={
                        "asset_mode": "auto_shot_assets",
                        "logical_member_tasks": members,
                        "tasks": tasks,
                    },
                ),
            )

            with (
                mock.patch.object(long_video, "_segment_generation_context") as context,
                self.assertRaises(ValueError) as raised,
            ):
                long_video.generate_long_video_segments(job)

            saved = json.loads(job.manifest_path.read_text(encoding="utf-8"))

        message = str(raised.exception)
        context.assert_not_called()
        self.assertIn("2/2 个镜头不可用", message)
        self.assertIn("远端服务无可用认证", message)
        self.assertIn("转绘质量门控未通过", message)
        self.assertIn("Seedance 尚未启动", message)
        self.assertEqual(saved["auto_asset_failure_summary"]["failed_shot_indexes"], [1, 2])

    def test_parallel_segment_generator_uses_same_asset_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "index": 1,
                "logical_segments": [1],
                "reference_package_status": "blocked_by_asset_failure",
                "status": "planned",
                "attempts": 0,
                "result": str(root / "segment.mp4"),
            }
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    engine="seedance",
                    resume=False,
                    force_rerun=False,
                    job_dir=root,
                    manifest_path=root / "manifest.json",
                    manifest={
                        "asset_mode": "auto_shot_assets",
                        "processing_contract_version": 2,
                        "auto_asset_options": {},
                        "logical_member_tasks": [
                            {
                                "index": 1,
                                "auto_asset_status": "analysis_failed",
                                "auto_asset_errors": [
                                    {
                                        "kind": "analysis",
                                        "attempts": 3,
                                        "message": 'HTTP 502: {"error":{"code":"server_is_overloaded","message":"busy"}}',
                                    }
                                ],
                            }
                        ],
                        "tasks": [task],
                    },
                ),
            )

            with self.assertRaises(RuntimeError) as raised:
                long_video.generate_long_video_segments_parallel(job, concurrency=3)

        message = str(raised.exception)
        self.assertIn("1/1 个镜头不可用", message)
        self.assertIn("远端服务过载或暂时不可用", message)
        self.assertIn("Seedance 尚未启动", message)

    def test_v3_rejects_parallel_generation_node(self) -> None:
        job = cast(
            long_video.LongVideoJob,
            SimpleNamespace(manifest={"processing_contract_version": 3}),
        )

        with self.assertRaisesRegex(ValueError, "不能接入并行分段生成节点"):
            long_video.generate_long_video_segments_parallel(job, concurrency=3)

    def test_manual_batch_pipeline_waits_for_global_integrated_frame_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = [
                {"index": 1, "auto_asset_status": "planned"},
                {"index": 2, "auto_asset_status": "planned"},
            ]
            tasks = [
                {"index": 1, "logical_segments": [1], "status": "planned"},
                {"index": 2, "logical_segments": [2], "status": "planned"},
            ]
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    manifest_path=root / "manifest.json",
                    manifest={
                        "job_id": "pipeline-overlap",
                        "asset_mode": "auto_shot_assets",
                        "processing_contract_version": 4,
                        "logical_member_tasks": members,
                        "tasks": tasks,
                    },
                ),
            )
            timeline: list[str] = []

            def fake_build_assets(_job, _concurrency, *, cancel_event=None, preserve_job_status=False):
                self.assertTrue(preserve_job_status)
                self.assertFalse(cancel_event.is_set())
                members[0]["auto_asset_status"] = "masters_ready"
                timeline.append("masters-1-ready")
                members[1]["auto_asset_status"] = "masters_ready"
                timeline.append("masters-2-ready")
                timeline.append("global-quality-calibrated")
                members[0]["auto_asset_status"] = "ready"
                members[1]["auto_asset_status"] = "ready"
                timeline.append("all-integrated-frames-ready")
                return job, json.dumps({"asset_stage_status": "auto_assets_ready"}), torch.empty((0,))

            def fake_pack(_job, task, _context):
                task["reference_package_status"] = "ready"
                task["status"] = "matched"
                timeline.append(f"pack-{task['index']}")
                return {"index": task["index"], "status": "ready"}

            continuation_inputs: list[object] = []

            def fake_generate(_job, task, _context, *, previous_end_frame):
                timeline.append(f"video-{task['index']}")
                continuation_inputs.append(previous_end_frame)
                task["status"] = "success"
                return f"end-frame-{task['index']}"

            with (
                mock.patch.object(long_video, "_preflight_auto_asset_analysis_gateway", return_value={"status": "ready"}),
                mock.patch.object(long_video, "build_long_video_auto_assets", side_effect=fake_build_assets),
                mock.patch.object(long_video, "_segment_generation_context", return_value={"state": None}),
                mock.patch.object(long_video, "_v3_group_pack_context", return_value={"members_by_index": {}}),
                mock.patch.object(long_video, "_v3_pack_single_group_reference", side_effect=fake_pack),
                mock.patch.object(long_video, "_generate_single_group_segment", side_effect=fake_generate),
                mock.patch.object(long_video, "PIPELINE_ASSET_WAIT_POLL_SECONDS", 0.01),
            ):
                result, report = long_video.generate_long_video_pipeline(job, image_concurrency=5)

        self.assertIs(result, job)
        self.assertEqual(json.loads(report)["status"], "segments_generated")
        self.assertEqual([task["status"] for task in tasks], ["success", "success"])
        self.assertLess(timeline.index("all-integrated-frames-ready"), timeline.index("video-1"))
        self.assertEqual(timeline[-1], "video-2")
        self.assertEqual(continuation_inputs, [None, None])

    def test_pipeline_stops_before_asset_producer_when_analysis_gateway_preflight_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    job_dir=root / "job",
                    manifest_path=root / "manifest.json",
                    manifest={
                        "job_id": "preflight-failure",
                        "asset_mode": "auto_shot_assets",
                        "processing_contract_version": 4,
                        "logical_member_tasks": [{"index": 1, "auto_asset_status": "planned"}],
                        "tasks": [{"index": 1, "logical_segments": [1], "status": "planned"}],
                    },
                ),
            )
            with (
                mock.patch.object(long_video, "_segment_generation_context", return_value={"state": None}),
                mock.patch.object(
                    long_video,
                    "_preflight_auto_asset_analysis_gateway",
                    side_effect=long_video.AnalysisGatewayUnavailableError("8317 不可用"),
                ),
                mock.patch.object(long_video, "build_long_video_auto_assets") as build_assets,
            ):
                with self.assertRaisesRegex(long_video.AnalysisGatewayUnavailableError, "8317 不可用"):
                    long_video.generate_long_video_pipeline(job, image_concurrency=5)

        build_assets.assert_not_called()
        self.assertEqual(job.manifest["status"], "analysis_gateway_unavailable")
        progress = job.manifest["auto_asset_progress"]
        self.assertEqual(progress["phase"], "analysis_gateway_unavailable")
        self.assertEqual(progress["extra"]["image_request_count"], 0)
        self.assertEqual(progress["extra"]["video_request_count"], 0)

    def test_analysis_gateway_preflight_checks_local_health_and_current_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    ai_model="gpt-5.6-terra",
                    manifest_path=root / "manifest.json",
                    manifest={"job_id": "gateway-ready"},
                ),
            )
            config = SimpleNamespace(
                name="gpttext",
                base_url="http://localhost:8317/v1",
                timeout_seconds=600,
            )
            response = mock.Mock(ok=True)
            response.json.return_value = {"status": "ok"}
            session = mock.Mock()
            session.get.return_value = response
            with (
                mock.patch.object(long_video, "_get_config", return_value=config),
                mock.patch.object(long_video.requests, "Session", return_value=session),
                mock.patch.object(long_video, "replace", return_value=config),
                mock.patch.object(long_video, "generate_openai_chat_text", return_value="OK") as chat_probe,
            ):
                report = long_video._preflight_auto_asset_analysis_gateway(job)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["health_url"], "http://localhost:8317/healthz")
        session.get.assert_called_once_with("http://localhost:8317/healthz", timeout=3)
        self.assertEqual(chat_probe.call_args.kwargs["model"], "gpt-5.6-terra")
        self.assertEqual(chat_probe.call_args.kwargs["max_tokens"], 4)

    def test_analysis_gateway_preflight_surfaces_health_failure_without_chat_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    ai_model="gpt-5.6-terra",
                    manifest_path=root / "manifest.json",
                    manifest={"job_id": "gateway-down"},
                ),
            )
            config = SimpleNamespace(
                name="gpttext",
                base_url="http://localhost:8317/v1",
                timeout_seconds=600,
            )
            session = mock.Mock()
            session.get.side_effect = long_video.requests.ConnectionError("connection refused")
            with (
                mock.patch.object(long_video, "_get_config", return_value=config),
                mock.patch.object(long_video.requests, "Session", return_value=session),
                mock.patch.object(long_video, "generate_openai_chat_text") as chat_probe,
            ):
                with self.assertRaisesRegex(long_video.AnalysisGatewayUnavailableError, "分析服务不可用"):
                    long_video._preflight_auto_asset_analysis_gateway(job)

        chat_probe.assert_not_called()
        self.assertEqual(job.manifest["analysis_gateway_preflight"]["status"], "failed")

    def test_pipeline_marks_group_blocked_when_required_asset_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"index": 1, "logical_segments": [1], "status": "planned"}
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    manifest_path=root / "manifest.json",
                    manifest={
                        "logical_member_tasks": [
                            {
                                "index": 1,
                                "auto_asset_status": "analysis_failed",
                                "auto_asset_analysis_attempts": 3,
                                "auto_asset_errors": [
                                    {
                                        "kind": "analysis",
                                        "attempts": 3,
                                        "message": 'HTTP 503: {"error":{"code":"internal_server_error","message":"auth_unavailable: no auth available"}}',
                                    }
                                ],
                            }
                        ],
                        "tasks": [task],
                    },
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "远端服务无可用认证") as raised:
                long_video._wait_pipeline_group_assets(
                    job,
                    task,
                    producer=threading.Thread(),
                    producer_error=[],
                )

            saved = json.loads(job.manifest_path.read_text(encoding="utf-8"))

        self.assertIn("Seedance 尚未启动", str(raised.exception))
        self.assertEqual(saved["auto_asset_failure_summary"]["failed_shot_indexes"], [1])
        self.assertEqual(task["reference_package_status"], "blocked_by_asset_failure")
        self.assertEqual(task["status"], "blocked_by_asset_failure")

    def test_pipeline_blocks_a_group_after_person_tos_upload_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"index": 1, "logical_segments": [1], "status": "planned"}
            job = cast(
                long_video.LongVideoJob,
                SimpleNamespace(
                    manifest_path=root / "manifest.json",
                    manifest={
                        "logical_member_tasks": [
                            {
                                "index": 1,
                                "auto_asset_status": "degraded",
                                "auto_asset_errors": [
                                    {"error_kind": "tos_upload_failed", "message": "TOS 写入失败"}
                                ],
                            }
                        ],
                        "tasks": [task],
                    },
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "人物素材上传失败") as raised:
                long_video._wait_pipeline_group_assets(
                    job,
                    task,
                    producer=threading.Thread(),
                    producer_error=[],
                )

        self.assertIn("TOS 写入失败", str(raised.exception))
        self.assertIn("Seedance 尚未启动", str(raised.exception))
        self.assertEqual(task["reference_package_status"], "blocked_by_asset_failure")
        self.assertEqual(task["status"], "blocked_by_asset_failure")

    def test_manual_batch_pipeline_workflow_replaces_only_intermediate_chain(self) -> None:
        root = Path(__file__).resolve().parents[3] / "user/default/workflows"
        original = json.loads((root / "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅.json").read_text(encoding="utf-8"))
        pipeline = json.loads(
            (root / "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_流水线.json").read_text(
                encoding="utf-8"
            )
        )
        original_types = {node["type"] for node in original["nodes"]}
        pipeline_nodes = {node["id"]: node for node in pipeline["nodes"]}
        pipeline_types = {node["type"] for node in pipeline["nodes"]}

        self.assertIn("CompanyLongVideoAutoAssetBuilder", original_types)
        self.assertIn("CompanyLongVideoAutoReferencePacker", original_types)
        self.assertIn("CompanyLongVideoSegmentGenerator", original_types)
        self.assertEqual(pipeline_nodes[5]["type"], "CompanyLongVideoPipelineAssetVideoGenerator")
        self.assertNotIn("CompanyLongVideoAutoAssetBuilder", pipeline_types)
        self.assertNotIn("CompanyLongVideoAutoReferencePacker", pipeline_types)
        self.assertNotIn("CompanyLongVideoSegmentGenerator", pipeline_types)
        self.assertTrue(
            any(link[1:5] == [5, 0, 10, 0] for link in pipeline["links"]),
            "流水线节点必须把完成任务交给连续性检查节点",
        )
        self.assertTrue(any(link[1:5] == [3, 3, 12, 1] for link in pipeline["links"]))
        self.assertFalse(pipeline["extra"]["long_video"]["asset_video_overlap"])
        self.assertTrue(pipeline["extra"]["long_video"]["global_integrated_frame_calibration"])

    def test_30_second_gateway_acceptance_workflow_preserves_production_settings(self) -> None:
        root = Path(__file__).resolve().parents[3] / "user/default/workflows"
        pipeline = json.loads(
            (root / "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_流水线.json").read_text(
                encoding="utf-8"
            )
        )
        acceptance = json.loads(
            (root / "人物视频多风格转绘_长视频_Seedance版_v3_首30秒真实验收.json").read_text(encoding="utf-8")
        )
        pipeline_nodes = {node["id"]: node for node in pipeline["nodes"]}
        acceptance_nodes = {node["id"]: node for node in acceptance["nodes"]}

        self.assertNotEqual(acceptance["id"], pipeline["id"])
        self.assertEqual(acceptance_nodes[1]["widgets_values"], pipeline_nodes[1]["widgets_values"])
        self.assertEqual(acceptance_nodes[4]["widgets_values"], pipeline_nodes[4]["widgets_values"])
        self.assertEqual(acceptance_nodes[5]["widgets_values"], pipeline_nodes[5]["widgets_values"])
        self.assertEqual(acceptance_nodes[3]["widgets_values"], ["新建系列", "", 0.5, 10])

    @staticmethod
    def _make_video_from_frames(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = int(frames[0].shape[1])
        stream.height = int(frames[0].shape[0])
        stream.pix_fmt = "yuv420p"
        for image in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(stream.encode(None))
        container.close()

    @staticmethod
    def _make_source_video(path: Path, *, duration: float = 2.4, fps: int = 10) -> None:
        container = av.open(str(path), mode="w")
        video_stream = container.add_stream("h264", rate=fps)
        video_stream.width = 64
        video_stream.height = 48
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=44100, layout="mono")
        for index in range(math.ceil(duration * fps)):
            image = np.full((48, 64, 3), index * 5 % 255, dtype=np.uint8)
            container.mux(video_stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(video_stream.encode(None))
        samples = np.zeros((1, math.ceil(duration * 44100)), dtype=np.float32)
        audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
        audio_frame.sample_rate = 44100
        audio_frame.pts = 0
        container.mux(audio_stream.encode(audio_frame))
        container.mux(audio_stream.encode(None))
        container.close()


if __name__ == "__main__":
    unittest.main()
