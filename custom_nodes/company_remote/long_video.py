from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as torch_functional
import av
import cv2
from PIL import Image, ImageDraw, ImageFont

import folder_paths
from comfy.utils import ProgressBar
from comfy_api.latest import IO, InputImpl

from .client import (
    CompanyRemoteAPIError,
    SeedanceAssetReference,
    generate_dashscope_video,
    generate_openai_chat_text,
    generate_openai_image,
    generate_openai_image_prompt_text,
    generate_video,
)
from .asset_gateway import publish_seedance_person_image
from .config_store import ConfigError, get_config, get_gpt_image_provider_config


PERSON_IDS = ("A", "B", "C")
BACKGROUND_IDS = tuple(f"BG{index:02d}" for index in range(1, 9))
ASSET_ROOT_NAME = "company_remote/long_video_assets"
JOB_ROOT_NAME = "company_remote/long_video_jobs"
MANUAL_BATCH_ROOT_NAME = "company_remote/manual_batch_series"
MANUAL_BATCH_PROCESSING_CONTRACT_VERSION = 4
MANUAL_BATCH_CONTRACT = "seedance_v3_manual_batch_v1"
MANUAL_BATCH_REFERENCE_PACKAGE_VERSION = "manual-batch-asset-library-v3"
SHOT_TEST_ROOT_NAME = "company_remote/shot_detection_tests"
MIN_REQUEST_DURATION = 2.0
SHOT_PREVIEW_LIMIT = 32
SHOT_PREVIEW_MAX_WIDTH = 512
SHOT_DETECTION_PROXY_MAX_WIDTH = 640
SHOT_DETECTION_CPU_THREADS = 2
AUTO_ASSET_MANIFEST_VERSION = 4
AUTO_ASSET_MAX_PEOPLE = 3
AUTO_ASSET_PROMPT_VERSION = "2026-08-15-replacement-assets-v1"
AUTO_ASSET_DEFAULT_REUSE_THRESHOLD = 0.92
AUTO_ASSET_CACHE_VERSION = 3
AUTO_ASSET_LIBRARY_VERSION = 1
AUTO_ASSET_DIFFERENT_THRESHOLD = 0.75
AUTO_ASSET_MAX_SOURCE_OBSERVATIONS = 6
AUTO_ASSET_QUALITY_GATE_VERSION = "replacement-quality-v2"
AUTO_ASSET_TARGET_STYLE_THRESHOLD = 0.85
AUTO_ASSET_SOURCE_RESIDUE_THRESHOLD = 0.15
AUTO_ASSET_COMPOSITION_THRESHOLD = 0.80
AUTO_ASSET_PERSON_IDENTITY_THRESHOLD = 0.85
AUTO_ASSET_SCENE_REPLACEMENT_THRESHOLD = 0.75
AUTO_ASSET_MIN_PERSON_MASTER_EDGE = 96
AUTO_ASSET_MIN_PERSON_MASTER_AREA = 12_000
AUTO_ASSET_MIN_NEW_PERSON_CONFIDENCE = 0.80
AUTO_ASSET_STYLE_WESTERN = "western"
AUTO_ASSET_STYLE_ANIME = "anime_2d"
AUTO_ASSET_STYLE_PHOTOREAL = "photoreal"
AUTO_ASSET_STYLE_CG_3D = "cg_3d"
AUTO_ASSET_STYLE_COMIC = "comic_illustration"
AUTO_ASSET_STYLE_CUSTOM = "custom"
AUTO_ASSET_PROGRESS_EVENT = "company_remote.auto_asset_progress"
V3_PROCESSING_CONTRACT_VERSION = 3
V3_GROUPING_VERSION = "adjacent-short-shots-v1"
V3_REFERENCE_PACKAGE_VERSION = "asset-library-v3"
V3_CONTACT_SHEET_ITEMS = 6
V3_CONTACT_SHEET_CELL_SIZE = 512
ANALYSIS_GATEWAY_HEALTH_TIMEOUT_SECONDS = 3
ANALYSIS_GATEWAY_PROBE_TIMEOUT_SECONDS = 15

SHOT_SENSITIVITY_PRESETS = {
    "低": {"adaptive_threshold": 4.0, "min_content_val": 20.0, "fade_threshold": 8.0},
    "标准": {"adaptive_threshold": 3.0, "min_content_val": 15.0, "fade_threshold": 12.0},
    "高": {"adaptive_threshold": 2.2, "min_content_val": 10.0, "fade_threshold": 16.0},
}


@dataclass(frozen=True)
class VideoEngineAdapter:
    key: str
    min_request_duration: float
    max_request_duration: float
    reference_image_limit: int

    def generate(
        self,
        *,
        source_segment: Any | None,
        references: list[Any],
        prompt: str,
        model: str,
        duration: float,
        negative_prompt: str,
        include_source_video: bool = True,
        generate_audio: bool | None = None,
    ) -> tuple[Any, str]:
        reference_videos = [source_segment] if include_source_video and source_segment is not None else []
        if not reference_videos and not references:
            raise ValueError("视频生成至少需要一张参考图或一段参考视频。")
        if self.key == "wan":
            return generate_dashscope_video(
                _get_config("aliyun_dashscope_video"),
                operation="dashscope_reference_to_video",
                prompt=prompt,
                model=model,
                resolution="720P",
                ratio="16:9",
                duration=int(round(duration)),
                negative_prompt=negative_prompt,
                prompt_extend=True,
                watermark=False,
                seed=0,
                reference_images=references,
                reference_videos=reference_videos,
            )
        extra_values = {
            "watermark": False,
            "auto_downscale": True,
            "auto_upscale": False,
        }
        if generate_audio is not None:
            extra_values["generate_audio"] = bool(generate_audio)
        return generate_video(
            _get_config("seedance2"),
            operation="seedance2_reference_video",
            prompt=prompt,
            model=model,
            resolution="720p",
            ratio="adaptive",
            duration=int(round(duration)),
            reference_images=references,
            reference_video=source_segment if reference_videos else None,
            reference_videos=reference_videos,
            extra_values=extra_values,
        )


@dataclass
class LongVideoAssets:
    manifest: dict[str, Any]
    people: dict[str, Any]
    backgrounds: dict[str, Any]


@dataclass
class LongVideoJob:
    video: Any
    assets: LongVideoAssets
    prompt: str
    engine: str
    model: str
    segment_duration: int
    ai_model: str
    max_retries: int
    resume: bool
    force_rerun: bool
    negative_prompt: str
    total_duration: float
    source_path: str
    job_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ShotBoundary:
    time: float
    frame: int
    kind: str
    detector: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 6),
            "frame": self.frame,
            "kind": self.kind,
            "detector": self.detector,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class LogicalShot:
    index: int
    start: float
    duration: float
    end: float
    boundary_in: str
    boundary_out: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 6),
            "duration": round(self.duration, 6),
            "end": round(self.end, 6),
            "boundary_in": self.boundary_in,
            "boundary_out": self.boundary_out,
        }


@dataclass(frozen=True)
class RequestSegment:
    index: int
    logical_shot: int
    start: float
    source_start: float
    source_duration: float
    request_duration: float
    trim_offset: float
    output_duration: float
    padding_start: float
    padding_end: float
    split_reason: str
    logical_shots: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "logical_segment": self.logical_shot,
            "logical_segments": list(self.logical_shots or (self.logical_shot,)),
            "start": round(self.start, 6),
            "duration": round(self.output_duration, 6),
            "source_start": round(self.source_start, 6),
            "source_duration": round(self.source_duration, 6),
            "request_duration": round(self.request_duration, 6),
            "trim_offset": round(self.trim_offset, 6),
            "padding_start": round(self.padding_start, 6),
            "padding_end": round(self.padding_end, 6),
            "split_reason": self.split_reason,
        }


@dataclass
class LongVideoShotPlan:
    video: Any
    total_duration: float
    fps: float
    requested_mode: str
    effective_mode: str
    fixed_duration: int
    sensitivity: str
    use_audio_silence: bool
    auto_fallback: bool
    detector: str
    config: dict[str, Any]
    boundaries: list[ShotBoundary]
    shots: list[LogicalShot]
    fallback_reason: str = ""

    def segmentation_dict(self) -> dict[str, Any]:
        boundaries = [item.to_dict() for item in self.boundaries]
        boundary_payload = json.dumps(boundaries, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "detector": self.detector,
            "config": self.config,
            "boundaries": boundaries,
            "boundaries_hash": hashlib.sha256(boundary_payload).hexdigest(),
            "fallback_reason": self.fallback_reason,
            "logical_shots": [item.to_dict() for item in self.shots],
            "fps": round(self.fps, 6),
        }


def get_video_engine_adapter(engine: str) -> VideoEngineAdapter:
    if str(engine).lower().startswith("wan"):
        return VideoEngineAdapter(key="wan", min_request_duration=2.0, max_request_duration=10.0, reference_image_limit=4)
    return VideoEngineAdapter(key="seedance", min_request_duration=4.0, max_request_duration=15.0, reference_image_limit=9)


def build_segment_windows(total_duration: float, target_duration: int, *, minimum: float = 2.0) -> list[tuple[float, float]]:
    """Build continuous windows and move a too-short tail into the previous window."""
    total = float(total_duration)
    target = float(target_duration)
    if total <= 0:
        raise ValueError("输入视频时长必须大于 0 秒。")
    if target not in (10.0, 15.0):
        raise ValueError("分段时长只能选择 10 秒或 15 秒。")
    if total < minimum:
        raise ValueError(f"输入视频只有 {total:.3f} 秒，短于模型允许的最小分段时长 {minimum:g} 秒。")

    lengths: list[float] = []
    remaining = total
    while remaining > target + 1e-6:
        lengths.append(target)
        remaining -= target
    if remaining > 1e-6:
        lengths.append(remaining)

    if len(lengths) > 1 and lengths[-1] < minimum:
        deficit = minimum - lengths[-1]
        if lengths[-2] - deficit < minimum:
            raise ValueError("视频尾段无法调整到模型允许的最小时长，请改用另一种分段长度。")
        lengths[-2] -= deficit
        lengths[-1] += deficit

    windows: list[tuple[float, float]] = []
    start = 0.0
    for length in lengths:
        windows.append((round(start, 6), round(length, 6)))
        start += length
    if windows:
        last_start, last_length = windows[-1]
        windows[-1] = (last_start, round(total - last_start, 6))
    return windows


def split_windows_for_engine(
    windows: list[tuple[float, float]],
    engine: str,
) -> list[tuple[float, float, int]]:
    """Split Wan reference-video windows to its 10-second request limit."""
    adapter = get_video_engine_adapter(engine)
    max_duration = adapter.max_request_duration
    min_duration = adapter.min_request_duration
    result: list[tuple[float, float, int]] = []
    for logical_index, (start, duration) in enumerate(windows, start=1):
        offset = 0.0
        while duration - offset > max_duration + 1e-6:
            result.append((round(start + offset, 6), max_duration, logical_index))
            offset += max_duration
        tail = duration - offset
        if tail > 1e-6:
            if tail < min_duration:
                raise ValueError(
                    f"第 {logical_index} 段经过 {engine} 时长拆分后剩余 {tail:.3f} 秒，低于模型最小 {min_duration:g} 秒。"
                )
            result.append((round(start + offset, 6), round(tail, 6), logical_index))
    return result


def _video_file_info(path: str, fallback_duration: float = 0.0) -> tuple[float, float, int]:
    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise ValueError("输入文件没有视频流。")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        duration = float(fallback_duration)
        if container.duration is not None:
            duration = float(container.duration / av.time_base)
        elif stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        frame_count = int(stream.frames or round(duration * fps))
    if duration <= 0 or fps <= 0:
        raise ValueError(f"无法读取有效视频信息：duration={duration}, fps={fps}")
    return duration, fps, frame_count


def _release_scene_video(video: Any) -> None:
    capture = getattr(video, "capture", None)
    if capture is None:
        capture = getattr(video, "_capture", None)
    if capture is not None and hasattr(capture, "release"):
        capture.release()


@contextmanager
def _background_processing_priority():
    process = None
    original_priority = None
    try:
        if os.name == "nt":
            import psutil

            process = psutil.Process()
            original_priority = process.nice()
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        process = None
        original_priority = None

    try:
        yield
    finally:
        if process is not None and original_priority is not None:
            try:
                process.nice(original_priority)
            except Exception:
                pass


def _scene_detection_source(source_path: str, directory: Path) -> tuple[str, dict[str, Any]]:
    width, height, _fps = _video_geometry(Path(source_path))
    metadata = {
        "proxy": False,
        "source_width": width,
        "source_height": height,
        "detection_width": width,
        "detection_height": height,
    }
    if width <= SHOT_DETECTION_PROXY_MAX_WIDTH:
        return source_path, metadata

    target_width = SHOT_DETECTION_PROXY_MAX_WIDTH
    target_height = max(2, int(round(height * target_width / width)))
    target_height += target_height % 2
    proxy_path = directory / "scene_detection_proxy.mp4"
    started = time.perf_counter()
    _run_ffmpeg(
        [
            _ffmpeg_exe(),
            "-y",
            "-i",
            source_path,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"scale={target_width}:{target_height}:flags=bilinear",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(SHOT_DETECTION_CPU_THREADS),
            str(proxy_path),
        ],
        error_prefix="分镜检测代理视频生成失败",
    )
    metadata.update(
        {
            "proxy": True,
            "detection_width": target_width,
            "detection_height": target_height,
            "proxy_elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    logging.info(
        "[company_remote] 分镜检测代理生成完成：%sx%s -> %sx%s，耗时 %.3f 秒",
        width,
        height,
        target_width,
        target_height,
        metadata["proxy_elapsed_seconds"],
    )
    return str(proxy_path), metadata


class _RecordingDetector:
    def __init__(self, detector: Any):
        self.detector = detector
        self.cuts: list[Any] = []

    @property
    def stats_manager(self):
        return self.detector.stats_manager

    @stats_manager.setter
    def stats_manager(self, value):
        self.detector.stats_manager = value

    @property
    def event_buffer_length(self) -> int:
        return int(self.detector.event_buffer_length)

    def get_metrics(self) -> list[str]:
        return self.detector.get_metrics()

    def process_frame(self, timecode: Any, frame_img: np.ndarray) -> list[Any]:
        cuts = self.detector.process_frame(timecode, frame_img)
        self.cuts.extend(cuts)
        return cuts

    def post_process(self, timecode: Any) -> list[Any]:
        cuts = self.detector.post_process(timecode)
        self.cuts.extend(cuts)
        return cuts


def _boundaries_from_detector_cuts(
    cuts: list[Any],
    *,
    detector_kind: str,
    stats: Any,
    config: dict[str, Any],
) -> list[ShotBoundary]:
    metric_name = "adaptive_ratio (w=2)" if detector_kind == "adaptive" else "average_rgb"
    boundary_kind = "hard_cut" if detector_kind == "adaptive" else "fade"
    boundaries: list[ShotBoundary] = []
    unique_cuts = {int(item.frame_num): item for item in cuts}
    for frame, cut in sorted(unique_cuts.items()):
        metric = None
        try:
            metric = stats.get_metrics(frame, [metric_name])[0]
        except (IndexError, KeyError, TypeError, ValueError):
            metric = None
        if detector_kind == "adaptive":
            threshold = max(0.001, float(config["adaptive_threshold"]))
            confidence = min(1.0, max(0.5, float(metric or threshold) / (threshold * 2.0)))
        else:
            confidence = 0.75
        boundaries.append(
            ShotBoundary(
                time=float(cut.seconds),
                frame=frame,
                kind=boundary_kind,
                detector=detector_kind,
                confidence=confidence,
            )
        )
    return boundaries


def _run_scene_detectors(
    source_path: str,
    *,
    fps: float,
    config: dict[str, Any],
) -> tuple[list[ShotBoundary], list[ShotBoundary]]:
    try:
        from scenedetect import SceneManager, StatsManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ThresholdDetector
    except ImportError as exc:
        raise RuntimeError("缺少 PySceneDetect，请安装 company_remote/requirements.txt。") from exc

    min_scene_len = max(3, int(round(fps * 0.3)))
    adaptive = _RecordingDetector(
        AdaptiveDetector(
            adaptive_threshold=float(config["adaptive_threshold"]),
            min_content_val=float(config["min_content_val"]),
            min_scene_len=min_scene_len,
        )
    )
    threshold = _RecordingDetector(
        ThresholdDetector(
            threshold=float(config["fade_threshold"]),
            min_scene_len=min_scene_len,
            add_final_scene=True,
        )
    )

    video = open_video(source_path, backend="opencv")
    stats = StatsManager()
    manager = SceneManager(stats)
    manager.add_detector(adaptive)
    manager.add_detector(threshold)
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(min(SHOT_DETECTION_CPU_THREADS, max(1, previous_cv_threads)))
    try:
        manager.detect_scenes(video=video, show_progress=False)
        return (
            _boundaries_from_detector_cuts(
                adaptive.cuts,
                detector_kind="adaptive",
                stats=stats,
                config=config,
            ),
            _boundaries_from_detector_cuts(
                threshold.cuts,
                detector_kind="threshold",
                stats=stats,
                config=config,
            ),
        )
    finally:
        cv2.setNumThreads(previous_cv_threads)
        _release_scene_video(video)


def _merge_shot_boundaries(
    boundaries: list[ShotBoundary],
    *,
    fps: float,
    duration: float,
    merge_frames: int = 3,
) -> list[ShotBoundary]:
    valid = sorted(
        (item for item in boundaries if 0 < item.time < duration and item.frame > 0),
        key=lambda item: (item.frame, item.detector),
    )
    groups: list[list[ShotBoundary]] = []
    for item in valid:
        if groups and item.frame - groups[-1][-1].frame <= merge_frames:
            groups[-1].append(item)
        else:
            groups.append([item])

    merged: list[ShotBoundary] = []
    for group in groups:
        adaptive = [item for item in group if item.detector == "adaptive"]
        selected = max(adaptive or group, key=lambda item: item.confidence)
        detectors = "+".join(sorted({item.detector for item in group}))
        kind = "fade" if any(item.detector == "threshold" for item in group) else "hard_cut"
        merged.append(
            ShotBoundary(
                time=round(selected.frame / fps, 6),
                frame=selected.frame,
                kind=kind,
                detector=detectors,
                confidence=max(item.confidence for item in group),
            )
        )
    return merged


def _shots_from_boundaries(duration: float, boundaries: list[ShotBoundary]) -> list[LogicalShot]:
    points = [0.0, *[item.time for item in boundaries], float(duration)]
    shots: list[LogicalShot] = []
    for index, (start, end) in enumerate(zip(points, points[1:]), start=1):
        shot_duration = end - start
        if shot_duration <= 1e-6:
            continue
        boundary_in = "video_start" if index == 1 else boundaries[index - 2].kind
        boundary_out = "video_end" if index == len(points) - 1 else boundaries[index - 1].kind
        shots.append(
            LogicalShot(
                index=index,
                start=round(start, 6),
                duration=round(shot_duration, 6),
                end=round(end, 6),
                boundary_in=boundary_in,
                boundary_out=boundary_out,
            )
        )
    if not shots:
        raise ValueError("镜头检测没有生成任何有效时间段。")
    return shots


def _fixed_shots(duration: float, target_duration: int) -> tuple[list[ShotBoundary], list[LogicalShot]]:
    if duration < MIN_REQUEST_DURATION:
        windows = [(0.0, duration)]
    else:
        windows = build_segment_windows(duration, target_duration, minimum=MIN_REQUEST_DURATION)
    boundaries = [
        ShotBoundary(
            time=start,
            frame=0,
            kind="fixed",
            detector="fixed_duration",
            confidence=1.0,
        )
        for start, _window_duration in windows[1:]
    ]
    shots = []
    for index, (start, window_duration) in enumerate(windows, start=1):
        shots.append(
            LogicalShot(
                index=index,
                start=start,
                duration=window_duration,
                end=round(start + window_duration, 6),
                boundary_in="video_start" if index == 1 else "fixed",
                boundary_out="video_end" if index == len(windows) else "fixed",
            )
        )
    return boundaries, shots


def _frames_at_times(source_path: str, times: list[float]) -> torch.Tensor:
    capture = cv2.VideoCapture(source_path)
    frames: list[torch.Tensor] = []
    try:
        for timestamp in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            if width > SHOT_PREVIEW_MAX_WIDTH:
                target_height = max(1, int(round(height * SHOT_PREVIEW_MAX_WIDTH / width)))
                frame = cv2.resize(
                    frame,
                    (SHOT_PREVIEW_MAX_WIDTH, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(rgb))
    finally:
        capture.release()
    if not frames:
        raise ValueError("无法读取镜头切点预览画面。")
    return torch.stack(frames, dim=0)


def detect_long_video_shots(
    *,
    video: Any,
    mode: str,
    fixed_duration: int,
    sensitivity: str,
    use_audio_silence: bool,
    auto_fallback: bool,
) -> tuple[LongVideoShotPlan, str, torch.Tensor]:
    total_duration = float(video.get_duration())
    requested_mode = "fixed" if "固定" in str(mode) else "shot_aware"
    sensitivity = sensitivity if sensitivity in SHOT_SENSITIVITY_PRESETS else "标准"
    config = dict(SHOT_SENSITIVITY_PRESETS[sensitivity])
    config.update(
        {
            "merge_frames": 3,
            "minimum_request_duration": MIN_REQUEST_DURATION,
            "detection_passes": 1,
            "detection_proxy_max_width": SHOT_DETECTION_PROXY_MAX_WIDTH,
            "detection_cpu_threads": SHOT_DETECTION_CPU_THREADS,
        }
    )

    with tempfile.TemporaryDirectory(prefix="company_shot_detect_") as temporary:
        source_path = _video_source_path(video, Path(temporary))
        _file_duration, fps, _frame_count = _video_file_info(source_path, total_duration)
        detector_name = "fixed_duration"
        effective_mode = "fixed"
        fallback_reason = ""
        if requested_mode == "fixed":
            boundaries, shots = _fixed_shots(total_duration, int(fixed_duration))
        else:
            try:
                detection_started = time.perf_counter()
                with _background_processing_priority():
                    detection_source, proxy_metadata = _scene_detection_source(source_path, Path(temporary))
                    adaptive, fades = _run_scene_detectors(
                        detection_source,
                        fps=fps,
                        config=config,
                    )
                config["detection_proxy"] = proxy_metadata
                config["detection_elapsed_seconds"] = round(time.perf_counter() - detection_started, 3)
                logging.info(
                    "[company_remote] 分镜检测完成：单次解码，adaptive=%s，threshold=%s，耗时 %.3f 秒",
                    len(adaptive),
                    len(fades),
                    config["detection_elapsed_seconds"],
                )
                boundaries = _merge_shot_boundaries(
                    [*adaptive, *fades],
                    fps=fps,
                    duration=total_duration,
                    merge_frames=int(config["merge_frames"]),
                )
                shots = _shots_from_boundaries(total_duration, boundaries)
                covered = sum(item.duration for item in shots)
                if abs(covered - total_duration) > max(1.0 / fps, 0.05):
                    raise ValueError(f"镜头时间范围覆盖异常：{covered:.6f}/{total_duration:.6f} 秒")
                detector_name = "PySceneDetect AdaptiveDetector + ThresholdDetector (single pass)"
                effective_mode = "shot_aware"
            except Exception as exc:
                if not auto_fallback:
                    raise
                fallback_reason = f"镜头检测失败，已改用固定 {fixed_duration} 秒切分：{exc}"
                boundaries, shots = _fixed_shots(total_duration, int(fixed_duration))
                detector_name = "fixed_duration_fallback"
                effective_mode = "fixed_fallback"

        preview_times = [
            min(item.end - 1e-3, item.start + min(0.05, item.duration / 2.0))
            for item in shots[:SHOT_PREVIEW_LIMIT]
        ]
        preview_started = time.perf_counter()
        previews = _frames_at_times(source_path, preview_times)
        logging.info(
            "[company_remote] 分镜预览生成完成：%s 张，尺寸 %sx%s，耗时 %.3f 秒",
            previews.shape[0],
            previews.shape[2],
            previews.shape[1],
            time.perf_counter() - preview_started,
        )

    plan = LongVideoShotPlan(
        video=video,
        total_duration=total_duration,
        fps=fps,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        fixed_duration=int(fixed_duration),
        sensitivity=sensitivity,
        use_audio_silence=bool(use_audio_silence),
        auto_fallback=bool(auto_fallback),
        detector=detector_name,
        config=config,
        boundaries=boundaries,
        shots=shots,
        fallback_reason=fallback_reason,
    )
    summary = plan.segmentation_dict()
    summary["preview_count"] = int(previews.shape[0])
    summary["preview_truncated"] = len(shots) > SHOT_PREVIEW_LIMIT
    return plan, json.dumps(summary, ensure_ascii=False, indent=2), previews


def inspect_long_video_shots(
    plan: LongVideoShotPlan,
    *,
    shot_index: int,
    export_all_shots: bool,
) -> tuple[Any, torch.Tensor, torch.Tensor, str, str]:
    if not plan.shots:
        raise ValueError("镜头计划中没有可检查的镜头。")
    selected_index = int(shot_index)
    if selected_index < 1 or selected_index > len(plan.shots):
        raise ValueError(f"镜头序号必须在 1-{len(plan.shots)} 之间，当前为 {selected_index}。")
    selected_shot = plan.shots[selected_index - 1]
    selected_video = plan.video.as_trimmed(
        selected_shot.start,
        selected_shot.duration,
        strict_duration=True,
    )
    if selected_video is None:
        raise ValueError(f"无法读取第 {selected_index} 个镜头的视频内容。")

    with tempfile.TemporaryDirectory(prefix="company_shot_inspect_") as temporary:
        source_path = _video_source_path(plan.video, Path(temporary))
        margin = min(max(1.0 / max(plan.fps, 1.0), 0.04), selected_shot.duration / 4.0)
        selected_times = [
            selected_shot.start + margin,
            selected_shot.start + selected_shot.duration / 2.0,
            max(selected_shot.start, selected_shot.end - margin),
        ]
        selected_frames = _frames_at_times(source_path, selected_times)

        boundary_times: list[float] = []
        boundary_order: list[dict[str, Any]] = []
        delta = max(1.0 / max(plan.fps, 1.0), 0.04)
        for boundary in plan.boundaries[:32]:
            before = max(0.0, boundary.time - delta)
            after = min(plan.total_duration - 1e-3, boundary.time + delta)
            boundary_times.extend([before, after])
            boundary_order.append(
                {
                    "boundary": boundary.to_dict(),
                    "preview_order": ["切点前", "切点后"],
                    "preview_times": [round(before, 6), round(after, 6)],
                }
            )
        boundary_frames = (
            _frames_at_times(source_path, boundary_times)
            if boundary_times
            else selected_frames[[0, -1]]
        )

    export_directory = ""
    exported_paths: list[str] = []
    if export_all_shots:
        segmentation = plan.segmentation_dict()
        identity = hashlib.sha256(
            json.dumps(
                {
                    "duration": plan.total_duration,
                    "fps": plan.fps,
                    "boundaries_hash": segmentation["boundaries_hash"],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        export_root = Path(folder_paths.get_output_directory()) / SHOT_TEST_ROOT_NAME / identity / "shots"
        export_root.mkdir(parents=True, exist_ok=True)
        for shot in plan.shots:
            output_path = export_root / f"shot_{shot.index:04d}_{shot.start:.3f}s_{shot.end:.3f}s.mp4"
            if not output_path.is_file():
                clip = plan.video.as_trimmed(shot.start, shot.duration, strict_duration=True)
                if clip is None:
                    raise ValueError(f"导出第 {shot.index} 个镜头失败：无法取得视频片段。")
                clip.save_to(str(output_path))
            exported_paths.append(str(output_path))
        export_directory = str(export_root)

    report = {
        "mode": plan.effective_mode,
        "detector": plan.detector,
        "total_duration": round(plan.total_duration, 6),
        "fps": round(plan.fps, 6),
        "shot_count": len(plan.shots),
        "boundary_count": len(plan.boundaries),
        "selected_shot": selected_shot.to_dict(),
        "selected_frame_order": ["镜头开始", "镜头中间", "镜头结束"],
        "boundary_preview_order": boundary_order,
        "boundary_preview_truncated": len(plan.boundaries) > 32,
        "fallback_reason": plan.fallback_reason,
        "exported_paths": exported_paths,
    }
    return (
        selected_video,
        selected_frames,
        boundary_frames,
        json.dumps(report, ensure_ascii=False, indent=2),
        export_directory,
    )


def select_continuous_shot_range(
    plan: LongVideoShotPlan,
    *,
    start_shot: int,
    shot_count: int,
) -> tuple[LongVideoShotPlan, Any, str]:
    """Trim and rebase a consecutive shot range."""
    if not plan.shots:
        raise ValueError("镜头计划为空，请先运行长视频镜头检测。")

    start_index = int(start_shot)
    requested_count = int(shot_count)
    count = requested_count
    if start_index < 1 or start_index > len(plan.shots):
        raise ValueError(f"起始镜头序号必须在 1-{len(plan.shots)} 之间，当前为 {start_index}。")
    if count == 0:
        count = len(plan.shots) - start_index + 1
    elif count < 0:
        raise ValueError(f"连续镜头数量必须为 0 或正整数，当前为 {count}。0 表示自动选择全部剩余镜头。")

    end_index = start_index + count - 1
    if end_index > len(plan.shots):
        available = len(plan.shots) - start_index + 1
        raise ValueError(
            f"从第 {start_index} 个镜头开始只剩 {available} 个镜头，不能选择连续 {count} 个。"
        )

    source_shots = plan.shots[start_index - 1 : end_index]
    source_start = float(source_shots[0].start)
    source_end = float(source_shots[-1].end)
    requested_duration = source_end - source_start
    selected_video = plan.video.as_trimmed(source_start, requested_duration, strict_duration=True)
    if selected_video is None:
        raise ValueError("无法从原视频取得选中的连续分镜范围。")

    selected_duration = float(selected_video.get_duration())
    tolerance = max(1.0 / max(plan.fps, 1.0), 0.05)
    if abs(selected_duration - requested_duration) > tolerance:
        raise ValueError(
            f"连续分镜截取时长异常：期望 {requested_duration:.3f} 秒，实际 {selected_duration:.3f} 秒。"
        )

    rebased_shots: list[LogicalShot] = []
    source_indices: list[int] = []
    for selected_index, shot in enumerate(source_shots, start=1):
        rebased_start = max(0.0, float(shot.start) - source_start)
        rebased_end = float(shot.end) - source_start
        if selected_index == len(source_shots):
            rebased_end = selected_duration
        rebased_end = min(selected_duration, max(rebased_start, rebased_end))
        rebased_shots.append(
            LogicalShot(
                index=selected_index,
                start=round(rebased_start, 6),
                duration=round(rebased_end - rebased_start, 6),
                end=round(rebased_end, 6),
                boundary_in=shot.boundary_in,
                boundary_out=shot.boundary_out,
            )
        )
        source_indices.append(int(shot.index))

    rebased_boundaries: list[ShotBoundary] = []
    for boundary in plan.boundaries:
        rebased_time = float(boundary.time) - source_start
        if rebased_time <= tolerance / 2 or rebased_time >= selected_duration - tolerance / 2:
            continue
        rebased_boundaries.append(
            ShotBoundary(
                time=round(rebased_time, 6),
                frame=max(0, int(round(rebased_time * plan.fps))),
                kind=boundary.kind,
                detector=boundary.detector,
                confidence=boundary.confidence,
            )
        )

    selection_config = dict(plan.config)
    selection_config["continuity_test_selection"] = {
        "start_shot": start_index,
        "requested_shot_count": requested_count,
        "shot_count": count,
        "auto_all_remaining": requested_count == 0,
        "source_shot_indices": source_indices,
        "source_start": round(source_start, 6),
        "source_end": round(source_end, 6),
    }
    selected_plan = LongVideoShotPlan(
        video=selected_video,
        total_duration=selected_duration,
        fps=plan.fps,
        requested_mode=plan.requested_mode,
        effective_mode=plan.effective_mode,
        fixed_duration=plan.fixed_duration,
        sensitivity=plan.sensitivity,
        use_audio_silence=plan.use_audio_silence,
        auto_fallback=plan.auto_fallback,
        detector=f"{plan.detector}+continuity_range",
        config=selection_config,
        boundaries=rebased_boundaries,
        shots=rebased_shots,
        fallback_reason=plan.fallback_reason,
    )
    report = {
        "source_shot_count": len(plan.shots),
        "requested_shot_count": requested_count,
        "selected_shot_count": count,
        "auto_all_remaining": requested_count == 0,
        "selected_source_shots": source_indices,
        "source_time_range": {
            "start": round(source_start, 6),
            "end": round(source_end, 6),
            "duration": round(requested_duration, 6),
        },
        "selected_video_duration": round(selected_duration, 6),
        "selected_shots": [item.to_dict() for item in rebased_shots],
        "continuity_rule": "第 2 个及后续生成分段自动使用上一生成分段末帧作为软连续性参考。",
    }
    return selected_plan, selected_video, json.dumps(report, ensure_ascii=False, indent=2)


def _shot_count_covering_duration(plan: LongVideoShotPlan, *, start_shot: int, target_duration: float) -> int:
    start_index = int(start_shot)
    if start_index < 1 or start_index > len(plan.shots):
        raise ValueError(f"起始镜头序号必须在 1-{len(plan.shots)} 之间，当前为 {start_index}。")
    if target_duration <= 0:
        return 0

    source_shots = plan.shots[start_index - 1 :]
    if not source_shots:
        return 0
    source_start = float(source_shots[0].start)
    target_end = source_start + float(target_duration)
    tolerance = max(1.0 / max(plan.fps, 1.0), 0.05)
    for count, shot in enumerate(source_shots, start=1):
        if float(shot.end) + tolerance >= target_end:
            return count
    return len(source_shots)


def select_long_video_length_range(
    plan: LongVideoShotPlan,
    *,
    start_shot: int,
    limit_mode: str,
    limit_minutes: float,
    limit_percent: float,
    shot_count: int,
) -> tuple[LongVideoShotPlan, Any, str]:
    """Select a whole-shot range by minutes, total-duration percentage, explicit count, or all remaining."""
    mode = str(limit_mode or "").strip()
    requested_target_seconds = 0.0
    requested_count = int(shot_count)
    if "分钟" in mode:
        minutes = float(limit_minutes)
        if minutes < 0:
            raise ValueError(f"生成时长分钟数必须为 0 或正数，当前为 {minutes:g}。")
        requested_target_seconds = minutes * 60.0
        requested_count = _shot_count_covering_duration(
            plan,
            start_shot=int(start_shot),
            target_duration=requested_target_seconds,
        )
    elif "百分比" in mode:
        percent = float(limit_percent)
        if percent < 0:
            raise ValueError(f"生成百分比必须为 0 或正数，当前为 {percent:g}。")
        requested_target_seconds = float(plan.total_duration) * percent / 100.0
        requested_count = _shot_count_covering_duration(
            plan,
            start_shot=int(start_shot),
            target_duration=requested_target_seconds,
        )
    elif "镜头数量" in mode:
        requested_count = int(shot_count)
    else:
        requested_count = 0

    selected_plan, selected_video, report_json = select_continuous_shot_range(
        plan,
        start_shot=int(start_shot),
        shot_count=requested_count,
    )
    selected_report = json.loads(report_json)
    selected_duration = float(selected_report["source_time_range"]["duration"])
    selected_count = int(selected_report["selected_shot_count"])
    length_selection = {
        "limit_mode": mode or "全部剩余",
        "start_shot": int(start_shot),
        "limit_minutes": float(limit_minutes),
        "limit_percent": float(limit_percent),
        "requested_shot_count": int(shot_count),
        "resolved_shot_count": selected_count,
        "target_duration_seconds": round(requested_target_seconds, 6),
        "selected_duration_seconds": round(selected_duration, 6),
        "over_target_seconds": round(max(0.0, selected_duration - requested_target_seconds), 6)
        if requested_target_seconds > 0
        else 0.0,
        "whole_shot_policy": "目标时长落在镜头中间时保留完整镜头，不硬切断。",
    }
    selected_config = dict(selected_plan.config)
    selected_config["length_range_selection"] = length_selection
    selected_plan.config = selected_config
    selected_plan.detector = f"{plan.detector}+length_range"
    selected_report["length_range_selection"] = length_selection
    selected_report["range_rule"] = "按完整镜头选择范围，最终时长可能略长于目标时长。"
    return selected_plan, selected_video, json.dumps(selected_report, ensure_ascii=False, indent=2)


def _manual_batch_root() -> Path:
    return (Path(folder_paths.get_output_directory()).resolve() / MANUAL_BATCH_ROOT_NAME).resolve()


def _manual_batch_series_id(value: str, *, allow_empty: bool = False) -> str:
    raw = str(value or "").strip()
    if not raw and allow_empty:
        return f"series_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 100000:05d}"
    if not raw:
        raise ValueError("继续或重试时必须填写系列 ID。")
    path_value = Path(raw)
    if path_value.is_absolute() or raw in {".", ".."} or "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("系列 ID 只能包含字母、数字、下划线和短横线，不能包含路径。")
    normalized = _safe_slug(raw, "series")
    if normalized != raw:
        raise ValueError("系列 ID 含有不支持的字符，请改用字母、数字、下划线或短横线。")
    return normalized


def _manual_batch_series_dir(series_id: str) -> Path:
    root = _manual_batch_root()
    directory = (root / _manual_batch_series_id(series_id)).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("系列状态路径越出了 ComfyUI output 目录。")
    return directory


def _manual_batch_state_path(series_id: str) -> Path:
    return _manual_batch_series_dir(series_id) / "series.json"


def _manual_batch_validate_state_path(series_id: str, value: str) -> Path:
    expected = _manual_batch_state_path(series_id).resolve()
    actual = Path(str(value or "")).resolve()
    if actual != expected:
        raise ValueError("手动批次状态文件必须位于对应系列目录内。")
    return expected


def _manual_batch_attempt_dir(series_id: str, batch_id: str, attempt: int) -> Path:
    series_dir = _manual_batch_series_dir(series_id)
    candidate = (series_dir / "batches" / _safe_slug(batch_id, "batch") / f"attempt_{int(attempt):03d}").resolve()
    if not candidate.is_relative_to(series_dir):
        raise ValueError("手动批次任务目录越出了对应系列目录。")
    return candidate


def _manual_batch_retry_manifest(manual_batch: dict[str, Any] | None) -> tuple[dict[str, Any], Path] | None:
    """Load the immediately preceding attempt for a retry of the same manual batch."""
    batch = manual_batch if isinstance(manual_batch, dict) else {}
    if str(batch.get("action") or "") != "重试当前批":
        return None
    try:
        attempt = int(batch.get("attempt", 0))
    except (TypeError, ValueError):
        return None
    if attempt <= 1:
        return None
    previous_path = _manual_batch_attempt_dir(
        str(batch.get("series_id") or ""),
        str(batch.get("batch_id") or ""),
        attempt - 1,
    ) / "job" / "manifest.json"
    if not previous_path.is_file():
        return None
    try:
        candidate = json.loads(previous_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    previous_batch = candidate.get("manual_batch") if isinstance(candidate, dict) else None
    if not isinstance(previous_batch, dict):
        return None
    for key in ("series_id", "batch_id"):
        if str(previous_batch.get(key) or "") != str(batch.get(key) or ""):
            return None
    for key in ("source_start", "source_end", "source_duration"):
        try:
            if abs(float(previous_batch.get(key)) - float(batch.get(key))) > 0.001:
                return None
        except (TypeError, ValueError):
            return None
    return candidate, previous_path


def _manual_batch_source_identity(video: Any) -> dict[str, Any]:
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    if not isinstance(source, str) or not Path(source).is_file():
        raise ValueError("手动批次模式需要可定位到本地文件的视频输入。")
    path = Path(source).resolve()
    trim_start, trim_duration = (0.0, 0.0)
    if hasattr(video, "get_active_trim_window"):
        trim_start, trim_duration = video.get_active_trim_window()
    duration, fps, frame_count = _video_file_info(str(path), float(video.get_duration()))
    size, quick_hash = _quick_file_fingerprint(path)
    stat = path.stat()
    effective_duration = float(video.get_duration())
    return {
        "path": str(path),
        "size": int(size),
        "mtime_ns": int(stat.st_mtime_ns),
        "quick_hash": quick_hash,
        "duration": round(effective_duration, 6),
        "file_duration": round(duration, 6),
        "fps": round(float(fps), 6),
        "frame_count": int(frame_count),
        "trim_start": round(float(trim_start), 6),
        "trim_duration": round(float(trim_duration), 6),
    }


def _manual_batch_source_identity_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key in ("path", "size", "mtime_ns", "quick_hash", "frame_count"):
        if expected.get(key) != actual.get(key):
            return False
    for key in ("duration", "file_duration", "fps", "trim_start", "trim_duration"):
        try:
            if abs(float(expected.get(key, 0.0)) - float(actual.get(key, 0.0))) > 0.05:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _manual_batch_read_state(series_id: str) -> tuple[dict[str, Any], Path]:
    path = _manual_batch_state_path(series_id)
    if not path.is_file():
        raise ValueError(f"没有找到系列 {series_id} 的状态文件，请先选择“新建系列”。")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"系列状态文件不可读取：{path}") from exc
    if not isinstance(state, dict) or state.get("contract") != MANUAL_BATCH_CONTRACT:
        raise ValueError("系列状态文件不是当前手动批次版本，不能继续使用。")
    return state, path


def _manual_batch_write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _atomic_write_json(path, state)


def _manual_batch_commit_record_valid(record: dict[str, Any]) -> bool:
    try:
        final_path = Path(str(record["final_video"])).resolve()
        frame_path = Path(str(record["final_frame"])).resolve()
        output_root = Path(folder_paths.get_output_directory()).resolve()
        return (
            final_path.is_file()
            and frame_path.is_file()
            and final_path.is_relative_to(output_root)
            and frame_path.is_relative_to(output_root)
            and record.get("contract") == MANUAL_BATCH_CONTRACT
        )
    except (KeyError, OSError, RuntimeError, ValueError, TypeError):
        return False


def _manual_batch_matches_current(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    try:
        if str(candidate.get("batch_id") or "") != str(current.get("batch_id") or ""):
            return False
        for key in ("batch_index", "attempt"):
            if int(candidate.get(key, -1)) != int(current.get(key, -2)):
                return False
        for key in ("source_start", "source_end", "source_duration"):
            if abs(float(candidate.get(key, -1.0)) - float(current.get(key, -2.0))) > 0.001:
                return False
    except (TypeError, ValueError):
        return False
    return True


def _manual_batch_commit_attempt(record: dict[str, Any]) -> int | None:
    try:
        return int(record["attempt"])
    except (KeyError, TypeError, ValueError):
        return None


def _manual_batch_commit_record_is_well_formed(record: dict[str, Any]) -> bool:
    return (
        _manual_batch_commit_attempt(record) is not None
        and "batch_index" in record
        and all(key in record for key in ("batch_id", "series_id", "source_start", "source_end", "source_duration"))
    )


def _manual_batch_apply_commit(state: dict[str, Any], record: dict[str, Any]) -> bool:
    if not _manual_batch_commit_record_is_well_formed(record):
        return False
    if not _manual_batch_commit_record_valid(record):
        return False
    current = state.get("current_batch")
    if not isinstance(current, dict):
        return False
    if str(record.get("series_id") or "") != str(state.get("series_id") or ""):
        return False
    if not _manual_batch_matches_current(current, record):
        return False
    completed = state.setdefault("completed_batches", [])
    previous = next(
        (item for item in completed if isinstance(item, dict) and str(item.get("batch_id")) == str(record.get("batch_id"))),
        None,
    )
    if previous is not None:
        current_attempt = _manual_batch_commit_attempt(record)
        previous_attempt = _manual_batch_commit_attempt(previous)
        if current_attempt is None or previous_attempt is None or current_attempt <= previous_attempt:
            return False
        history = list(previous.get("attempt_history") or [])
        history.append(dict(previous))
        record["attempt_history"] = history
        completed.remove(previous)
    completed.append(dict(record))
    completed.sort(key=lambda item: int(item.get("batch_index", 0)))
    state["next_cursor"] = round(max(float(state.get("next_cursor", 0.0)), float(record["source_end"])), 6)
    state["current_cursor"] = state["next_cursor"]
    state["last_committed_batch_id"] = record["batch_id"]
    state["last_committed_final_video"] = record["final_video"]
    state["last_committed_final_frame"] = record["final_frame"]
    current_batch = dict(state.get("current_batch") or {})
    current_batch.update(record)
    current_batch["status"] = "completed"
    state["current_batch"] = current_batch
    state["series_complete"] = bool(state["next_cursor"] >= float(state["source_duration"]) - 0.05)
    return True


def _manual_batch_reconcile_state(state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    changed = False
    series_dir = state_path.parent
    commit_dir = series_dir / "commits"
    if commit_dir.is_dir():
        for marker in sorted(commit_dir.glob("*.json")):
            try:
                record = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and _manual_batch_apply_commit(state, record):
                changed = True
    if changed:
        _manual_batch_write_state(state_path, state)
    return state


def _manual_batch_range_plan(
    plan: LongVideoShotPlan,
    *,
    start: float,
    target_duration: float,
    boundary_tolerance: float,
) -> tuple[float, float, list[dict[str, Any]]]:
    total = float(plan.total_duration)
    start = max(0.0, min(float(start), total))
    if start >= total - 0.05:
        raise ValueError("原视频已经生成到结尾，没有可继续的批次。")
    target = min(total, start + float(target_duration))
    candidates = [
        float(boundary.time)
        for boundary in plan.boundaries
        if start + 0.05 < float(boundary.time) < total - 0.05
    ]
    nearby = [item for item in candidates if abs(item - target) <= float(boundary_tolerance)]
    end = min(nearby, key=lambda item: (abs(item - target), item)) if nearby else target
    end = min(total, max(start + 0.05, end))
    slices: list[dict[str, Any]] = []
    for shot in plan.shots:
        slice_start = max(start, float(shot.start))
        slice_end = min(end, float(shot.end))
        if slice_end - slice_start <= 0.05:
            continue
        starts_inside = slice_start > float(shot.start) + 0.05
        ends_inside = slice_end < float(shot.end) - 0.05
        slices.append(
            {
                "source_start": round(slice_start, 6),
                "source_end": round(slice_end, 6),
                "source_duration": round(slice_end - slice_start, 6),
                "parent_shot_index": int(shot.index),
                "is_inside_shot_split": starts_inside or ends_inside,
                "starts_inside_shot_split": starts_inside,
                "ends_inside_shot_split": ends_inside,
                "asset_inheritance": "parent_logical_shot_assets",
                "reference_inheritance": "parent_logical_shot_references",
            }
        )
    if not slices:
        raise ValueError("无法在当前源游标和批次长度下建立有效范围。")
    return round(start, 6), round(end, 6), slices


def _manual_batch_starts_inside_logical_shot(slices: list[dict[str, Any]]) -> bool:
    """Whether a batch starts by continuing a logical shot from the prior batch."""
    if not slices:
        return False
    first_slice = slices[0]
    return bool(first_slice.get("starts_inside_shot_split", first_slice.get("is_inside_shot_split")))


def select_manual_batch_range(
    plan: LongVideoShotPlan,
    *,
    action: str,
    series_id: str,
    batch_minutes: float,
    boundary_tolerance: float = 10.0,
    start_shot: int = 1,
    start_second: float = 0.0,
    end_second: float = 0.0,
) -> tuple[LongVideoShotPlan, Any, str, str]:
    if batch_minutes < 0.5 or batch_minutes > 5.0:
        raise ValueError("每批生成时长必须在 0.5-5 分钟之间。")
    if boundary_tolerance < 0 or boundary_tolerance > 30:
        raise ValueError("镜头边界容差必须在 0-30 秒之间。")
    start_second = max(0.0, float(start_second))
    end_second = max(0.0, float(end_second))
    if end_second > 0.0 and end_second <= start_second + 0.05:
        raise ValueError("指定结束秒必须大于起始秒。")
    use_absolute_range = start_second > 0.0 or end_second > 0.0
    if use_absolute_range and start_second >= float(plan.total_duration) - 0.05:
        raise ValueError(
            f"指定起始秒 {start_second:g} 超出或接近视频总长 {float(plan.total_duration):g} 秒。"
        )
    normalized_action = str(action or "新建系列").strip()
    if normalized_action not in {"新建系列", "继续下一批", "重试当前批"}:
        raise ValueError("批次动作必须是新建系列、继续下一批或重试当前批。")
    actual_series_id = _manual_batch_series_id(series_id, allow_empty=normalized_action == "新建系列")
    source_identity = _manual_batch_source_identity(plan.video)
    state_path = _manual_batch_state_path(actual_series_id)
    if normalized_action == "新建系列":
        if state_path.is_file():
            raise ValueError(f"系列 {actual_series_id} 已存在，请改用“继续下一批”或换一个系列 ID。")
        state = {
            "version": 1,
            "contract": MANUAL_BATCH_CONTRACT,
            "processing_contract_version": MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
            "series_id": actual_series_id,
            "source_identity": source_identity,
            "source_duration": round(float(plan.total_duration), 6),
            "source_fps": round(float(plan.fps), 6),
            "next_cursor": 0.0,
            "current_cursor": 0.0,
            "next_batch_index": 1,
            "completed_batches": [],
            "series_complete": False,
            "status": "ready",
        }
    else:
        state, state_path = _manual_batch_read_state(actual_series_id)
        state = _manual_batch_reconcile_state(state, state_path)
        if not _manual_batch_source_identity_matches(state.get("source_identity", {}), source_identity):
            raise ValueError("当前视频与系列创建时的视频身份不一致，已拒绝继续或重试。")
        if normalized_action == "继续下一批" and state.get("series_complete"):
            raise ValueError("这个系列已经生成到原视频结尾，不能继续创建新批次。")
        if normalized_action == "继续下一批" and state.get("current_batch", {}).get("status") not in {None, "completed"}:
            raise ValueError("当前批次还没有完成，请选择“重试当前批”。")
        if normalized_action == "重试当前批" and not isinstance(state.get("current_batch"), dict):
            raise ValueError("当前系列没有可以重试的批次。")

    current = state.get("current_batch") if isinstance(state.get("current_batch"), dict) else None
    is_retry = normalized_action == "重试当前批"
    if is_retry:
        source_start = float(current["source_start"])
        source_end = float(current["source_end"])
        slices = list(current.get("virtual_slices") or [])
        batch_index = int(current["batch_index"])
        attempt = int(current.get("attempt", 0)) + 1
    else:
        if use_absolute_range:
            plan_start = start_second
            if end_second > 0.0:
                plan_target = end_second - start_second
                plan_tolerance = 0.0
            else:
                plan_target = float(batch_minutes) * 60.0
                plan_tolerance = float(boundary_tolerance)
        else:
            plan_start = float(state.get("next_cursor", 0.0))
            plan_target = float(batch_minutes) * 60.0
            plan_tolerance = float(boundary_tolerance)
        source_start, source_end, slices = _manual_batch_range_plan(
            plan,
            start=plan_start,
            target_duration=plan_target,
            boundary_tolerance=plan_tolerance,
        )
        batch_index = int(state.get("next_batch_index", 1))
        attempt = 1
    cross_batch_continuity = _manual_batch_starts_inside_logical_shot(slices)
    continuity_path = (
        str(current.get("cross_batch_final_frame") or "")
        if is_retry and cross_batch_continuity
        else str(state.get("last_committed_final_frame") or "")
        if cross_batch_continuity
        else ""
    )
    batch_id = f"B{batch_index}_{int(round(source_start * 1000))}_{int(round(source_end * 1000))}"
    batch_dir = _manual_batch_attempt_dir(actual_series_id, batch_id, attempt)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch = {
        "series_id": actual_series_id,
        "batch_id": batch_id,
        "batch_index": batch_index,
        "attempt": attempt,
        "action": normalized_action,
        "status": "planned",
        "source_start": source_start,
        "source_end": source_end,
        "source_duration": round(source_end - source_start, 6),
        "source_shot_indices": sorted({int(item["parent_shot_index"]) for item in slices}),
        "virtual_slices": slices,
        "cross_batch_continuity": cross_batch_continuity,
        "cross_batch_final_frame": continuity_path,
        "attempt_dir": str(batch_dir),
        "state_path": str(state_path),
    }
    if use_absolute_range and not is_retry:
        batch["absolute_range"] = {
            "requested_start_second": round(start_second, 6),
            "requested_end_second": round(end_second, 6) if end_second > 0.0 else None,
            "exact_endpoints": end_second > 0.0,
        }
    state["current_batch"] = batch
    state["status"] = "planning"
    state["next_batch_index"] = max(int(state.get("next_batch_index", 1)), batch_index + (0 if is_retry else 1))
    _manual_batch_write_state(state_path, state)

    start_offset = source_start
    selected_video = plan.video.as_trimmed(start_offset, source_end - source_start, strict_duration=True)
    if selected_video is None:
        raise ValueError("无法取得当前批次的原视频范围。")
    selected_shots: list[LogicalShot] = []
    for index, item in enumerate(slices, start=1):
        local_start = float(item["source_start"]) - source_start
        local_end = float(item["source_end"]) - source_start
        selected_shots.append(
            LogicalShot(
                index=index,
                start=round(local_start, 6),
                duration=round(local_end - local_start, 6),
                end=round(local_end, 6),
                boundary_in="manual_batch_split" if item["is_inside_shot_split"] else "logical_shot",
                boundary_out="manual_batch_split" if item["is_inside_shot_split"] else "logical_shot",
            )
        )
    selected_config = dict(plan.config)
    selected_config["manual_batch"] = batch
    selected_plan = LongVideoShotPlan(
        video=selected_video,
        total_duration=float(selected_video.get_duration()),
        fps=plan.fps,
        requested_mode=plan.requested_mode,
        effective_mode=plan.effective_mode,
        fixed_duration=plan.fixed_duration,
        sensitivity=plan.sensitivity,
        use_audio_silence=plan.use_audio_silence,
        auto_fallback=plan.auto_fallback,
        detector=f"{plan.detector}+manual_batch",
        config=selected_config,
        boundaries=[],
        shots=selected_shots,
        fallback_reason=plan.fallback_reason,
    )
    report = {
        "contract": MANUAL_BATCH_CONTRACT,
        "series_id": actual_series_id,
        "state_path": str(state_path),
        "action": normalized_action,
        "batch": batch,
        "actual_selected_duration": round(float(selected_plan.total_duration), 6),
        "manual_pause_after_batch": True,
        "source_cursor_rule": "源视频时间线权威，生成结果时长漂移不改变下一批起点。",
    }
    return selected_plan, selected_video, json.dumps(report, ensure_ascii=False, indent=2), json.dumps(state, ensure_ascii=False, indent=2)


def _detect_silence_points(source_path: str) -> list[float]:
    command = [
        _ffmpeg_exe(),
        "-hide_banner",
        "-nostats",
        "-i",
        source_path,
        "-af",
        "silencedetect=noise=-35dB:d=0.25",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    output = f"{process.stdout}\n{process.stderr}"
    starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", output)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", output)]
    points = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) and ends[index] >= start else start
        points.append(round((start + end) / 2.0, 6))
    return points


def _motion_score_at(capture: cv2.VideoCapture, timestamp: float) -> float | None:
    frames = []
    for offset in (-0.06, 0.06):
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp + offset) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA))
    return float(cv2.absdiff(frames[0], frames[1]).mean() / 255.0)


def _select_low_motion_split(
    source_path: str,
    *,
    desired: float,
    minimum: float,
    maximum: float,
    silence_points: list[float],
) -> tuple[float, str]:
    search_start = max(minimum, desired - 2.0)
    search_end = min(maximum, desired + 2.0)
    if search_end < search_start + 1e-6:
        return min(max(desired, minimum), maximum), "fixed_limit_fallback"

    count = max(2, int(math.floor((search_end - search_start) / 0.25)) + 1)
    candidates = np.linspace(search_start, search_end, num=count).tolist()
    capture = cv2.VideoCapture(source_path)
    scored: list[tuple[float, float, bool]] = []
    try:
        for timestamp in candidates:
            motion = _motion_score_at(capture, timestamp)
            if motion is None:
                continue
            near_silence = any(abs(timestamp - point) <= 0.45 for point in silence_points)
            distance = abs(timestamp - desired) / max(0.25, search_end - search_start)
            score = motion * 0.65 + distance * 0.35 - (0.20 if near_silence else 0.0)
            scored.append((score, timestamp, near_silence))
    finally:
        capture.release()
    if not scored:
        return min(max(desired, minimum), maximum), "fixed_limit_fallback"
    _score, timestamp, near_silence = min(scored, key=lambda item: (item[0], abs(item[1] - desired)))
    return round(float(timestamp), 6), "low_motion_silence" if near_silence else "low_motion"


def _request_segment(
    *,
    index: int,
    logical_shot: int,
    start: float,
    duration: float,
    min_request_duration: float,
    max_request_duration: float,
    split_reason: str,
) -> RequestSegment:
    if duration < min_request_duration - 1e-6:
        missing = min_request_duration - duration
        padding_start = missing / 2.0
        padding_end = missing - padding_start
        request_duration = min_request_duration
        trim_offset = padding_start
    else:
        padding_start = 0.0
        padding_end = 0.0
        request_duration = min(
            max_request_duration,
            max(min_request_duration, float(math.ceil(duration - 1e-6))),
        )
        trim_offset = 0.0
    return RequestSegment(
        index=index,
        logical_shot=logical_shot,
        start=start,
        source_start=start,
        source_duration=duration,
        request_duration=request_duration,
        trim_offset=trim_offset,
        output_duration=duration,
        padding_start=padding_start,
        padding_end=padding_end,
        split_reason=split_reason,
    )


def adapt_shot_plan_to_requests(
    plan: LongVideoShotPlan,
    *,
    engine: str,
    source_path: str,
) -> tuple[list[RequestSegment], dict[str, Any]]:
    adapter = get_video_engine_adapter(engine)
    max_duration = adapter.max_request_duration
    min_duration = adapter.min_request_duration
    silence_points = _detect_silence_points(source_path) if plan.use_audio_silence else []
    requests: list[RequestSegment] = []

    for shot in plan.shots:
        cursor = shot.start
        remaining = shot.duration
        pieces: list[tuple[float, float, str]] = []
        while remaining > max_duration + 1e-6:
            desired = cursor + max_duration
            latest = shot.end - min_duration
            earliest = cursor + min_duration
            if plan.effective_mode == "shot_aware":
                split_at, reason = _select_low_motion_split(
                    source_path,
                    desired=desired,
                    minimum=earliest,
                    maximum=latest,
                    silence_points=silence_points,
                )
            else:
                split_at = min(desired, latest)
                reason = "engine_limit_fixed"
            piece_duration = split_at - cursor
            if piece_duration < min_duration - 1e-6 or piece_duration > max_duration + 1e-6:
                split_at = min(desired, latest)
                piece_duration = split_at - cursor
                reason = "fixed_limit_fallback"
            pieces.append((cursor, piece_duration, reason))
            cursor = split_at
            remaining = shot.end - cursor
        if remaining > 1e-6:
            reason = "short_shot_padding" if remaining < min_duration else "logical_shot"
            pieces.append((cursor, remaining, reason))

        for start, duration, reason in pieces:
            requests.append(
                _request_segment(
                    index=len(requests) + 1,
                    logical_shot=shot.index,
                    start=round(start, 6),
                    duration=round(duration, 6),
                    min_request_duration=min_duration,
                    max_request_duration=max_duration,
                    split_reason=reason,
                )
            )

    output_duration = sum(item.output_duration for item in requests)
    tolerance = max(1.0 / max(plan.fps, 1.0), 0.05)
    if abs(output_duration - plan.total_duration) > tolerance:
        raise ValueError(
            f"请求片段总时长与原视频不一致：{output_duration:.6f}/{plan.total_duration:.6f} 秒"
        )
    if any(item.request_duration < min_duration - 1e-6 or item.request_duration > max_duration + 1e-6 for item in requests):
        raise ValueError(
            f"存在不符合 {engine} 请求时长范围 {min_duration:g}-{max_duration:g} 秒的片段。"
        )
    details = {
        "engine_max_request_duration": max_duration,
        "engine_min_request_duration": min_duration,
        "silence_points": silence_points,
        "requests": [item.to_dict() for item in requests],
    }
    return requests, details


def _v3_member_request(shot: LogicalShot, *, min_duration: float, max_duration: float) -> RequestSegment:
    """Create a logical-shot member task without padding or rounding its output."""
    request = _request_segment(
        index=shot.index,
        logical_shot=shot.index,
        start=round(shot.start, 6),
        duration=round(shot.duration, 6),
        min_request_duration=min_duration,
        max_request_duration=max_duration,
        split_reason="logical_member",
    )
    return RequestSegment(
        **{
            **request.__dict__,
            "request_duration": round(shot.duration, 6),
            "trim_offset": 0.0,
            "padding_start": 0.0,
            "padding_end": 0.0,
            "logical_shots": (shot.index,),
        }
    )


def _v3_group_members(
    members: list[RequestSegment],
    *,
    min_duration: float,
    max_duration: float,
) -> list[RequestSegment]:
    """Group adjacent short shots deterministically without changing shot boundaries."""
    groups: list[list[RequestSegment]] = []
    cursor = 0
    while cursor < len(members):
        member = members[cursor]
        current = [member]
        duration = member.source_duration
        if duration < min_duration - 1e-6:
            while duration < min_duration - 1e-6 and cursor + 1 < len(members):
                candidate = members[cursor + 1]
                if duration + candidate.source_duration > max_duration + 1e-6:
                    break
                cursor += 1
                current.append(candidate)
                duration += candidate.source_duration
            if duration < min_duration - 1e-6 and groups:
                previous_duration = sum(item.source_duration for item in groups[-1])
                if previous_duration + duration <= max_duration + 1e-6:
                    groups[-1].extend(current)
                    cursor += 1
                    continue
        groups.append(current)
        cursor += 1

    requests: list[RequestSegment] = []
    for index, group in enumerate(groups, start=1):
        source_duration = round(sum(item.source_duration for item in group), 6)
        short_padding = max(0.0, min_duration - source_duration)
        request_duration = min(
            max_duration,
            max(min_duration, float(math.ceil(source_duration - 1e-6))),
        )
        logical_shots = tuple(dict.fromkeys(item.logical_shot for item in group))
        requests.append(
            RequestSegment(
                index=index,
                logical_shot=logical_shots[0],
                logical_shots=logical_shots,
                start=group[0].start,
                source_start=group[0].source_start,
                source_duration=source_duration,
                request_duration=request_duration,
                trim_offset=0.0,
                output_duration=request_duration,
                padding_start=0.0,
                padding_end=short_padding,
                split_reason="merged_short_shots" if len(group) > 1 else (
                    "short_shot_padding" if short_padding > 1e-6 else "logical_shot"
                ),
            )
        )
    return requests


def _request_group_logical_shot_ids(group: RequestSegment | dict[str, Any]) -> set[int]:
    values = group.logical_shots if isinstance(group, RequestSegment) else group.get("logical_segments", [])
    if not values:
        values = (group.logical_shot,) if isinstance(group, RequestSegment) else (group.get("logical_segment"),)
    return {int(value) for value in values if value is not None}


def _request_group_continues_same_logical_shot(
    previous: RequestSegment | dict[str, Any] | None,
    current: RequestSegment | dict[str, Any],
) -> bool:
    if previous is None:
        return False
    return bool(_request_group_logical_shot_ids(previous) & _request_group_logical_shot_ids(current))


def build_v3_logical_members_and_request_groups(
    plan: LongVideoShotPlan,
    *,
    engine: str,
    source_path: str,
) -> tuple[list[RequestSegment], list[RequestSegment], dict[str, Any]]:
    adapter = get_video_engine_adapter(engine)
    members = [
        _v3_member_request(
            shot,
            min_duration=adapter.min_request_duration,
            max_duration=adapter.max_request_duration,
        )
        for shot in plan.shots
    ]
    legacy_requests, legacy_details = adapt_shot_plan_to_requests(
        plan,
        engine=engine,
        source_path=source_path,
    )
    request_pieces = [
        RequestSegment(
            **{
                **item.__dict__,
                "request_duration": item.source_duration,
                "trim_offset": 0.0,
                "output_duration": item.source_duration,
                "padding_start": 0.0,
                "padding_end": 0.0,
                "logical_shots": (item.logical_shot,),
            }
        )
        for item in legacy_requests
    ]
    groups = _v3_group_members(
        request_pieces,
        min_duration=adapter.min_request_duration,
        max_duration=adapter.max_request_duration,
    )
    tolerance = max(1.0 / max(plan.fps, 1.0), 0.05)
    source_total = sum(item.source_duration for item in groups)
    if abs(source_total - plan.total_duration) > tolerance:
        raise ValueError(f"v3 请求组没有完整覆盖输入视频：{source_total:.6f}/{plan.total_duration:.6f} 秒")
    details = {
        "processing_contract_version": V3_PROCESSING_CONTRACT_VERSION,
        "grouping_version": V3_GROUPING_VERSION,
        "engine_min_request_duration": adapter.min_request_duration,
        "engine_max_request_duration": adapter.max_request_duration,
        "logical_members": [item.to_dict() for item in members],
        "request_groups": [item.to_dict() for item in groups],
        "legacy_request_count": len(legacy_requests),
        "legacy_request_seconds": round(sum(item.request_duration for item in legacy_requests), 6),
        "request_count": len(groups),
        "request_seconds": round(sum(item.request_duration for item in groups), 6),
        "legacy_adaptation": legacy_details,
    }
    return members, groups, details


def _safe_slug(value: str, default: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip()).strip("._")
    return text[:80] or default


_JSON_WRITE_LOCK = threading.RLock()
_JSON_WRITE_REPLACE_ATTEMPTS = 5
_JSON_WRITE_REPLACE_DELAY_SECONDS = 0.05


def _atomic_write_json(path: Path, value: Any) -> None:
    with _JSON_WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for attempt in range(1, _JSON_WRITE_REPLACE_ATTEMPTS + 1):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError:
                    if attempt >= _JSON_WRITE_REPLACE_ATTEMPTS:
                        raise
                    time.sleep(_JSON_WRITE_REPLACE_DELAY_SECONDS * attempt)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _load_json_or_path(value: str, *, field_name: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} 不能为空。")
    if not text.startswith(("{", "[")):
        candidates = [Path(text)]
        output_root = Path(folder_paths.get_output_directory())
        candidates.append(output_root / text)
        for candidate in candidates:
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False
            if is_file:
                text = candidate.read_text(encoding="utf-8")
                break
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} 必须是合法 JSON 或 manifest.json 路径：{exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{field_name} 的根节点必须是 JSON 对象。")
    return result


def _first_frame(image: Any) -> Any:
    if image is None:
        return None
    if getattr(image, "ndim", 0) == 4:
        return image[0]
    return image


def _save_image_tensor(image: Any, path: Path) -> None:
    frame = _first_frame(image)
    if frame is None:
        raise ValueError("不能保存空的图片资产。")
    array = np.asarray(frame.detach().cpu() if hasattr(frame, "detach") else frame)
    if array.ndim != 3:
        raise ValueError(f"图片资产形状无效：{array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = np.clip(array[..., :3] * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def _load_image_file(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _relative_output_path(path: Path) -> str:
    return path.relative_to(Path(folder_paths.get_output_directory())).as_posix()


def _output_image_preview(path: Path | None) -> dict[str, str] | None:
    if path is None or not path.is_file():
        return None
    output_root = Path(folder_paths.get_output_directory()).resolve()
    try:
        relative = path.resolve().relative_to(output_root)
    except ValueError:
        return None
    return {
        "filename": relative.name,
        "subfolder": "" if str(relative.parent) == "." else relative.parent.as_posix(),
        "type": "output",
    }


def _auto_asset_preview_payload(
    task: dict[str, Any],
    *,
    source_frames: dict[str, str],
    people: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    integrated_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def image(value: Any) -> dict[str, str] | None:
        return _output_image_preview(_asset_path_if_file(value))

    converted: list[dict[str, Any]] = []
    for item in people:
        preview = image(item.get("path"))
        if preview is None:
            continue
        publication = item.get("publication") if isinstance(item.get("publication"), dict) else {}
        tos = publication.get("tos") if isinstance(publication.get("tos"), dict) else {}
        library = publication.get("asset_library") if isinstance(publication.get("asset_library"), dict) else {}
        converted.append(
            {
                "kind": "person",
                "label": f"人物 {item.get('slot') or ''}".strip(),
                "preview": preview,
                "tos_status": str(tos.get("status") or "pending"),
                "asset_library_status": str(library.get("status") or "pending"),
                "asset_id": str(library.get("asset_id") or ""),
                "warning": str(library.get("error") or ""),
            }
        )
    for item in scenes:
        preview = image(item.get("path"))
        if preview is None:
            continue
        converted.append(
            {
                "kind": "scene",
                "label": "场景开始" if item.get("role") == "scene_start" else "场景结束",
                "preview": preview,
                "tos_status": "direct_seedance",
                "asset_library_status": "not_registered",
                "asset_id": "",
                "warning": "",
            }
        )
    for item in integrated_frames or []:
        preview = image(item.get("path"))
        if preview is None:
            continue
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        converted.append(
            {
                "kind": "integrated_frame",
                "label": "整帧开始" if item.get("role") == "frame_start" else "整帧结束",
                "preview": preview,
                "tos_status": "direct_seedance",
                "asset_library_status": str(quality.get("verdict") or "pending"),
                "asset_id": "",
                "warning": "; ".join(_as_string_list(quality.get("reasons"))),
                "quality": dict(quality),
            }
        )
    return {
        "task_index": int(task.get("index", 0) or 0),
        "source_start": image(source_frames.get("source_start")),
        "source_end": image(source_frames.get("source_end")),
        "converted": converted,
        "identity_review": [
            {
                "slot": str(item.get("slot") or ""),
                "state": str(item.get("identity_state") or ""),
                "global_person_id": str(item.get("global_person_id") or item.get("person_id") or ""),
                "reason": str(item.get("identity_reason") or ""),
            }
            for item in people
            if str(item.get("identity_state") or "")
        ],
    }


def _asset_path(value: Any) -> Path | None:
    if not value:
        return None
    if isinstance(value, dict):
        value = value.get("styled") or value.get("styled_filename") or value.get("path")
    path = Path(str(value))
    if path.is_file():
        return path
    output_path = Path(folder_paths.get_output_directory()) / path
    return output_path if output_path.is_file() else None


def _string_mapping(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {"raw": str(value or "")}
    return parsed if isinstance(parsed, dict) else {"raw": str(value or "")}


def _manifest_entries(manifest: dict[str, Any], key: str, ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    source = manifest.get(key, [])
    if isinstance(source, dict):
        source = [dict(item, id=entry_id) if isinstance(item, dict) else {"id": entry_id, "styled": item}
                  for entry_id, item in source.items()]
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(source if isinstance(source, list) else []):
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("id") or (ids[index] if index < len(ids) else "")).strip()
        if entry_id in ids:
            result[entry_id] = item
    return result


def _resolve_assets(
    manifest: dict[str, Any],
    connected_people: dict[str, Any],
    connected_backgrounds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    people_entries = _manifest_entries(manifest, "people", PERSON_IDS)
    background_entries = _manifest_entries(manifest, "backgrounds", BACKGROUND_IDS)
    people: dict[str, Any] = {}
    backgrounds: dict[str, Any] = {}
    for entry_id in PERSON_IDS:
        image = connected_people.get(entry_id)
        if image is None:
            asset = _asset_path(people_entries.get(entry_id, {}).get("styled"))
            image = _load_image_file(asset) if asset else None
        if image is not None:
            people[entry_id] = image
    for entry_id in BACKGROUND_IDS:
        image = connected_backgrounds.get(entry_id)
        if image is None:
            asset = _asset_path(background_entries.get(entry_id, {}).get("styled"))
            image = _load_image_file(asset) if asset else None
        if image is not None:
            backgrounds[entry_id] = image
    if not people:
        raise ValueError("资产清单中没有可用的人物图片。请连接人物图片，或确认 manifest 中的 styled 文件仍存在。")
    if not backgrounds:
        raise ValueError("资产清单中没有可用的背景图片。请连接背景图片，或确认 manifest 中的 styled 文件仍存在。")
    return people, backgrounds


def _get_config(name: str):
    try:
        return get_config(name)
    except ConfigError as exc:
        raise ValueError(f"未找到 {name} 配置，请先在 Company Remote 配置面板中配置。") from exc


def _video_source_path(video: Any, directory: Path) -> str:
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    trim_start = 0.0
    trim_duration = 0.0
    if hasattr(video, "get_active_trim_window"):
        trim_start, trim_duration = video.get_active_trim_window()
    has_trim_window = abs(float(trim_start)) > 1e-9 or float(trim_duration) > 0
    if not has_trim_window and isinstance(source, str) and Path(source).is_file():
        return source

    path = directory / "source_video.mp4"
    if isinstance(source, str) and Path(source).is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            _ffmpeg_exe(),
            "-y",
        ]
        if float(trim_start) > 1e-9:
            command.extend(["-ss", f"{float(trim_start):.6f}"])
        command.extend(["-i", source])
        if float(trim_duration) > 1e-9:
            command.extend(["-t", f"{float(trim_duration):.6f}"])
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(SHOT_DETECTION_CPU_THREADS),
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(path),
            ]
        )
        with _background_processing_priority():
            _run_ffmpeg(command, error_prefix="长视频源片段流式截取失败")
        return str(path)

    video.save_to(str(path))
    return str(path)


def _sample_frames(video: Any) -> torch.Tensor:
    source = video.get_stream_source()
    if hasattr(source, "seek"):
        source.seek(0)
    start, duration = video.get_active_trim_window()
    if duration <= 0:
        duration = float(video.get_duration())
    end = start + duration
    margin = min(0.05, duration / 10.0)
    targets = [start + margin, start + duration / 2.0, max(start, end - margin)]
    selected: list[torch.Tensor] = []
    last_frame: torch.Tensor | None = None
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("视频分段没有视频流。")
        stream = container.streams.video[0]
        if start > 0 and stream.time_base:
            container.seek(max(0, int(start / stream.time_base)), stream=stream, backward=True)
        target_index = 0
        for frame in container.decode(stream):
            timestamp = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if timestamp + 1e-6 < start:
                continue
            if timestamp > end + margin:
                break
            array = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            current = torch.from_numpy(array)
            last_frame = current
            while target_index < len(targets) and timestamp + 1e-6 >= targets[target_index]:
                selected.append(current)
                target_index += 1
            if target_index == len(targets):
                break
    if last_frame is None:
        raise ValueError("视频分段没有可读取的画面。")
    while len(selected) < len(targets):
        selected.append(last_frame)
    return torch.stack(selected, dim=0)


def _auto_asset_root(job: LongVideoJob, task: dict[str, Any]) -> Path:
    return job.job_dir / "shot_assets" / f"shot_{int(task['index']):04d}"


def _auto_asset_member_tasks(job: LongVideoJob) -> list[dict[str, Any]]:
    members = job.manifest.get("logical_member_tasks")
    return members if isinstance(members, list) else job.manifest.get("tasks", [])


def _auto_asset_cache_path(job: LongVideoJob) -> Path:
    return job.job_dir / "asset_cache" / "index.json"


def _auto_asset_library_path() -> Path:
    return Path(folder_paths.get_user_directory()) / "default" / "company_remote" / "long_video_asset_library.json"


def _persistent_auto_asset_library_enabled() -> bool:
    """Avoid making a user library from a test-only patched output directory."""
    configured = getattr(folder_paths, "output_directory", "")
    try:
        return Path(folder_paths.get_output_directory()).resolve() == Path(str(configured)).resolve()
    except (OSError, TypeError, ValueError):
        return False


def _empty_auto_asset_cache() -> dict[str, Any]:
    return {
        "version": AUTO_ASSET_CACHE_VERSION,
        "people": {"by_id": {}, "candidate_index": {}},
        "scenes": {
            "by_id": {},
            "candidate_index": {},
            "style_master_id": "",
            "style_master_path": "",
            "style_master_quality_score": 0.0,
        },
        "legacy": {"people": {}, "scenes": {}},
    }


def _normalized_id_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    return []


def _normalize_auto_asset_cache(value: Any) -> dict[str, Any]:
    cache = _empty_auto_asset_cache()
    if not isinstance(value, dict):
        return cache

    try:
        version = int(value.get("version", 1) or 1)
    except (TypeError, ValueError):
        version = 1
    if version >= 2:
        people = value.get("people") if isinstance(value.get("people"), dict) else {}
        scenes = value.get("scenes") if isinstance(value.get("scenes"), dict) else {}
        cache["people"]["by_id"].update(people.get("by_id") if isinstance(people.get("by_id"), dict) else {})
        cache["scenes"]["by_id"].update(scenes.get("by_id") if isinstance(scenes.get("by_id"), dict) else {})
        cache["people"]["candidate_index"].update(
            {
                str(key): _normalized_id_list(ids)
                for key, ids in (people.get("candidate_index") if isinstance(people.get("candidate_index"), dict) else {}).items()
            }
        )
        cache["scenes"]["candidate_index"].update(
            {
                str(key): _normalized_id_list(ids)
                for key, ids in (scenes.get("candidate_index") if isinstance(scenes.get("candidate_index"), dict) else {}).items()
            }
        )
        cache["scenes"]["style_master_id"] = str(scenes.get("style_master_id") or "")
        cache["scenes"]["style_master_path"] = str(scenes.get("style_master_path") or "")
        cache["scenes"]["style_master_quality_score"] = _score01(scenes.get("style_master_quality_score"))
        legacy = value.get("legacy") if isinstance(value.get("legacy"), dict) else {}
        cache["legacy"]["people"].update(legacy.get("people") if isinstance(legacy.get("people"), dict) else {})
        cache["legacy"]["scenes"].update(legacy.get("scenes") if isinstance(legacy.get("scenes"), dict) else {})
        return cache

    # v1 caches only stored converted paths. Keep them as legacy hints, but do
    # not use them for reuse because source observations are missing.
    people = value.get("people") if isinstance(value.get("people"), dict) else {}
    scenes = value.get("scenes") if isinstance(value.get("scenes"), dict) else {}
    cache["legacy"]["people"].update(people)
    cache["legacy"]["scenes"].update(scenes)
    return cache


def _empty_identity_mapping() -> dict[str, Any]:
    return {"version": 1, "expected_distinct_people": 0, "global_people": {}, "shot_people": {}}


def _load_identity_mapping(value: Any) -> dict[str, Any]:
    """Load a non-secret human identity mapping from a JSON object, string, or file path."""
    if isinstance(value, dict):
        payload = deepcopy(value)
    else:
        text = str(value or "").strip()
        if not text:
            return _empty_identity_mapping()
        path = Path(text)
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"人物映射文件无法读取：{path}") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"人物映射不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("人物映射根节点必须是对象。")
    global_people = payload.get("global_people") if isinstance(payload.get("global_people"), dict) else {}
    shot_people = payload.get("shot_people") if isinstance(payload.get("shot_people"), dict) else {}
    normalized_global: dict[str, dict[str, Any]] = {}
    for key, item in global_people.items():
        if not isinstance(item, dict):
            continue
        identifier = str(key).strip()
        if not identifier:
            continue
        normalized_global[identifier] = {
            "name": str(item.get("name") or identifier),
            "asset_id": str(item.get("asset_id") or "").strip(),
            "path": str(item.get("path") or "").strip(),
            "status": str(item.get("status") or "confirmed").strip().lower(),
        }
    normalized_shot: dict[str, str] = {}
    for key, assignment_value in shot_people.items():
        assignment = str(assignment_value or "").strip()
        if str(key).strip() and assignment:
            normalized_shot[str(key).strip()] = assignment
    return {
        "version": int(payload.get("version", 1) or 1),
        "expected_distinct_people": int(payload.get("expected_distinct_people", 0) or 0),
        "global_people": normalized_global,
        "shot_people": normalized_shot,
    }


def _apply_identity_mapping_to_analysis(task: dict[str, Any], mapping: dict[str, Any]) -> None:
    analysis = task.get("auto_asset_analysis")
    if not isinstance(analysis, dict):
        return
    global_people = mapping.get("global_people") if isinstance(mapping.get("global_people"), dict) else {}
    shot_people = mapping.get("shot_people") if isinstance(mapping.get("shot_people"), dict) else {}
    task_index = int(task.get("index", 0) or 0)
    for person in analysis.get("people", []) if isinstance(analysis.get("people"), list) else []:
        if not isinstance(person, dict):
            continue
        slot = str(person.get("slot") or "")
        assignment = str(
            shot_people.get(f"{task_index}:{slot}")
            or shot_people.get(f"shot_{task_index:04d}:{slot}")
            or ""
        ).strip()
        if not assignment:
            continue
        if assignment.lower() in {"ignore", "partial", "none"}:
            person.update(
                {
                    "identity_state": "partial",
                    "global_person_id": "",
                    "identity_reason": "人工映射要求忽略该局部人物观察。",
                }
            )
            continue
        record = global_people.get(assignment)
        if not isinstance(record, dict):
            raise ValueError(f"人物映射引用了未定义的全局人物：{assignment}")
        if str(record.get("status") or "confirmed").lower() not in {"confirmed", "active", "linked"}:
            raise ValueError(f"全局人物 {assignment} 尚未确认，不能用于付费生成。")
        person.update(
            {
                "identity_state": "linked",
                "global_person_id": assignment,
                "mapped_asset_id": str(record.get("asset_id") or "").strip(),
                "mapped_asset_path": str(record.get("path") or "").strip(),
                "identity_reason": "使用人工确认的人物映射。",
            }
        )


IDENTITY_MAPPING_ROOT_NAME = "company_remote/identity_mappings"


def _identity_mapping_store_path(series_id: str) -> Path:
    root = (Path(folder_paths.get_output_directory()).resolve() / IDENTITY_MAPPING_ROOT_NAME).resolve()
    path = (root / f"{_manual_batch_series_id(series_id)}.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError("人物映射路径越出了 ComfyUI output 目录。")
    return path


def load_identity_mapping_record(series_id: str) -> dict[str, Any]:
    path = _identity_mapping_store_path(series_id)
    if not path.is_file():
        return {
            "series_id": _manual_batch_series_id(series_id),
            "exists": False,
            "path": str(path),
            "mapping": _empty_identity_mapping(),
        }
    return {
        "series_id": _manual_batch_series_id(series_id),
        "exists": True,
        "path": str(path),
        "mapping": _load_identity_mapping(str(path)),
    }


def save_identity_mapping_record(series_id: str, payload: Any) -> dict[str, Any]:
    mapping = _load_identity_mapping(payload)
    path = _identity_mapping_store_path(series_id)
    _atomic_write_json(path, mapping)
    return {
        "series_id": _manual_batch_series_id(series_id),
        "exists": True,
        "path": str(path),
        "mapping": mapping,
    }


def _load_auto_asset_cache(job: LongVideoJob) -> dict[str, Any]:
    path = _auto_asset_cache_path(job)
    if not path.is_file():
        return _empty_auto_asset_cache()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_auto_asset_cache()
    return _normalize_auto_asset_cache(value)


def _save_auto_asset_cache(job: LongVideoJob, value: dict[str, Any]) -> None:
    _atomic_write_json(_auto_asset_cache_path(job), _normalize_auto_asset_cache(value))


def _cache_observation_entries(value: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and str(item.get("path") or "").strip():
            entries.append({str(key): str(item[key]) for key in ("kind", "index", "path", "description") if item.get(key) is not None})
    return entries


def _merge_auto_asset_cache_data(
    destination: dict[str, Any],
    source: dict[str, Any],
    *,
    persistent_library: bool = False,
) -> None:
    """Merge verified cache records without copying the underlying image files."""
    normalized_destination = _normalize_auto_asset_cache(destination)
    normalized_source = _normalize_auto_asset_cache(source)
    destination.clear()
    destination.update(normalized_destination)

    target_people = destination["people"].setdefault("by_id", {})
    for person_id, value in normalized_source["people"].get("by_id", {}).items():
        if not isinstance(value, dict):
            continue
        incoming = deepcopy(value)
        incoming_id = str(person_id)
        if incoming_id in target_people and target_people[incoming_id].get("converted_path") != incoming.get("converted_path"):
            incoming_id = f"library_{hashlib.sha256(str(incoming.get('converted_path') or person_id).encode('utf-8')).hexdigest()[:16]}"
            incoming["person_id"] = incoming_id
        if persistent_library:
            incoming["persistent_library"] = True
        record = target_people.setdefault(incoming_id, incoming)
        if record is not incoming:
            if not record.get("converted_path"):
                record["converted_path"] = str(incoming.get("converted_path") or "")
            if not record.get("appearance"):
                record["appearance"] = str(incoming.get("appearance") or "")
            incoming_publication = incoming.get("publication")
            recorded_publication = record.get("publication")
            incoming_library = (
                incoming_publication.get("asset_library") if isinstance(incoming_publication, dict) else {}
            )
            recorded_library = (
                recorded_publication.get("asset_library") if isinstance(recorded_publication, dict) else {}
            )
            incoming_is_active = str(incoming_library.get("status") or "").lower() == "active"
            recorded_is_active = str(recorded_library.get("status") or "").lower() == "active"
            if isinstance(incoming_publication, dict) and (not isinstance(recorded_publication, dict) or incoming_is_active and not recorded_is_active):
                record["publication"] = deepcopy(incoming_publication)
            if persistent_library:
                record["persistent_library"] = True
            keys = record.setdefault("identity_keys", [])
            for identity_key in _normalized_id_list(incoming.get("identity_keys")):
                if identity_key not in keys:
                    keys.append(identity_key)
            _append_source_observations(record, _cache_observation_entries(incoming.get("source_observations")))
        for identity_key in _normalized_id_list(record.get("identity_keys")):
            _append_candidate_index(destination, "person", identity_key, incoming_id)

    target_places = destination["scenes"].setdefault("by_id", {})
    for place_id, value in normalized_source["scenes"].get("by_id", {}).items():
        if not isinstance(value, dict):
            continue
        incoming = deepcopy(value)
        incoming_id = str(place_id)
        if incoming_id in target_places and target_places[incoming_id].get("style_master_path") != incoming.get("style_master_path"):
            incoming_id = f"library_{hashlib.sha256(str(incoming.get('style_master_path') or place_id).encode('utf-8')).hexdigest()[:16]}"
            incoming["place_id"] = incoming_id
        if persistent_library:
            incoming["persistent_library"] = True
        place = target_places.setdefault(incoming_id, incoming)
        if place is not incoming:
            if not place.get("description"):
                place["description"] = str(incoming.get("description") or "")
            if not place.get("style_master_path"):
                place["style_master_path"] = str(incoming.get("style_master_path") or "")
            if persistent_library:
                place["persistent_library"] = True
            scene_keys = place.setdefault("scene_keys", [])
            for scene_key in _normalized_id_list(incoming.get("scene_keys")):
                if scene_key not in scene_keys:
                    scene_keys.append(scene_key)
            target_versions = place.setdefault("versions", {})
            for version_id, version in (incoming.get("versions") if isinstance(incoming.get("versions"), dict) else {}).items():
                if not isinstance(version, dict):
                    continue
                if str(version_id) not in target_versions:
                    target_versions[str(version_id)] = deepcopy(version)
                else:
                    _append_source_observations(
                        target_versions[str(version_id)],
                        _cache_observation_entries(version.get("source_observations")),
                    )
        for scene_key in _normalized_id_list(place.get("scene_keys")):
            _append_candidate_index(destination, "scene", scene_key, incoming_id)


def _auto_asset_cache_contract(cache_path: Path) -> tuple[str, str]:
    manifest_path = cache_path.parent.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "", ""
    options = manifest.get("auto_asset_options") if isinstance(manifest, dict) else None
    options = options if isinstance(options, dict) else {}
    style = _normalize_auto_asset_style(options.get("visual_style")) if options.get("visual_style") else ""
    return style, str(options.get("prompt_version") or manifest.get("prompt_version") or "")


def _auto_asset_library_seed_paths(visual_style: str) -> list[Path]:
    root = Path(folder_paths.get_output_directory()) / "company_remote"
    if not root.is_dir():
        return []
    style = _normalize_auto_asset_style(visual_style)
    return sorted(
        (
            path
            for path in root.glob("**/asset_cache/index.json")
            if _auto_asset_cache_contract(path) == (style, AUTO_ASSET_PROMPT_VERSION)
        ),
        key=lambda path: path.stat().st_mtime,
    )


def _load_auto_asset_library(visual_style: str) -> dict[str, Any]:
    if not _persistent_auto_asset_library_enabled():
        return _empty_auto_asset_cache()
    style = _normalize_auto_asset_style(visual_style)
    path = _auto_asset_library_path()
    expected_output_root = str(Path(folder_paths.get_output_directory()).resolve())
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            styles = payload.get("styles") if isinstance(payload, dict) and payload.get("output_root") == expected_output_root else None
            style_record = styles.get(style) if isinstance(styles, dict) else None
            cache = style_record.get("cache") if isinstance(style_record, dict) else None
            if isinstance(cache, dict):
                return _normalize_auto_asset_cache(cache)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logging.warning("Unable to load the persistent auto-asset library.", exc_info=True)

    cache = _empty_auto_asset_cache()
    source_count = 0
    for cache_path in _auto_asset_library_seed_paths(style):
        try:
            source = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(source, dict):
            continue
        _merge_auto_asset_cache_data(cache, source, persistent_library=True)
        source_count += 1
    if source_count:
        _save_auto_asset_library(cache, visual_style=style, imported_cache_count=source_count)
    return cache


def _save_auto_asset_library(
    cache: dict[str, Any],
    *,
    visual_style: str,
    imported_cache_count: int | None = None,
) -> None:
    if not _persistent_auto_asset_library_enabled():
        return
    normalized = _normalize_auto_asset_cache(cache)
    path = _auto_asset_library_path()
    styles: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and isinstance(previous.get("styles"), dict):
                styles = dict(previous["styles"])
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logging.warning("Unable to preserve other persistent auto-asset library styles.", exc_info=True)
    style_record: dict[str, Any] = {"cache": normalized}
    if imported_cache_count is not None:
        style_record["imported_cache_count"] = int(imported_cache_count)
    styles[_normalize_auto_asset_style(visual_style)] = style_record
    payload: dict[str, Any] = {
        "version": AUTO_ASSET_LIBRARY_VERSION,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_root": str(Path(folder_paths.get_output_directory()).resolve()),
        "styles": styles,
    }
    _atomic_write_json(path, payload)


ASSET_LIBRARY_ALL_STYLES = "all"
ASSET_LIBRARY_GRID_CELL = 320
ASSET_LIBRARY_GRID_LABEL_HEIGHT = 92


def _read_asset_library_styles() -> dict[str, dict[str, Any]]:
    """Read every style cache from the persistent library file without seeding.

    Read-only: unlike ``_load_auto_asset_library`` this never rebuilds the library
    from job caches, so a viewer node can list what is already registered without
    side effects.
    """
    path = _auto_asset_library_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        logging.warning("Unable to read the persistent auto-asset library for viewing.", exc_info=True)
        return {}
    expected_output_root = str(Path(folder_paths.get_output_directory()).resolve())
    if not isinstance(payload, dict) or payload.get("output_root") != expected_output_root:
        return {}
    styles = payload.get("styles") if isinstance(payload.get("styles"), dict) else {}
    result: dict[str, dict[str, Any]] = {}
    for style, record in styles.items():
        cache = record.get("cache") if isinstance(record, dict) else None
        if isinstance(cache, dict):
            result[str(style)] = _normalize_auto_asset_cache(cache)
    return result


def _person_library_entry(style: str, person_id: str, record: dict[str, Any]) -> dict[str, Any]:
    publication = record.get("publication") if isinstance(record.get("publication"), dict) else {}
    library = publication.get("asset_library") if isinstance(publication.get("asset_library"), dict) else {}
    active_asset_id = _active_person_asset_id(record)
    path = _asset_path_if_file(record.get("converted_path"))
    return {
        "type": "person",
        "style": style,
        "id": str(person_id),
        "appearance": str(record.get("appearance") or ""),
        "asset_id": active_asset_id or str(library.get("asset_id") or ""),
        "registered": bool(active_asset_id),
        "library_status": str(library.get("status") or ("none" if not library else "")),
        "path": str(path) if path is not None else "",
        "has_image": path is not None,
        "identity_keys": _normalized_id_list(record.get("identity_keys")),
        "persistent_library": bool(record.get("persistent_library")),
    }


def _scene_library_entry(style: str, place_id: str, record: dict[str, Any]) -> dict[str, Any]:
    path = _asset_path_if_file(record.get("style_master_path")) or _asset_path_if_file(record.get("path"))
    return {
        "type": "scene",
        "style": style,
        "id": str(place_id),
        "description": str(record.get("description") or ""),
        "asset_id": "",
        "registered": False,
        "library_status": "local",
        "path": str(path) if path is not None else "",
        "has_image": path is not None,
        "scene_keys": _normalized_id_list(record.get("scene_keys")),
        "persistent_library": bool(record.get("persistent_library")),
    }


def collect_registered_asset_inventory(
    visual_style: str = "",
    *,
    job: LongVideoJob | None = None,
) -> dict[str, Any]:
    """Enumerate registered person and scene assets from the persistent library.

    Optionally fold in the current job cache so in-progress assets that have not yet
    reached the shared library are also visible. Read-only; no remote calls.
    """
    requested = str(visual_style or "").strip()
    wanted = None if requested in {"", ASSET_LIBRARY_ALL_STYLES, "全部", "全部样式"} else {_normalize_auto_asset_style(requested)}

    people_entries: list[dict[str, Any]] = []
    scene_entries: list[dict[str, Any]] = []
    seen_people: set[str] = set()
    seen_scenes: set[str] = set()

    def add_cache(style_label: str, cache: dict[str, Any]) -> None:
        people = cache.get("people") if isinstance(cache.get("people"), dict) else {}
        for person_id, record in (people.get("by_id") or {}).items():
            if not isinstance(record, dict):
                continue
            entry = _person_library_entry(style_label, str(person_id), record)
            dedup_key = entry["asset_id"] or entry["path"] or f"{style_label}:{person_id}"
            if dedup_key in seen_people:
                continue
            seen_people.add(dedup_key)
            people_entries.append(entry)
        scenes = cache.get("scenes") if isinstance(cache.get("scenes"), dict) else {}
        for place_id, record in (scenes.get("by_id") or {}).items():
            if not isinstance(record, dict):
                continue
            entry = _scene_library_entry(style_label, str(place_id), record)
            dedup_key = entry["path"] or f"{style_label}:{place_id}"
            if dedup_key in seen_scenes:
                continue
            seen_scenes.add(dedup_key)
            scene_entries.append(entry)

    for style, cache in sorted(_read_asset_library_styles().items()):
        if wanted is None or style in wanted:
            add_cache(style, cache)

    if job is not None:
        try:
            job_cache = _load_auto_asset_cache(job)
        except Exception:  # noqa: BLE001 - viewing must never break on a bad cache
            job_cache = None
        if isinstance(job_cache, dict):
            job_style = _normalize_auto_asset_style((job.manifest.get("auto_asset_options") or {}).get("visual_style"))
            add_cache(f"{job_style}·当前任务", job_cache)

    entries = people_entries + scene_entries
    summary = {
        "styles": sorted({entry["style"] for entry in entries}),
        "total": len(entries),
        "people": len(people_entries),
        "scenes": len(scene_entries),
        "registered_volcano_assets": sum(1 for entry in people_entries if entry["registered"]),
        "library_path": str(_auto_asset_library_path()),
    }
    return {"summary": summary, "people": people_entries, "scenes": scene_entries, "entries": entries}


def _asset_inventory_grid(entries: list[dict[str, Any]], *, columns: int = 3) -> torch.Tensor:
    columns = max(1, min(6, int(columns)))
    display = [entry for entry in entries if entry.get("has_image") and _asset_path_if_file(entry.get("path"))]
    if not display:
        canvas = Image.new("RGB", (760, 200), "#202124")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (24, 84),
            "暂无入库资源：尚未有已保存的人物或场景素材。",
            fill="#ffffff",
            font=ImageFont.load_default(size=26),
        )
        value = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(value)[None,]

    cols = min(columns, len(display))
    rows = int(math.ceil(len(display) / cols))
    cell_w = ASSET_LIBRARY_GRID_CELL
    cell_h = ASSET_LIBRARY_GRID_CELL + ASSET_LIBRARY_GRID_LABEL_HEIGHT
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), "#15171a")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=22)
    detail_font = ImageFont.load_default(size=18)
    image_box = ASSET_LIBRARY_GRID_CELL - 16
    for position, entry in enumerate(display):
        cell_x = (position % cols) * cell_w
        cell_y = (position // cols) * cell_h
        path = _asset_path_if_file(entry.get("path"))
        try:
            with Image.open(path) as source:
                source = source.convert("RGB")
                source.thumbnail((image_box, image_box), Image.Resampling.LANCZOS)
                canvas.paste(
                    source,
                    (
                        cell_x + (cell_w - source.width) // 2,
                        cell_y + 8 + (image_box - source.height) // 2,
                    ),
                )
        except Exception:  # noqa: BLE001 - a broken thumbnail must not abort the whole grid
            draw.rectangle(
                (cell_x + 8, cell_y + 8, cell_x + cell_w - 8, cell_y + ASSET_LIBRARY_GRID_CELL - 8),
                outline="#5f6368",
                width=2,
            )
        label_top = cell_y + ASSET_LIBRARY_GRID_CELL
        draw.rectangle((cell_x, label_top, cell_x + cell_w - 1, cell_y + cell_h - 1), fill="#0f1113")
        if entry.get("registered"):
            badge, color = "已入库", "#34a853"
        elif entry["type"] == "scene":
            badge, color = "场景", "#8ab4f8"
        else:
            badge, color = "本地", "#fbbc04"
        draw.text((cell_x + 10, label_top + 8), f"[{badge}] {str(entry['id'])[:26]}", fill=color, font=title_font)
        detail = entry.get("asset_id") or str(entry.get("appearance") or entry.get("description") or "")
        draw.text(
            (cell_x + 10, label_top + 40),
            f"{entry.get('style', '')} · {detail[:30]}",
            fill="#e8eaed",
            font=detail_font,
        )
    value = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(value)[None,]


def build_asset_library_view(
    visual_style: str = "",
    columns: int = 3,
    *,
    job: LongVideoJob | None = None,
) -> tuple[torch.Tensor, str, str]:
    inventory = collect_registered_asset_inventory(visual_style, job=job)
    grid = _asset_inventory_grid(inventory["entries"], columns=columns)
    report = json.dumps(inventory, ensure_ascii=False, indent=2)
    summary_data = inventory["summary"]
    summary = (
        f"入库资源共 {summary_data['total']} 项：人物 {summary_data['people']}、场景 {summary_data['scenes']}，"
        f"其中已注册火山素材 {summary_data['registered_volcano_assets']} 个。"
        f"样式：{'、'.join(summary_data['styles']) or '无'}"
    )
    return grid, report, summary


def _source_frames_for_task(job: LongVideoJob, task: dict[str, Any]) -> torch.Tensor:
    source_start = float(task.get("source_start", task["start"]))
    source_duration = float(task.get("source_duration", task["duration"]))
    source_segment = job.video.as_trimmed(source_start, source_duration, strict_duration=True)
    if source_segment is None:
        raise ValueError(f"第 {task['index']} 段无法读取原始镜头画面。")
    return _sample_frames(source_segment)


def _save_source_frames(frames: torch.Tensor, root: Path) -> dict[str, str]:
    names = ("source_start", "source_middle", "source_end")
    result: dict[str, str] = {}
    for index, name in enumerate(names):
        path = root / f"{name}.png"
        _save_image_tensor(frames[index], path)
        result[name] = str(path)
    return result


def _saved_source_frame_paths(root: Path, assets: dict[str, Any]) -> dict[str, str]:
    names = ("source_start", "source_middle", "source_end")
    saved = assets.get("source_frames") if isinstance(assets.get("source_frames"), dict) else {}
    resolved: dict[str, str] = {}
    for name in names:
        configured = _asset_path_if_file(saved.get(name))
        path = configured or _asset_path_if_file(root / f"{name}.png")
        if path is None:
            return {}
        resolved[name] = str(path)
    return resolved


def _load_saved_source_frames(root: Path, assets: dict[str, Any]) -> tuple[torch.Tensor | None, dict[str, str]]:
    resolved = _saved_source_frame_paths(root, assets)
    if not resolved:
        return None, {}
    paths = [Path(resolved[name]) for name in ("source_start", "source_middle", "source_end")]
    return _vision_batch([_load_image_file(path) for path in paths]), resolved


def _extract_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型没有返回合法 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("模型 JSON 根节点必须是对象。")
    return parsed


def _normalized_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    left, top = max(0.0, min(1.0, left)), max(0.0, min(1.0, top))
    right, bottom = max(0.0, min(1.0, right)), max(0.0, min(1.0, bottom))
    if right - left < 0.02 or bottom - top < 0.02:
        return None
    return [round(left, 5), round(top, 5), round(right, 5), round(bottom, 5)]


def _normalize_auto_asset_analysis(value: dict[str, Any]) -> dict[str, Any]:
    stability = str(value.get("shot_stability") or "uncertain").strip().lower()
    if stability not in {"stable", "transition", "crowded", "uncertain"}:
        stability = "uncertain"
    people: list[dict[str, Any]] = []
    for index, item in enumerate(value.get("people", []) if isinstance(value.get("people"), list) else []):
        if len(people) >= AUTO_ASSET_MAX_PEOPLE or not isinstance(item, dict):
            break
        first_bbox = _normalized_bbox(item.get("first_bbox"))
        last_bbox = _normalized_bbox(item.get("last_bbox"))
        if first_bbox is None and last_bbox is None:
            continue
        persistence = str(item.get("persistence") or "both").lower()
        if persistence not in {"both", "first_only", "last_only"}:
            persistence = "both" if first_bbox and last_bbox else ("first_only" if first_bbox else "last_only")
        try:
            reuse_confidence = float(item.get("reuse_confidence", 0.0))
        except (TypeError, ValueError):
            reuse_confidence = 0.0
        people.append(
            {
                "slot": f"P{len(people) + 1}",
                "appearance": str(item.get("appearance") or "画面中的主要人物").strip(),
                "first_bbox": first_bbox,
                "last_bbox": last_bbox,
                "persistence": persistence,
                "identity_key": _safe_slug(str(item.get("identity_key") or ""), f"shot_person_{index + 1}"),
                "reuse_confidence": max(0.0, min(1.0, reuse_confidence)),
            }
        )
    background = value.get("background") if isinstance(value.get("background"), dict) else {}
    try:
        same_scene_confidence = float(background.get("same_scene_confidence", 0.0))
    except (TypeError, ValueError):
        same_scene_confidence = 0.0
    return {
        "shot_stability": stability,
        "story_action": str(value.get("story_action") or "按照原视频镜头的剧情、动作、表演和镜头运动重新演绎。").strip(),
        "people": people,
        "background": {
            "first_description": str(background.get("first_description") or "镜头开始时的主要环境").strip(),
            "last_description": str(background.get("last_description") or "镜头结束时的主要环境").strip(),
            "scene_key": _safe_slug(str(background.get("scene_key") or ""), "shot_scene"),
            "same_scene_confidence": max(0.0, min(1.0, same_scene_confidence)),
        },
    }


def _auto_asset_analysis(frames: torch.Tensor, *, model: str) -> dict[str, Any]:
    skill = (
        "你是视频镜头的结构化视觉分析器。只输出合法 JSON，不要 Markdown。"
        "输入顺序固定为镜头开始、镜头中间、镜头结束。识别该镜头最多三位叙事主体及环境变化，"
        "不得臆造画面外人物。人物框使用相对于原图的 [left, top, right, bottom]，范围 0-1。"
        "输出字段 shot_stability、story_action、people、background；people 每项包含 appearance、"
        "first_bbox、last_bbox、persistence、identity_key、reuse_confidence；background 包含"
        "first_description、last_description、scene_key、same_scene_confidence。"
    )
    request = (
        "根据首、中、尾帧判断镜头是否稳定。若首尾明显跨场景，shot_stability 使用 transition；"
        "若主要人物超过三人，使用 crowded。identity_key 必须优先描述可见的脸部轮廓、年龄感、体型、性别表达、"
        "稳定发型等可跨镜头识别的特征；服装和道具只能作为辅助，不能使用真实姓名。"
        "reuse_confidence 只表示本镜头的可见信息完整度，不能阻止后续素材库核验。"
    )
    raw = generate_openai_image_prompt_text(
        _get_config("gpttext"),
        skill=skill,
        modification_target=request,
        image=_vision_batch([frames[0], frames[1], frames[2]]),
        model=model,
        temperature=0.1,
        max_tokens=1800,
    )
    try:
        return _normalize_auto_asset_analysis(_extract_json_object(raw))
    except ValueError:
        repaired = generate_openai_image_prompt_text(
            _get_config("gpttext"),
            skill="把下方分析改写为符合指定字段的合法 JSON。只输出 JSON，不要 Markdown。",
            modification_target=f"原始分析如下：\n{raw}\n\n必须保留实际观察到的内容，不能虚构。",
            image=_vision_batch([frames[0], frames[1], frames[2]]),
            model=model,
            temperature=0.0,
            max_tokens=1800,
        )
        return _normalize_auto_asset_analysis(_extract_json_object(repaired))


def _crop_person_frame(frame: Any, bbox: list[float]) -> torch.Tensor:
    value = _first_frame(frame)
    tensor = value.detach().cpu().float() if hasattr(value, "detach") else torch.as_tensor(value).float()
    height, width = tensor.shape[:2]
    left, top, right, bottom = bbox
    padding_x = (right - left) * 0.12
    padding_y = (bottom - top) * 0.12
    left = max(0.0, left - padding_x)
    top = max(0.0, top - padding_y)
    right = min(1.0, right + padding_x)
    bottom = min(1.0, bottom + padding_y)
    x0, y0 = int(left * width), int(top * height)
    x1, y1 = max(x0 + 1, int(math.ceil(right * width))), max(y0 + 1, int(math.ceil(bottom * height)))
    return tensor[y0:y1, x0:x1, :3][None,]


def _openai_image_files(images: list[Any]) -> list[tuple[str, tuple[str, BytesIO, str]]]:
    files: list[tuple[str, tuple[str, BytesIO, str]]] = []
    normalized = [_first_frame(item) for item in images if item is not None]
    for index, frame in enumerate(normalized):
        value = frame.detach().cpu() if hasattr(frame, "detach") else torch.as_tensor(frame)
        array = np.clip(np.asarray(value)[..., :3] * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        field_name = "image" if len(normalized) == 1 else "image[]"
        files.append((field_name, (f"reference_{index + 1}.png", buffer, "image/png")))
    if not files:
        raise ValueError("自动资产生成缺少有效的源图片。")
    return files


def _generate_auto_asset_image(
    images: list[Any],
    *,
    prompt: str,
    image_model: str,
    image_quality: str,
    image_provider: str,
) -> Any:
    return generate_openai_image(
        get_gpt_image_provider_config(image_provider),
        prompt=prompt,
        model=image_model,
        size="auto",
        quality=image_quality,
        background="auto",
        n=1,
        files=_openai_image_files(images),
    )


def _provider_policy_error(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in ("inputimagesensitivecontent", "sensitive content", "content policy", "moderation"))


class AnalysisGatewayUnavailableError(RuntimeError):
    """Raised before asset production when the configured analysis gateway is unusable."""


def _analysis_error_is_recoverable(value: Any) -> bool:
    """Return whether an analysis failure can reasonably succeed on a short retry."""
    text = str(value or "").lower()
    if _provider_policy_error(text):
        return False
    status_match = re.search(r"\bhttp\s+(\d{3})\b", text)
    if status_match:
        status = int(status_match.group(1))
        return status in {408, 429} or 500 <= status <= 599
    nonrecoverable_markers = (
        "authentication",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid model",
        "model not found",
        "bad request",
        "validation error",
        "config must",
        "not found: gpttext",
    )
    if any(marker in text for marker in nonrecoverable_markers):
        return False
    recoverable_markers = (
        "connection refused",
        "winerror 10061",
        "max retries exceeded",
        "connecttimeout",
        "readtimeout",
        "timed out",
        "timeout",
        "temporary",
        "eof",
        "connection reset",
        "connection aborted",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    return any(marker in text for marker in recoverable_markers)


def _real_person_privacy_error(value: Any) -> bool:
    text = str(value or "").lower()
    return (
        "inputimagesensitivecontentdetected.privacyinformation" in text
        or "input image may contain real person" in text
    )


def _auto_asset_match_result(
    decision: str,
    *,
    confidence: float = 0.0,
    reasons: list[str] | None = None,
    hard_mismatches: list[str] | None = None,
    missing_fields: list[str] | None = None,
    suspected_candidate_id: str = "",
    matched_features: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "decision": decision if decision in {"same", "different", "uncertain", "same_place_new_version"} else "uncertain",
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "reasons": reasons or [],
        "hard_mismatches": hard_mismatches or [],
        "missing_fields": missing_fields or [],
        "suspected_candidate_id": suspected_candidate_id,
        "matched_features": matched_features or [],
    }


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _normalize_auto_asset_match(
    value: Any,
    *,
    reuse_threshold: float,
    candidate_id: str = "",
    kind: str = "人物",
) -> dict[str, Any]:
    if isinstance(value, bool):
        return _auto_asset_match_result("same" if value else "uncertain", confidence=1.0 if value else 0.0)
    if not isinstance(value, dict):
        return _auto_asset_match_result("uncertain", reasons=["模型没有返回对象"], suspected_candidate_id=candidate_id)

    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    hard_mismatches = _as_string_list(value.get("hard_mismatches"))
    missing_fields = _as_string_list(value.get("missing_fields"))
    matched_features = _as_string_list(value.get("matched_features"))
    reasons = _as_string_list(value.get("reasons"))
    verdict = str(value.get("verdict") or value.get("decision") or "").strip().lower()

    if kind == "场景":
        same_place = bool(value.get("same_place"))
        same_version = bool(value.get("same_version") or value.get("same_subject"))
        if hard_mismatches:
            return _auto_asset_match_result(
                "different",
                confidence=confidence,
                reasons=reasons,
                hard_mismatches=hard_mismatches,
                missing_fields=missing_fields,
                suspected_candidate_id=candidate_id,
                matched_features=matched_features,
            )
        if verdict in {"same_place_new_version", "same_place"} or (same_place and not same_version):
            return _auto_asset_match_result(
                "same_place_new_version",
                confidence=max(confidence, reuse_threshold),
                reasons=reasons,
                missing_fields=missing_fields,
                suspected_candidate_id=candidate_id,
                matched_features=matched_features,
            )
        if (verdict in {"same", "same_version"} or same_version) and confidence >= reuse_threshold:
            return _auto_asset_match_result(
                "same",
                confidence=confidence,
                reasons=reasons,
                suspected_candidate_id=candidate_id,
                matched_features=matched_features,
            )
    else:
        same_subject = bool(value.get("same_subject"))
        if hard_mismatches:
            return _auto_asset_match_result(
                "different",
                confidence=confidence,
                reasons=reasons,
                hard_mismatches=hard_mismatches,
                missing_fields=missing_fields,
                suspected_candidate_id=candidate_id,
                matched_features=matched_features,
            )
        if (verdict == "same" or same_subject) and confidence >= reuse_threshold and not missing_fields:
            return _auto_asset_match_result(
                "same",
                confidence=confidence,
                reasons=reasons,
                suspected_candidate_id=candidate_id,
                matched_features=matched_features,
            )

    if confidence <= AUTO_ASSET_DIFFERENT_THRESHOLD:
        return _auto_asset_match_result(
            "different",
            confidence=confidence,
            reasons=reasons,
            hard_mismatches=hard_mismatches,
            missing_fields=missing_fields,
            suspected_candidate_id=candidate_id,
            matched_features=matched_features,
        )
    return _auto_asset_match_result(
        "uncertain",
        confidence=confidence,
        reasons=reasons,
        hard_mismatches=hard_mismatches,
        missing_fields=missing_fields,
        suspected_candidate_id=candidate_id,
        matched_features=matched_features,
    )


def _verify_auto_asset_cache(
    source: Any,
    candidate: Any,
    *,
    model: str,
    kind: str,
    current_features: str = "",
    candidate_features: str = "",
    candidate_id: str = "",
    reuse_threshold: float = AUTO_ASSET_DEFAULT_REUSE_THRESHOLD,
) -> dict[str, Any]:
    if kind == "场景":
        output_contract = (
            "{\"verdict\":\"same|same_place_new_version|different|uncertain\","
            "\"same_place\":true|false,\"same_version\":true|false,\"confidence\":0-1,"
            "\"hard_mismatches\":[],\"matched_features\":[],\"missing_fields\":[],\"reasons\":[]}"
        )
        request = (
            "第一张是当前镜头的原始场景帧，第二张是素材库中保存的原始场景观察帧。"
            "判断是否同一物理地点，以及是否同一机位/时间/光照版本。"
            f"当前描述：{current_features}。候选描述：{candidate_features}。"
            f"只输出合法 JSON：{output_contract}。"
        )
    else:
        output_contract = (
            "{\"verdict\":\"same|different|uncertain\",\"same_subject\":true|false,"
            "\"confidence\":0-1,\"hard_mismatches\":[],\"matched_features\":[],"
            "\"missing_fields\":[],\"reasons\":[]}"
        )
        request = (
            "第一张是当前镜头的人物原始裁剪，第二张是素材库保存的人物原始裁剪。"
            "判断是否是同一位会在多个镜头反复出现的叙事人物。优先比较脸部轮廓和五官、年龄感、体型、"
            "性别表达、稳定发型等身份特征；允许机位、表情、姿势、光线、背景和服装变化。"
            "只有脸部/体型/年龄等稳定身份特征明显冲突时才判为 different；脸不可见或证据不足时判为 uncertain。"
            f"当前特征：{current_features}。候选特征：{candidate_features}。"
            f"只输出合法 JSON：{output_contract}。"
        )
    try:
        raw = generate_openai_image_prompt_text(
            _get_config("gpttext"),
            skill="你是保守的源素材一致性校验器。不要根据名字猜测，只依据两张原始画面的可见特征判断。",
            modification_target=request,
            image=_vision_batch([source, candidate]),
            model=model,
            temperature=0.0,
            max_tokens=520,
        )
        return _normalize_auto_asset_match(
            _extract_json_object(raw),
            reuse_threshold=reuse_threshold,
            candidate_id=candidate_id,
            kind=kind,
        )
    except Exception as exc:
        return _auto_asset_match_result(
            "uncertain",
            reasons=[str(exc)],
            suspected_candidate_id=candidate_id,
        )


def _auto_asset_preview(job: LongVideoJob) -> torch.Tensor:
    paths: list[Path] = []
    for task in _auto_asset_member_tasks(job):
        assets = task.get("auto_assets") if isinstance(task, dict) else None
        if not isinstance(assets, dict):
            continue
        for value in assets.get("source_frames", {}).values():
            path = Path(str(value))
            if path.is_file():
                paths.append(path)
        for item in assets.get("people", []):
            path = Path(str(item.get("path", "")))
            if path.is_file():
                paths.append(path)
        for item in assets.get("scenes", []):
            path = Path(str(item.get("path", "")))
            if path.is_file():
                paths.append(path)
        for item in assets.get("integrated_frames", []):
            path = Path(str(item.get("path", "")))
            if path.is_file():
                paths.append(path)
    if not paths:
        raise ValueError("尚未生成可预览的自动资产。")
    return _vision_batch([_load_image_file(path) for path in paths[:SHOT_PREVIEW_LIMIT]])


def _normalize_auto_asset_style(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    supported = {
        AUTO_ASSET_STYLE_WESTERN,
        AUTO_ASSET_STYLE_ANIME,
        AUTO_ASSET_STYLE_PHOTOREAL,
        AUTO_ASSET_STYLE_CG_3D,
        AUTO_ASSET_STYLE_COMIC,
        AUTO_ASSET_STYLE_CUSTOM,
    }
    return normalized if normalized in supported else AUTO_ASSET_STYLE_WESTERN


def _person_asset_prompt(
    person: dict[str, Any],
    *,
    visual_style: str = AUTO_ASSET_STYLE_WESTERN,
    style_prompt: str = "",
) -> str:
    style = _normalize_auto_asset_style(visual_style)
    if style == AUTO_ASSET_STYLE_ANIME:
        return (
            "将参考图中的同一位真人完整重绘为高质量二维动漫角色设定图。"
            "保留可识别的年龄感、脸部轮廓、体型、发型、服装结构、配饰、道具和身份特征，"
            "但所有皮肤、五官、头发、衣物和材质都必须使用统一的二维动画造型、清晰线稿、赛璐璐分层上色和动漫光影。"
            "禁止保留照片纹理、真实皮肤毛孔、真人摄影质感、写实镜头噪点、3D 写实渲染或半真人面孔。"
            "使用干净、与人物颜色明显区分的纯色色键背景，画面中仅保留该人物，完整呈现其可见身体，"
            "不添加其他人物，不复制肢体，不改变人物身份。"
            f"人物可见特征：{person['appearance']}。"
        )
    if style == AUTO_ASSET_STYLE_PHOTOREAL:
        return (
            "将参考图中的同一位人物制作成高质量真人影视角色参考图。"
            "严格保留可识别的年龄感、脸部轮廓、五官比例、体型、发型、服装结构、配饰、道具和身份特征，"
            "使用自然真实的皮肤、头发和布料材质，符合真实摄影的光线、镜头和人体结构。"
            "禁止动漫线稿、卡通脸、插画笔触、塑料皮肤、游戏建模感或夸张美化。"
            "使用干净、与人物颜色明显区分的纯色棚拍背景，画面中仅保留该人物，完整呈现其可见身体，"
            "不添加其他人物，不复制肢体，不改变人物身份。"
            f"人物可见特征：{person['appearance']}。"
        )
    if style == AUTO_ASSET_STYLE_CG_3D:
        return (
            "将参考图中的同一位人物重制为高质量 3D 游戏 CG 角色设定图。"
            "严格保留可识别的年龄感、脸部轮廓、体型、发型、服装结构、配饰、道具和身份特征，"
            "使用一致的三维造型、PBR 材质、可信的体积光和影视级游戏过场渲染。"
            "禁止真人摄影画面、二维线稿、平面插画、低模、塑料质感、材质穿帮或半真人半卡通。"
            "使用干净、与人物颜色明显区分的纯色背景，画面中仅保留该人物，完整呈现其可见身体，"
            "不添加其他人物，不复制肢体，不改变人物身份。"
            f"人物可见特征：{person['appearance']}。"
        )
    if style == AUTO_ASSET_STYLE_COMIC:
        return (
            "将参考图中的同一位人物完整重绘为高质量漫画插画角色设定图。"
            "严格保留可识别的年龄感、脸部轮廓、体型、发型、服装结构、配饰、道具和身份特征，"
            "使用统一的手绘墨线、明确的明暗块面、细腻插画上色和有表现力但稳定的五官造型。"
            "禁止真人摄影纹理、3D 建模感、廉价卡通贴纸感、画风漂移或半写实拼贴。"
            "使用干净、与人物颜色明显区分的纯色背景，画面中仅保留该人物，完整呈现其可见身体，"
            "不添加其他人物，不复制肢体，不改变人物身份。"
            f"人物可见特征：{person['appearance']}。"
        )
    if style == AUTO_ASSET_STYLE_CUSTOM:
        return (
            "根据用户设定的目标视觉方向，将参考图中的同一位人物制作成稳定的角色参考图。"
            "严格保留人物身份、年龄感、脸部轮廓、体型、发型、服装结构、配饰和道具，"
            "使用干净、与人物颜色明显区分的纯色背景，画面中仅保留该人物，不添加其他人物或肢体。"
            f"目标视觉方向：{str(style_prompt or '保持用户指定风格').strip()}。"
            f"人物可见特征：{person['appearance']}。"
        )
    return (
        "为后续镜头建立可复用的欧美人物替换母版。用一位符合当代欧美审美的外国人物，"
        "替换参考图中的原人物，而不是只改脸或轻微调色。保留性别、年龄段、体型、人物关系和剧情身份功能，"
        "但人物的面部地域特征、发型发色、妆容、服装鞋履、配饰、版型剪裁、材质配色和整体气质可以发生明显变化。"
        "必须清除中式/东方服饰结构、纹样和本土造型残留，使替换后人物自然且文化风格统一。"
        "若原图是动漫、漫画、插画或 CG，保持媒介并替换为欧美版本；若是真人，使用高质量欧美真人影视质感。"
        "使用干净、与服装颜色明显区分的纯色色键背景，画面中仅保留完整单人，不添加其他人物或肢体。"
        f"人物可见特征：{person['appearance']}。"
    )


def _scene_asset_prompt(
    description: str,
    *,
    position: str,
    visual_style: str = AUTO_ASSET_STYLE_WESTERN,
    style_prompt: str = "",
) -> str:
    style = _normalize_auto_asset_style(visual_style)
    if style == AUTO_ASSET_STYLE_ANIME:
        return (
            "将参考图中的整个环境完整重绘为高质量二维动漫背景。移除所有真人、人物、人体残留和原照片人物投影，"
            "并自然补全被遮挡区域。保留原镜头的空间布局、建筑与物体位置、时间氛围、光线方向、机位关系和叙事用途，"
            "但所有建筑、地面、家具、植被、天空、光影和材质都必须变成统一的二维动画美术、清晰线稿、"
            "赛璐璐分层上色和动漫场景光影。禁止保留照片纹理、真人摄影质感、真实镜头噪点、写实 3D 渲染、"
            "文字、水印或拼贴边缘。"
            f"这是镜头{position}的场景，画面描述：{description}。"
        )
    if style == AUTO_ASSET_STYLE_PHOTOREAL:
        return (
            "以参考图中的完整环境为基础，移除所有人物、人体残留和人物投影，并自然补全被遮挡区域。"
            "严格保留空间布局、建筑与物体位置、时间氛围、光线方向、机位关系和叙事用途，"
            "把环境制作成高质量真人影视实景，使用可信的真实材质、自然光影和摄影镜头质感。"
            "禁止动漫线稿、插画笔触、游戏 CG、塑料材质、人物、文字、水印或拼贴边缘。"
            f"这是镜头{position}的场景，画面描述：{description}。"
        )
    if style == AUTO_ASSET_STYLE_CG_3D:
        return (
            "以参考图中的完整环境为基础，移除所有人物并自然补全被遮挡区域，"
            "严格保留空间布局、建筑与物体位置、时间氛围、光线方向、机位关系和叙事用途。"
            "将整个环境重制为统一的高质量 3D 游戏 CG 场景，使用完整三维结构、PBR 材质、体积光和影视级过场渲染。"
            "禁止真人摄影纹理、二维线稿、平面插画、低模、人物、文字、水印或拼贴边缘。"
            f"这是镜头{position}的场景，画面描述：{description}。"
        )
    if style == AUTO_ASSET_STYLE_COMIC:
        return (
            "以参考图中的完整环境为基础，移除所有人物并自然补全被遮挡区域，"
            "严格保留空间布局、建筑与物体位置、时间氛围、光线方向、机位关系和叙事用途。"
            "将整个环境重绘为统一的高质量漫画插画背景，使用手绘墨线、明确明暗块面、细腻插画上色和稳定透视。"
            "禁止真人摄影纹理、3D 建模感、廉价卡通贴纸感、人物、文字、水印或拼贴边缘。"
            f"这是镜头{position}的场景，画面描述：{description}。"
        )
    if style == AUTO_ASSET_STYLE_CUSTOM:
        return (
            "根据用户设定的目标视觉方向重制参考图中的完整环境。移除所有人物并自然补全被遮挡区域，"
            "严格保留空间布局、物体位置、时间氛围、光线方向、机位关系和叙事用途，"
            "不得出现人物、人体残留、文字、水印或拼贴边缘。"
            f"目标视觉方向：{str(style_prompt or '保持用户指定风格').strip()}。"
            f"这是镜头{position}的场景，画面描述：{description}。"
        )
    return (
        "为后续镜头建立可复用的欧美场景替换母版。移除参考图中的所有人物、人体残留、倒影和人物投影，"
        "用完整、可信、地域统一的北美或欧洲环境替换原建筑、道路、公共设施、家具陈设、材质、植被、照明和生活细节。"
        "不要轻微调色或只换少量道具；允许环境外观发生明显变化。只保留镜头的画幅、机位、景别、透视、"
        "人物可活动空间、基本叙事用途和光线方向，以便后续镜头能沿用同一个替换场景。"
        "必须清除中式建筑、家具、纹样、中文招牌及本土化视觉符号。保持输入图的原始视觉媒介和合理透视；"
        "不得出现人物、人体残留、可辨识文字、乱码、Logo、水印或拼贴边缘。"
        f"这是镜头{position}的场景，画面描述：{description}。"
    )


def _integrated_frame_prompt(
    analysis: dict[str, Any],
    *,
    role: str,
    person_master_count: int,
    has_scene_master: bool,
    has_style_master: bool,
    visual_style: str,
    style_prompt: str,
    retry_reasons: list[str] | None = None,
) -> str:
    style = _normalize_auto_asset_style(visual_style)
    style_targets = {
        AUTO_ASSET_STYLE_ANIME: "统一、明确的高质量二维动漫造型、线稿、上色和光影",
        AUTO_ASSET_STYLE_PHOTOREAL: "统一的高质量真人影视摄影、真实材质和自然电影光影",
        AUTO_ASSET_STYLE_CG_3D: "统一的高质量 3D 游戏 CG、PBR 材质和影视级体积光",
        AUTO_ASSET_STYLE_COMIC: "统一的高质量漫画插画、手绘墨线、块面和插画上色",
        AUTO_ASSET_STYLE_CUSTOM: str(style_prompt or "用户指定的目标视觉风格").strip(),
        AUTO_ASSET_STYLE_WESTERN: (
            "彻底、明显的欧美化影视视觉：人物整体造型、服装、发型与气质，以及建筑、材质、陈设和地域语言"
        ),
    }
    references: list[str] = [
        "图片 1 是当前镜头原始整帧，只负责构图、机位、景别、动作、人物数量、空间关系和叙事物体位置，"
        "不提供人物身份、服装、建筑、地域、材质或视觉风格约束"
    ]
    next_index = 2
    if person_master_count:
        end = next_index + person_master_count - 1
        references.append(
            f"图片 {next_index}" + (f" 至图片 {end}" if end > next_index else "") +
            " 是人物替换母版，只负责对应人物的欧美身份、脸、发型、服装、配饰和整体造型"
        )
        next_index = end + 1
    if has_scene_master:
        references.append(f"图片 {next_index} 是场景替换母版，只负责欧美环境、建筑设计、材质和色彩")
        next_index += 1
    if has_style_master:
        references.append(
            f"图片 {next_index} 是全局强风格母版，只迁移风格强度和视觉语言，严禁复制其中的人物、建筑布局或构图"
        )
    failure_instruction = ""
    if retry_reasons:
        failure_instruction = (
            "上一版因以下问题未通过验收，本次必须逐项修正："
            + "；".join(str(item) for item in retry_reasons if str(item).strip())
            + "。"
        )
    position = "开始" if role == "frame_start" else "结束"
    people = "；".join(str(item.get("appearance") or "主要人物") for item in analysis.get("people", []))
    background = analysis.get("background") if isinstance(analysis.get("background"), dict) else {}
    scene_description = background.get("first_description") if role == "frame_start" else background.get("last_description")
    return (
        "执行多图整帧替换。" + "；".join(references) + "。"
        f"输出必须保持图片 1 的完整画幅、{position}帧构图、透视、景别、人物数量、人物站位、动作状态和叙事物体，"
        "把人物与背景一次性替换为完整画面，不能输出人物抠图、空背景、拼图、设定图或局部裁剪。"
        f"目标风格：{style_targets[style]}。"
        "不得只做调色。人物和环境与图片 1 明显不同是预期结果；不得保留与目标风格冲突的服装、建筑、材质、纹样或摄影质感；"
        "不得新增、删除、复制、融合人物，不得把母版中的人物或场景布局搬到当前镜头。"
        f"当前人物：{people or '按原整帧中的实际人物数量与外观处理'}。"
        f"当前场景：{str(scene_description or '只保持当前镜头的空间关系和叙事用途')}。"
        f"{failure_instruction}只输出一张完成后的整帧画面，不要文字、水印、边框或说明。"
    )


def _score01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_integrated_frame_quality(
    value: Any,
    *,
    require_person_identity: bool,
) -> dict[str, Any]:
    result = value if isinstance(value, dict) else {}
    target_style = _score01(result.get("target_style_score"))
    source_residue = _score01(result.get("source_style_residue"))
    composition = _score01(result.get("composition_preservation"))
    person_identity = _score01(result.get("person_identity_score")) if require_person_identity else 1.0
    scene_replacement = _score01(result.get("scene_replacement_score", result.get("scene_identity_score")))
    person_count_match = bool(result.get("person_count_match"))
    reasons = _as_string_list(result.get("reasons"))
    failed: list[str] = []
    if target_style < AUTO_ASSET_TARGET_STYLE_THRESHOLD:
        failed.append(f"目标风格强度不足（{target_style:.2f}）")
    if source_residue > AUTO_ASSET_SOURCE_RESIDUE_THRESHOLD:
        failed.append(f"原风格残留过多（{source_residue:.2f}）")
    if composition < AUTO_ASSET_COMPOSITION_THRESHOLD:
        failed.append(f"构图保留不足（{composition:.2f}）")
    if not person_count_match:
        failed.append("人物数量与源帧不一致")
    if require_person_identity and person_identity < AUTO_ASSET_PERSON_IDENTITY_THRESHOLD:
        failed.append(f"人物身份一致性不足（{person_identity:.2f}）")
    if scene_replacement < AUTO_ASSET_SCENE_REPLACEMENT_THRESHOLD:
        failed.append(f"场景替换或空间关系不合格（{scene_replacement:.2f}）")
    for item in failed:
        if item not in reasons:
            reasons.append(item)
    return {
        "version": AUTO_ASSET_QUALITY_GATE_VERSION,
        "target_style_score": target_style,
        "source_style_residue": source_residue,
        "composition_preservation": composition,
        "person_count_match": person_count_match,
        "person_identity_score": person_identity,
        "scene_replacement_score": scene_replacement,
        "verdict": "approved" if not failed else "retry",
        "reasons": reasons,
        "thresholds": {
            "target_style_score": AUTO_ASSET_TARGET_STYLE_THRESHOLD,
            "source_style_residue": AUTO_ASSET_SOURCE_RESIDUE_THRESHOLD,
            "composition_preservation": AUTO_ASSET_COMPOSITION_THRESHOLD,
            "person_identity_score": AUTO_ASSET_PERSON_IDENTITY_THRESHOLD,
            "scene_replacement_score": AUTO_ASSET_SCENE_REPLACEMENT_THRESHOLD,
        },
    }


def _evaluate_integrated_frame_quality(
    *,
    source_frame: Any,
    candidate: Any,
    person_masters: list[Any],
    scene_master: Any | None,
    style_master: Any | None,
    analysis: dict[str, Any],
    model: str,
    visual_style: str,
    style_prompt: str,
) -> dict[str, Any]:
    images = [source_frame, candidate]
    labels = ["图片1=当前源整帧", "图片2=候选整帧替换结果"]
    for person in person_masters:
        images.append(person)
        labels.append(f"图片{len(images)}=人物替换母版")
    if scene_master is not None:
        images.append(scene_master)
        labels.append(f"图片{len(images)}=场景替换母版")
    if style_master is not None:
        images.append(style_master)
        labels.append(f"图片{len(images)}=全局强风格母版")
    output_contract = (
        '{"target_style_score":0-1,"source_style_residue":0-1,'
        '"composition_preservation":0-1,"person_count_match":true|false,'
        '"person_identity_score":0-1,"scene_replacement_score":0-1,"reasons":[]}'
    )
    request = (
        f"输入顺序：{'；'.join(labels)}。"
        "严格比较图片2是否保留图片1的构图、机位、景别、人物数量、动作、空间关系和叙事物体，"
        "并是否用人物/场景替换母版完成明显的目标替换。不要因为人物长相、服装、建筑或环境与图片1不同而扣分，"
        "这种改变正是目标。若仍接近源图、只调色、人物数量变化、身份未遵循人物母版、环境未遵循场景母版、"
        "空间关系丢失或复制了母版构图，必须降低对应分数。"
        f"目标风格类型：{_normalize_auto_asset_style(visual_style)}；目标说明：{str(style_prompt or '')[:1200]}。"
        f"镜头分析：{json.dumps(analysis, ensure_ascii=False)[:1800]}。只输出合法 JSON：{output_contract}。"
    )
    raw = generate_openai_image_prompt_text(
        _get_config("gpttext"),
        skill="你是严格的整帧替换验收器。宁可要求再次生成，也不能让弱替换、身份错误或空间关系错误进入视频参考包。",
        modification_target=request,
        image=_vision_batch(images),
        model=model,
        temperature=0.0,
        max_tokens=700,
    )
    return _normalize_integrated_frame_quality(
        _extract_json_object(raw),
        require_person_identity=bool(person_masters),
    )


def _last_frame(video_path: Path) -> torch.Tensor:
    last: torch.Tensor | None = None
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"远程结果没有视频流：{video_path}")
        for frame in container.decode(container.streams.video[0]):
            last = torch.from_numpy(frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0)
    if last is None:
        raise ValueError(f"远程结果没有可用画面：{video_path}")
    return last[None,]


def _video_geometry(video_path: Path) -> tuple[int, int, float]:
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"远程结果没有视频流：{video_path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        return int(stream.width), int(stream.height), max(1.0, fps)


def _video_track_info(video_path: Path) -> tuple[float, float, int, bool]:
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"远程结果没有视频流：{video_path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        frame_count = int(stream.frames or 0)
        if frame_count <= 0:
            frame_count = sum(1 for _frame in container.decode(stream))
        duration = frame_count / max(fps, 1.0)
        return duration, fps, frame_count, bool(container.streams.audio)


def _normalize_segment(
    source_path: Path,
    output_path: Path,
    *,
    duration: float,
    width: int,
    height: int,
    fps: float,
    trim_offset: float = 0.0,
) -> None:
    ffmpeg = _ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"tpad=stop_mode=clone:stop_duration={trim_offset + duration:.6f},"
        f"trim=start={trim_offset:.6f}:duration={duration:.6f},"
        f"setpts=PTS-STARTPTS,fps={fps:.6f}"
    )
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        error_prefix="远程分段规范化失败",
    )


def _normalize_segment_v3(
    source_path: Path,
    output_path: Path,
    *,
    request_duration: float,
    width: int,
    height: int,
    fps: float,
    keep_generated_audio: bool,
) -> dict[str, Any]:
    source_duration, _source_fps, source_frames, has_audio = _video_track_info(source_path)
    if keep_generated_audio and not has_audio:
        raise ValueError("Seedance 返回的视频没有生成音轨，不能按生成音频模式继续。")
    target_duration = max(float(request_duration), source_duration)
    padding = max(0.0, target_duration - source_duration)
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={fps:.6f}"
    )
    if padding > 1e-6:
        video_filter += f",tpad=stop_mode=clone:stop_duration={padding:.6f}"
    args = [_ffmpeg_exe(), "-y", "-i", str(source_path), "-map", "0:v:0"]
    if keep_generated_audio:
        args.extend(
            [
                "-map",
                "0:a:0",
                "-af",
                f"apad=pad_dur={target_duration:.6f},atrim=duration={target_duration:.6f},asetpts=PTS-STARTPTS",
            ]
        )
    else:
        args.append("-an")
    args.extend(
        [
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if keep_generated_audio:
        args.extend(["-c:a", "aac"])
    args.append(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(args, error_prefix="v3 远程分段规范化失败")
    final_duration, final_fps, final_frames, final_has_audio = _video_track_info(output_path)
    if keep_generated_audio and not final_has_audio:
        raise ValueError("v3 规范化结果缺少音轨。")
    return {
        "remote_duration": round(source_duration, 6),
        "remote_frame_count": source_frames,
        "request_duration": round(float(request_duration), 6),
        "output_duration": round(final_duration, 6),
        "output_fps": round(final_fps, 6),
        "output_frame_count": final_frames,
        "video_padding": round(padding, 6),
        "has_audio": final_has_audio,
    }
def _source_segment_for_task(job: LongVideoJob, task: dict[str, Any]) -> Any:
    source_start = float(task.get("source_start", task["start"]))
    source_duration = float(task.get("source_duration", task["duration"]))
    padding_start = float(task.get("padding_start", 0.0))
    padding_end = float(task.get("padding_end", 0.0))
    request_path = job.job_dir / "requests" / f"request_{int(task['index']):04d}.mp4"
    if job.force_rerun or not request_path.is_file():
        request_path.parent.mkdir(parents=True, exist_ok=True)
        if padding_start <= 1e-6 and padding_end <= 1e-6:
            _run_ffmpeg(
                [
                    _ffmpeg_exe(),
                    "-y",
                    "-ss",
                    f"{source_start:.6f}",
                    "-i",
                    job.source_path,
                    "-t",
                    f"{source_duration:.6f}",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(request_path),
                ],
                error_prefix=f"第 {task['index']} 段参考视频截取失败",
            )
            return InputImpl.VideoFromFile(str(request_path))

        source_clip_path = request_path.with_name(request_path.stem + "_source.mp4")
        _run_ffmpeg(
            [
                _ffmpeg_exe(),
                "-y",
                "-ss",
                f"{source_start:.6f}",
                "-i",
                job.source_path,
                "-t",
                f"{source_duration:.6f}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source_clip_path),
            ],
            error_prefix=f"第 {task['index']} 段短镜头截取失败",
        )
        _width, _height, source_fps = _video_geometry(Path(job.source_path))
        padding_start_frames = max(0, int(round(padding_start * source_fps)))
        padding_end_frames = max(0, int(round(padding_end * source_fps)))
        video_filter = (
            f"tpad=start={padding_start_frames}:start_mode=clone:"
            f"stop={padding_end_frames}:stop_mode=clone,"
            f"fps={source_fps:.6f}"
        )
        _run_ffmpeg(
            [
                _ffmpeg_exe(),
                "-y",
                "-i",
                str(source_clip_path),
                "-an",
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(request_path),
            ],
            error_prefix=f"第 {task['index']} 段短镜头补帧失败",
        )
    return InputImpl.VideoFromFile(str(request_path))


def _person_contact_sheet(images: list[Any]) -> torch.Tensor:
    frames = [_first_frame(image) for image in images if image is not None]
    if not frames:
        raise ValueError("无法从空人物列表创建参考拼图。")
    tensors = []
    for frame in frames:
        value = frame.detach().cpu().float() if hasattr(frame, "detach") else torch.as_tensor(frame).float()
        value = value[..., :3].movedim(-1, 0)[None,]
        value = torch_functional.interpolate(value, size=(768, 512), mode="bilinear", align_corners=False)
        tensors.append(value[0].movedim(0, -1))
    return torch.cat(tensors, dim=1)[None,]


def _vision_batch(items: list[Any], *, size: int = 512) -> torch.Tensor:
    prepared = []
    for item in items:
        frame = _first_frame(item)
        value = frame.detach().cpu().float() if hasattr(frame, "detach") else torch.as_tensor(frame).float()
        value = value[..., :3].movedim(-1, 0)[None,]
        height, width = value.shape[-2:]
        scale = min(size / max(1, width), size / max(1, height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        value = torch_functional.interpolate(
            value,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        pad_left = (size - resized_width) // 2
        pad_right = size - resized_width - pad_left
        pad_top = (size - resized_height) // 2
        pad_bottom = size - resized_height - pad_top
        value = torch_functional.pad(value, (pad_left, pad_right, pad_top, pad_bottom), value=0.5)
        prepared.append(value[0].movedim(0, -1))
    return torch.stack(prepared, dim=0)


def _reference_images_for_task(
    adapter: VideoEngineAdapter,
    *,
    selected_people: list[Any],
    selected_backgrounds: list[Any],
    previous_end_frame: Any,
) -> tuple[list[Any], list[str]]:
    references: list[Any] = []
    roles: list[str] = []
    if adapter.key == "wan" and len(selected_people) > 1:
        references.append(_person_contact_sheet(selected_people))
        roles.append("people_contact_sheet")
    else:
        references.extend(selected_people)
        roles.extend(f"person_{index + 1}" for index in range(len(selected_people)))
    if selected_backgrounds:
        references.append(selected_backgrounds[0])
        roles.append("background")
    if previous_end_frame is not None:
        references.append(previous_end_frame)
        roles.append("previous_segment_end_frame")
    return references[: adapter.reference_image_limit], roles[: adapter.reference_image_limit]


def _segment_analysis(
    frames: torch.Tensor,
    *,
    manifest: dict[str, Any],
    model: str,
    people_images: dict[str, Any],
    background_images: dict[str, Any],
) -> dict[str, Any]:
    people = _manifest_entries(manifest, "people", PERSON_IDS)
    backgrounds = _manifest_entries(manifest, "backgrounds", BACKGROUND_IDS)
    person_order = [entry_id for entry_id in PERSON_IDS if entry_id in people_images]
    background_order = [entry_id for entry_id in BACKGROUND_IDS if entry_id in background_images]
    identity_text = json.dumps({"people": people, "backgrounds": backgrounds}, ensure_ascii=False)
    skill = (
        "你是长视频转绘的分段素材识别器。只输出合法 JSON，不要 Markdown。"
        "根据输入视频片段的关键帧，选择实际出现的人物和场景资产。人物只能从 A、B、C 中选择，"
        "背景只能从 BG01-BG08 中选择。输出字段 people、backgrounds、action_prompt，"
        "其中 people 和 backgrounds 必须是字符串数组，action_prompt 必须简洁描述该片段的剧情、动作、"
        "镜头和表演，不要改变原剧情。"
    )
    request = (
        f"已确认资产身份表：{identity_text}\n"
        "输入图片顺序：前 3 张是当前视频分段的开始、中间、结束关键帧；"
        f"随后人物参考图依次为 {', '.join(person_order) or '无'}；"
        f"最后背景参考图依次为 {', '.join(background_order) or '无'}。"
        "请通过视觉对照识别这一段实际出现的人物和场景资产。"
    )
    visual_inputs = [frames[index] for index in range(frames.shape[0])]
    visual_inputs.extend(people_images[entry_id] for entry_id in person_order)
    visual_inputs.extend(background_images[entry_id] for entry_id in background_order)
    try:
        text = generate_openai_image_prompt_text(
            _get_config("gpttext"),
            skill=skill,
            modification_target=request,
            image=_vision_batch(visual_inputs),
            model=model,
            temperature=0.1,
            max_tokens=1200,
        )
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("分段识别结果根节点不是对象。")
        selected_people = [item for item in parsed.get("people", []) if item in PERSON_IDS and item in people]
        selected_backgrounds = [item for item in parsed.get("backgrounds", []) if item in BACKGROUND_IDS and item in backgrounds]
        if not selected_people or not selected_backgrounds:
            raise ValueError("分段识别没有返回可用的人物或背景资产。")
        return {
            "people": selected_people,
            "backgrounds": selected_backgrounds,
            "action_prompt": str(parsed.get("action_prompt") or "按照参考视频片段的完整剧情、动作、镜头和表演重新演绎。"),
            "analysis_source": "gpttext",
        }
    except Exception as exc:
        fallback_people = list(people)
        fallback_backgrounds = list(backgrounds)[:1]
        return {
            "people": fallback_people,
            "backgrounds": fallback_backgrounds,
            "action_prompt": "按照参考视频片段的完整剧情、动作、镜头和表演重新演绎。",
            "analysis_source": "fallback",
            "analysis_error": str(exc),
        }


def _build_video_prompt(base_prompt: str, analysis: dict[str, Any], *, soft_continuity: bool) -> str:
    continuity = (
        "上一段末帧仅作为动作和空间连续性的额外参考，属于软约束；不得复制错误、不得改变剧情。"
        if soft_continuity
        else "严格使用提供的首帧/尾帧约束相邻片段连续性。"
    )


def _build_auto_asset_video_prompt(
    base_prompt: str,
    analysis: dict[str, Any],
    *,
    reference_roles: list[str],
    soft_continuity: bool,
    visual_style: str = AUTO_ASSET_STYLE_WESTERN,
    send_source_video: bool = True,
    reference_timeline: list[dict[str, Any]] | None = None,
) -> str:
    continuity = (
        "上一段末帧是动作与空间连续性的软参考；优先保持本段剧情和镜头逻辑。"
        if soft_continuity
        else "本段按独立镜头生成，不引用上一段的末帧。"
    )
    people = "; ".join(item.get("appearance", "主要人物") for item in analysis.get("people", [])) or "按原视频实际人物处理"
    style = _normalize_auto_asset_style(visual_style)
    if style == AUTO_ASSET_STYLE_ANIME:
        source_instruction = (
            "未向视频模型提供原真人分镜视频；只能依据本段剧情分析文字重新演绎动作和镜头。"
            if not send_source_video
            else "原分镜视频只定义剧情、动作、镜头和时间过程，不定义真人外观。"
        )
        style_instruction = (
            "参考图片是最终人物和环境的唯一视觉依据。整段输出必须从第一帧到最后一帧保持统一的高质量二维动漫风格，"
            "人物、皮肤、五官、头发、服装、道具、建筑、地面、天空和全部背景都必须动漫化。"
            "禁止出现真人脸、真实皮肤、照片纹理、真人摄影画面、半真人半动漫、写实 3D 人物或风格回退。"
        )
    elif style == AUTO_ASSET_STYLE_PHOTOREAL:
        source_instruction = (
            "未向视频模型提供原分镜视频；只能依据本段剧情分析文字重新演绎动作和镜头。"
            if not send_source_video
            else "原分镜视频只定义剧情、动作、镜头和时间过程，不定义人物与环境外观。"
        )
        style_instruction = (
            "参考图片是最终人物和环境的唯一视觉依据。整段输出必须从第一帧到最后一帧保持高质量真人影视质感，"
            "人物身份、服装、道具和完整环境必须稳定，皮肤、头发、布料、建筑和光影均符合真实摄影。"
            "禁止出现动漫线稿、漫画笔触、卡通脸、游戏 CG、塑料皮肤、半真人半卡通或风格回退。"
        )
    elif style == AUTO_ASSET_STYLE_CG_3D:
        source_instruction = (
            "未向视频模型提供原分镜视频；只能依据本段剧情分析文字重新演绎动作和镜头。"
            if not send_source_video
            else "原分镜视频只定义剧情、动作、镜头和时间过程，不定义人物与环境外观。"
        )
        style_instruction = (
            "参考图片是最终人物和环境的唯一视觉依据。整段输出必须保持统一的高质量 3D 游戏 CG 风格，"
            "人物、服装、道具和完整环境使用稳定三维造型、PBR 材质、体积光与影视级游戏过场渲染。"
            "禁止真人摄影画面、二维线稿、平面插画、低模、塑料质感、材质跳变或风格回退。"
        )
    elif style == AUTO_ASSET_STYLE_COMIC:
        source_instruction = (
            "未向视频模型提供原分镜视频；只能依据本段剧情分析文字重新演绎动作和镜头。"
            if not send_source_video
            else "原分镜视频只定义剧情、动作、镜头和时间过程，不定义人物与环境外观。"
        )
        style_instruction = (
            "参考图片是最终人物和环境的唯一视觉依据。整段输出必须保持统一的高质量漫画插画风格，"
            "人物、服装、道具和完整环境使用稳定手绘墨线、明确明暗块面、细腻插画上色与一致透视。"
            "禁止真人摄影纹理、3D 建模感、廉价卡通贴纸感、拼贴、画风漂移或风格回退。"
        )
    elif style == AUTO_ASSET_STYLE_CUSTOM:
        source_instruction = (
            "未向视频模型提供原分镜视频；动作和镜头必须依据本段剧情分析文字重新演绎。"
            if not send_source_video
            else "原分镜视频只定义剧情、动作、镜头和时间过程。"
        )
        style_instruction = (
            "参考图片定义最终人物身份、服装、道具和环境；严格遵守用户提示词指定的目标视觉方向，"
            "并保证整段人物、完整背景、材质、色彩和画风稳定一致。"
        )
    else:
        source_instruction = (
            "原分镜视频只定义剧情、动作、镜头和时间过程，不定义人物与环境外观。"
            if send_source_video
            else "未向视频模型提供原分镜视频；只能依据本段剧情分析文字重新演绎动作和镜头。"
        )
        style_instruction = (
            "参考图片分别定义用于替换原人物和原环境的唯一欧美视觉身份，不得混用："
            "人物必须保持参考图中整体欧美化后的面孔、发型、妆容、服装、鞋履、配饰和气质，不能只换脸或恢复原造型；"
            "完整背景必须保持参考图中地域统一的欧美建筑、道路、家具、材质、植被、照明和生活细节，"
            "不能恢复原背景或退化为轻微调色。人物和环境可以与原视频显著不同，但必须保留原镜头的人物数量、"
            "动作、空间关系、叙事物体和镜头运动。保持参考图原有视觉媒介，真人保持欧美真人电影质感，"
            "动漫、漫画或 CG 保持同一媒介的欧美版本；从第一帧到最后一帧维持人物身份、整体造型、环境和光影稳定。"
        )
    timeline_lines: list[str] = []
    for fallback_index, item in enumerate(reference_timeline or [], start=1):
        image_index = int(item.get("image_index") or fallback_index)
        ranges = item.get("ranges") if isinstance(item.get("ranges"), list) else []
        range_text = "、".join(
            f"{float(value.get('start', 0.0)):.2f}-{float(value.get('end', 0.0)):.2f} 秒"
            for value in ranges
            if isinstance(value, dict)
        )
        if range_text:
            timeline_lines.append(
                f"{range_text}以图片 {image_index} 定义该时间范围的人物、场景、构图和目标视觉；"
            )
    timeline_instruction = ""
    if timeline_lines:
        timeline_instruction = (
            "参考图片按 content 中的实际顺序编号，时间映射如下：\n"
            + "\n".join(timeline_lines)
            + "\n这些时间范围属于同一次连续生成请求，不得理解为视频切段或额外转场。\n"
        )
    role_instruction = (
        f"自动参考图角色顺序：{', '.join(reference_roles) or '无额外参考图'}。\n"
        if not timeline_lines
        else ""
    )
    return (
        f"{str(base_prompt or '').strip()}\n\n"
        f"本段原视频规定剧情、动作、镜头运动、表演节奏和时间过程：{analysis.get('story_action', '')}\n"
        f"自动识别的主要人物：{people}\n"
        f"{role_instruction}{timeline_instruction}"
        f"{style_instruction}{source_instruction}"
        "保持原视频实际人物数量、人物关系和场景变化，不增加、删除、复制或融合人物。"
        "允许人物动作、表情和镜头内互动自然演绎，但画面必须符合真实的身体结构、空间关系和叙事因果。\n"
        f"{continuity}"
    )


def _asset_path_if_file(value: Any) -> Path | None:
    path = Path(str(value or ""))
    return path if path.is_file() else None


def _asset_record_id(*parts: Any) -> str:
    return _safe_slug("_".join(str(part) for part in parts if str(part or "").strip()), "asset")


def _source_observation_paths(record: dict[str, Any]) -> list[Path]:
    values = record.get("source_observations")
    paths: list[Path] = []
    for item in values if isinstance(values, list) else []:
        value = item.get("path") if isinstance(item, dict) else item
        path = _asset_path_if_file(value)
        if path is not None:
            paths.append(path)
    return paths


def _source_observation_entries(paths: list[str], *, kind: str, description: str = "") -> list[dict[str, str]]:
    entries = []
    for index, path in enumerate(paths[:AUTO_ASSET_MAX_SOURCE_OBSERVATIONS], start=1):
        entries.append({"kind": kind, "index": str(index), "path": str(path), "description": description})
    return entries


def _append_source_observations(record: dict[str, Any], observations: list[dict[str, str]]) -> None:
    existing = record.setdefault("source_observations", [])
    if not isinstance(existing, list):
        existing = []
        record["source_observations"] = existing
    seen = {str(item.get("path")) for item in existing if isinstance(item, dict)}
    for observation in observations:
        path = str(observation.get("path") or "")
        if path and path not in seen:
            existing.append(observation)
            seen.add(path)
    del existing[AUTO_ASSET_MAX_SOURCE_OBSERVATIONS:]


def _append_candidate_index(cache: dict[str, Any], category: str, key: str, identifier: str) -> None:
    if not key or not identifier:
        return
    group = "people" if category == "person" else "scenes"
    index = cache[group].setdefault("candidate_index", {})
    values = index.setdefault(str(key), [])
    if identifier not in values:
        values.append(identifier)


def _candidate_ids(index: dict[str, Any], hint: str, available: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for candidate_id in _normalized_id_list(index.get(hint)):
        if candidate_id in available and candidate_id not in ordered:
            ordered.append(candidate_id)
    for candidate_id in available:
        if candidate_id not in ordered:
            ordered.append(candidate_id)
    return ordered


def _save_person_source_observations(images: list[Any], root: Path, slot: str) -> tuple[list[str], list[Any]]:
    paths: list[str] = []
    normalized: list[Any] = []
    source_root = root / "source_people"
    source_root.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images, start=1):
        path = source_root / f"{slot}_{index}.png"
        _save_image_tensor(image, path)
        paths.append(str(path))
        normalized.append(image)
    return paths, normalized


def _person_master_resolution(paths: list[str]) -> dict[str, Any]:
    widths: list[int] = []
    heights: list[int] = []
    for value in paths:
        path = _asset_path_if_file(value)
        if path is None:
            continue
        with Image.open(path) as image:
            widths.append(int(image.width))
            heights.append(int(image.height))
    width = max(widths, default=0)
    height = max(heights, default=0)
    eligible = min(width, height) >= AUTO_ASSET_MIN_PERSON_MASTER_EDGE and width * height >= AUTO_ASSET_MIN_PERSON_MASTER_AREA
    return {
        "width": width,
        "height": height,
        "eligible_for_identity_master": eligible,
        "reason": "" if eligible else f"人物源裁剪仅 {width}x{height}，不足以作为可靠身份母版。",
    }


def _legacy_auto_asset_suspicions(cache: dict[str, Any], category: str, hint: str) -> list[dict[str, Any]]:
    legacy_group = cache.get("legacy", {}).get("people" if category == "person" else "scenes", {})
    candidate = legacy_group.get(hint) if isinstance(legacy_group, dict) else None
    if not isinstance(candidate, dict):
        return []
    return [
        {
            "category": category,
            "candidate_id": f"legacy:{hint}",
            "decision": "uncertain",
            "confidence": 0.0,
            "reason": "旧缓存缺少源图观察，不能进行源图对源图校验。",
            "path": str(candidate.get("path") or ""),
        }
    ]


def _uncertain_match_needs_review(verdict: dict[str, Any]) -> bool:
    """Whether an ``uncertain`` cache comparison is a genuine identity ambiguity.

    Only a comparison that had usable evidence on both sides and still landed in the
    middle should hold up the paid pipeline for human review. A verdict is uncertain
    for two non-signal reasons that must NOT block: the candidate frame has no
    comparable subject (``missing_fields`` populated), or the request failed and the
    verifier fabricated an uncertain result (confidence at/under the different floor).
    """
    if str(verdict.get("decision") or "") != "uncertain":
        return False
    if _as_string_list(verdict.get("missing_fields")):
        return False
    try:
        confidence = float(verdict.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence > AUTO_ASSET_DIFFERENT_THRESHOLD


def _find_person_cache_match(
    cache: dict[str, Any],
    *,
    person: dict[str, Any],
    source_image: Any,
    model: str,
    reuse_threshold: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool]:
    people = cache.get("people", {})
    by_id = people.get("by_id") if isinstance(people, dict) else {}
    by_id = by_id if isinstance(by_id, dict) else {}
    index = people.get("candidate_index") if isinstance(people, dict) else {}
    index = index if isinstance(index, dict) else {}
    identity_key = str(person.get("identity_key") or "")
    suspected = _legacy_auto_asset_suspicions(cache, "person", identity_key)
    uncertain_candidate_found = False
    for person_id in _candidate_ids(index, identity_key, by_id):
        candidate = by_id.get(person_id)
        if not isinstance(candidate, dict):
            continue
        converted_path = _asset_path_if_file(candidate.get("converted_path"))
        observations = _source_observation_paths(candidate)
        if converted_path is None or not observations:
            suspected.append(
                {
                    "category": "person",
                    "candidate_id": person_id,
                    "decision": "uncertain",
                    "confidence": 0.0,
                    "reason": "候选缺少转换图或源图观察。",
                }
            )
            continue
        verdict = _verify_auto_asset_cache(
            source_image,
            _load_image_file(observations[0]),
            model=model,
            kind="人物",
            current_features=str(person.get("appearance") or ""),
            candidate_features=str(candidate.get("appearance") or ""),
            candidate_id=person_id,
            reuse_threshold=reuse_threshold,
        )
        decision = str(verdict.get("decision") or "uncertain")
        if decision == "same":
            return candidate, suspected, True
        if decision == "uncertain":
            needs_review = _uncertain_match_needs_review(verdict)
            if needs_review:
                uncertain_candidate_found = True
            suspected.append(
                {
                    "category": "person",
                    "candidate_id": person_id,
                    "decision": decision,
                    "confidence": verdict.get("confidence", 0.0),
                    "reason": "; ".join(verdict.get("reasons", [])) or "人物相似但不足以自动复用。",
                    "hard_mismatches": verdict.get("hard_mismatches", []),
                    "missing_fields": verdict.get("missing_fields", []),
                    "needs_review": needs_review,
                }
            )
    return None, suspected, uncertain_candidate_found


def _find_scene_cache_match(
    cache: dict[str, Any],
    *,
    scene_key: str,
    description: str,
    source_frame: Any,
    model: str,
    reuse_threshold: float,
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]]]:
    scenes = cache.get("scenes", {})
    by_id = scenes.get("by_id") if isinstance(scenes, dict) else {}
    by_id = by_id if isinstance(by_id, dict) else {}
    index = scenes.get("candidate_index") if isinstance(scenes, dict) else {}
    index = index if isinstance(index, dict) else {}
    suspected = _legacy_auto_asset_suspicions(cache, "scene", scene_key)
    matched_place_id: str | None = None
    for place_id in _candidate_ids(index, scene_key, by_id):
        place = by_id.get(place_id)
        if not isinstance(place, dict):
            continue
        versions = place.get("versions") if isinstance(place.get("versions"), dict) else {}
        for version_id, version in versions.items():
            if not isinstance(version, dict):
                continue
            converted_path = _asset_path_if_file(version.get("converted_path"))
            observations = _source_observation_paths(version)
            if converted_path is None or not observations:
                suspected.append(
                    {
                        "category": "scene",
                        "candidate_id": f"{place_id}:{version_id}",
                        "decision": "uncertain",
                        "confidence": 0.0,
                        "reason": "候选缺少转换图或源图观察。",
                    }
                )
                continue
            verdict = _verify_auto_asset_cache(
                source_frame,
                _load_image_file(observations[0]),
                model=model,
                kind="场景",
                current_features=description,
                candidate_features=str(version.get("description") or place.get("description") or ""),
                candidate_id=f"{place_id}:{version_id}",
                reuse_threshold=reuse_threshold,
            )
            decision = str(verdict.get("decision") or "uncertain")
            if decision == "same":
                matched = dict(version)
                matched["place_id"] = place_id
                return matched, place_id, suspected
            if decision == "same_place_new_version":
                matched_place_id = place_id
                continue
            if decision == "uncertain":
                suspected.append(
                    {
                        "category": "scene",
                        "candidate_id": f"{place_id}:{version_id}",
                        "decision": decision,
                        "confidence": verdict.get("confidence", 0.0),
                        "reason": "; ".join(verdict.get("reasons", [])) or "场景相似但不足以自动复用。",
                        "hard_mismatches": verdict.get("hard_mismatches", []),
                        "missing_fields": verdict.get("missing_fields", []),
                    }
                )
    return None, matched_place_id, suspected


def _scene_cache_place(cache: dict[str, Any], place_id: str) -> dict[str, Any] | None:
    scenes = cache.get("scenes") if isinstance(cache.get("scenes"), dict) else {}
    places = scenes.get("by_id") if isinstance(scenes.get("by_id"), dict) else {}
    value = places.get(str(place_id or ""))
    return value if isinstance(value, dict) else None


def _scene_cache_version(cache: dict[str, Any], place_id: str, view_id: str) -> dict[str, Any] | None:
    place = _scene_cache_place(cache, place_id)
    versions = place.get("versions") if isinstance(place, dict) and isinstance(place.get("versions"), dict) else {}
    value = versions.get(str(view_id or ""))
    return value if isinstance(value, dict) else None


def _scene_style_master_path(cache: dict[str, Any], place_id: str) -> Path | None:
    place = _scene_cache_place(cache, place_id)
    if not isinstance(place, dict):
        return None
    direct = _asset_path_if_file(place.get("style_master_path"))
    if direct is not None:
        return direct
    versions = place.get("versions") if isinstance(place.get("versions"), dict) else {}
    for version in versions.values():
        if isinstance(version, dict):
            path = _asset_path_if_file(version.get("converted_path"))
            if path is not None:
                return path
    return None


def _write_auto_asset_manifest(root: Path, task: dict[str, Any]) -> None:
    assets = task.get("auto_assets", {})
    _atomic_write_json(
        root / "manifest.json",
        {
            "version": 1,
            "task_index": task.get("index"),
            "source_start": task.get("source_start"),
            "source_duration": task.get("source_duration"),
            "status": task.get("auto_asset_status"),
            "analysis": task.get("auto_asset_analysis"),
            "analysis_attempts": int(task.get("auto_asset_analysis_attempts") or 0),
            "assets": assets,
            "errors": task.get("auto_asset_errors", []),
            "warnings": task.get("auto_asset_warnings", []),
        },
    )


def _valid_auto_asset_entries(assets: dict[str, Any], category: str, key: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    values = assets.get(category, []) if isinstance(assets, dict) else []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get(key) or "")
        if identifier and (_asset_path_if_file(item.get("path")) or _active_person_asset_id(item)):
            entries[identifier] = item
    return entries


def _approved_integrated_frames(assets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    approved: dict[str, dict[str, Any]] = {}
    values = assets.get("integrated_frames", []) if isinstance(assets, dict) else []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        if role and quality.get("verdict") == "approved" and _asset_path_if_file(item.get("path")):
            approved[role] = item
    return approved


def _auto_asset_task_complete(
    task: dict[str, Any],
    root: Path,
    *,
    reuse_threshold: float,
    require_integrated_frames: bool = False,
) -> bool:
    if task.get("auto_asset_status") != "ready" or task.get("auto_asset_errors"):
        return False
    if not _auto_asset_task_masters_complete(task, root, reuse_threshold=reuse_threshold):
        return False
    if not require_integrated_frames:
        return True
    assets = task.get("auto_assets")
    integrated = _approved_integrated_frames(assets)
    expected_scenes = _auto_asset_expected_scene_roles(task, reuse_threshold=reuse_threshold)
    expected_frames = ["frame_start"]
    if len(expected_scenes) > 1:
        expected_frames.append("frame_end")
    return all(role in integrated for role in expected_frames)


def _auto_asset_expected_scene_roles(task: dict[str, Any], *, reuse_threshold: float) -> list[str]:
    analysis = task.get("auto_asset_analysis")
    assets = task.get("auto_assets")
    if not isinstance(analysis, dict):
        return ["scene_start"]
    background = analysis.get("background") if isinstance(analysis.get("background"), dict) else {}
    if (
        analysis.get("shot_stability") in {"transition", "uncertain"}
        or float(background.get("same_scene_confidence", 0.0)) < reuse_threshold
    ):
        return ["scene_start", "scene_end"]
    return ["scene_start"]


def _auto_asset_task_masters_complete(
    task: dict[str, Any],
    root: Path,
    *,
    reuse_threshold: float,
) -> bool:
    analysis = task.get("auto_asset_analysis")
    assets = task.get("auto_assets")
    if not isinstance(analysis, dict) or not isinstance(assets, dict) or not (root / "manifest.json").is_file():
        return False
    if not _saved_source_frame_paths(root, assets):
        return False
    people = _valid_auto_asset_entries(assets, "people", "slot")
    ignored_people = {
        str(item.get("slot") or "")
        for item in assets.get("people", []) if isinstance(assets.get("people"), list)
        if isinstance(item, dict) and str(item.get("identity_state") or "") in {"partial", "ignored"}
    }
    scenes = _valid_auto_asset_entries(assets, "scenes", "role")
    expected_people = [] if analysis.get("shot_stability") == "crowded" else [
        str(item.get("slot") or "")
        for item in analysis.get("people", [])
        if (
            isinstance(item, dict)
            and item.get("slot")
            and str(item.get("slot") or "") not in ignored_people
        )
    ]
    expected_scenes = _auto_asset_expected_scene_roles(task, reuse_threshold=reuse_threshold)
    return all(slot in people for slot in expected_people) and all(role in scenes for role in expected_scenes)


def _checkpoint_auto_asset_task(
    job: LongVideoJob,
    root: Path,
    task: dict[str, Any],
    *,
    source_frames: dict[str, str],
    analysis: dict[str, Any] | None,
    people: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    status: str,
    suspected_matches: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    integrated_frames: list[dict[str, Any]] | None = None,
) -> None:
    if analysis is not None:
        task["auto_asset_analysis"] = analysis
    previous_assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
    if integrated_frames is None:
        integrated_frames = [
            dict(item)
            for item in previous_assets.get("integrated_frames", [])
            if isinstance(item, dict)
        ]
    task["auto_assets"] = {
        "prompt_version": AUTO_ASSET_PROMPT_VERSION,
        "cache_version": AUTO_ASSET_CACHE_VERSION,
        "source_frames": source_frames,
        "people": people,
        "scenes": scenes,
        "integrated_frames": integrated_frames,
        "suspected_matches": suspected_matches or [],
    }
    task["auto_asset_suspected_matches"] = suspected_matches or []
    task["auto_asset_errors"] = errors
    task["auto_asset_warnings"] = warnings or []
    task["auto_asset_status"] = status
    _write_auto_asset_manifest(root, task)
    _atomic_write_json(job.manifest_path, job.manifest)


def _auto_asset_progress_path(job: LongVideoJob) -> Path:
    return job.job_dir / "progress.json"


def _auto_asset_status_counts(job: LongVideoJob) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in _auto_asset_member_tasks(job):
        if not isinstance(task, dict):
            continue
        status = str(task.get("auto_asset_status") or "planned")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _send_auto_asset_progress_event(payload: dict[str, Any]) -> None:
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        if context is not None:
            payload["node"] = context.node_id
            payload["prompt_id"] = context.prompt_id
    except Exception:
        logging.debug("Unable to read ComfyUI executing context for auto asset progress.", exc_info=True)

    try:
        from server import PromptServer

        instance = PromptServer.instance
        send_sync = getattr(instance, "send_sync", None)
        if callable(send_sync):
            client_id = getattr(instance, "client_id", None)
            send_sync(AUTO_ASSET_PROGRESS_EVENT, payload, client_id)
    except Exception:
        logging.debug("Unable to send auto asset progress event.", exc_info=True)


_LOCAL_EVENT_PATH_PATTERN = re.compile(
    r"(?i)(?:[a-z]:[\\/]|/(?:mnt|home|tmp|var|private|users|workspace|output)/)[^\s\"'`，,;；:：]+"
)


def _public_auto_asset_progress_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip local-only fields before a progress event leaves the backend."""
    private_keys = {"manifest", "progress_path", "path", "source_path", "result"}

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items() if key not in private_keys}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, str):
            return _LOCAL_EVENT_PATH_PATTERN.sub("[本机路径]", value)
        return value

    return sanitize(payload)


def _emit_auto_asset_progress(
    job: LongVideoJob,
    progress_bar: ProgressBar | None,
    *,
    value: float,
    total: int,
    phase: str,
    message: str,
    task: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = max(1, int(total))
    value = min(max(0.0, float(value)), float(total))
    percent = round((value / total) * 100.0, 1)
    if progress_bar is not None:
        progress_bar.update_absolute(value, total)

    payload: dict[str, Any] = {
        "event": AUTO_ASSET_PROGRESS_EVENT,
        "job_id": str(job.manifest.get("job_id") or ""),
        "status": str(job.manifest.get("status") or ""),
        "manifest": str(job.manifest_path),
        "progress_path": str(_auto_asset_progress_path(job)),
        "value": round(value, 4),
        "total": total,
        "percent": percent,
        "phase": phase,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "auto_asset_status_counts": _auto_asset_status_counts(job),
    }
    if task is not None:
        payload.update(
            {
                "task_index": int(task.get("index", 0)),
                "task_start": float(task.get("start", 0.0)),
                "task_duration": float(task.get("duration", 0.0)),
                "task_status": str(task.get("auto_asset_status") or "planned"),
            }
        )
    if extra:
        payload["extra"] = extra
        preview = extra.get("preview") if isinstance(extra, dict) else None
        if isinstance(preview, dict):
            payload["preview"] = preview

    job.manifest["auto_asset_progress"] = payload
    try:
        _atomic_write_json(_auto_asset_progress_path(job), payload)
    except Exception:
        logging.warning("Unable to write auto asset progress file.", exc_info=True)
    # The manifest keeps absolute paths for local resume, but WebSocket clients only
    # receive `/view`-compatible preview descriptors and must not learn those paths.
    _send_auto_asset_progress_event(_public_auto_asset_progress_payload(payload))
    logging.info("[company_remote] 自动资产进度 %.1f%%：%s", percent, message)
    return payload


def _retry_auto_asset_creation(factory: Any, *, max_retries: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    last_error: dict[str, Any] | None = None
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, attempts + 1):
        entry, error = factory()
        if entry is not None:
            entry["generation_attempts"] = attempt
            return entry, None
        last_error = dict(error or {"message": "自动资产生成没有返回结果。"})
        last_error["attempts"] = attempt
        if last_error.get("error_kind") == "provider_policy" or attempt >= attempts:
            break
        time.sleep(min(2.0 * attempt, 6.0))
    return None, last_error


def _retry_auto_asset_analysis(
    factory: Any,
    *,
    max_retries: int,
    on_retry: Any | None = None,
) -> tuple[dict[str, Any], int]:
    total_attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            return factory(), attempt
        except Exception as exc:
            if attempt >= total_attempts or not _analysis_error_is_recoverable(exc):
                raise
            delay = min(float(attempt), 3.0)
            if on_retry is not None:
                on_retry(attempt, total_attempts, delay, exc)
            time.sleep(delay)
    raise RuntimeError("自动资产分析重试异常结束。")


def _local_gateway_health_url(config: Any) -> str:
    """Return the CLIProxyAPI health endpoint for loopback gpttext configurations."""
    parsed = urllib.parse.urlsplit(str(config.base_url or ""))
    if (parsed.hostname or "").lower() not in {"localhost", "127.0.0.1", "::1"}:
        return ""
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/healthz", "", ""))


def _preflight_auto_asset_analysis_gateway(job: LongVideoJob) -> dict[str, Any]:
    """Verify that text analysis can run before any asset or video request is submitted."""
    config = _get_config("gpttext")
    health_url = _local_gateway_health_url(config)
    checked_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    preflight = {
        "status": "checking",
        "config_name": str(config.name),
        "model": str(job.ai_model),
        "health_url": health_url,
        "checked_at": checked_at,
    }
    job.manifest["analysis_gateway_preflight"] = preflight
    _atomic_write_json(job.manifest_path, job.manifest)

    try:
        if health_url:
            session = requests.Session()
            session.trust_env = False
            response = session.get(health_url, timeout=ANALYSIS_GATEWAY_HEALTH_TIMEOUT_SECONDS)
            if not response.ok:
                raise CompanyRemoteAPIError(f"本机分析网关健康检查失败（HTTP {response.status_code}）。")
            try:
                health = response.json()
            except ValueError as exc:
                raise CompanyRemoteAPIError("本机分析网关健康检查未返回 JSON。") from exc
            if not isinstance(health, dict) or str(health.get("status") or "").lower() != "ok":
                raise CompanyRemoteAPIError("本机分析网关健康检查未返回 status=ok。")

        probe_config = replace(
            config,
            timeout_seconds=min(int(config.timeout_seconds), ANALYSIS_GATEWAY_PROBE_TIMEOUT_SECONDS),
        )
        probe = generate_openai_chat_text(
            probe_config,
            skill="你是连接检测器。只回复 OK。",
            user_prompt="回复 OK。",
            model=job.ai_model,
            temperature=0.0,
            max_tokens=4,
        ).strip()
        if not probe:
            raise CompanyRemoteAPIError("分析网关探测未返回文本。")
    except Exception as exc:
        preflight.update({"status": "failed", "error": str(exc), "checked_at": checked_at})
        job.manifest["analysis_gateway_preflight"] = preflight
        _atomic_write_json(job.manifest_path, job.manifest)
        raise AnalysisGatewayUnavailableError(
            "自动资产分析服务不可用：请确认 CLIProxyAPI 已在 8317 启动并检查账号授权。"
        ) from exc

    preflight.update({"status": "ready", "checked_at": checked_at})
    job.manifest["analysis_gateway_preflight"] = preflight
    _atomic_write_json(job.manifest_path, job.manifest)
    return preflight


def _mark_analysis_gateway_unavailable(
    job: LongVideoJob,
    context: dict[str, Any],
    error: BaseException,
) -> None:
    """Checkpoint a preflight failure without creating failed logical-shot assets."""
    job.manifest["status"] = "analysis_gateway_unavailable"
    job.manifest.setdefault("pipeline", {})["error"] = str(error)
    job.manifest["pipeline"]["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    total = max(1, len(_auto_asset_member_tasks(job)))
    _emit_auto_asset_progress(
        job,
        None,
        value=0,
        total=total,
        phase="analysis_gateway_unavailable",
        message="自动资产分析服务不可用，尚未提交图片或 Seedance 视频请求。",
        extra={"error": str(error), "image_request_count": 0, "video_request_count": 0},
    )
    state = context.get("state")
    if state is not None:
        state["status"] = "failed"
        state["current_batch"]["status"] = "failed"
        state["current_batch"]["last_error"] = str(error)
        _manual_batch_write_state(context["state_path"], state)
    _atomic_write_json(job.manifest_path, job.manifest)


def _publish_auto_person_asset(
    entry: dict[str, Any],
    *,
    task_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Publish one generated person image and return ``(warning, blocking_error)``."""
    try:
        path = _asset_path_if_file(entry.get("path"))
        if path is None:
            raise ValueError("人物转绘图不存在，不能上传到 TOS。")
        publication = publish_seedance_person_image(
            _load_image_file(path),
            character_label=f"第 {task_index} 段人物 {entry.get('slot') or '未命名'}",
            reuse_cached=True,
        )
    except Exception as exc:
        entry["publication"] = {
            "tos": {"status": "failed", "object_key": ""},
            "asset_library": {"status": "skipped", "asset_id": "", "error": str(exc)},
        }
        return None, {
            "kind": "person",
            "slot": str(entry.get("slot") or ""),
            "error_kind": "tos_upload_failed",
            "message": f"人物转绘图上传 TOS 失败：{exc}",
        }

    entry["publication"] = publication
    library = publication.get("asset_library") if isinstance(publication, dict) else {}
    if str(library.get("status") or "") == "warning":
        return {
            "kind": "person_asset_library",
            "slot": str(entry.get("slot") or ""),
            "message": str(library.get("error") or "素材库登记未完成。"),
        }, None
    return None, None


def _merge_auto_asset_cache_entry(cache: dict[str, Any], entry: dict[str, Any]) -> None:
    update = entry.get("cache_update")
    if not isinstance(update, dict):
        return
    normalized = _normalize_auto_asset_cache(cache)
    cache.clear()
    cache.update(normalized)
    if update.get("category") == "person":
        person_id = str(update.get("person_id") or "")
        if not person_id:
            return
        people = cache["people"].setdefault("by_id", {})
        record = people.setdefault(
            person_id,
            {
                "person_id": person_id,
                "identity_keys": [],
                "appearance": str(update.get("appearance") or ""),
                "converted_path": str(update.get("converted_path") or ""),
                "source_observations": [],
                "suspected_candidates": [],
            },
        )
        if not record.get("converted_path"):
            record["converted_path"] = str(update.get("converted_path") or "")
        if update.get("appearance"):
            record["appearance"] = str(update["appearance"])
        identity_key = str(update.get("identity_key") or "")
        if identity_key:
            identity_keys = record.setdefault("identity_keys", [])
            if identity_key not in identity_keys:
                identity_keys.append(identity_key)
            _append_candidate_index(cache, "person", identity_key, person_id)
        _append_source_observations(record, update.get("source_observations") or [])
        if isinstance(entry.get("publication"), dict):
            record["publication"] = deepcopy(entry["publication"])
        if isinstance(update.get("publication"), dict):
            record["publication"] = deepcopy(update["publication"])
        suspected = record.setdefault("suspected_candidates", [])
        if isinstance(suspected, list):
            suspected.extend(item for item in update.get("suspected_candidates", []) if isinstance(item, dict))
        return
    if update.get("category") == "scene":
        place_id = str(update.get("place_id") or "")
        version_id = str(update.get("version_id") or "")
        if not place_id or not version_id:
            return
        places = cache["scenes"].setdefault("by_id", {})
        place = places.setdefault(
            place_id,
            {
                "place_id": place_id,
                "scene_keys": [],
                "description": str(update.get("description") or ""),
                "style_master_path": str(update.get("style_master_path") or update.get("converted_path") or ""),
                "versions": {},
                "suspected_candidates": [],
            },
        )
        if update.get("description"):
            place["description"] = str(update["description"])
        if update.get("style_master_path") or not place.get("style_master_path"):
            place["style_master_path"] = str(update.get("style_master_path") or update.get("converted_path") or "")
        scene_key = str(update.get("scene_key") or "")
        if scene_key:
            scene_keys = place.setdefault("scene_keys", [])
            if scene_key not in scene_keys:
                scene_keys.append(scene_key)
            _append_candidate_index(cache, "scene", scene_key, place_id)
        versions = place.setdefault("versions", {})
        version = versions.setdefault(
            version_id,
            {
                "version_id": version_id,
                "view_id": version_id,
                "scene_key": scene_key,
                "description": str(update.get("description") or ""),
                "converted_path": str(update.get("converted_path") or ""),
                "source_observations": [],
            },
        )
        if not version.get("converted_path"):
            version["converted_path"] = str(update.get("converted_path") or "")
        version["view_id"] = version_id
        if update.get("description"):
            version["description"] = str(update["description"])
        _append_source_observations(version, update.get("source_observations") or [])
        suspected = place.setdefault("suspected_candidates", [])
        if isinstance(suspected, list):
            suspected.extend(item for item in update.get("suspected_candidates", []) if isinstance(item, dict))


def _ordered_auto_asset_entries(
    entries: dict[str, dict[str, Any]],
    order: list[str],
) -> list[dict[str, Any]]:
    return [entries[key] for key in order if key in entries]


def _create_auto_person_asset(
    *,
    root: Path,
    person: dict[str, Any],
    frames: torch.Tensor,
    cache: dict[str, Any],
    analysis_model: str,
    image_model: str,
    image_quality: str,
    image_provider: str,
    reuse_threshold: float,
    visual_style: str = AUTO_ASSET_STYLE_WESTERN,
    style_prompt: str = "",
    enforce_identity_gate: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_images: list[Any] = []
    if person.get("first_bbox"):
        source_images.append(_crop_person_frame(frames[0], person["first_bbox"]))
    if person.get("last_bbox"):
        source_images.append(_crop_person_frame(frames[2], person["last_bbox"]))
    if not source_images:
        return None, {"kind": "person", "slot": person["slot"], "message": "人物没有可用裁剪框。"}
    source_paths, source_images = _save_person_source_observations(source_images, root, str(person["slot"]))
    source_observations = _source_observation_entries(
        source_paths,
        kind="person_crop",
        description=str(person.get("appearance") or ""),
    )
    source_resolution = _person_master_resolution(source_paths)

    mapped_asset_id = str(person.get("mapped_asset_id") or "").strip()
    mapped_asset_path = _asset_path_if_file(person.get("mapped_asset_path"))
    mapped_global_id = str(person.get("global_person_id") or "").strip()
    if mapped_global_id and mapped_asset_id:
        publication = {
            "tos": {"status": "already_registered", "object_key": ""},
            "asset_library": {"status": "active", "asset_id": mapped_asset_id, "cache_reused": True},
        }
        return {
            "slot": person["slot"],
            "identity_key": person["identity_key"],
            "person_id": mapped_global_id,
            "global_person_id": mapped_global_id,
            "identity_state": "linked",
            "identity_reason": "使用人工确认的火山素材 asset_id。",
            "appearance": person["appearance"],
            "path": str(mapped_asset_path) if mapped_asset_path is not None else "",
            "source_observations": source_paths,
            "source_resolution": source_resolution,
            "reused_from_cache": True,
            "publication": publication,
            "suspected_matches": [],
        }, None

    # reuse_confidence only describes the current analysis evidence. The cache
    # verifier still needs a chance to recognize the same recurring character
    # when wardrobe, pose, or shot framing changed.
    suspected_matches: list[dict[str, Any]] = []
    candidate, suspected_matches, uncertain_candidate_found = _find_person_cache_match(
        cache,
        person=person,
        source_image=source_images[0],
        model=analysis_model,
        reuse_threshold=reuse_threshold,
    )
    if candidate is not None:
        candidate_path = _asset_path_if_file(candidate.get("converted_path"))
        if candidate_path is not None:
            person_id = str(candidate.get("person_id") or "")
            cache_update = {
                "category": "person",
                "person_id": person_id,
                "identity_key": str(person.get("identity_key") or ""),
                "appearance": str(person.get("appearance") or ""),
                "converted_path": str(candidate_path),
                "source_observations": source_observations,
                "suspected_candidates": suspected_matches,
                "publication": deepcopy(candidate.get("publication"))
                if isinstance(candidate.get("publication"), dict)
                else None,
            }
            return {
                "slot": person["slot"],
                "identity_key": person["identity_key"],
                "person_id": person_id,
                "global_person_id": person_id,
                "identity_state": "linked",
                "identity_reason": "已通过原始观察图核验并复用固定人物素材。",
                "appearance": person["appearance"],
                "path": str(candidate_path),
                "source_observations": source_paths,
                "source_resolution": source_resolution,
                "reused_from_cache": True,
                "cache_update": cache_update,
                "publication": deepcopy(candidate.get("publication"))
                if isinstance(candidate.get("publication"), dict)
                else None,
                "suspected_matches": suspected_matches,
            }, None

    evidence_confidence = float(person.get("reuse_confidence", 0.0) or 0.0)
    eligible_for_master = bool(source_resolution.get("eligible_for_identity_master"))
    if enforce_identity_gate and (
        not eligible_for_master or evidence_confidence < AUTO_ASSET_MIN_NEW_PERSON_CONFIDENCE
    ):
        reason = str(source_resolution.get("reason") or "").strip()
        if evidence_confidence < AUTO_ASSET_MIN_NEW_PERSON_CONFIDENCE:
            reason = (
                f"人物可见证据置信度仅 {evidence_confidence:.2f}，低于首次建立人物母版所需的 "
                f"{AUTO_ASSET_MIN_NEW_PERSON_CONFIDENCE:.2f}。"
            )
        return {
            "slot": person["slot"],
            "identity_key": person["identity_key"],
            "person_id": "",
            "global_person_id": "",
            "appearance": person["appearance"],
            "path": "",
            "source_observations": source_paths,
            "source_resolution": source_resolution,
            "reused_from_cache": False,
            "identity_state": "partial",
            "identity_reason": reason or "只有局部、背影或远景证据，不建立独立人物素材。",
            "suspected_matches": suspected_matches,
        }, None
    if enforce_identity_gate and uncertain_candidate_found:
        return None, {
            "kind": "person_identity",
            "slot": str(person.get("slot") or ""),
            "error_kind": "identity_review_required",
            "identity_state": "unresolved",
            "message": "人物与已有素材存在候选关系，但原始画面证据不足以自动确认；已停止新建人物素材，等待人工映射。",
            "suspected_matches": suspected_matches,
            "source_observations": source_paths,
            "source_resolution": source_resolution,
        }

    try:
        image = _generate_auto_asset_image(
            source_images,
            prompt=_person_asset_prompt(person, visual_style=visual_style, style_prompt=style_prompt),
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
        )
        path = root / "people" / f"{person['slot']}.png"
        _save_image_tensor(image, path)
        person_id = _asset_record_id(root.name, person["slot"], person.get("identity_key") or "person")
        cache_update = {
            "category": "person",
            "person_id": person_id,
            "identity_key": str(person.get("identity_key") or ""),
            "appearance": str(person.get("appearance") or ""),
            "converted_path": str(path),
            "source_observations": source_observations,
            "suspected_candidates": suspected_matches,
        }
        entry = {
            "slot": person["slot"],
            "identity_key": person["identity_key"],
            "person_id": person_id,
            "global_person_id": person_id,
            "identity_state": "confirmed",
            "identity_reason": "清晰原始观察首次建立固定人物素材。",
            "appearance": person["appearance"],
            "path": str(path),
            "source_observations": source_paths,
            "source_resolution": source_resolution,
            "reused_from_cache": False,
            "cache_update": cache_update,
            "suspected_matches": suspected_matches,
        }
        return entry, None
    except Exception as exc:
        return None, {
            "kind": "person",
            "slot": person["slot"],
            "message": str(exc),
            "error_kind": "provider_policy" if _provider_policy_error(exc) else "generation_failed",
        }


def _create_auto_scene_asset(
    *,
    root: Path,
    role: str,
    source_frame: Any,
    source_frame_path: Path,
    description: str,
    scene_key: str,
    cache: dict[str, Any],
    analysis_model: str,
    image_model: str,
    image_quality: str,
    image_provider: str,
    reuse_threshold: float,
    allow_cache: bool,
    visual_style: str = AUTO_ASSET_STYLE_WESTERN,
    style_prompt: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_observations = _source_observation_entries([str(source_frame_path)], kind=role, description=description)
    suspected_matches: list[dict[str, Any]] = []
    matched_place_id: str | None = None
    if allow_cache:
        candidate, matched_place_id, suspected_matches = _find_scene_cache_match(
            cache,
            scene_key=scene_key,
            description=description,
            source_frame=source_frame,
            model=analysis_model,
            reuse_threshold=reuse_threshold,
        )
        if candidate is not None:
            candidate_path = _asset_path_if_file(candidate.get("converted_path"))
            if candidate_path is not None:
                place_id = str(matched_place_id or candidate.get("place_id") or "")
                version_id = str(candidate.get("version_id") or "")
                cache_update = {
                    "category": "scene",
                    "place_id": place_id,
                    "version_id": version_id,
                    "view_id": version_id,
                    "scene_key": scene_key,
                    "description": description,
                    "converted_path": str(candidate_path),
                    "source_observations": source_observations,
                    "suspected_candidates": suspected_matches,
                }
                return {
                    "role": role,
                    "description": description,
                    "scene_key": scene_key,
                    "place_id": place_id,
                    "version_id": version_id,
                    "view_id": str(candidate.get("view_id") or version_id),
                    "path": str(candidate_path),
                    "style_master_path": str(candidate.get("style_master_path") or candidate_path),
                    "integrated_frame_path": str(candidate.get("integrated_frame_path") or ""),
                    "integrated_frame_quality": candidate.get("integrated_frame_quality")
                    if isinstance(candidate.get("integrated_frame_quality"), dict)
                    else {},
                    "source_observations": [str(source_frame_path)],
                    "reused_from_cache": True,
                    "cache_update": cache_update,
                    "suspected_matches": suspected_matches,
                }, None

        # A different camera/time view of the same physical place can use the
        # established replacement environment as-is. The final integrated frame
        # still follows this shot's composition, so another scene image request
        # is unnecessary.
        same_place_master = _scene_style_master_path(cache, matched_place_id or "")
        if same_place_master is not None:
            place_id = str(matched_place_id or "")
            version_id = f"{place_id}:replacement_master"
            cache_update = {
                "category": "scene",
                "place_id": place_id,
                "version_id": version_id,
                "view_id": version_id,
                "scene_key": scene_key,
                "description": description,
                "converted_path": str(same_place_master),
                "style_master_path": str(same_place_master),
                "source_observations": source_observations,
                "suspected_candidates": suspected_matches,
            }
            return {
                "role": role,
                "description": description,
                "scene_key": scene_key,
                "place_id": place_id,
                "version_id": version_id,
                "view_id": version_id,
                "path": str(same_place_master),
                "style_master_path": str(same_place_master),
                "source_observations": [str(source_frame_path)],
                "reused_from_cache": True,
                "reuse_scope": "same_place",
                "cache_update": cache_update,
                "suspected_matches": suspected_matches,
            }, None

    try:
        style_master_path = _scene_style_master_path(cache, matched_place_id or "")
        image_inputs = [source_frame]
        if style_master_path is not None:
            image_inputs.append(_load_image_file(style_master_path))
        image = _generate_auto_asset_image(
            image_inputs,
            prompt=_scene_asset_prompt(
                description,
                position="开始" if role == "scene_start" else "结束",
                visual_style=visual_style,
                style_prompt=style_prompt,
            ),
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
        )
        path = root / "scenes" / f"{role}.png"
        _save_image_tensor(image, path)
        place_id = matched_place_id or _asset_record_id(root.name, scene_key, "place")
        version_id = _asset_record_id(root.name, role, scene_key)
        cache_update = {
            "category": "scene",
            "place_id": place_id,
            "version_id": version_id,
            "scene_key": scene_key,
            "description": description,
            "converted_path": str(path),
            "style_master_path": str(style_master_path or path),
            "source_observations": source_observations,
            "suspected_candidates": suspected_matches,
        }
        entry = {
            "role": role,
            "description": description,
            "scene_key": scene_key,
            "place_id": place_id,
            "version_id": version_id,
            "view_id": version_id,
            "path": str(path),
            "style_master_path": str(style_master_path or path),
            "style_master_reused": style_master_path is not None,
            "source_observations": [str(source_frame_path)],
            "reused_from_cache": False,
            "cache_update": cache_update,
            "suspected_matches": suspected_matches,
        }
        return entry, None
    except Exception as exc:
        return None, {
            "kind": "scene",
            "role": role,
            "message": str(exc),
            "error_kind": "provider_policy" if _provider_policy_error(exc) else "generation_failed",
        }


def _integrated_frame_specs(task: dict[str, Any]) -> list[dict[str, Any]]:
    assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
    source_frames = assets.get("source_frames") if isinstance(assets.get("source_frames"), dict) else {}
    scenes = {
        str(item.get("role") or ""): item
        for item in assets.get("scenes", [])
        if isinstance(item, dict)
    }
    specs: list[dict[str, Any]] = []
    start_source = _asset_path_if_file(source_frames.get("source_start"))
    start_scene = scenes.get("scene_start")
    if start_source is not None and isinstance(start_scene, dict):
        specs.append({"role": "frame_start", "source_path": start_source, "scene": start_scene})
    end_source = _asset_path_if_file(source_frames.get("source_end"))
    end_scene = scenes.get("scene_end")
    if end_source is not None and isinstance(end_scene, dict):
        specs.append({"role": "frame_end", "source_path": end_source, "scene": end_scene})
    return specs


def _eligible_person_master_entries(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
    eligible: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for item in assets.get("people", []):
        if not isinstance(item, dict) or _asset_path_if_file(item.get("path")) is None:
            continue
        resolution = item.get("source_resolution") if isinstance(item.get("source_resolution"), dict) else {}
        if not resolution:
            resolution = _person_master_resolution(
                [str(value) for value in item.get("source_observations", []) if str(value or "").strip()]
            )
            item["source_resolution"] = resolution
        if bool(resolution.get("eligible_for_identity_master")):
            eligible.append(item)
        else:
            warnings.append(
                {
                    "kind": "person_master_too_small",
                    "slot": str(item.get("slot") or ""),
                    "message": str(resolution.get("reason") or "人物源裁剪过小，不作为身份母版。"),
                }
            )
    return eligible, warnings


def _integrated_frame_attempt(
    *,
    task: dict[str, Any],
    spec: dict[str, Any],
    root: Path,
    attempt: int,
    style_master_path: Path | None,
    analysis_model: str,
    image_model: str,
    image_quality: str,
    image_provider: str,
    visual_style: str,
    style_prompt: str,
    retry_reasons: list[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    role = str(spec["role"])
    source_path = Path(spec["source_path"])
    scene = spec["scene"]
    source_frame = _load_image_file(source_path)
    person_entries, _warnings = _eligible_person_master_entries(task)
    exact_integrated_path = _asset_path_if_file(scene.get("integrated_frame_path"))
    exact_quality = scene.get("integrated_frame_quality") if isinstance(scene.get("integrated_frame_quality"), dict) else {}
    if (
        attempt == 1
        and not (task.get("auto_asset_analysis") or {}).get("people")
        and exact_integrated_path is not None
        and exact_quality.get("verdict") == "approved"
    ):
        quality = dict(exact_quality)
        quality["reused_exact_view"] = True
        quality.setdefault("version", AUTO_ASSET_QUALITY_GATE_VERSION)
        return {
            "role": role,
            "path": str(exact_integrated_path),
            "source_path": str(source_path),
            "place_id": str(scene.get("place_id") or ""),
            "view_id": str(scene.get("view_id") or scene.get("version_id") or ""),
            "scene_master_path": str(scene.get("path") or ""),
            "style_master_path": str(scene.get("style_master_path") or ""),
            "person_master_ids": [],
            "generation_attempts": 0,
            "quality": quality,
            "reused_from_cache": True,
        }, None
    person_paths = [path for item in person_entries if (path := _asset_path_if_file(item.get("path"))) is not None]
    person_masters = [_load_image_file(path) for path in person_paths]
    scene_path = _asset_path_if_file(scene.get("path"))
    scene_master = _load_image_file(scene_path) if scene_path is not None else None
    normalized_style_path = style_master_path if style_master_path is not None and style_master_path.is_file() else None
    if normalized_style_path is not None and scene_path is not None:
        try:
            if normalized_style_path.resolve() == scene_path.resolve():
                normalized_style_path = None
        except OSError:
            pass
    style_master = _load_image_file(normalized_style_path) if normalized_style_path is not None else None
    inputs: list[Any] = [source_frame, *person_masters]
    if scene_master is not None:
        inputs.append(scene_master)
    if style_master is not None:
        inputs.append(style_master)
    try:
        candidate = _generate_auto_asset_image(
            inputs,
            prompt=_integrated_frame_prompt(
                task.get("auto_asset_analysis") or {},
                role=role,
                person_master_count=len(person_masters),
                has_scene_master=scene_master is not None,
                has_style_master=style_master is not None,
                visual_style=visual_style,
                style_prompt=style_prompt,
                retry_reasons=retry_reasons,
            ),
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
        )
        path = root / "integrated_frames" / f"{role}_attempt_{attempt:02d}.png"
        _save_image_tensor(candidate, path)
        quality = _evaluate_integrated_frame_quality(
            source_frame=source_frame,
            candidate=candidate,
            person_masters=person_masters,
            scene_master=scene_master,
            style_master=style_master,
            analysis=task.get("auto_asset_analysis") or {},
            model=analysis_model,
            visual_style=visual_style,
            style_prompt=style_prompt,
        )
        return {
            "role": role,
            "path": str(path),
            "source_path": str(source_path),
            "place_id": str(scene.get("place_id") or ""),
            "view_id": str(scene.get("view_id") or scene.get("version_id") or ""),
            "scene_master_path": str(scene_path or ""),
            "style_master_path": str(normalized_style_path or ""),
            "person_master_ids": [str(item.get("person_id") or item.get("slot") or "") for item in person_entries],
            "generation_attempts": attempt,
            "quality": quality,
            "reused_from_cache": False,
        }, None
    except Exception as exc:
        return None, {
            "kind": "integrated_frame",
            "role": role,
            "message": str(exc),
            "error_kind": "provider_policy" if _provider_policy_error(exc) else "generation_or_quality_failed",
            "attempts": attempt,
        }


def _adaptive_integrated_frame_floor(entries: list[dict[str, Any]]) -> float:
    approved_scores = [
        _score01((item.get("quality") or {}).get("target_style_score"))
        for item in entries
        if isinstance(item.get("quality"), dict) and item["quality"].get("verdict") == "approved"
    ]
    if not approved_scores:
        return AUTO_ASSET_TARGET_STYLE_THRESHOLD
    return max(AUTO_ASSET_TARGET_STYLE_THRESHOLD, max(approved_scores) - 0.08)


def _apply_integrated_frame_group_floor(entries: list[dict[str, Any]], floor: float) -> None:
    for entry in entries:
        quality = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
        score = _score01(quality.get("target_style_score"))
        if quality.get("verdict") == "approved" and score + 1e-9 < floor:
            quality["verdict"] = "retry"
            reasons = quality.setdefault("reasons", [])
            message = f"低于本组自适应风格下限（{score:.2f} < {floor:.2f}）"
            if message not in reasons:
                reasons.append(message)
            quality["adaptive_group_style_floor"] = floor


def _record_integrated_frame_cache(cache: dict[str, Any], entry: dict[str, Any], *, has_people: bool) -> None:
    if (entry.get("quality") or {}).get("verdict") != "approved":
        return
    path = _asset_path_if_file(entry.get("path"))
    place_id = str(entry.get("place_id") or "")
    view_id = str(entry.get("view_id") or "")
    if path is None or not place_id:
        return
    place = _scene_cache_place(cache, place_id)
    if isinstance(place, dict):
        current_style = _asset_path_if_file(place.get("style_master_path"))
        if current_style is None:
            place["style_master_path"] = str(path)
        version = _scene_cache_version(cache, place_id, view_id)
        if isinstance(version, dict) and not has_people:
            version["integrated_frame_path"] = str(path)
            version["integrated_frame_quality"] = dict(entry.get("quality") or {})
    scenes = cache.get("scenes") if isinstance(cache.get("scenes"), dict) else {}
    current_global = _asset_path_if_file(scenes.get("style_master_path"))
    quality = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
    score = _score01(quality.get("target_style_score"))
    current_score = _score01(scenes.get("style_master_quality_score"))
    if current_global is None or score > current_score + 1e-9:
        scenes["style_master_path"] = str(path)
        scenes["style_master_id"] = f"{place_id}:{view_id}"
        scenes["style_master_quality_score"] = score


def _build_integrated_frames(
    job: LongVideoJob,
    cache: dict[str, Any],
    *,
    image_concurrency: int,
    image_model: str,
    image_quality: str,
    image_provider: str,
    visual_style: str,
    style_prompt: str,
    progress_bar: ProgressBar | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    tasks = _auto_asset_member_tasks(job)
    work: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_specs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    entries_by_task: dict[int, dict[str, dict[str, Any]]] = {}
    warnings_by_task: dict[int, list[dict[str, Any]]] = {}
    errors_by_task: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        if not isinstance(task, dict) or task.get("auto_asset_status") not in {"ready", "masters_ready"}:
            continue
        task_index = int(task["index"])
        assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
        existing = _approved_integrated_frames(assets)
        entries_by_task[task_index] = dict(existing)
        _eligible, resolution_warnings = _eligible_person_master_entries(task)
        warnings_by_task[task_index] = [
            dict(item) for item in task.get("auto_asset_warnings", []) if isinstance(item, dict)
        ]
        warnings_by_task[task_index].extend(resolution_warnings)
        errors_by_task[task_index] = [
            dict(item)
            for item in task.get("auto_asset_errors", [])
            if isinstance(item, dict) and item.get("kind") != "integrated_frame"
        ]
        for spec in _integrated_frame_specs(task):
            all_specs.append((task, spec))
            if str(spec["role"]) not in existing:
                work.append((task, spec))

    generated_count = 0

    def emit_result(task: dict[str, Any], *, phase: str, message: str) -> None:
        if progress_bar is None:
            return
        assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
        integrated = list(entries_by_task.get(int(task["index"]), {}).values())
        _emit_auto_asset_progress(
            job,
            progress_bar,
            value=max(0.0, len(tasks) - 0.02),
            total=max(1, len(tasks)),
            phase=phase,
            message=message,
            task=task,
            extra={
                "preview": _auto_asset_preview_payload(
                    task,
                    source_frames=assets.get("source_frames", {}),
                    people=[dict(item) for item in assets.get("people", []) if isinstance(item, dict)],
                    scenes=[dict(item) for item in assets.get("scenes", []) if isinstance(item, dict)],
                    integrated_frames=integrated,
                )
            },
        )

    def run_attempt(
        task: dict[str, Any],
        spec: dict[str, Any],
        attempt: int,
        style_anchor: Path | None,
        reasons: list[str] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        entry, error = _integrated_frame_attempt(
            task=task,
            spec=spec,
            root=_auto_asset_root(job, task),
            attempt=attempt,
            style_master_path=style_anchor,
            analysis_model=job.ai_model,
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
            visual_style=visual_style,
            style_prompt=style_prompt,
            retry_reasons=reasons,
        )
        return task, spec, entry, error

    worker_count = max(1, len(work)) if image_concurrency == 0 else max(1, min(image_concurrency, len(work) or 1))
    # An approved frame belongs to one shot composition. Reusing it as a global
    # style reference leaks that shot's people, layout, and background into
    # unrelated shots. Each integrated request already has its own scene master
    # and target-style text, so keep the reference set shot-local.
    global_style_path = None
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="company-integrated") as executor:
        futures = [executor.submit(run_attempt, task, spec, 1, None, None) for task, spec in work]
        for future in as_completed(futures):
            task, spec, entry, error = future.result()
            generated_count += 1
            task_index = int(task["index"])
            role = str(spec["role"])
            if entry is not None:
                entries_by_task[task_index][role] = entry
                verdict = str((entry.get("quality") or {}).get("verdict") or "retry")
                emit_result(
                    task,
                    phase="integrated_frame_ready" if verdict == "approved" else "integrated_frame_retrying",
                    message=(
                        f"第 {task_index} 段{role}整帧转换已通过。"
                        if verdict == "approved"
                        else f"第 {task_index} 段{role}转换效果偏弱，将自动重试。"
                    ),
                )
            elif error is not None:
                errors_by_task[task_index].append(error)
                emit_result(task, phase="integrated_frame_failed", message=f"第 {task_index} 段{role}整帧转换失败。")

    all_entries = [entry for values in entries_by_task.values() for entry in values.values()]
    floor = _adaptive_integrated_frame_floor(all_entries)
    _apply_integrated_frame_group_floor(all_entries, floor)

    approved = [
        item for item in all_entries if (item.get("quality") or {}).get("verdict") == "approved"
    ]
    approved.sort(
        key=lambda item: (
            _score01((item.get("quality") or {}).get("target_style_score")),
            _score01((item.get("quality") or {}).get("composition_preservation")),
        ),
        reverse=True,
    )
    style_anchor = None

    total_attempts = max(1, int(job.max_retries) + 1)
    for attempt in range(2, total_attempts + 1):
        retry_work: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []
        for task, spec in all_specs:
            entry = entries_by_task[int(task["index"])].get(str(spec["role"]))
            quality = entry.get("quality") if isinstance(entry, dict) and isinstance(entry.get("quality"), dict) else {}
            if quality.get("verdict") != "approved":
                retry_work.append((task, spec, _as_string_list(quality.get("reasons"))))
        if not retry_work:
            break
        with ThreadPoolExecutor(
            max_workers=max(1, len(retry_work)) if image_concurrency == 0 else max(1, min(image_concurrency, len(retry_work))),
            thread_name_prefix="company-integrated-retry",
        ) as executor:
            futures = [
                executor.submit(run_attempt, task, spec, attempt, None, reasons)
                for task, spec, reasons in retry_work
            ]
            for future in as_completed(futures):
                task, spec, entry, error = future.result()
                generated_count += 1
                task_index = int(task["index"])
                role = str(spec["role"])
                errors_by_task[task_index] = [
                    item for item in errors_by_task[task_index]
                    if not (item.get("kind") == "integrated_frame" and item.get("role") == role)
                ]
                if entry is not None:
                    entries_by_task[task_index][role] = entry
                    verdict = str((entry.get("quality") or {}).get("verdict") or "retry")
                    emit_result(
                        task,
                        phase="integrated_frame_ready" if verdict == "approved" else "integrated_frame_retrying",
                        message=(
                            f"第 {task_index} 段{role}第 {attempt} 次转换已通过。"
                            if verdict == "approved"
                            else f"第 {task_index} 段{role}第 {attempt} 次转换仍偏弱。"
                        ),
                    )
                elif error is not None:
                    errors_by_task[task_index].append(error)
                    emit_result(
                        task,
                        phase="integrated_frame_failed",
                        message=f"第 {task_index} 段{role}第 {attempt} 次转换失败。",
                    )
        all_entries = [entry for values in entries_by_task.values() for entry in values.values()]
        floor = _adaptive_integrated_frame_floor(all_entries)
        _apply_integrated_frame_group_floor(all_entries, floor)
        approved = [item for item in all_entries if (item.get("quality") or {}).get("verdict") == "approved"]
        approved.sort(key=lambda item: _score01((item.get("quality") or {}).get("target_style_score")), reverse=True)
        # Do not turn a frame from one shot into a style/layout reference for
        # the next shot. Retries must remain grounded in their own source frame
        # and shot-local masters.

    reports: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict) or int(task.get("index", 0)) not in entries_by_task:
            continue
        task_index = int(task["index"])
        assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
        integrated = [entries_by_task[task_index][spec["role"]] for spec in _integrated_frame_specs(task) if spec["role"] in entries_by_task[task_index]]
        expected_roles = [str(spec["role"]) for spec in _integrated_frame_specs(task)]
        approved_roles = {
            str(item.get("role"))
            for item in integrated
            if (item.get("quality") or {}).get("verdict") == "approved"
        }
        for item in integrated:
            _record_integrated_frame_cache(cache, item, has_people=bool((task.get("auto_asset_analysis") or {}).get("people")))
        missing = [role for role in expected_roles if role not in approved_roles]
        errors = errors_by_task[task_index]
        for role in missing:
            entry = entries_by_task[task_index].get(role) or {}
            quality = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
            errors.append(
                {
                    "kind": "integrated_frame",
                    "role": role,
                    "error_kind": "quality_gate_failed",
                    "message": "整帧转绘在自动重试后仍未通过：" + ("；".join(_as_string_list(quality.get("reasons"))) or "没有合格结果"),
                    "attempts": int(entry.get("generation_attempts") or total_attempts),
                }
            )
        status = "ready" if expected_roles and not missing and not errors else "degraded"
        _checkpoint_auto_asset_task(
            job,
            _auto_asset_root(job, task),
            task,
            source_frames=assets.get("source_frames", {}),
            analysis=task.get("auto_asset_analysis") if isinstance(task.get("auto_asset_analysis"), dict) else None,
            people=[dict(item) for item in assets.get("people", []) if isinstance(item, dict)],
            scenes=[dict(item) for item in assets.get("scenes", []) if isinstance(item, dict)],
            integrated_frames=integrated,
            errors=errors,
            status=status,
            suspected_matches=task.get("auto_asset_suspected_matches", []),
            warnings=warnings_by_task[task_index],
        )
        emit_result(
            task,
            phase="integrated_frame_task_ready" if status == "ready" else "integrated_frame_task_failed",
            message=(
                f"第 {task_index} 段整帧转换全部通过。"
                if status == "ready"
                else f"第 {task_index} 段仍有整帧转换未通过，已阻止进入 Seedance。"
            ),
        )
        reports.append(
            {
                "index": task_index,
                "status": status,
                "integrated_frames": len(integrated),
                "approved_integrated_frames": len(approved_roles),
                "adaptive_style_floor": round(floor, 4),
                "errors": errors,
            }
        )
    return generated_count, reports


def build_long_video_auto_assets(
    job: LongVideoJob,
    image_concurrency: int = 0,
    *,
    cancel_event: threading.Event | None = None,
    preserve_job_status: bool = False,
) -> tuple[LongVideoJob, str, torch.Tensor]:
    if job.manifest.get("asset_mode") != "auto_shot_assets":
        raise ValueError("该任务不是按镜头自动资产任务，请使用原有手工资产节点链路。")
    options = job.manifest.get("auto_asset_options", {})
    if not isinstance(options, dict):
        options = {}
    image_model = str(options.get("image_model") or "gpt-image-2")
    image_quality = str(options.get("image_quality") or "medium")
    image_provider = str(options.get("image_provider") or "WisArt")
    visual_style = _normalize_auto_asset_style(options.get("visual_style"))
    style_prompt = str(job.prompt or "")
    reuse_threshold = float(options.get("reuse_threshold", AUTO_ASSET_DEFAULT_REUSE_THRESHOLD))
    force_rerun_assets = bool(options.get("force_rerun_assets", False))
    requires_integrated_frames = (
        get_video_engine_adapter(job.engine).key == "seedance"
        and bool(options.get("use_integrated_frame_references", False))
    )
    enforce_identity_gate = int(job.manifest.get("processing_contract_version", 0)) in {
        V3_PROCESSING_CONTRACT_VERSION,
        MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
    }
    identity_mapping = (
        _load_identity_mapping(options.get("identity_mapping"))
        if enforce_identity_gate
        else _empty_identity_mapping()
    )
    image_concurrency = max(0, int(image_concurrency))
    cache = _empty_auto_asset_cache() if force_rerun_assets else _load_auto_asset_cache(job)
    if not force_rerun_assets:
        # A job cache only covers one attempt. Merge the persistent library
        # first so a new manual series still recognizes already-approved
        # recurring people and locations.
        _merge_auto_asset_cache_data(cache, _load_auto_asset_library(visual_style), persistent_library=True)
    tasks = _auto_asset_member_tasks(job)
    task_count = len(tasks)
    progress = ProgressBar(max(1, task_count))
    reports: list[dict[str, Any]] = []
    can_resume_assets = job.resume and not job.force_rerun and not force_rerun_assets
    _emit_auto_asset_progress(
        job,
        progress,
        value=0,
        total=task_count,
        phase="started",
        message=f"准备处理 {task_count} 个分镜的自动人物和背景资产。",
    )

    submitted_count = 0
    cancelled = False
    worker_limit = (
        max(1, task_count * (AUTO_ASSET_MAX_PEOPLE + 2))
        if image_concurrency == 0
        else image_concurrency
    )

    with ThreadPoolExecutor(max_workers=worker_limit, thread_name_prefix="company-image") as executor:
        for task_offset, task in enumerate(tasks):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            task_index = int(task.get("index", task_offset + 1))

            def update_task_progress(
                fraction: float,
                phase: str,
                message: str,
                *,
                extra: dict[str, Any] | None = None,
            ) -> None:
                _emit_auto_asset_progress(
                    job,
                    progress,
                    value=task_offset + fraction,
                    total=task_count,
                    phase=phase,
                    message=message,
                    task=task,
                    extra=extra,
                )

            root = _auto_asset_root(job, task)
            previous_assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
            if can_resume_assets and _auto_asset_task_complete(
                task,
                root,
                reuse_threshold=reuse_threshold,
                require_integrated_frames=requires_integrated_frames,
            ):
                reused_assets = task.get("auto_assets") if isinstance(task.get("auto_assets"), dict) else {}
                reused_people = [dict(item) for item in reused_assets.get("people", []) if isinstance(item, dict)]
                reused_scenes = [dict(item) for item in reused_assets.get("scenes", []) if isinstance(item, dict)]
                reused_warnings = [
                    dict(item) for item in task.get("auto_asset_warnings", []) if isinstance(item, dict)
                ]
                publication_errors: list[dict[str, Any]] = []
                publication_changed = False
                for person in reused_people:
                    if _active_person_asset_id(person):
                        continue
                    publication_changed = True
                    warning, publication_error = _publish_auto_person_asset(person, task_index=task_index)
                    if warning is not None:
                        reused_warnings.append(warning)
                    if publication_error is not None:
                        publication_errors.append(publication_error)

                if publication_changed:
                    _checkpoint_auto_asset_task(
                        job,
                        root,
                        task,
                        source_frames=reused_assets.get("source_frames", {}),
                        analysis=task.get("auto_asset_analysis") if isinstance(task.get("auto_asset_analysis"), dict) else None,
                        people=reused_people,
                        scenes=reused_scenes,
                        errors=publication_errors,
                        status="degraded" if publication_errors else "ready",
                        suspected_matches=task.get("auto_asset_suspected_matches", []),
                        warnings=reused_warnings,
                    )
                current_status = str(task.get("auto_asset_status") or "ready")
                reports.append(
                    {
                        "index": task["index"],
                        "status": current_status,
                        "reused": True,
                        "errors": publication_errors,
                        "warnings": reused_warnings,
                    }
                )
                update_task_progress(
                    1.0,
                    "task_reused" if not publication_errors else "task_blocked",
                    (
                        f"第 {task_index} 段已有完整自动资产，直接复用。"
                        if not publication_errors
                        else f"第 {task_index} 段人物素材上传 TOS 失败，已阻断后续视频请求。"
                    ),
                    extra={
                        "errors": publication_errors,
                        "warnings": reused_warnings,
                        "preview": _auto_asset_preview_payload(
                            task,
                            source_frames=reused_assets.get("source_frames", {}),
                            people=reused_people,
                            scenes=reused_scenes,
                            integrated_frames=[
                                dict(item)
                                for item in reused_assets.get("integrated_frames", [])
                                if isinstance(item, dict)
                            ],
                        )
                    },
                )
                continue

            root.mkdir(parents=True, exist_ok=True)
            update_task_progress(
                0.02,
                "task_started",
                f"开始处理第 {task_index} 段自动资产。",
                extra={
                    "preview": _auto_asset_preview_payload(
                        task,
                        source_frames={},
                        people=[],
                        scenes=[],
                    )
                },
            )
            errors: list[dict[str, Any]] = []
            warnings: list[dict[str, Any]] = []
            frames: torch.Tensor | None = None
            source_frames: dict[str, str] = {}
            source_frames_reused = False
            if can_resume_assets:
                frames, source_frames = _load_saved_source_frames(root, previous_assets)
                source_frames_reused = frames is not None
            update_task_progress(
                0.08,
                "source_frames_started",
                f"第 {task_index} 段正在读取首帧、中帧和尾帧。",
            )
            try:
                if frames is None:
                    frames = _source_frames_for_task(job, task)
                    source_frames = _save_source_frames(frames, root)
            except Exception as exc:
                error = {"kind": "source_frames", "message": str(exc)}
                _checkpoint_auto_asset_task(
                    job,
                    root,
                    task,
                    source_frames=source_frames,
                    analysis=None,
                    people=[],
                    scenes=[],
                    errors=[error],
                    status="source_frames_failed",
                )
                reports.append({"index": task["index"], "status": "source_frames_failed", "error": str(exc)})
                update_task_progress(
                    1.0,
                    "source_frames_failed",
                    f"第 {task_index} 段源帧读取失败：{exc}",
                    extra={"error": str(exc)},
                )
                continue
            update_task_progress(
                0.22,
                "source_frames_ready",
                f"第 {task_index} 段源帧{'已复用' if source_frames_reused else '已提取'}。",
                extra={
                    "preview": _auto_asset_preview_payload(
                        task,
                        source_frames=source_frames,
                        people=[],
                        scenes=[],
                    )
                },
            )

            analysis = task.get("auto_asset_analysis") if can_resume_assets else None
            analysis_reused = isinstance(analysis, dict)
            analysis_attempts = int(task.get("auto_asset_analysis_attempts") or 0) if analysis_reused else 0
            if not analysis_reused:
                update_task_progress(
                    0.28,
                    "analysis_started",
                    f"第 {task_index} 段正在分析人物和背景。",
                )

                def analyze_current_task() -> dict[str, Any]:
                    nonlocal analysis_attempts
                    analysis_attempts += 1
                    return _auto_asset_analysis(frames, model=job.ai_model)

                def report_analysis_retry(
                    failed_attempt: int,
                    total_attempts: int,
                    delay: float,
                    exc: Exception,
                ) -> None:
                    retries_remaining = total_attempts - failed_attempt
                    update_task_progress(
                        0.28,
                        "analysis_retrying",
                        (
                            f"第 {task_index} 段人物和背景分析第 {failed_attempt} 次失败，"
                            f"{delay:g} 秒后重试（剩余 {retries_remaining} 次）：{exc}"
                        ),
                        extra={
                            "attempt": failed_attempt,
                            "total_attempts": total_attempts,
                            "retries_remaining": retries_remaining,
                            "retry_delay_seconds": delay,
                            "error": str(exc),
                        },
                    )

                try:
                    analysis, analysis_attempts = _retry_auto_asset_analysis(
                        analyze_current_task,
                        max_retries=job.max_retries,
                        on_retry=report_analysis_retry,
                    )
                except Exception as exc:
                    task["auto_asset_analysis_attempts"] = analysis_attempts
                    error = {
                        "kind": "analysis",
                        "message": str(exc),
                        "attempts": analysis_attempts,
                    }
                    _checkpoint_auto_asset_task(
                        job,
                        root,
                        task,
                        source_frames=source_frames,
                        analysis=None,
                        people=[],
                        scenes=[],
                        errors=[error],
                        status="analysis_failed",
                    )
                    reports.append({"index": task["index"], "status": "analysis_failed", "error": str(exc)})
                    update_task_progress(
                        1.0,
                        "analysis_failed",
                        f"第 {task_index} 段人物和背景分析在 {analysis_attempts} 次尝试后失败：{exc}",
                        extra={"error": str(exc), "attempts": analysis_attempts},
                    )
                    continue
                task["auto_asset_analysis_attempts"] = analysis_attempts
            update_task_progress(
                0.40,
                "analysis_ready",
                f"第 {task_index} 段人物和背景分析{'已复用' if analysis_reused else '完成'}。",
                extra={
                    "analysis_attempts": analysis_attempts,
                    "people_count": len(analysis.get("people", [])) if isinstance(analysis, dict) else 0,
                    "shot_stability": str(analysis.get("shot_stability", "")) if isinstance(analysis, dict) else "",
                    "preview": _auto_asset_preview_payload(
                        task,
                        source_frames=source_frames,
                        people=[],
                        scenes=[],
                    ),
                },
            )

            existing_people = _valid_auto_asset_entries(previous_assets, "people", "slot") if can_resume_assets else {}
            existing_scenes = _valid_auto_asset_entries(previous_assets, "scenes", "role") if can_resume_assets else {}
            people_by_slot: dict[str, dict[str, Any]] = {}
            scenes_by_role: dict[str, dict[str, Any]] = {}
            reused_items: list[str] = []
            task["auto_asset_analysis"] = analysis
            task["auto_asset_status"] = "building"
            task["auto_asset_errors"] = []
            task["auto_asset_warnings"] = []
            task["auto_asset_suspected_matches"] = []
            try:
                _apply_identity_mapping_to_analysis(task, identity_mapping)
            except ValueError as exc:
                error = {"kind": "identity_mapping", "message": str(exc), "error_kind": "identity_mapping_invalid"}
                _checkpoint_auto_asset_task(
                    job,
                    root,
                    task,
                    source_frames=source_frames,
                    analysis=analysis,
                    people=[],
                    scenes=[],
                    errors=[error],
                    status="identity_mapping_failed",
                )
                reports.append({"index": task["index"], "status": "identity_mapping_failed", "error": str(exc)})
                update_task_progress(
                    1.0,
                    "identity_mapping_failed",
                    f"第 {task_index} 段人物映射无效，已阻止付费生成：{exc}",
                    extra={"error": str(exc)},
                )
                continue
            _write_auto_asset_manifest(root, task)
            _atomic_write_json(job.manifest_path, job.manifest)

            background = analysis["background"]
            needs_end_scene = (
                analysis["shot_stability"] in {"transition", "uncertain"}
                or background["same_scene_confidence"] < reuse_threshold
            )
            scene_specs = [
                ("scene_start", frames[0], background["first_description"], f"{background['scene_key']}_start")
            ]
            if needs_end_scene:
                scene_specs.append(
                    ("scene_end", frames[2], background["last_description"], f"{background['scene_key']}_end")
                )
            person_order = [str(item["slot"]) for item in analysis["people"]]
            scene_order = [str(item[0]) for item in scene_specs]
            task_pending: list[tuple[str, str, Any]] = []

            if analysis["shot_stability"] != "crowded":
                for person in analysis["people"]:
                    slot = str(person.get("slot") or "")
                    existing = existing_people.get(slot)
                    if existing is not None:
                        existing = dict(existing)
                        if not _active_person_asset_id(existing):
                            warning, publication_error = _publish_auto_person_asset(
                                existing,
                                task_index=task_index,
                            )
                            if warning is not None:
                                warnings.append(warning)
                            if publication_error is not None:
                                errors.append(publication_error)
                        people_by_slot[slot] = existing
                        reused_items.append(f"person:{slot}")
                        continue
                    frame_paths = tuple(
                        Path(source_frames[name])
                        for name in ("source_start", "source_middle", "source_end")
                    )

                    def create_person(
                        *,
                        person=person,
                        root=root,
                        frame_paths=frame_paths,
                    ):
                        request_frames = _vision_batch([_load_image_file(path) for path in frame_paths])
                        return _retry_auto_asset_creation(
                            lambda: _create_auto_person_asset(
                                root=root,
                                person=person,
                                frames=request_frames,
                                cache=deepcopy(cache),
                                analysis_model=job.ai_model,
                                image_model=image_model,
                                image_quality=image_quality,
                                image_provider=image_provider,
                                reuse_threshold=reuse_threshold,
                                visual_style=visual_style,
                                style_prompt=style_prompt,
                                enforce_identity_gate=enforce_identity_gate,
                            ),
                            max_retries=job.max_retries,
                        )

                    task_pending.append(("person", slot, create_person))

            for role, source_frame, description, scene_key in scene_specs:
                existing = existing_scenes.get(role)
                if existing is not None:
                    scenes_by_role[role] = existing
                    reused_items.append(f"scene:{role}")
                    continue
                source_frame_path = Path(
                    source_frames["source_start" if role == "scene_start" else "source_end"]
                )

                def create_scene(
                    *,
                    role=role,
                    root=root,
                    source_frame_path=source_frame_path,
                    description=description,
                    scene_key=scene_key,
                    allow_cache=True,
                ):
                    request_frame = _load_image_file(source_frame_path)
                    return _retry_auto_asset_creation(
                        lambda: _create_auto_scene_asset(
                            root=root,
                            role=role,
                            source_frame=request_frame,
                            source_frame_path=source_frame_path,
                            description=description,
                            scene_key=scene_key,
                            cache=deepcopy(cache),
                            analysis_model=job.ai_model,
                            image_model=image_model,
                            image_quality=image_quality,
                            image_provider=image_provider,
                            reuse_threshold=reuse_threshold,
                            allow_cache=allow_cache,
                            visual_style=visual_style,
                            style_prompt=style_prompt,
                        ),
                        max_retries=job.max_retries,
                    )

                task_pending.append(("scene", role, create_scene))

            suspected_matches: list[dict[str, Any]] = []
            future_jobs = {executor.submit(factory): (kind, key) for kind, key, factory in task_pending}
            submitted_count += len(future_jobs)
            update_task_progress(
                0.48,
                "assets_queued",
                f"第 {task_index} 段待生成或复用 {len(task_pending)} 张人物/背景素材。",
                extra={
                    "image_request_count": len(task_pending),
                    "reused_items": reused_items,
                    "image_worker_count": len(future_jobs)
                    if image_concurrency == 0
                    else min(image_concurrency, len(future_jobs)),
                },
            )
            completed_asset_count = 0
            if not future_jobs:
                update_task_progress(
                    0.86,
                    "assets_reused",
                    f"第 {task_index} 段没有新的图片请求，素材全部来自缓存或已有文件。",
                    extra={
                        "reused_items": reused_items,
                        "preview": _auto_asset_preview_payload(
                            task,
                            source_frames=source_frames,
                            people=_ordered_auto_asset_entries(people_by_slot, person_order),
                            scenes=_ordered_auto_asset_entries(scenes_by_role, scene_order),
                        ),
                    },
                )
            for future in as_completed(list(future_jobs)):
                kind, key = future_jobs[future]
                publication_error: dict[str, Any] | None = None
                try:
                    entry, error = future.result()
                except Exception as exc:
                    entry = None
                    error = {"kind": kind, "message": str(exc), "error_kind": "generation_failed"}
                if entry:
                    if kind == "person":
                        if (
                            str(entry.get("identity_state") or "") != "partial"
                            and not _active_person_asset_id(entry)
                        ):
                            warning, publication_error = _publish_auto_person_asset(entry, task_index=task_index)
                            if warning is not None:
                                warnings.append(warning)
                            if publication_error is not None:
                                errors.append(publication_error)
                    if entry.get("cache_update"):
                        _merge_auto_asset_cache_entry(cache, entry)
                        _save_auto_asset_cache(job, cache)
                        _save_auto_asset_library(cache, visual_style=visual_style)
                    distributed_entry = dict(entry)
                    distributed_entry["reused_from_cache"] = bool(entry.get("reused_from_cache"))
                    distributed_entry["slot" if kind == "person" else "role"] = key
                    target = people_by_slot if kind == "person" else scenes_by_role
                    target[key] = distributed_entry
                    for suspected in entry.get("suspected_matches", []):
                        if isinstance(suspected, dict):
                            suspected_matches.append(dict(suspected))
                if error:
                    errors.append(dict(error))
                    for suspected in error.get("suspected_matches", []):
                        if isinstance(suspected, dict):
                            suspected_matches.append(dict(suspected))
                people = _ordered_auto_asset_entries(people_by_slot, person_order)
                scenes = _ordered_auto_asset_entries(scenes_by_role, scene_order)
                _checkpoint_auto_asset_task(
                    job,
                    root,
                    task,
                    source_frames=source_frames,
                    analysis=analysis,
                    people=people,
                    scenes=scenes,
                    errors=errors,
                    status="building",
                    suspected_matches=suspected_matches,
                    warnings=warnings,
                )
                completed_asset_count += 1
                update_task_progress(
                    0.48 + 0.38 * (completed_asset_count / max(1, len(future_jobs))),
                    "asset_done",
                    f"第 {task_index} 段素材进度 {completed_asset_count}/{len(future_jobs)}。",
                    extra={
                        "asset_kind": kind,
                        "asset_key": key,
                        "asset_status": "failed" if publication_error is not None else "ready" if entry else "failed",
                        "error": publication_error or error,
                        "warnings": warnings,
                        "asset_done": completed_asset_count,
                        "asset_total": len(future_jobs),
                        "preview": _auto_asset_preview_payload(
                            task,
                            source_frames=source_frames,
                            people=people,
                            scenes=scenes,
                        ),
                    },
                )

            people = _ordered_auto_asset_entries(people_by_slot, person_order)
            scenes = _ordered_auto_asset_entries(scenes_by_role, scene_order)
            unresolved_people = [
                item
                for item in errors
                if isinstance(item, dict) and item.get("error_kind") == "identity_review_required"
            ]
            ignored_people = [
                item for item in people if str(item.get("identity_state") or "") in {"partial", "ignored"}
            ]
            expected_people = 0 if analysis["shot_stability"] == "crowded" else len(analysis["people"])
            effective_people = len(people) + len(unresolved_people)
            expected_scenes = len(scene_specs)
            blocking_errors = [
                item
                for item in errors
                if not isinstance(item, dict) or item.get("error_kind") != "identity_review_required"
            ]
            complete = effective_people == expected_people and len(scenes) == expected_scenes and not blocking_errors
            status = (
                "identity_review_required"
                if unresolved_people
                else "masters_ready"
                if complete and requires_integrated_frames
                else "ready"
                if complete
                else "degraded"
                if scenes or people
                else "failed"
            )
            _checkpoint_auto_asset_task(
                job,
                root,
                task,
                source_frames=source_frames,
                analysis=analysis,
                people=people,
                scenes=scenes,
                errors=errors,
                status=status,
                suspected_matches=suspected_matches,
                warnings=warnings,
            )
            reports.append(
                {
                    "index": task["index"],
                    "status": status,
                    "people": len(people),
                    "ignored_people": len(ignored_people),
                    "unresolved_people": len(unresolved_people),
                    "scenes": len(scenes),
                    "errors": errors,
                    "warnings": warnings,
                    "suspected_matches": suspected_matches,
                    "source_frames_reused": source_frames_reused,
                    "analysis_reused": analysis_reused,
                    "reused_items": reused_items,
                    "image_concurrency": image_concurrency,
                    "image_worker_count": len(future_jobs) if image_concurrency == 0 else min(image_concurrency, len(future_jobs)),
                    "image_request_count": len(task_pending),
                }
            )
            update_task_progress(
                1.0,
                f"task_{status}",
                f"第 {task_index} 段自动资产生成完成，状态：{status}。",
                extra={
                    "people": len(people),
                    "scenes": len(scenes),
                    "errors": errors,
                    "warnings": warnings,
                    "suspected_matches": suspected_matches,
                    "image_request_count": len(task_pending),
                    "reused_items": reused_items,
                    "preview": _auto_asset_preview_payload(
                        task,
                        source_frames=source_frames,
                        people=people,
                        scenes=scenes,
                    ),
                },
            )

    integrated_request_count = 0
    integrated_reports: list[dict[str, Any]] = []
    if not cancelled and requires_integrated_frames:
        _emit_auto_asset_progress(
            job,
            progress,
            value=max(0.0, task_count - 0.05),
            total=task_count,
            phase="integrated_frames_started",
            message="人物和场景母版已准备，开始整帧融合转绘与自适应质量验收。",
        )
        integrated_request_count, integrated_reports = _build_integrated_frames(
            job,
            cache,
            image_concurrency=image_concurrency,
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
            visual_style=visual_style,
            style_prompt=style_prompt,
            progress_bar=progress,
        )
        submitted_count += integrated_request_count

    worker_count = submitted_count if image_concurrency == 0 else min(image_concurrency, submitted_count)

    reports.sort(key=lambda item: int(item.get("index", 0)))
    _save_auto_asset_cache(job, cache)
    _save_auto_asset_library(cache, visual_style=visual_style)
    asset_stage_status = (
        "auto_assets_cancelled"
        if cancelled
        else "auto_assets_ready"
        if all(item.get("auto_asset_status") == "ready" for item in tasks)
        else "auto_assets_partial_failure"
    )
    if not preserve_job_status:
        job.manifest["status"] = asset_stage_status
    _atomic_write_json(job.manifest_path, job.manifest)
    _emit_auto_asset_progress(
        job,
        progress,
        value=task_count,
        total=task_count,
        phase="completed",
        message=f"自动资产阶段结束：{asset_stage_status}。",
        extra={
            "image_request_count": submitted_count,
            "integrated_frame_request_count": integrated_request_count,
            "image_worker_count": worker_count,
            "status_counts": _auto_asset_status_counts(job),
        },
    )
    report = {
        "job_id": job.manifest["job_id"],
        "asset_mode": "auto_shot_assets",
        "prompt_version": AUTO_ASSET_PROMPT_VERSION,
        "cache_version": AUTO_ASSET_CACHE_VERSION,
        "image_concurrency": image_concurrency,
        "image_worker_count": worker_count,
        "image_request_count": submitted_count,
        "integrated_frame_request_count": integrated_request_count,
        "integrated_frame_quality_gate": AUTO_ASSET_QUALITY_GATE_VERSION,
        "integrated_frame_tasks": integrated_reports,
        "tasks": reports,
        "manifest": str(job.manifest_path),
        "cancelled": cancelled,
        "asset_stage_status": asset_stage_status,
        "check": (
            "检查每个镜头的首尾源帧、人物/场景母版和素材库状态；默认参考包优先发送人物 asset_id 与场景母版。"
            if not requires_integrated_frames
            else "检查每个镜头的首尾源帧、人物/场景母版和 approved 整帧图；整帧参考已明确启用。"
        ),
    }
    return job, json.dumps(report, ensure_ascii=False, indent=2), _auto_asset_preview(job)


def _save_reference_package_image(image: Any, path: Path) -> str:
    _save_image_tensor(image, path)
    return str(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _v3_asset_identity(kind: str, item: dict[str, Any], path: Path) -> tuple[str, str]:
    if kind == "person":
        verified = str(item.get("person_id") or "")
    else:
        place_id = str(item.get("place_id") or "")
        version_id = str(item.get("version_id") or "")
        verified = f"{place_id}:{version_id}" if place_id and version_id else ""
    content_hash = _file_sha256(path)
    return (f"verified:{kind}:{verified}" if verified else f"hash:{content_hash}", content_hash)


def _active_person_asset_id(item: dict[str, Any]) -> str:
    publication = item.get("publication")
    if not isinstance(publication, dict):
        return ""
    library = publication.get("asset_library")
    if not isinstance(library, dict) or str(library.get("status") or "").lower() != "active":
        return ""
    asset_id = str(library.get("asset_id") or "").strip()
    return asset_id if asset_id.startswith("asset-") else ""


def _v3_labeled_contact_sheet(items: list[dict[str, Any]]) -> torch.Tensor:
    if not items or len(items) > V3_CONTACT_SHEET_ITEMS:
        raise ValueError(f"单张参考拼图只能容纳 1-{V3_CONTACT_SHEET_ITEMS} 个资源。")
    label_height = 72
    columns = 1 if len(items) == 1 else 2 if len(items) <= 4 else 3
    rows = int(math.ceil(len(items) / columns))
    cell_height = V3_CONTACT_SHEET_CELL_SIZE + label_height
    width = V3_CONTACT_SHEET_CELL_SIZE * columns
    canvas = Image.new("RGB", (width, cell_height * rows), "#202124")
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.load_default(size=30)
    for index, item in enumerate(items):
        column = index % columns
        row = index // columns
        cell_x = column * V3_CONTACT_SHEET_CELL_SIZE
        cell_y = row * cell_height
        path = Path(str(item["path"]))
        with Image.open(path) as source:
            source = source.convert("RGB")
            source.thumbnail((V3_CONTACT_SHEET_CELL_SIZE, V3_CONTACT_SHEET_CELL_SIZE), Image.Resampling.LANCZOS)
            x = cell_x + (V3_CONTACT_SHEET_CELL_SIZE - source.width) // 2
            y = cell_y + label_height + (V3_CONTACT_SHEET_CELL_SIZE - source.height) // 2
            canvas.paste(source, (x, y))
        label = str(item["label"])
        draw.rectangle(
            (cell_x, cell_y, cell_x + V3_CONTACT_SHEET_CELL_SIZE - 1, cell_y + label_height - 1),
            outline="#f1f3f4",
            width=2,
        )
        draw.text((cell_x + 16, cell_y + 16), label, fill="#ffffff", font=label_font)
    value = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(value)[None,]


def _v3_reference_package_resume_key(tasks: list[dict[str, Any]]) -> str:
    payload = [
        {
            "index": int(task["index"]),
            "package_key": str((task.get("reference_package") or {}).get("package_key") or ""),
        }
        for task in tasks
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _v3_group_pack_context(job: LongVideoJob) -> dict[str, Any]:
    """Collect the loop-invariant values shared by every request group packing call."""
    options = job.manifest.get("auto_asset_options") or {}
    is_manual_batch = int(job.manifest.get("processing_contract_version", 0)) == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION
    manual_batch = job.manifest.get("manual_batch") if isinstance(job.manifest.get("manual_batch"), dict) else {}
    return {
        "members_by_index": {
            int(item["index"]): item for item in _auto_asset_member_tasks(job) if isinstance(item, dict)
        },
        "visual_style": _normalize_auto_asset_style(options.get("visual_style")),
        "send_source_video": bool(options.get("send_source_video", True)),
        "is_manual_batch": is_manual_batch,
        "cross_batch_frame": (
            str(manual_batch.get("cross_batch_final_frame") or "")
            if bool(manual_batch.get("cross_batch_continuity"))
            else ""
        ),
        "package_version": MANUAL_BATCH_REFERENCE_PACKAGE_VERSION if is_manual_batch else V3_REFERENCE_PACKAGE_VERSION,
    }


def _reference_timeline_ranges(
    task: dict[str, Any],
    member_ids: list[int],
    members_by_index: dict[int, dict[str, Any]],
    path_to_package_index: dict[str, int],
) -> list[dict[str, Any]]:
    group_start = float(task.get("source_start", task.get("start", 0.0)))
    timeline_by_index: dict[int, dict[str, Any]] = {}
    for member_id in member_ids:
        member = members_by_index.get(member_id)
        if not isinstance(member, dict):
            continue
        assets = member.get("auto_assets") if isinstance(member.get("auto_assets"), dict) else {}
        integrated = [
            item
            for item in assets.get("integrated_frames", [])
            if isinstance(item, dict) and (item.get("quality") or {}).get("verdict") == "approved"
        ]
        if not integrated:
            continue
        start = max(0.0, float(member.get("source_start", member.get("start", 0.0))) - group_start)
        end = start + float(member.get("source_duration", member.get("duration", 0.0)))
        ordered_paths = [
            str(path)
            for role in ("frame_start", "frame_end")
            for item in integrated
            if item.get("role") == role and (path := _asset_path_if_file(item.get("path"))) is not None
        ]
        for value in ordered_paths:
            package_index = path_to_package_index.get(value)
            if package_index is None:
                continue
            entry = timeline_by_index.setdefault(package_index, {"image_index": package_index, "ranges": []})
            entry["ranges"].append(
                {
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "logical_segment": member_id,
                }
            )
    return [timeline_by_index[index] for index in sorted(timeline_by_index)]


def _evenly_spaced_reference_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indexes = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in dict.fromkeys(indexes)]


def _select_integrated_reference_items(
    items: list[dict[str, Any]],
    *,
    package_limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep temporal coverage without reintroducing source or contact-sheet references."""
    if package_limit <= 0:
        raise ValueError("Seedance 参考图位置已被连续性参考占满，无法发送合格整帧图。")
    if len(items) <= package_limit:
        return list(items), 0

    primary_by_member: dict[int, dict[str, Any]] = {}
    for item in items:
        primary_by_member.setdefault(int(item["member_index"]), item)
    primary = list(primary_by_member.values())
    selected = _evenly_spaced_reference_items(primary, package_limit)
    if len(selected) < package_limit:
        selected_identities = {str(item["identity"]) for item in selected}
        remaining = [item for item in items if str(item["identity"]) not in selected_identities]
        selected.extend(remaining[: package_limit - len(selected)])
    selected.sort(key=lambda item: int(item["temporal_order"]))
    return selected, len(items) - len(selected)


def _v3_pack_single_group_reference(
    job: LongVideoJob,
    task: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Pack the reference package for one request group and return its report entry."""
    members_by_index = context["members_by_index"]
    visual_style = context["visual_style"]
    send_source_video = context["send_source_video"]
    is_manual_batch = context["is_manual_batch"]
    cross_batch_frame = context["cross_batch_frame"]
    package_version = context["package_version"]
    options = job.manifest.get("auto_asset_options") or {}
    use_integrated_frame_references = bool(options.get("use_integrated_frame_references", False))

    member_ids = list(
        dict.fromkeys(int(item) for item in task.get("logical_segments", [task.get("logical_segment")]))
    )
    members = [members_by_index.get(index) for index in member_ids]
    if any(member is None or member.get("auto_asset_status") != "ready" for member in members):
        task["reference_package_status"] = "blocked_by_asset_failure"
        task["reference_package"] = {"version": package_version, "items": []}
        task["reference_roles"] = []
        task["status"] = "blocked_by_asset_failure"
        return {"index": task["index"], "status": task["reference_package_status"]}
    ready_members = [member for member in members if member is not None]

    master_items: list[dict[str, Any]] = []
    integrated_items: list[dict[str, Any]] = []
    seen_master_asset_ids: set[str] = set()
    seen_master_content_hashes: set[str] = set()
    seen_integrated: set[str] = set()
    story_parts: list[str] = []
    people_descriptions: list[str] = []
    for member_index, member in zip(member_ids, ready_members, strict=True):
        analysis = member.get("auto_asset_analysis") or {}
        story_parts.append(f"镜头 {member_index}：{analysis.get('story_action', '')}")
        assets = member.get("auto_assets") or {}
        people = [item for item in assets.get("people", []) if isinstance(item, dict)]
        for person in sorted(people, key=lambda item: 0 if _active_person_asset_id(item) else 1):
            people_descriptions.append(str(person.get("appearance") or "主要人物"))
            asset_id = _active_person_asset_id(person)
            path = _asset_path_if_file(person.get("path"))
            if not asset_id and path is None:
                continue
            content_hash = _file_sha256(path) if path is not None else ""
            identity = f"asset:{asset_id}" if asset_id else f"hash:{content_hash}"
            if (
                (asset_id and asset_id in seen_master_asset_ids)
                or (content_hash and content_hash in seen_master_content_hashes)
            ):
                continue
            if asset_id:
                seen_master_asset_ids.add(asset_id)
            if content_hash:
                seen_master_content_hashes.add(content_hash)
            master_items.append(
                {
                    "kind": "person_asset",
                    "source_role": str(person.get("slot") or "person"),
                    "member_index": member_index,
                    "temporal_order": len(master_items),
                    "path": str(path) if path is not None else "",
                    "asset_id": asset_id,
                    "identity": identity,
                    "content_hash": content_hash,
                    "reused_from_cache": bool(person.get("reused_from_cache")),
                }
            )
        for scene in assets.get("scenes", []):
            if not isinstance(scene, dict):
                continue
            path = _asset_path_if_file(scene.get("style_master_path")) or _asset_path_if_file(scene.get("path"))
            if path is None:
                continue
            content_hash = _file_sha256(path)
            identity = f"hash:{content_hash}"
            if content_hash in seen_master_content_hashes:
                continue
            seen_master_content_hashes.add(content_hash)
            master_items.append(
                {
                    "kind": "scene_master",
                    "source_role": str(scene.get("role") or "scene"),
                    "member_index": member_index,
                    "temporal_order": len(master_items),
                    "path": str(path),
                    "asset_id": "",
                    "identity": identity,
                    "content_hash": content_hash,
                    "reused_from_cache": bool(scene.get("reused_from_cache")),
                }
            )
        if not use_integrated_frame_references:
            continue
        integrated = [
            item
            for item in assets.get("integrated_frames", [])
            if isinstance(item, dict) and (item.get("quality") or {}).get("verdict") == "approved"
        ]
        integrated.sort(key=lambda item: 0 if item.get("role") == "frame_start" else 1)
        for item in integrated:
            path = _asset_path_if_file(item.get("path"))
            if path is None:
                continue
            content_hash = _file_sha256(path)
            identity = f"hash:{content_hash}"
            if identity in seen_integrated:
                continue
            seen_integrated.add(identity)
            integrated_items.append(
                {
                    "kind": "integrated_frame",
                    "source_role": str(item.get("role") or "frame_start"),
                    "member_index": member_index,
                    "temporal_order": len(integrated_items),
                    "path": str(path),
                    "asset_id": "",
                    "identity": identity,
                    "content_hash": content_hash,
                    "quality": dict(item.get("quality") or {}),
                }
            )

    root = job.job_dir / "request_groups" / f"group_{int(task['index']):04d}" / "reference_package"
    root.mkdir(parents=True, exist_ok=True)
    package: list[dict[str, Any]] = []
    adapter_limit = get_video_engine_adapter(job.engine).reference_image_limit
    uses_previous_end_frame = bool(task.get("continuity_from_previous_group"))
    uses_cross_batch_frame = is_manual_batch and int(task["index"]) == 1 and bool(cross_batch_frame)
    continuity_reserved = uses_previous_end_frame or uses_cross_batch_frame
    package_limit = adapter_limit - (1 if continuity_reserved else 0)
    if package_limit <= 0:
        raise ValueError("Seedance 参考图位置已被连续性参考占满，无法发送人物或场景素材。")
    ordered_master_items = sorted(
        master_items,
        key=lambda item: (0 if item["kind"] == "person_asset" else 1, int(item["temporal_order"])),
    )
    selected_masters = list(ordered_master_items[:package_limit])
    remaining_limit = package_limit - len(selected_masters)
    selected_integrated, omitted_integrated_count = _select_integrated_reference_items(
        integrated_items,
        package_limit=remaining_limit,
    ) if use_integrated_frame_references and remaining_limit > 0 else ([], len(integrated_items))
    selected = [*selected_masters, *selected_integrated]
    omitted_master_count = max(0, len(ordered_master_items) - len(selected_masters))
    package = []
    person_index = 0
    scene_index = 0
    integrated_index = 0
    for item in selected:
        kind = str(item["kind"])
        if kind == "person_asset":
            person_index += 1
            role = f"person_{person_index}"
            label = f"PERSON-{person_index:02d}"
        elif kind == "scene_master":
            scene_index += 1
            role = f"scene_{scene_index}"
            label = f"SCENE-{scene_index:02d}"
        else:
            integrated_index += 1
            role = f"integrated_frame_{integrated_index}"
            label = f"FRAME-{integrated_index:02d}"
        package_item = {
            "role": role,
            "kind": kind,
            "path": item["path"],
            "label": label,
            "source_role": item["source_role"],
            "logical_segment": item["member_index"],
            "reused_from_cache": bool(item.get("reused_from_cache")),
        }
        if item.get("asset_id"):
            package_item["asset_id"] = item["asset_id"]
        if kind == "integrated_frame":
            package_item["quality"] = item["quality"]
        package.append(package_item)

    roles = [item["role"] for item in package]
    group_analysis = {
        "story_action": "；".join(story_parts),
        "people": [{"appearance": value} for value in dict.fromkeys(people_descriptions)],
    }
    path_to_package_index = {
        str(Path(item["path"])): index
        for index, item in enumerate(package, start=1)
        if item.get("kind") == "integrated_frame" and item.get("path")
    }
    reference_timeline = _reference_timeline_ranges(task, member_ids, members_by_index, path_to_package_index)
    task["auto_asset_analysis"] = group_analysis
    task["prompt"] = _build_auto_asset_video_prompt(
        job.prompt,
        group_analysis,
        reference_roles=roles,
        soft_continuity=continuity_reserved,
        visual_style=visual_style,
        send_source_video=send_source_video,
        reference_timeline=reference_timeline,
    )
    package_key_source = {
        "version": package_version,
        "logical_segments": member_ids,
        "analysis": group_analysis,
        "prompt": task["prompt"],
        "assets": [
            {
                "identity": item["identity"],
                "content_hash": item["content_hash"],
                "asset_id": item.get("asset_id") or "",
                "kind": item["kind"],
                "quality": item.get("quality") or {},
            }
            for item in selected
        ],
    }
    package_key = hashlib.sha256(
        json.dumps(package_key_source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    task["reference_package"] = {
        "version": package_version,
        "package_key": package_key,
        "items": package,
        "unique_asset_count": len(master_items) + len(integrated_items),
        "packed_asset_count": len(selected),
        "omitted_master_count": omitted_master_count,
        "omitted_integrated_frame_count": omitted_integrated_count,
        "asset_hashes": [item["content_hash"] for item in selected if item["content_hash"]],
        "asset_ids": [item["asset_id"] for item in selected if item.get("asset_id")],
        "uses_integrated_frame_references": use_integrated_frame_references,
        "logical_segments": member_ids,
        "reference_timeline": reference_timeline,
        "reserved_continuity_slots": adapter_limit - package_limit,
    }
    task["reference_roles"] = roles
    task["reference_package_status"] = "ready" if package else "empty"
    task["uses_previous_end_frame"] = continuity_reserved
    if package and task.get("generated_reference_package_key") == package_key:
        task["status"] = "success"
        task.pop("resume_invalidated", None)
    else:
        task["status"] = "matched" if package else "auto_asset_failed"
    return {
        "index": task["index"],
        "status": task["reference_package_status"],
        "logical_segments": member_ids,
        "unique_assets": len(master_items) + len(integrated_items),
        "packed_assets": len(selected),
        "omitted_masters": omitted_master_count,
        "omitted_integrated_frames": omitted_integrated_count,
        "uses_integrated_frames": use_integrated_frame_references,
        "roles": roles,
    }


def _v3_pack_group_references(job: LongVideoJob) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    previous_resume_key = str(job.manifest.get("reference_package_resume_key") or "")
    context = _v3_group_pack_context(job)
    reports: list[dict[str, Any]] = [
        _v3_pack_single_group_reference(job, task, context) for task in job.manifest["tasks"]
    ]
    tasks = job.manifest["tasks"]
    current_resume_key = _v3_reference_package_resume_key(tasks)
    if previous_resume_key and previous_resume_key != current_resume_key:
        for task in tasks:
            package_key = str((task.get("reference_package") or {}).get("package_key") or "")
            if task.get("generated_reference_package_key") == package_key:
                continue
            if task.get("status") == "success":
                task["status"] = "matched"
            task["resume_invalidated"] = "reference_package_changed"
            task.pop("generated_reference_package_key", None)
    job.manifest["reference_package_resume_key"] = current_resume_key
    return reports, tasks


def pack_long_video_auto_references(job: LongVideoJob) -> tuple[LongVideoJob, str, torch.Tensor]:
    if job.manifest.get("asset_mode") != "auto_shot_assets":
        raise ValueError("该任务不是按镜头自动资产任务。")
    adapter = get_video_engine_adapter(job.engine)
    options = job.manifest.get("auto_asset_options", {})
    options = options if isinstance(options, dict) else {}
    visual_style = _normalize_auto_asset_style(options.get("visual_style"))
    send_source_video = bool(options.get("send_source_video", True))
    reports: list[dict[str, Any]] = []
    if int(job.manifest.get("processing_contract_version", 0)) in {
        V3_PROCESSING_CONTRACT_VERSION,
        MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
    }:
        reports, _tasks = _v3_pack_group_references(job)
        job.manifest["status"] = "auto_references_packed"
        _atomic_write_json(job.manifest_path, job.manifest)
        preview_paths = [
            Path(item["path"])
            for task in job.manifest["tasks"]
            for item in task.get("reference_package", {}).get("items", [])
            if isinstance(item, dict) and _asset_path_if_file(item.get("path"))
        ]
        previews = (
            _vision_batch([_load_image_file(path) for path in preview_paths[:SHOT_PREVIEW_LIMIT]])
            if preview_paths
            else _auto_asset_preview(job)
        )
        report = {
            "job_id": job.manifest["job_id"],
            "processing_contract_version": int(job.manifest.get("processing_contract_version", 0)),
            "reference_package_version": (
                MANUAL_BATCH_REFERENCE_PACKAGE_VERSION
                if int(job.manifest.get("processing_contract_version", 0)) == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION
                else V3_REFERENCE_PACKAGE_VERSION
            ),
            "reference_image_limit": adapter.reference_image_limit,
            "tasks": reports,
            "manifest": str(job.manifest_path),
        }
        return job, json.dumps(report, ensure_ascii=False, indent=2), previews
    for task in job.manifest["tasks"]:
        assets = task.get("auto_assets")
        if not isinstance(assets, dict) or task.get("auto_asset_status") != "ready":
            task["reference_package_status"] = "blocked_by_asset_failure"
            reports.append({"index": task["index"], "status": task["reference_package_status"]})
            continue
        root = _auto_asset_root(job, task) / "reference_package"
        root.mkdir(parents=True, exist_ok=True)
        people = [
            (item.get("slot", "person"), _asset_path_if_file(item.get("path")))
            for item in assets.get("people", [])
            if isinstance(item, dict)
        ]
        scenes = [
            (item.get("role", "scene"), _asset_path_if_file(item.get("path")))
            for item in assets.get("scenes", [])
            if isinstance(item, dict)
        ]
        people = [(role, path) for role, path in people if path]
        scenes = [(role, path) for role, path in scenes if path]
        reserve_for_continuity = 1 if int(task["index"]) > 1 else 0
        base_limit = max(0, adapter.reference_image_limit - reserve_for_continuity)
        package: list[dict[str, str]] = []
        if adapter.key == "wan":
            if people and len(package) < base_limit:
                person_sheet = _person_contact_sheet([_load_image_file(path) for _role, path in people])
                package.append({"role": "people_contact_sheet", "path": _save_reference_package_image(person_sheet, root / "people_contact_sheet.png")})
            if scenes and len(package) < base_limit:
                scene_images = [_load_image_file(path) for _role, path in scenes[:2]]
                scene_value = _person_contact_sheet(scene_images) if len(scene_images) > 1 else scene_images[0]
                package.append({"role": "scene_sequence", "path": _save_reference_package_image(scene_value, root / "scene_sequence.png")})
        else:
            for role, path in [*people, *scenes]:
                if len(package) >= base_limit:
                    break
                package.append({"role": role, "path": str(path)})
        task["reference_package"] = {"items": package, "reserved_continuity_slots": reserve_for_continuity}
        task["reference_roles"] = [item["role"] for item in package]
        task["reference_package_status"] = "ready" if package else "empty"
        analysis = task.get("auto_asset_analysis", {})
        task["prompt"] = _build_auto_asset_video_prompt(
            job.prompt,
            analysis if isinstance(analysis, dict) else {},
            reference_roles=task["reference_roles"],
            soft_continuity=int(task["index"]) > 1,
            visual_style=visual_style,
            send_source_video=send_source_video,
        )
        if task.get("status") != "success":
            task["status"] = "matched" if package else "auto_asset_failed"
        reports.append({"index": task["index"], "status": task["reference_package_status"], "roles": task["reference_roles"]})
    job.manifest["status"] = "auto_references_packed"
    _atomic_write_json(job.manifest_path, job.manifest)
    preview_paths = [
        Path(item["path"])
        for task in job.manifest["tasks"]
        for item in task.get("reference_package", {}).get("items", [])
        if isinstance(item, dict) and _asset_path_if_file(item.get("path"))
    ]
    previews = _vision_batch([_load_image_file(path) for path in preview_paths[:SHOT_PREVIEW_LIMIT]]) if preview_paths else _auto_asset_preview(job)
    report = {
        "job_id": job.manifest["job_id"],
        "engine": adapter.key,
        "reference_image_limit": adapter.reference_image_limit,
        "visual_style": visual_style,
        "send_source_video": send_source_video,
        "tasks": reports,
        "manifest": str(job.manifest_path),
    }
    return job, json.dumps(report, ensure_ascii=False, indent=2), previews


def _run_ffmpeg(args: list[str], *, error_prefix: str) -> None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("当前 portable Python 找不到 FFmpeg，无法切分/合并长视频。") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"{error_prefix}: {detail}") from exc


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("portable Python 缺少 imageio_ffmpeg，无法处理长视频。") from exc


def _concat_and_mux(segment_paths: list[Path], source_path: str, output_path: Path, duration: float, work_dir: Path) -> None:
    if not segment_paths:
        raise ValueError("没有可合并的视频分段。")
    ffmpeg = _ffmpeg_exe()
    concat_list = work_dir / "segments.txt"
    concat_list.write_text(
        "\n".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(39))}'" for path in segment_paths) + "\n",
        encoding="utf-8",
    )
    silent_video = work_dir / "video_no_audio.mp4"
    _run_ffmpeg(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent_video)],
        error_prefix="分段视频合并失败",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_video),
            "-i",
            source_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-t",
            f"{duration:.6f}",
            str(output_path),
        ],
        error_prefix="原音频封装失败",
    )


def _concat_v3_segments(
    tasks: list[dict[str, Any]],
    source_path: str,
    output_path: Path,
    work_dir: Path,
    *,
    use_original_audio: bool,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("没有可合并的 v3 视频分段。")
    ffmpeg = _ffmpeg_exe()
    if use_original_audio:
        _source_duration, _source_fps, _source_frames, source_has_audio = _video_track_info(Path(source_path))
        if not source_has_audio:
            raise ValueError("已开启使用原视频音频，但源视频没有音轨。")
    prepared: list[Path] = []
    segment_reports: list[dict[str, Any]] = []
    for task in tasks:
        segment = Path(str(task["result"]))
        output_duration = float(task.get("media", {}).get("output_duration", 0.0))
        if output_duration <= 0:
            output_duration, _fps, _frames, _has_audio = _video_track_info(segment)
        if use_original_audio:
            prepared_path = work_dir / f"original_audio_{int(task['index']):04d}.mp4"
            source_duration = float(task.get("source_duration", task.get("duration", 0.0)))
            source_start = float(task.get("source_start", task.get("start", 0.0)))
            _run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(segment),
                    "-ss",
                    f"{source_start:.6f}",
                    "-t",
                    f"{source_duration:.6f}",
                    "-i",
                    source_path,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-af",
                    f"apad=pad_dur={output_duration:.6f},atrim=duration={output_duration:.6f},asetpts=PTS-STARTPTS",
                    "-c:a",
                    "aac",
                    str(prepared_path),
                ],
                error_prefix=f"第 {task['index']} 组原音频对齐失败",
            )
            prepared.append(prepared_path)
        else:
            _duration, _fps, _frames, has_audio = _video_track_info(segment)
            if not has_audio:
                raise ValueError(f"第 {task['index']} 组缺少 Seedance 生成音轨。")
            prepared.append(segment)
        segment_reports.append(
            {
                "index": task["index"],
                "source_duration": round(float(task.get("source_duration", 0.0)), 6),
                "output_duration": round(output_duration, 6),
                "added_duration": round(max(0.0, output_duration - float(task.get("source_duration", 0.0))), 6),
            }
        )

    concat_list = work_dir / "v3_segments.txt"
    concat_list.write_text(
        "\n".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(39))}'" for path in prepared) + "\n",
        encoding="utf-8",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output_path),
        ],
        error_prefix="v3 音视频分段合并失败",
    )
    final_duration, final_fps, final_frames, final_has_audio = _video_track_info(output_path)
    if not final_has_audio:
        raise ValueError("v3 最终视频缺少音轨。")
    return {
        "source_duration": round(sum(float(item.get("source_duration", 0.0)) for item in tasks), 6),
        "output_duration": round(final_duration, 6),
        "added_duration": round(
            max(0.0, final_duration - sum(float(item.get("source_duration", 0.0)) for item in tasks)), 6
        ),
        "output_fps": round(final_fps, 6),
        "output_frame_count": final_frames,
        "use_original_audio": bool(use_original_audio),
        "segments": segment_reports,
    }


def _job_signature(source_path: str, manifest: dict[str, Any], values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    with open(source_path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    digest.update(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _asset_image_signatures(people: dict[str, Any], backgrounds: dict[str, Any]) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for prefix, collection in (("person", people), ("background", backgrounds)):
        for entry_id, image in collection.items():
            frame = _first_frame(image)
            array = np.asarray(frame.detach().cpu() if hasattr(frame, "detach") else frame)
            signatures[f"{prefix}_{entry_id}"] = hashlib.sha256(array.tobytes()).hexdigest()
    return signatures


def create_asset_manifest(
    *,
    asset_name: str,
    mapping_json: str,
    people: dict[str, Any],
    backgrounds: dict[str, Any],
) -> tuple[str, str]:
    version = str(time.time_ns())
    root = Path(folder_paths.get_output_directory()) / ASSET_ROOT_NAME / _safe_slug(asset_name, "assets") / version
    root.mkdir(parents=True, exist_ok=True)
    mapping = _string_mapping(mapping_json)
    people_meta = mapping.get("people", {}) if isinstance(mapping.get("people", {}), dict) else {}
    background_meta = mapping.get("backgrounds", {}) if isinstance(mapping.get("backgrounds", {}), dict) else {}
    people_entries = []
    for person_id in PERSON_IDS:
        image = people.get(person_id)
        if image is None:
            continue
        filename = f"person_{person_id}.png"
        path = root / filename
        _save_image_tensor(image, path)
        meta = people_meta.get(person_id, {})
        meta = meta if isinstance(meta, dict) else {"source": str(meta)}
        people_entries.append({
            "id": person_id,
            "source": meta.get("source", f"原人物 {person_id}"),
            "styled": _relative_output_path(path),
            "identity": meta.get("identity", ""),
        })
    background_entries = []
    for background_id in BACKGROUND_IDS:
        image = backgrounds.get(background_id)
        if image is None:
            continue
        filename = f"background_{background_id}.png"
        path = root / filename
        _save_image_tensor(image, path)
        meta = background_meta.get(background_id, {})
        meta = meta if isinstance(meta, dict) else {"source": str(meta)}
        background_entries.append({
            "id": background_id,
            "source": meta.get("source", f"原背景 {background_id}"),
            "styled": _relative_output_path(path),
            "description": meta.get("description", ""),
        })
    if not people_entries:
        raise ValueError("至少需要连接一张欧美化人物图片。")
    if not background_entries:
        raise ValueError("至少需要连接一张欧美化背景图片。")
    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "people": people_entries,
        "backgrounds": background_entries,
        "mapping": mapping,
    }
    manifest_path = root / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return json.dumps(manifest, ensure_ascii=False, indent=2), str(manifest_path)


def analyze_asset_mapping(
    *,
    people: dict[str, Any],
    backgrounds: dict[str, Any] | None = None,
    model: str,
) -> str:
    if not people:
        raise ValueError("资产映射分析至少需要一张欧美化人物图片。")
    backgrounds = backgrounds or {}
    if not backgrounds:
        raise ValueError("资产映射分析至少需要一张欧美化背景图片。")
    frames = []
    person_order = []
    background_order = []
    for person_id in PERSON_IDS:
        image = people.get(person_id)
        if image is None:
            continue
        frames.append(_first_frame(image))
        person_order.append(person_id)
    for background_id in BACKGROUND_IDS:
        image = backgrounds.get(background_id)
        if image is None:
            continue
        frames.append(_first_frame(image))
        background_order.append(background_id)
    image_batch = _vision_batch(frames)
    skill = (
        "你是长视频欧美化人物与背景资产映射分析器。输入图片按给定顺序对应人物和背景编号。"
        "只输出合法 JSON，不要 Markdown、标题或解释。JSON 根节点必须包含 people、backgrounds 和 mapping。"
        "people 是以人物编号为键的对象，每项包含 source 和 identity；identity 记录年龄感、性别、"
        "脸型、发型发色、服装、配饰和气质。backgrounds 是以背景编号为键的对象，每项包含 source 和 description；"
        "description 记录地点类型、空间结构、建筑或陈设、时间、天气、光线和欧美化后的视觉特征。"
        "mapping 必须包含 people 和 backgrounds 两个对象，分别明确原人物到欧美化人物、原背景到欧美化背景的对应关系。"
        "不得交换编号，不得把不同人物或不同背景合并。"
    )
    target = (
        f"人物图片顺序为：{', '.join(person_order)}；"
        f"背景图片顺序为：{', '.join(background_order)}。"
        "请生成供用户人工确认和修改的完整人物/背景映射 JSON。"
    )
    text = generate_openai_image_prompt_text(
        _get_config("gpttext"),
        skill=skill,
        modification_target=target,
        image=image_batch,
        model=model,
        temperature=0.1,
        max_tokens=2400,
    )
    parsed = _load_json_or_path(text, field_name="人物/背景映射分析结果")
    if not isinstance(parsed.get("people"), dict) or not isinstance(parsed.get("backgrounds"), dict):
        raise ValueError("人物/背景映射分析结果必须同时包含 people 和 backgrounds 对象。")
    mapping = parsed.get("mapping")
    if not isinstance(mapping, dict) or not isinstance(mapping.get("people"), dict) or not isinstance(mapping.get("backgrounds"), dict):
        raise ValueError("人物/背景映射分析结果的 mapping 必须同时包含 people 和 backgrounds 对象。")
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def load_long_video_assets(
    assets_manifest: str,
    *,
    people: dict[str, Any] | None = None,
    backgrounds: dict[str, Any] | None = None,
) -> tuple[LongVideoAssets, str]:
    manifest = _load_json_or_path(assets_manifest, field_name="资产清单")
    people_images, background_images = _resolve_assets(manifest, people or {}, backgrounds or {})
    assets = LongVideoAssets(manifest=manifest, people=people_images, backgrounds=background_images)
    summary = {
        "people": list(people_images),
        "backgrounds": list(background_images),
        "mapping": manifest.get("mapping", {}),
    }
    return assets, json.dumps(summary, ensure_ascii=False, indent=2)


def _migrate_job_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if int(manifest.get("version", 1)) >= 3:
        return manifest
    migrated = json.loads(json.dumps(manifest, ensure_ascii=False))
    legacy_adapter = get_video_engine_adapter(str(migrated.get("engine", "seedance")))
    for task in migrated.get("tasks", []):
        if not isinstance(task, dict):
            continue
        task.setdefault("source_start", task.get("start", 0.0))
        task.setdefault("source_duration", task.get("duration", 0.0))
        legacy_duration = float(task.get("duration", 0.0))
        task.setdefault(
            "request_duration",
            min(
                legacy_adapter.max_request_duration,
                max(legacy_adapter.min_request_duration, float(math.ceil(legacy_duration - 1e-6))),
            ),
        )
        task.setdefault("trim_offset", 0.0)
        task.setdefault("padding_start", 0.0)
        task.setdefault("padding_end", 0.0)
        task.setdefault("split_reason", "legacy_fixed")
    empty_hash = hashlib.sha256(b"[]").hexdigest()
    migrated["segmentation"] = {
        "requested_mode": "fixed",
        "effective_mode": "legacy_fixed_v2",
        "detector": "legacy_fixed_duration",
        "config": {"segment_duration": migrated.get("segment_duration", 10)},
        "boundaries": [],
        "boundaries_hash": empty_hash,
        "fallback_reason": "",
        "logical_shots": [],
    }
    migrated["version"] = 3
    migrated["migrated_from_version"] = int(manifest.get("version", 1))
    return migrated


def _task_timing_matches(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    fields = (
        "start",
        "duration",
        "source_start",
        "source_duration",
        "request_duration",
        "trim_offset",
        "padding_start",
        "padding_end",
    )
    return all(abs(float(previous.get(field, current[field])) - float(current[field])) <= 1e-6 for field in fields)


def _task_source_timing_matches(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    fields = ("start", "duration", "source_start", "source_duration")
    return all(abs(float(previous.get(field, current[field])) - float(current[field])) <= 1e-6 for field in fields)


def _quick_file_fingerprint(path: Path) -> tuple[int, str]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    return size, digest.hexdigest()


def _auto_resume_settings_match(candidate: dict[str, Any], values: dict[str, Any], segmentation: dict[str, Any]) -> bool:
    if candidate.get("asset_mode") != "auto_shot_assets":
        return False
    candidate_settings = candidate.get("settings", {})
    if not isinstance(candidate_settings, dict):
        return False
    for field in ("engine", "model", "prompt", "negative_prompt", "ai_model"):
        if candidate_settings.get(field) != values.get(field):
            return False
    candidate_options = dict(candidate.get("auto_asset_options") or {})
    current_options = dict(values.get("auto_asset_options") or {})
    candidate_options.pop("force_rerun_assets", None)
    current_options.pop("force_rerun_assets", None)
    candidate_options.setdefault("target_resource_type", "")
    current_options.setdefault("target_resource_type", "")
    if (
        not candidate_options["target_resource_type"]
        and candidate_options.get("visual_style") == AUTO_ASSET_STYLE_ANIME
        and current_options.get("target_resource_type") == "二维动漫资源"
    ):
        candidate_options["target_resource_type"] = current_options["target_resource_type"]
    if candidate_options != current_options:
        return False
    candidate_segmentation = candidate.get("segmentation", {})
    return (
        isinstance(candidate_segmentation, dict)
        and candidate_segmentation.get("boundaries_hash") == segmentation.get("boundaries_hash")
        and candidate_segmentation.get("effective_mode") == segmentation.get("effective_mode")
    )


def _find_compatible_auto_resume_manifest(
    *,
    output_root: Path,
    source_path: Path,
    values: dict[str, Any],
    segmentation: dict[str, Any],
    source_duration: float,
    skip_path: Path,
) -> tuple[dict[str, Any], Path] | None:
    try:
        source_fingerprint = _quick_file_fingerprint(source_path)
    except OSError:
        return None
    candidates = sorted(
        (path for path in (output_root / JOB_ROOT_NAME).glob("*/manifest.json") if path != skip_path),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:32]
    for candidate_path in candidates:
        try:
            candidate = _migrate_job_manifest(json.loads(candidate_path.read_text(encoding="utf-8")))
            candidate_source = Path(str(candidate.get("source_path") or ""))
            if not candidate_source.is_file() or abs(float(candidate.get("source_duration", -1.0)) - source_duration) > 0.05:
                continue
            if not _auto_resume_settings_match(candidate, values, segmentation):
                continue
            if _quick_file_fingerprint(candidate_source) != source_fingerprint:
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        return candidate, candidate_path
    return None


def _rebase_auto_asset_paths(value: Any, *, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rebase_auto_asset_paths(item, old_root=old_root, new_root=new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_auto_asset_paths(item, old_root=old_root, new_root=new_root) for item in value]
    if isinstance(value, str):
        try:
            relative = Path(value).relative_to(old_root)
        except ValueError:
            return value
        return str(new_root / relative)
    return value


def _manual_batch_retry_assets_compatible(
    candidate: dict[str, Any],
    *,
    values: dict[str, Any],
    segmentation: dict[str, Any],
    processing_contract_version: int,
) -> bool:
    if int(candidate.get("processing_contract_version", 0)) != int(processing_contract_version):
        return False
    if candidate.get("asset_mode") != "auto_shot_assets":
        return False
    candidate_settings = candidate.get("settings") if isinstance(candidate.get("settings"), dict) else {}
    for field in ("engine", "model", "prompt", "negative_prompt", "ai_model"):
        if candidate_settings.get(field) != values.get(field):
            return False
    candidate_options = dict(candidate.get("auto_asset_options") or {})
    current_options = dict(values.get("auto_asset_options") or {})
    candidate_options.pop("manual_batch", None)
    current_options.pop("manual_batch", None)
    candidate_options.pop("force_rerun_assets", None)
    current_options.pop("force_rerun_assets", None)
    if candidate_options != current_options:
        return False
    candidate_segmentation = candidate.get("segmentation") if isinstance(candidate.get("segmentation"), dict) else {}
    return (
        candidate_segmentation.get("boundaries_hash") == segmentation.get("boundaries_hash")
        and candidate_segmentation.get("effective_mode") == segmentation.get("effective_mode")
    )


def _inherit_manual_batch_retry_asset_root(
    *,
    old_job_dir: Path,
    new_job_dir: Path,
    task: dict[str, Any],
    reuse_threshold: float = AUTO_ASSET_DEFAULT_REUSE_THRESHOLD,
) -> None:
    old_root = old_job_dir / "shot_assets" / f"shot_{int(task['index']):04d}"
    new_root = new_job_dir / "shot_assets" / f"shot_{int(task['index']):04d}"
    if not old_root.is_dir() or old_root == new_root:
        return
    shutil.copytree(old_root, new_root, dirs_exist_ok=True)
    for key in (
        "auto_asset_analysis",
        "auto_assets",
        "auto_asset_errors",
        "auto_asset_warnings",
        "auto_asset_suspected_matches",
    ):
        if key in task:
            task[key] = _rebase_auto_asset_paths(task[key], old_root=old_root, new_root=new_root)
    task["auto_asset_reused_from"] = str(old_root)
    _reset_manual_batch_quality_retry_state(task, new_root, reuse_threshold=reuse_threshold)
    _write_auto_asset_manifest(new_root, task)


def _reset_manual_batch_quality_retry_state(
    task: dict[str, Any],
    root: Path,
    *,
    reuse_threshold: float,
) -> bool:
    """Reopen only a retry whose base masters are intact and frame quality failed."""
    if task.get("auto_asset_status") not in {"degraded", "failed"}:
        return False
    errors = task.get("auto_asset_errors") if isinstance(task.get("auto_asset_errors"), list) else []
    if not errors or any(str(item.get("kind") or "") != "integrated_frame" for item in errors if isinstance(item, dict)):
        return False
    if any(not isinstance(item, dict) or str(item.get("error_kind") or "") not in {
        "quality_gate_failed",
        "generation_or_quality_failed",
    } for item in errors):
        return False
    if not _auto_asset_task_masters_complete(task, root, reuse_threshold=reuse_threshold):
        return False
    assets = task.get("auto_assets")
    if isinstance(assets, dict):
        # The old candidates are retained on disk for inspection, but are not
        # eligible for packing after a retry starts.
        assets["integrated_frames"] = []
    task["auto_asset_errors"] = []
    task["auto_asset_status"] = "masters_ready"
    task["auto_asset_retry_reason"] = "integrated_frame_quality_gate"
    return True


def plan_long_video_job(
    *,
    video: Any,
    assets: LongVideoAssets,
    prompt: str,
    engine: str,
    model: str,
    segment_duration: int,
    ai_model: str,
    max_retries: int,
    resume: bool,
    force_rerun: bool,
    negative_prompt: str,
    shot_plan: LongVideoShotPlan | None = None,
    asset_mode: str = "manual",
    auto_asset_options: dict[str, Any] | None = None,
) -> LongVideoJob:
    total_duration = float(video.get_duration())
    with tempfile.TemporaryDirectory(prefix="company_long_video_plan_") as temporary:
        temporary_root = Path(temporary).resolve()
        source_path = _video_source_path(video, temporary_root)
        _file_duration, fps, _frame_count = _video_file_info(source_path, total_duration)
        if shot_plan is None:
            boundaries, shots = _fixed_shots(total_duration, int(segment_duration))
            shot_plan = LongVideoShotPlan(
                video=video,
                total_duration=total_duration,
                fps=fps,
                requested_mode="fixed",
                effective_mode="fixed",
                fixed_duration=int(segment_duration),
                sensitivity="标准",
                use_audio_silence=False,
                auto_fallback=True,
                detector="fixed_duration",
                config={"segment_duration": int(segment_duration), "minimum_request_duration": MIN_REQUEST_DURATION},
                boundaries=boundaries,
                shots=shots,
            )
        elif abs(shot_plan.total_duration - total_duration) > max(1.0 / max(fps, 1.0), 0.05):
            raise ValueError("镜头计划与当前输入视频时长不一致，请重新运行镜头检测节点。")
        request_segments, duration_adaptation = adapt_shot_plan_to_requests(
            shot_plan,
            engine=engine,
            source_path=source_path,
        )
        segmentation = shot_plan.segmentation_dict()
        segmentation["duration_adaptation"] = duration_adaptation
        values = {
            "engine": engine,
            "model": model,
            "segment_duration": int(segment_duration),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "ai_model": ai_model,
            "asset_images": _asset_image_signatures(assets.people, assets.backgrounds),
            "segmentation": {
                "effective_mode": segmentation["effective_mode"],
                "boundaries_hash": segmentation["boundaries_hash"],
                "config": segmentation["config"],
                "requests": duration_adaptation["requests"],
            },
            "asset_mode": asset_mode,
            "auto_asset_options": auto_asset_options or {},
        }
        signature = _job_signature(source_path, assets.manifest, values)
        legacy_values = dict(values)
        legacy_values.pop("segmentation", None)
        legacy_values.pop("asset_mode", None)
        legacy_values.pop("auto_asset_options", None)
        legacy_signature = _job_signature(source_path, assets.manifest, legacy_values)
        job_id = signature[:20]
        job_dir = Path(folder_paths.get_output_directory()) / JOB_ROOT_NAME / job_id
        legacy_manifest_path = (
            Path(folder_paths.get_output_directory()) / JOB_ROOT_NAME / legacy_signature[:20] / "manifest.json"
        )
        job_dir.mkdir(parents=True, exist_ok=True)
        source_candidate = Path(source_path).resolve()
        if source_candidate.is_relative_to(temporary_root):
            persisted_source = job_dir / "source" / "source_video.mp4"
            persisted_source.parent.mkdir(parents=True, exist_ok=True)
            if force_rerun or not persisted_source.is_file():
                shutil.copy2(source_candidate, persisted_source)
        else:
            persisted_source = source_candidate

    manifest_path = job_dir / "manifest.json"
    existing = None
    resumed_from_manifest = ""
    reuse_auto_assets = (
        asset_mode == "auto_shot_assets"
        and resume
        and not force_rerun
        and not bool((auto_asset_options or {}).get("force_rerun_assets", False))
    )
    if resume and not force_rerun:
        candidates = [manifest_path]
        if legacy_manifest_path != manifest_path:
            candidates.append(legacy_manifest_path)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                existing = _migrate_job_manifest(json.loads(candidate.read_text(encoding="utf-8")))
                resumed_from_manifest = str(candidate)
                break
            except json.JSONDecodeError:
                existing = None
    if existing is None and reuse_auto_assets:
        compatible = _find_compatible_auto_resume_manifest(
            output_root=Path(folder_paths.get_output_directory()),
            source_path=Path(persisted_source),
            values=values,
            segmentation=segmentation,
            source_duration=total_duration,
            skip_path=manifest_path,
        )
        if compatible is not None:
            existing, source_manifest_path = compatible
            resumed_from_manifest = str(source_manifest_path)
    existing_tasks = {
        int(item.get("index")): item
        for item in (existing or {}).get("tasks", [])
        if isinstance(item, dict) and item.get("index") is not None
    }
    tasks = []
    for request in request_segments:
        index = request.index
        previous = existing_tasks.get(index, {})
        result_path = job_dir / "segments" / f"segment_{index:04d}.mp4"
        task = {
            **request.to_dict(),
            "status": "planned",
            "attempts": 0,
            "result": str(result_path),
        }
        previous_result = _asset_path_if_file(previous.get("result")) if previous else None
        can_reuse_completed_result = (
            previous.get("status") == "success"
            and previous_result is not None
            and _task_source_timing_matches(previous, task)
        )
        if previous and not force_rerun and (_task_timing_matches(previous, task) or can_reuse_completed_result):
            task.update(previous)
        elif previous and reuse_auto_assets and _task_source_timing_matches(previous, task):
            old_source_frames = previous.get("auto_assets", {}).get("source_frames", {})
            old_start_frame = _asset_path_if_file(old_source_frames.get("source_start"))
            old_root = old_start_frame.parent if old_start_frame else None
            new_root = job_dir / "shot_assets" / f"shot_{index:04d}"
            if old_root and old_root.is_dir() and (old_root / "manifest.json").is_file():
                shutil.copytree(old_root, new_root, dirs_exist_ok=True)
                for key in (
                    "auto_asset_analysis",
                    "auto_assets",
                    "auto_asset_errors",
                    "auto_asset_warnings",
                    "auto_asset_status",
                ):
                    if key in previous:
                        task[key] = _rebase_auto_asset_paths(previous[key], old_root=old_root, new_root=new_root)
                task["auto_asset_reused_from"] = str(old_root)
                _write_auto_asset_manifest(new_root, task)
        tasks.append(task)
    job_manifest = {
        "version": AUTO_ASSET_MANIFEST_VERSION if asset_mode == "auto_shot_assets" else 3,
        "job_id": job_id,
        "signature": signature,
        "source_path": str(persisted_source),
        "source_duration": total_duration,
        "engine": engine,
        "model": model,
        "segment_duration": int(segment_duration),
        "continuity": "soft_previous_end_frame",
        "status": "planned",
        "assets": assets.manifest,
        "asset_mode": asset_mode,
        "settings": values,
        "segmentation": segmentation,
        "tasks": tasks,
    }
    if auto_asset_options:
        job_manifest["auto_asset_options"] = auto_asset_options
    if resumed_from_manifest and Path(resumed_from_manifest) != manifest_path:
        job_manifest["resumed_from_manifest"] = resumed_from_manifest
    for key in ("output_geometry", "final"):
        if existing and key in existing and not force_rerun:
            job_manifest[key] = existing[key]
    _atomic_write_json(manifest_path, job_manifest)
    return LongVideoJob(
        video=video,
        assets=assets,
        prompt=prompt,
        engine=engine,
        model=model,
        segment_duration=int(segment_duration),
        ai_model=ai_model,
        max_retries=int(max_retries),
        resume=bool(resume),
        force_rerun=bool(force_rerun),
        negative_prompt=negative_prompt,
        total_duration=total_duration,
        source_path=str(persisted_source),
        job_dir=job_dir,
        manifest_path=manifest_path,
        manifest=job_manifest,
    )


def plan_long_video_auto_asset_job(
    *,
    shot_plan: LongVideoShotPlan,
    prompt: str,
    engine: str,
    model: str,
    ai_model: str,
    image_model: str,
    image_quality: str,
    reuse_threshold: float,
    max_retries: int,
    resume: bool,
    force_rerun: bool,
    force_rerun_assets: bool,
    negative_prompt: str,
    visual_style: str = AUTO_ASSET_STYLE_WESTERN,
    send_source_video: bool = True,
    image_provider: str = "WisArt",
    target_resource_type: str = "",
    processing_contract_version: int = 0,
    use_original_audio: bool = True,
    use_integrated_frame_references: bool = False,
    manual_batch: dict[str, Any] | None = None,
    identity_mapping: Any = "",
) -> LongVideoJob:
    options = {
        "image_model": str(image_model or "gpt-image-2"),
        "image_quality": str(image_quality or "medium"),
        "image_provider": str(image_provider or "WisArt"),
        "reuse_threshold": max(0.5, min(1.0, float(reuse_threshold))),
        "force_rerun_assets": bool(force_rerun_assets),
        "prompt_version": AUTO_ASSET_PROMPT_VERSION,
        "max_people_per_shot": AUTO_ASSET_MAX_PEOPLE,
        "visual_style": _normalize_auto_asset_style(visual_style),
        "send_source_video": bool(send_source_video),
        "use_integrated_frame_references": bool(use_integrated_frame_references),
        "target_resource_type": str(target_resource_type or ""),
    }
    contract_version = int(processing_contract_version)
    if contract_version in {V3_PROCESSING_CONTRACT_VERSION, MANUAL_BATCH_PROCESSING_CONTRACT_VERSION}:
        options.update(
            {
                "processing_contract_version": contract_version,
                "use_original_audio": bool(use_original_audio),
                "grouping_version": V3_GROUPING_VERSION,
                "reference_package_version": (
                    MANUAL_BATCH_REFERENCE_PACKAGE_VERSION
                    if contract_version == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION
                    else V3_REFERENCE_PACKAGE_VERSION
                ),
            }
        )
        if contract_version == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION:
            options["manual_batch"] = dict(manual_batch or {})
        normalized_identity_mapping = _load_identity_mapping(identity_mapping)
        if (
            normalized_identity_mapping["global_people"]
            or normalized_identity_mapping["shot_people"]
            or normalized_identity_mapping["expected_distinct_people"]
        ):
            options["identity_mapping"] = normalized_identity_mapping
    auto_assets = LongVideoAssets(
        manifest={
            "version": 1,
            "mode": "auto_shot_assets",
            "prompt_version": AUTO_ASSET_PROMPT_VERSION,
            "auto_asset_options": options,
            "people": [],
            "backgrounds": [],
        },
        people={},
        backgrounds={},
    )
    if contract_version not in {V3_PROCESSING_CONTRACT_VERSION, MANUAL_BATCH_PROCESSING_CONTRACT_VERSION}:
        return plan_long_video_job(
            video=shot_plan.video,
            assets=auto_assets,
            prompt=prompt,
            engine=engine,
            model=model,
            segment_duration=int(shot_plan.fixed_duration),
            ai_model=ai_model,
            max_retries=max_retries,
            resume=resume,
            force_rerun=force_rerun,
            negative_prompt=negative_prompt,
            shot_plan=shot_plan,
            asset_mode="auto_shot_assets",
            auto_asset_options=options,
        )
    return plan_long_video_auto_asset_v3_job(
        shot_plan=shot_plan,
        assets=auto_assets,
        prompt=prompt,
        engine=engine,
        model=model,
        ai_model=ai_model,
        max_retries=max_retries,
        resume=resume,
        force_rerun=force_rerun,
        negative_prompt=negative_prompt,
        auto_asset_options=options,
        processing_contract_version=contract_version,
        manual_batch=manual_batch,
    )


def plan_long_video_auto_asset_v3_job(
    *,
    shot_plan: LongVideoShotPlan,
    assets: LongVideoAssets,
    prompt: str,
    engine: str,
    model: str,
    ai_model: str,
    max_retries: int,
    resume: bool,
    force_rerun: bool,
    negative_prompt: str,
    auto_asset_options: dict[str, Any],
    processing_contract_version: int = V3_PROCESSING_CONTRACT_VERSION,
    manual_batch: dict[str, Any] | None = None,
) -> LongVideoJob:
    total_duration = float(shot_plan.video.get_duration())
    with tempfile.TemporaryDirectory(prefix="company_long_video_v3_plan_") as temporary:
        temporary_root = Path(temporary).resolve()
        source_path = _video_source_path(shot_plan.video, temporary_root)
        members, groups, adaptation = build_v3_logical_members_and_request_groups(
            shot_plan,
            engine=engine,
            source_path=source_path,
        )
        segmentation = shot_plan.segmentation_dict()
        segmentation["duration_adaptation"] = adaptation
        values = {
            "engine": engine,
            "model": model,
            "segment_duration": int(shot_plan.fixed_duration),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "ai_model": ai_model,
            "asset_mode": "auto_shot_assets",
            "auto_asset_options": auto_asset_options,
            "processing_contract_version": int(processing_contract_version),
            "request_groups": adaptation["request_groups"],
        }
        signature = _job_signature(source_path, assets.manifest, values)
        job_id = signature[:20]
        if int(processing_contract_version) == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION:
            batch_root = Path(str((manual_batch or {}).get("attempt_dir") or "")).resolve()
            output_root = Path(folder_paths.get_output_directory()).resolve()
            if not batch_root.is_relative_to(output_root):
                raise ValueError("手动批次任务目录必须位于 ComfyUI output 目录内。")
            expected_batch_root = _manual_batch_attempt_dir(
                str((manual_batch or {}).get("series_id") or ""),
                str((manual_batch or {}).get("batch_id") or ""),
                int((manual_batch or {}).get("attempt", 0)),
            )
            if batch_root != expected_batch_root:
                raise ValueError("手动批次任务目录与系列状态中的批次不一致。")
            _manual_batch_validate_state_path(
                str((manual_batch or {}).get("series_id") or ""),
                str((manual_batch or {}).get("state_path") or ""),
            )
            job_dir = (batch_root / "job").resolve()
            if not job_dir.is_relative_to(batch_root):
                raise ValueError("手动批次 job 目录必须位于当前 attempt 目录内。")
        else:
            job_dir = Path(folder_paths.get_output_directory()) / JOB_ROOT_NAME / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        source_candidate = Path(source_path).resolve()
        if source_candidate.is_relative_to(temporary_root):
            persisted_source = job_dir / "source" / "source_video.mp4"
            persisted_source.parent.mkdir(parents=True, exist_ok=True)
            if force_rerun or not persisted_source.is_file():
                shutil.copy2(source_candidate, persisted_source)
        else:
            persisted_source = source_candidate

    manifest_path = job_dir / "manifest.json"
    existing = None
    resumed_from_manifest = ""
    if resume and not force_rerun and manifest_path.is_file():
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                candidate.get("signature") == signature
                and int(candidate.get("processing_contract_version", 0)) == int(processing_contract_version)
            ):
                existing = candidate
                resumed_from_manifest = str(manifest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing = None

    retry_source_job_dir: Path | None = None
    if (
        existing is None
        and resume
        and not force_rerun
        and int(processing_contract_version) == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION
    ):
        retry_candidate = _manual_batch_retry_manifest(manual_batch)
        if retry_candidate is not None:
            candidate, candidate_path = retry_candidate
            if _manual_batch_retry_assets_compatible(
                candidate,
                values=values,
                segmentation=segmentation,
                processing_contract_version=processing_contract_version,
            ):
                existing = candidate
                resumed_from_manifest = str(candidate_path)
                retry_source_job_dir = candidate_path.parent
                source_cache = retry_source_job_dir / "asset_cache" / "index.json"
                if source_cache.is_file():
                    try:
                        cache = json.loads(source_cache.read_text(encoding="utf-8"))
                        _atomic_write_json(
                            job_dir / "asset_cache" / "index.json",
                            _rebase_auto_asset_paths(cache, old_root=retry_source_job_dir, new_root=job_dir),
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        logging.warning("Unable to inherit auto asset cache for manual batch retry.", exc_info=True)

    existing_members = {
        int(item["index"]): item
        for item in (existing or {}).get("logical_member_tasks", [])
        if isinstance(item, dict) and item.get("index") is not None
    }
    member_tasks = []
    for member in members:
        task = {**member.to_dict(), "status": "planned"}
        previous = existing_members.get(member.index)
        if previous and not force_rerun and _task_source_timing_matches(previous, task):
            task.update(previous)
            if retry_source_job_dir is not None:
                _inherit_manual_batch_retry_asset_root(
                    old_job_dir=retry_source_job_dir,
                    new_job_dir=job_dir,
                    task=task,
                    reuse_threshold=float((auto_asset_options or {}).get("reuse_threshold", AUTO_ASSET_DEFAULT_REUSE_THRESHOLD)),
                )
        member_tasks.append(task)

    existing_groups = {
        int(item["index"]): item
        for item in (existing or {}).get("tasks", [])
        if isinstance(item, dict) and item.get("index") is not None
    }
    tasks = []
    previous_group: RequestSegment | None = None
    for group in groups:
        task = {
            **group.to_dict(),
            "status": "planned",
            "attempts": 0,
            "result": str(job_dir / "segments" / f"segment_{group.index:04d}.mp4"),
            "continuity_from_previous_group": _request_group_continues_same_logical_shot(previous_group, group),
        }
        previous = existing_groups.get(group.index)
        if previous and not force_rerun and _task_timing_matches(previous, task):
            task.update(previous)
        task["continuity_from_previous_group"] = _request_group_continues_same_logical_shot(previous_group, group)
        tasks.append(task)
        previous_group = group

    manifest = {
        "version": AUTO_ASSET_MANIFEST_VERSION + 1,
        "processing_contract_version": int(processing_contract_version),
        "job_id": job_id,
        "signature": signature,
        "source_path": str(persisted_source),
        "source_duration": total_duration,
        "engine": engine,
        "model": model,
        "segment_duration": int(shot_plan.fixed_duration),
        "continuity": "same_logical_shot_only",
        "status": "planned",
        "assets": assets.manifest,
        "asset_mode": "auto_shot_assets",
        "settings": values,
        "segmentation": segmentation,
        "auto_asset_options": auto_asset_options,
        "logical_member_tasks": member_tasks,
        "tasks": tasks,
    }
    if int(processing_contract_version) == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION:
        manifest["batch_contract"] = MANUAL_BATCH_CONTRACT
        manifest["manual_batch"] = dict(manual_batch or {})
    if resumed_from_manifest and Path(resumed_from_manifest) != manifest_path:
        manifest["resumed_from_manifest"] = resumed_from_manifest
    if existing and not force_rerun and "reference_package_resume_key" in existing:
        manifest["reference_package_resume_key"] = existing["reference_package_resume_key"]
    if existing and not force_rerun and "output_geometry" in existing:
        manifest["output_geometry"] = existing["output_geometry"]
    _atomic_write_json(manifest_path, manifest)
    return LongVideoJob(
        video=shot_plan.video,
        assets=assets,
        prompt=prompt,
        engine=engine,
        model=model,
        segment_duration=int(shot_plan.fixed_duration),
        ai_model=ai_model,
        max_retries=max_retries,
        resume=resume,
        force_rerun=force_rerun,
        negative_prompt=negative_prompt,
        total_duration=total_duration,
        source_path=str(persisted_source),
        job_dir=job_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def analyze_long_video_job(job: LongVideoJob) -> LongVideoJob:
    progress = ProgressBar(len(job.manifest["tasks"]))
    for task in job.manifest["tasks"]:
        if job.resume and not job.force_rerun and task.get("analysis"):
            progress.update(1)
            continue
        source_start = float(task.get("source_start", task["start"]))
        source_duration = float(task.get("source_duration", task["duration"]))
        source_segment = job.video.as_trimmed(source_start, source_duration, strict_duration=True)
        if source_segment is None:
            raise ValueError(
                f"第 {task['index']} 段无法从原视频取得 {source_duration:.3f} 秒画面。"
            )
        task["analysis"] = _segment_analysis(
            _sample_frames(source_segment),
            manifest=job.assets.manifest,
            model=job.ai_model,
            people_images=job.assets.people,
            background_images=job.assets.backgrounds,
        )
        if task.get("status") != "success":
            task["status"] = "analyzed"
        _atomic_write_json(job.manifest_path, job.manifest)
        progress.update(1)
    job.manifest["status"] = "analyzed"
    _atomic_write_json(job.manifest_path, job.manifest)
    return job


def match_long_video_references(job: LongVideoJob) -> LongVideoJob:
    adapter = get_video_engine_adapter(job.engine)
    for task in job.manifest["tasks"]:
        analysis = task.get("analysis")
        if not isinstance(analysis, dict):
            raise ValueError(f"第 {task['index']} 段尚未完成 GPT 人物/背景分析。")
        people_ids = [item for item in analysis.get("people", []) if item in job.assets.people]
        background_ids = [item for item in analysis.get("backgrounds", []) if item in job.assets.backgrounds]
        if not people_ids or not background_ids:
            raise ValueError(f"第 {task['index']} 段没有可用的人物或背景资产匹配结果。")
        roles = (["people_contact_sheet"] if adapter.key == "wan" and len(people_ids) > 1 else [f"person_{item}" for item in people_ids])
        roles.append(f"background_{background_ids[0]}")
        if task["index"] > 1:
            roles.append("previous_segment_end_frame")
        task["selected_assets"] = {"people": people_ids, "backgrounds": background_ids[:1]}
        task["reference_roles"] = roles[: adapter.reference_image_limit]
        task["prompt"] = _build_video_prompt(job.prompt, analysis, soft_continuity=task["index"] > 1)
        if task.get("status") != "success":
            task["status"] = "matched"
    job.manifest["status"] = "matched"
    _atomic_write_json(job.manifest_path, job.manifest)
    return job


def _references_from_auto_package(
    adapter: VideoEngineAdapter,
    task: dict[str, Any],
    *,
    previous_end_frame: Any,
    continuity_role: str = "previous_segment_end_frame",
) -> tuple[list[Any], list[str]]:
    package = task.get("reference_package")
    if not isinstance(package, dict):
        raise ValueError(f"第 {task['index']} 段尚未完成自动参考素材打包。")
    images: list[Any] = []
    roles: list[str] = []
    for item in package.get("items", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "auto_asset")
        asset_id = str(item.get("asset_id") or "").strip()
        if adapter.key == "seedance" and asset_id.startswith("asset-"):
            images.append(SeedanceAssetReference(asset_id=asset_id, role=role))
            roles.append(role)
            continue
        path = _asset_path_if_file(item.get("path"))
        if path is None:
            continue
        images.append(_load_image_file(path))
        roles.append(role)
    if previous_end_frame is not None:
        images.append(previous_end_frame)
        roles.append(continuity_role)
    return images[: adapter.reference_image_limit], roles[: adapter.reference_image_limit]


def _parallel_video_progress_path(job: LongVideoJob) -> Path:
    return job.job_dir / "parallel_video_progress.json"


def _parallel_video_saved_result(path: Path) -> dict[str, str] | None:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    try:
        relative = path.resolve().relative_to(output_root)
    except ValueError:
        return None
    return {
        "filename": relative.name,
        "subfolder": "" if str(relative.parent) == "." else relative.parent.as_posix(),
        "type": "output",
    }


def _send_parallel_video_progress_event(payload: dict[str, Any]) -> None:
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
        if context is not None:
            payload["node"] = context.node_id
            payload["prompt_id"] = context.prompt_id
    except Exception:
        logging.debug("Unable to read ComfyUI executing context for parallel video progress.", exc_info=True)

    try:
        from server import PromptServer

        instance = PromptServer.instance
        send_sync = getattr(instance, "send_sync", None)
        if callable(send_sync):
            client_id = getattr(instance, "client_id", None)
            send_sync("company_remote.parallel_video_progress", payload, client_id)
    except Exception:
        logging.debug("Unable to send parallel video progress event.", exc_info=True)


def _emit_parallel_video_progress(
    job: LongVideoJob,
    *,
    phase: str,
    message: str,
    completed_segments: list[dict[str, Any]],
    running_segments: list[int],
    failed_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(job.manifest.get("tasks", []))
    completed_segments = sorted(completed_segments, key=lambda item: int(item.get("sequence", 0)))
    running_segments = sorted({int(item) for item in running_segments})
    failed_segments = sorted(failed_segments, key=lambda item: int(item.get("sequence", 0)))
    payload: dict[str, Any] = {
        "event": "company_remote.parallel_video_progress",
        "job_id": str(job.manifest.get("job_id") or ""),
        "manifest": str(job.manifest_path),
        "progress_path": str(_parallel_video_progress_path(job)),
        "value": len(completed_segments),
        "total": total,
        "percent": round((len(completed_segments) / max(1, total)) * 100.0, 1),
        "phase": phase,
        "message": message,
        "completed_segments": completed_segments,
        "running_segments": running_segments,
        "failed_segments": failed_segments,
        "completed_count": len(completed_segments),
        "running_count": len(running_segments),
        "failed_count": len(failed_segments),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    job.manifest["parallel_video_progress"] = payload
    _atomic_write_json(job.manifest_path, job.manifest)
    _atomic_write_json(_parallel_video_progress_path(job), payload)
    _send_parallel_video_progress_event(dict(payload))
    return payload


def _parallel_video_task_worker(
    job: LongVideoJob,
    task: dict[str, Any],
    adapter: VideoEngineAdapter,
    *,
    send_source_video: bool,
) -> dict[str, Any]:
    task_copy = deepcopy(task)
    index = int(task_copy["index"])
    source_segment = _source_segment_for_task(job, task_copy) if send_source_video else None
    if send_source_video and source_segment is None:
        raise ValueError(f"第 {index} 段无法读取原视频画面。")

    if job.manifest.get("asset_mode") == "auto_shot_assets":
        if task_copy.get("reference_package_status") != "ready":
            raise ValueError(
                f"第 {index} 段自动参考素材不可用，状态为 {task_copy.get('reference_package_status')}。"
            )
        references, roles = _references_from_auto_package(adapter, task_copy, previous_end_frame=None)
    else:
        selected = task_copy.get("selected_assets")
        if not isinstance(selected, dict):
            raise ValueError(f"第 {index} 段尚未完成参考素材匹配。")
        selected_people = [
            job.assets.people[item] for item in selected["people"] if item in job.assets.people
        ]
        selected_backgrounds = [
            job.assets.backgrounds[item]
            for item in selected["backgrounds"]
            if item in job.assets.backgrounds
        ]
        references, roles = _reference_images_for_task(
            adapter,
            selected_people=selected_people,
            selected_backgrounds=selected_backgrounds,
            previous_end_frame=None,
        )

    # Parallel requests deliberately do not wait for the previous segment's end frame.
    prompt = str(task_copy.get("prompt") or "").strip()
    prompt = (
        f"{prompt}\n\n并行生成模式：本段独立生成，不提供上一段末帧，也不等待其他分段；"
        "严格依据本段参考人物、背景图片和剧情分析完成完整镜头。"
    )
    include_source_video = send_source_video
    privacy_fallback_used = False
    last_error = ""
    provider_policy_blocked = False
    attempts = max(1, int(job.max_retries) + 1)
    for attempt in range(1, attempts + 2):
        try:
            request_duration = float(task_copy.get("request_duration", task_copy["duration"]))
            _, remote_path = adapter.generate(
                source_segment=source_segment,
                references=references,
                prompt=prompt,
                model=job.model,
                duration=request_duration,
                negative_prompt=job.negative_prompt,
                include_source_video=include_source_video,
            )
            return {
                "index": index,
                "status": "success",
                "attempts": attempt,
                "remote_path": str(remote_path),
                "reference_roles": roles,
                "source_video_sent": include_source_video,
                "prompt": prompt,
                "privacy_fallback": task_copy.get("privacy_fallback"),
            }
        except Exception as exc:
            last_error = str(exc)
            provider_policy_blocked = _provider_policy_error(exc)
            if include_source_video and _real_person_privacy_error(exc) and not privacy_fallback_used:
                privacy_fallback_used = True
                include_source_video = False
                task_copy["privacy_fallback"] = {
                    "used": True,
                    "reason": "real_person_privacy_detected",
                    "original_error": last_error,
                    "fallback": "reference_images_and_story_text_only",
                }
                if job.manifest.get("asset_mode") == "auto_shot_assets":
                    prompt = _build_auto_asset_video_prompt(
                        job.prompt,
                        task_copy.get("auto_asset_analysis", {}),
                        reference_roles=roles,
                        soft_continuity=False,
                        visual_style=str(
                            (job.manifest.get("auto_asset_options") or {}).get("visual_style")
                            or AUTO_ASSET_STYLE_WESTERN
                        ),
                        send_source_video=False,
                    )
                    prompt = (
                        f"{prompt}\n\n并行生成模式：本段独立生成，不提供上一段末帧，也不等待其他分段；"
                        "严格依据本段参考人物、背景图片和剧情分析完成完整镜头。"
                    )
                continue
            if provider_policy_blocked or privacy_fallback_used or attempt > job.max_retries:
                break
            time.sleep(min(2.0 * attempt, 8.0))

    return {
        "index": index,
        "status": "blocked_by_provider_policy" if provider_policy_blocked else "failed",
        "attempts": min(attempts, job.max_retries + 1),
        "error": last_error,
        "reference_roles": roles,
        "source_video_sent": include_source_video,
        "prompt": prompt,
        "privacy_fallback": task_copy.get("privacy_fallback"),
    }


def generate_long_video_segments_parallel(job: LongVideoJob, concurrency: int = 3) -> LongVideoJob:
    """Generate independent segments concurrently and stream completed previews to the UI."""
    if int(job.manifest.get("processing_contract_version", 0)) == V3_PROCESSING_CONTRACT_VERSION:
        raise ValueError("v3 请求组依赖顺序连续帧和音频合同，不能接入并行分段生成节点。")
    adapter = get_video_engine_adapter(job.engine)
    options = job.manifest.get("auto_asset_options", {})
    options = options if isinstance(options, dict) else {}
    send_source_video = bool(options.get("send_source_video", True))
    concurrency = max(1, min(8, int(concurrency)))
    tasks = job.manifest["tasks"]
    total = len(tasks)
    completed_segments: list[dict[str, Any]] = []
    failed_segments: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for task in tasks:
        result_path = Path(task["result"])
        if job.resume and not job.force_rerun and task.get("status") == "success" and result_path.is_file():
            saved = _parallel_video_saved_result(result_path)
            completed_segments.append(
                {
                    "sequence": int(task["index"]),
                    "video": str(result_path),
                    **({"preview": saved} if saved else {}),
                    "reused": True,
                }
            )
            continue
        if job.manifest.get("asset_mode") == "auto_shot_assets" and task.get("reference_package_status") != "ready":
            task["status"] = "blocked_by_asset_failure"
            error = {
                "sequence": int(task["index"]),
                "error": f"自动参考素材不可用：{task.get('reference_package_status')}",
            }
            failed_segments.append(error)
            continue
        task["status"] = "running"
        task["attempts"] = 0
        pending.append(task)

    progress = ProgressBar(max(1, total))
    progress.update_absolute(len(completed_segments), max(1, total))
    _emit_parallel_video_progress(
        job,
        phase="submitted" if pending else "completed",
        message=(
            f"已提交 {len(pending)} 个分段，并发数 {concurrency}；完成一个立即显示。"
            if pending
            else "所有分段已从断点复用。"
        ),
        completed_segments=completed_segments,
        running_segments=[int(task["index"]) for task in pending],
        failed_segments=failed_segments,
    )

    if not pending:
        if failed_segments:
            job.manifest["status"] = "failed"
            _atomic_write_json(job.manifest_path, job.manifest)
            raise RuntimeError(f"并发视频生成有 {len(failed_segments)} 个分段不可用，断点已保存到 {job.manifest_path}")
        job.manifest["status"] = "segments_generated"
        _atomic_write_json(job.manifest_path, job.manifest)
        return job

    output_geometry = job.manifest.get("output_geometry")
    running = {int(task["index"]) for task in pending}
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="company-video") as executor:
        futures = {
            executor.submit(
                _parallel_video_task_worker,
                job,
                task,
                adapter,
                send_source_video=send_source_video,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            index = int(task["index"])
            running.discard(index)
            try:
                result = future.result()
            except Exception as exc:
                result = {"index": index, "status": "failed", "error": str(exc), "attempts": task.get("attempts", 0)}

            task["attempts"] = int(result.get("attempts", task.get("attempts", 0)))
            task["reference_roles"] = list(result.get("reference_roles") or task.get("reference_roles") or [])
            task["source_video_sent"] = bool(result.get("source_video_sent", send_source_video))
            task["prompt"] = str(result.get("prompt") or task.get("prompt") or "")
            if result.get("privacy_fallback"):
                task["privacy_fallback"] = result["privacy_fallback"]

            if result.get("status") == "success":
                remote_path = Path(str(result["remote_path"]))
                try:
                    if not output_geometry:
                        width, height, fps = _video_geometry(remote_path)
                        output_geometry = {"width": width, "height": height, "fps": fps}
                        job.manifest["output_geometry"] = output_geometry
                    _normalize_segment(
                        remote_path,
                        Path(task["result"]),
                        duration=float(task["duration"]),
                        width=int(output_geometry["width"]),
                        height=int(output_geometry["height"]),
                        fps=float(output_geometry["fps"]),
                        trim_offset=float(task.get("trim_offset", 0.0)),
                    )
                    task["status"] = "success"
                    task["error"] = ""
                    saved = _parallel_video_saved_result(Path(task["result"]))
                    completed_segments.append(
                        {
                            "sequence": index,
                            "video": str(task["result"]),
                            **({"preview": saved} if saved else {}),
                            "reused": False,
                        }
                    )
                except Exception as exc:
                    task["status"] = "failed"
                    task["error"] = str(exc)
                    failed_segments.append({"sequence": index, "error": str(exc)})
            else:
                task["status"] = str(result.get("status") or "failed")
                task["error"] = str(result.get("error") or "视频分段生成失败")
                failed_segments.append({"sequence": index, "error": task["error"]})

            _atomic_write_json(job.manifest_path, job.manifest)
            progress.update_absolute(len(completed_segments), max(1, total))
            _emit_parallel_video_progress(
                job,
                phase="segment_completed" if task["status"] == "success" else "segment_failed",
                message=(
                    f"第 {index} 段已完成，已完成 {len(completed_segments)}/{total} 段。"
                    if task["status"] == "success"
                    else f"第 {index} 段失败，已保存断点；已完成 {len(completed_segments)}/{total} 段。"
                ),
                completed_segments=completed_segments,
                running_segments=sorted(running),
                failed_segments=failed_segments,
            )

    if failed_segments:
        job.manifest["status"] = "failed"
        _atomic_write_json(job.manifest_path, job.manifest)
        _emit_parallel_video_progress(
            job,
            phase="completed_with_failures",
            message=f"并发视频生成结束，但有 {len(failed_segments)} 个分段失败；可复用已完成分段继续。",
            completed_segments=completed_segments,
            running_segments=[],
            failed_segments=failed_segments,
        )
        raise RuntimeError(
            f"并发视频生成有 {len(failed_segments)} 个分段失败，已完成分段和断点保存在 {job.manifest_path}"
        )

    job.manifest["status"] = "segments_generated"
    _atomic_write_json(job.manifest_path, job.manifest)
    _emit_parallel_video_progress(
        job,
        phase="completed",
        message=f"全部 {total} 个视频分段已完成。",
        completed_segments=completed_segments,
        running_segments=[],
        failed_segments=[],
    )
    return job


def _segment_generation_context(job: LongVideoJob) -> dict[str, Any]:
    """Prepare the loop-invariant context shared by sequential and pipeline segment generation."""
    options = job.manifest.get("auto_asset_options", {})
    options = options if isinstance(options, dict) else {}
    seedance_contract = int(job.manifest.get("processing_contract_version", 0)) in {
        V3_PROCESSING_CONTRACT_VERSION,
        MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
    }
    is_manual_batch = int(job.manifest.get("processing_contract_version", 0)) == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION
    manual_batch = job.manifest.get("manual_batch") if isinstance(job.manifest.get("manual_batch"), dict) else {}
    cross_batch_path = (
        _asset_path_if_file(manual_batch.get("cross_batch_final_frame"))
        if is_manual_batch and bool(manual_batch.get("cross_batch_continuity"))
        else None
    )
    state_path = (
        _manual_batch_validate_state_path(str(manual_batch.get("series_id") or ""), str(manual_batch.get("state_path")))
        if is_manual_batch and manual_batch.get("state_path")
        else None
    )
    if state_path is not None:
        state, _ = _manual_batch_read_state(str(manual_batch.get("series_id")))
        current_batch = state.get("current_batch")
        if not isinstance(current_batch, dict) or not _manual_batch_matches_current(current_batch, manual_batch):
            raise ValueError("当前任务与系列状态中的批次不一致，已停止生成。")
        state["status"] = "generating"
        current_batch["status"] = "generating"
        _manual_batch_write_state(state_path, state)
    else:
        state = None
    return {
        "adapter": get_video_engine_adapter(job.engine),
        "options": options,
        "send_source_video": bool(options.get("send_source_video", True)),
        "seedance_contract": seedance_contract,
        "is_manual_batch": is_manual_batch,
        "use_original_audio": bool(options.get("use_original_audio", True)) if seedance_contract else True,
        "cross_batch_frame": _load_image_file(cross_batch_path) if cross_batch_path else None,
        "state_path": state_path,
        "state": state,
    }


def _generate_single_group_segment(
    job: LongVideoJob,
    task: dict[str, Any],
    context: dict[str, Any],
    *,
    previous_end_frame: Any,
) -> Any:
    """Generate the video for one request group and return its end frame for continuity."""
    adapter = context["adapter"]
    options = context["options"]
    send_source_video = context["send_source_video"]
    seedance_contract = context["seedance_contract"]
    is_manual_batch = context["is_manual_batch"]
    use_original_audio = context["use_original_audio"]
    cross_batch_frame = context["cross_batch_frame"]
    state_path = context["state_path"]
    state = context["state"]
    output_geometry = job.manifest.get("output_geometry")

    segment_path = Path(task["result"])
    can_reuse_result = (
        job.resume
        and not job.force_rerun
        and task.get("status") == "success"
        and segment_path.is_file()
    )
    if seedance_contract:
        package = task.get("reference_package") or {}
        package_key = str(package.get("package_key") or "") if isinstance(package, dict) else ""
        can_reuse_result = bool(
            can_reuse_result
            and package_key
            and task.get("generated_reference_package_key") == package_key
        )
    if can_reuse_result:
        return _last_frame(segment_path)
    source_segment = _source_segment_for_task(job, task) if send_source_video else None
    if send_source_video and source_segment is None:
        raise ValueError(f"第 {task['index']} 段无法读取原视频画面。")
    if job.manifest.get("asset_mode") == "auto_shot_assets":
        if task.get("reference_package_status") != "ready":
            raise ValueError(f"第 {task['index']} 段自动参考素材不可用，状态为 {task.get('reference_package_status')}。")
        if is_manual_batch and int(task["index"]) == 1 and cross_batch_frame is not None:
            continuity_frame = cross_batch_frame
            continuity_role = "cross_batch_final_frame"
        elif bool(task.get("continuity_from_previous_group")):
            continuity_frame = previous_end_frame
            continuity_role = "previous_segment_end_frame"
        else:
            continuity_frame = None
            continuity_role = "previous_segment_end_frame"
        references, roles = _references_from_auto_package(
            adapter,
            task,
            previous_end_frame=continuity_frame,
            continuity_role=continuity_role,
        )
    else:
        selected = task.get("selected_assets")
        if not isinstance(selected, dict):
            raise ValueError(f"第 {task['index']} 段尚未完成参考素材匹配。")
        selected_people = [job.assets.people[item] for item in selected["people"] if item in job.assets.people]
        selected_backgrounds = [job.assets.backgrounds[item] for item in selected["backgrounds"] if item in job.assets.backgrounds]
        references, roles = _reference_images_for_task(
            adapter,
            selected_people=selected_people,
            selected_backgrounds=selected_backgrounds,
            previous_end_frame=previous_end_frame,
        )
    task["reference_roles"] = roles
    task["source_video_sent"] = send_source_video
    include_source_video = send_source_video
    privacy_fallback_used = False
    last_error = ""
    normal_attempts = max(1, job.max_retries + 1)
    for attempt in range(1, normal_attempts + 2):
        task["attempts"] = attempt
        task["status"] = "running"
        try:
            request_duration = float(task.get("request_duration", task["duration"]))
            generate_kwargs = {
                "source_segment": source_segment,
                "references": references,
                "prompt": task["prompt"],
                "model": job.model,
                "duration": request_duration,
                "negative_prompt": job.negative_prompt,
                "include_source_video": include_source_video,
            }
            if seedance_contract:
                generate_kwargs["generate_audio"] = not use_original_audio
            _, remote_path = adapter.generate(
                **generate_kwargs,
            )
            remote_path = Path(remote_path)
            if not output_geometry:
                width, height, fps = _video_geometry(remote_path)
                output_geometry = {"width": width, "height": height, "fps": fps}
                job.manifest["output_geometry"] = output_geometry
            if seedance_contract:
                task["media"] = _normalize_segment_v3(
                    remote_path,
                    segment_path,
                    request_duration=request_duration,
                    width=int(output_geometry["width"]),
                    height=int(output_geometry["height"]),
                    fps=float(output_geometry["fps"]),
                    keep_generated_audio=not use_original_audio,
                )
            else:
                _normalize_segment(
                    remote_path,
                    segment_path,
                    duration=task["duration"],
                    width=int(output_geometry["width"]),
                    height=int(output_geometry["height"]),
                    fps=float(output_geometry["fps"]),
                    trim_offset=float(task.get("trim_offset", 0.0)),
                )
            task["status"] = "success"
            task["error"] = ""
            task["source_video_sent"] = include_source_video
            if seedance_contract:
                task["generated_reference_package_key"] = str(
                    (task.get("reference_package") or {}).get("package_key") or ""
                )
            end_frame = _last_frame(segment_path)
            break
        except Exception as exc:
            last_error = str(exc)
            task["error"] = last_error
            if include_source_video and _real_person_privacy_error(exc) and not privacy_fallback_used:
                privacy_fallback_used = True
                include_source_video = False
                task["status"] = "retrying_without_source_video"
                task["source_video_sent"] = False
                task["privacy_fallback"] = {
                    "used": True,
                    "reason": "real_person_privacy_detected",
                    "original_error": last_error,
                    "fallback": "reference_images_and_story_text_only",
                }
                task["prompt"] = _build_auto_asset_video_prompt(
                    job.prompt,
                    task.get("auto_asset_analysis", {}),
                    reference_roles=roles,
                    soft_continuity=bool(task.get("uses_previous_end_frame")),
                    visual_style=str(options.get("visual_style") or AUTO_ASSET_STYLE_WESTERN),
                    send_source_video=False,
                )
                _atomic_write_json(job.manifest_path, job.manifest)
                continue
            if _provider_policy_error(exc):
                task["status"] = "blocked_by_provider_policy"
                task["failure_kind"] = "provider_policy"
                _atomic_write_json(job.manifest_path, job.manifest)
                break
            if privacy_fallback_used:
                task["status"] = "failed"
                _atomic_write_json(job.manifest_path, job.manifest)
                break
            task["status"] = "retrying" if attempt <= job.max_retries else "failed"
            _atomic_write_json(job.manifest_path, job.manifest)
            if attempt <= job.max_retries:
                time.sleep(min(2.0 * attempt, 8.0))
            else:
                break
    if task["status"] != "success":
        job.manifest["status"] = "failed"
        if state is not None:
            state["status"] = "failed"
            state["current_batch"]["status"] = "failed"
            state["current_batch"]["last_error"] = last_error
            _manual_batch_write_state(state_path, state)
        _atomic_write_json(job.manifest_path, job.manifest)
        raise RuntimeError(
            f"第 {task['index']} 段生成失败，断点已保存到 {job.manifest_path}\n{last_error}"
        )
    _atomic_write_json(job.manifest_path, job.manifest)
    return end_frame


def _finish_segment_generation(job: LongVideoJob, context: dict[str, Any]) -> None:
    """Mark the batch as generated and hand the manual-batch series over to merging."""
    state = context["state"]
    job.manifest["status"] = "segments_generated"
    if state is not None:
        state["status"] = "merging"
        state["current_batch"]["status"] = "merging"
        _manual_batch_write_state(context["state_path"], state)
    _atomic_write_json(job.manifest_path, job.manifest)


def generate_long_video_segments(job: LongVideoJob) -> LongVideoJob:
    context = _segment_generation_context(job)
    previous_end_frame = None
    progress = ProgressBar(len(job.manifest["tasks"]))
    for task in job.manifest["tasks"]:
        previous_end_frame = _generate_single_group_segment(
            job,
            task,
            context,
            previous_end_frame=previous_end_frame,
        )
        progress.update(1)
    _finish_segment_generation(job, context)
    return job


# Keep the consumer responsive once the producer checkpoints a ready group.
PIPELINE_ASSET_WAIT_POLL_SECONDS = 0.2
_PIPELINE_ASSET_FAILURE_STATUSES = {
    "source_frames_failed",
    "analysis_failed",
    "failed",
    "degraded",
    "identity_review_required",
}


def _pipeline_group_member_ids(task: dict[str, Any]) -> list[int]:
    """Read the logical member indexes a request group depends on."""
    values = task.get("logical_segments", [task.get("logical_segment", task.get("index"))])
    if not isinstance(values, list):
        values = [values]
    member_ids = list(dict.fromkeys(int(item) for item in values if item is not None))
    if not member_ids:
        raise ValueError(f"第 {task.get('index')} 组没有可等待的逻辑镜头资产。")
    return member_ids


def _wait_pipeline_group_assets(
    job: LongVideoJob,
    task: dict[str, Any],
    *,
    producer: threading.Thread,
    producer_error: list[BaseException],
) -> None:
    """Block until every member asset of this request group is ready, or fail fast."""
    member_ids = _pipeline_group_member_ids(task)
    while True:
        if producer_error:
            raise RuntimeError(f"资产生成线程已失败：{producer_error[0]}") from producer_error[0]
        members_by_index = {
            int(item["index"]): item
            for item in _auto_asset_member_tasks(job)
            if isinstance(item, dict) and item.get("index") is not None
        }
        statuses = {
            index: str((members_by_index.get(index) or {}).get("auto_asset_status") or "planned")
            for index in member_ids
        }
        if all(status == "ready" for status in statuses.values()):
            return
        blocked = {index: status for index, status in statuses.items() if status in _PIPELINE_ASSET_FAILURE_STATUSES}
        if blocked:
            detail = "，".join(f"镜头 {index} 状态 {status}" for index, status in sorted(blocked.items()))
            task["reference_package_status"] = "blocked_by_asset_failure"
            task["status"] = "blocked_by_asset_failure"
            _atomic_write_json(job.manifest_path, job.manifest)
            raise RuntimeError(
                f"第 {task['index']} 组依赖的自动资产不可用（{detail}），断点已保存到 {job.manifest_path}"
            )
        if not producer.is_alive():
            pending = "，".join(f"镜头 {index} 状态 {status}" for index, status in sorted(statuses.items()) if status != "ready")
            task["reference_package_status"] = "blocked_by_asset_failure"
            task["status"] = "blocked_by_asset_failure"
            _atomic_write_json(job.manifest_path, job.manifest)
            raise RuntimeError(
                f"资产生成已结束，但第 {task['index']} 组仍缺少素材（{pending}），断点已保存到 {job.manifest_path}"
            )
        time.sleep(PIPELINE_ASSET_WAIT_POLL_SECONDS)


def generate_long_video_pipeline(job: LongVideoJob, image_concurrency: int = 0) -> tuple[LongVideoJob, str]:
    """Build shot assets in the background while Seedance generates each ready group in order.

    The producer thread walks the logical shots and checkpoints every finished asset into the
    manifest. The consumer generates each ready group in order. It only carries the previous end
    frame across a boundary that splits one logical shot; normal shot and scene changes stay
    independent.
    """
    if job.manifest.get("asset_mode") != "auto_shot_assets":
        raise ValueError("流水线生成只支持按镜头自动资产任务，请使用原有节点链路。")
    if int(job.manifest.get("processing_contract_version", 0)) != MANUAL_BATCH_PROCESSING_CONTRACT_VERSION:
        raise ValueError("流水线生成只支持 contract=4 的手动批次任务。")

    context = _segment_generation_context(job)
    options = job.manifest.get("auto_asset_options") or {}
    job.manifest["status"] = "analysis_gateway_preflight"
    job.manifest["pipeline"] = {
        "asset_video_overlap": False,
        "global_integrated_frame_calibration": bool(options.get("use_integrated_frame_references", False)),
        "image_concurrency": max(0, int(image_concurrency)),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_write_json(job.manifest_path, job.manifest)
    try:
        _preflight_auto_asset_analysis_gateway(job)
    except AnalysisGatewayUnavailableError as exc:
        _mark_analysis_gateway_unavailable(job, context, exc)
        raise

    producer_error: list[BaseException] = []
    producer_cancel = threading.Event()
    producer_report: dict[str, Any] = {}

    def run_producer() -> None:
        try:
            _job, report, _previews = build_long_video_auto_assets(
                job,
                int(image_concurrency),
                cancel_event=producer_cancel,
                preserve_job_status=True,
            )
            producer_report.update(json.loads(report))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer below
            producer_error.append(exc)
            logging.exception("Long video pipeline asset producer failed.")

    job.manifest["status"] = "pipeline_generating"
    _atomic_write_json(job.manifest_path, job.manifest)
    pack_context = _v3_group_pack_context(job)
    tasks = job.manifest["tasks"]
    progress = ProgressBar(len(tasks))
    reports: list[dict[str, Any]] = []
    previous_end_frame = None

    producer = threading.Thread(
        target=run_producer,
        name="company-pipeline-assets",
        daemon=True,
    )
    producer.start()
    try:
        for task in tasks:
            _wait_pipeline_group_assets(job, task, producer=producer, producer_error=producer_error)
            pack_context["members_by_index"] = {
                int(item["index"]): item
                for item in _auto_asset_member_tasks(job)
                if isinstance(item, dict) and item.get("index") is not None
            }
            pack_report = _v3_pack_single_group_reference(job, task, pack_context)
            reports.append(pack_report)
            _atomic_write_json(job.manifest_path, job.manifest)
            continuity_frame = previous_end_frame if bool(task.get("continuity_from_previous_group")) else None
            previous_end_frame = _generate_single_group_segment(
                job,
                task,
                context,
                previous_end_frame=continuity_frame,
            )
            progress.update(1)
        producer.join()
        if producer_error:
            raise RuntimeError(f"资产生成线程已失败：{producer_error[0]}") from producer_error[0]
        asset_stage_status = str(producer_report.get("asset_stage_status") or "")
        if asset_stage_status != "auto_assets_ready":
            raise RuntimeError(f"自动资产阶段没有完整完成，当前状态：{asset_stage_status or 'unknown'}。")

        _finish_segment_generation(job, context)
        job.manifest["pipeline"]["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _atomic_write_json(job.manifest_path, job.manifest)
        summary = {
            "job_id": job.manifest["job_id"],
            "mode": "asset_video_pipeline",
            "status": job.manifest["status"],
            "image_concurrency": int(image_concurrency),
            "groups": len(tasks),
            "reference_packages": reports,
            "auto_asset_status_counts": _auto_asset_status_counts(job),
            "asset_stage": producer_report,
            "manifest": str(job.manifest_path),
        }
        return job, json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as exc:
        producer_cancel.set()
        state = context.get("state")
        if state is not None:
            state["status"] = "failed"
            state["current_batch"]["status"] = "failed"
            state["current_batch"]["last_error"] = str(exc)
            _manual_batch_write_state(context["state_path"], state)
        job.manifest["status"] = "failed"
        job.manifest.setdefault("pipeline", {})["error"] = str(exc)
        _atomic_write_json(job.manifest_path, job.manifest)
        raise
    finally:
        producer_cancel.set()
        producer.join()


def collect_long_video_results(job: LongVideoJob) -> tuple[LongVideoJob, str, torch.Tensor]:
    paths = []
    previews = []
    for task in job.manifest["tasks"]:
        path = Path(task["result"])
        if task.get("status") != "success" or not path.is_file():
            raise ValueError(f"第 {task['index']} 段尚未成功生成，不能进入合并阶段。")
        paths.append(str(path))
        previews.append(_first_frame(_last_frame(path)))
    summary = {
        "job_id": job.manifest["job_id"],
        "completed": len(paths),
        "segments": paths,
        "manifest": str(job.manifest_path),
    }
    return job, json.dumps(summary, ensure_ascii=False, indent=2), _vision_batch(previews)


def merge_long_video_job(job: LongVideoJob) -> tuple[Any, str, str, str]:
    completed_paths = [Path(task["result"]) for task in job.manifest["tasks"]]
    is_manual_batch = int(job.manifest.get("processing_contract_version", 0)) == MANUAL_BATCH_PROCESSING_CONTRACT_VERSION
    final_path = job.job_dir / "final" / ("batch_video.mp4" if is_manual_batch else "long_video_final.mp4")
    manual_batch = job.manifest.get("manual_batch") if isinstance(job.manifest.get("manual_batch"), dict) else {}
    state_path = None
    state = None
    if is_manual_batch:
        series_id = str(manual_batch.get("series_id") or "")
        state_path = _manual_batch_validate_state_path(series_id, str(manual_batch.get("state_path")))
        state, _ = _manual_batch_read_state(series_id)
        current_batch = state.get("current_batch")
        if not isinstance(current_batch, dict) or not _manual_batch_matches_current(current_batch, manual_batch):
            raise ValueError("当前合并任务与系列状态中的批次或 attempt 不一致。")
        existing_commit = next(
            (
                item
                for item in state.get("completed_batches", [])
                if isinstance(item, dict) and _manual_batch_matches_current(current_batch, item)
            ),
            None,
        )
        if isinstance(existing_commit, dict) and _manual_batch_commit_record_valid(existing_commit):
            job.manifest["manual_batch_commit"] = existing_commit
            job.manifest["status"] = "success"
            job.manifest["final"] = str(existing_commit["final_video"])
            _atomic_write_json(job.manifest_path, job.manifest)
            return (
                InputImpl.VideoFromFile(str(existing_commit["final_video"])),
                str(existing_commit["final_video"]),
                str(job.manifest_path),
                json.dumps(job.manifest, ensure_ascii=False, indent=2),
            )
    with tempfile.TemporaryDirectory(prefix="company_long_video_merge_") as temporary:
        if int(job.manifest.get("processing_contract_version", 0)) in {
            V3_PROCESSING_CONTRACT_VERSION,
            MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
        }:
            options = job.manifest.get("auto_asset_options") or {}
            job.manifest["final_media"] = _concat_v3_segments(
                job.manifest["tasks"],
                job.source_path,
                final_path,
                Path(temporary),
                use_original_audio=bool(options.get("use_original_audio", True)),
            )
        else:
            _concat_and_mux(completed_paths, job.source_path, final_path, job.total_duration, Path(temporary))
    job.manifest["status"] = "success"
    job.manifest["final"] = str(final_path)
    if is_manual_batch:
        series_id = str(manual_batch.get("series_id") or "")
        assert state_path is not None and state is not None
        state["status"] = "committing"
        state["current_batch"]["status"] = "committing"
        _manual_batch_write_state(state_path, state)
        final_frame_path = job.job_dir / "final" / "batch_final_frame.png"
        _save_image_tensor(_last_frame(final_path), final_frame_path)
        current_batch = dict(state.get("current_batch") or {})
        record = {
            "contract": MANUAL_BATCH_CONTRACT,
            "series_id": series_id,
            "batch_id": current_batch.get("batch_id"),
            "batch_index": int(current_batch.get("batch_index", 0)),
            "attempt": int(current_batch.get("attempt", 1)),
            "source_start": float(current_batch.get("source_start", 0.0)),
            "source_end": float(current_batch.get("source_end", 0.0)),
            "source_duration": float(current_batch.get("source_duration", 0.0)),
            "final_video": str(final_path),
            "final_frame": str(final_frame_path),
            "manifest": str(job.manifest_path),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        commit_dir = state_path.parent / "commits"
        commit_path = commit_dir / f"{_safe_slug(str(record['batch_id']), 'batch')}_attempt_{int(record['attempt']):03d}.json"
        _atomic_write_json(commit_path, record)
        if not _manual_batch_apply_commit(state, record):
            raise RuntimeError("手动批次提交标记校验失败，源游标没有推进。")
        state["status"] = "completed"
        state["current_batch"]["status"] = "completed"
        state["current_batch"]["final_video"] = str(final_path)
        state["current_batch"]["final_frame"] = str(final_frame_path)
        _manual_batch_write_state(state_path, state)
        job.manifest["manual_batch_commit"] = record
    _atomic_write_json(job.manifest_path, job.manifest)
    return (
        InputImpl.VideoFromFile(str(final_path)),
        str(final_path),
        str(job.manifest_path),
        json.dumps(job.manifest, ensure_ascii=False, indent=2),
    )


def process_long_video(
    *,
    video: Any,
    assets_manifest: str,
    prompt: str,
    engine: str,
    model: str,
    segment_duration: int,
    ai_model: str,
    max_retries: int,
    resume: bool,
    force_rerun: bool,
    negative_prompt: str,
    people: dict[str, Any],
    backgrounds: dict[str, Any],
) -> tuple[Any, str, str, str]:
    assets, _ = load_long_video_assets(
        assets_manifest,
        people=people,
        backgrounds=backgrounds,
    )
    job = plan_long_video_job(
        video=video,
        assets=assets,
        prompt=prompt,
        engine=engine,
        model=model,
        segment_duration=segment_duration,
        ai_model=ai_model,
        max_retries=max_retries,
        resume=resume,
        force_rerun=force_rerun,
        negative_prompt=negative_prompt,
    )
    analyze_long_video_job(job)
    match_long_video_references(job)
    generate_long_video_segments(job)
    collect_long_video_results(job)
    return merge_long_video_job(job)


def asset_image_inputs() -> list[Any]:
    return [
        *[IO.Image.Input(f"person_{person_id}", display_name=f"欧美化人物 {person_id}", optional=True) for person_id in PERSON_IDS],
        *[IO.Image.Input(background_id, display_name=f"欧美化背景 {background_id}", optional=True) for background_id in BACKGROUND_IDS],
    ]


def connected_asset_dict(kwargs: dict[str, Any], prefix: str, ids: tuple[str, ...]) -> dict[str, Any]:
    return {entry_id: kwargs.get(f"{prefix}{entry_id}") for entry_id in ids if kwargs.get(f"{prefix}{entry_id}") is not None}
