from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
from comfy_api.latest import IO, InputImpl

import folder_paths

from .asset_gateway import create_seedance_image_asset, create_seedance_video_asset
from .asset_gateway_video import MODEL_OPTIONS, generate_three_person_asset_video
from .long_video import (
    _concat_and_mux,
    _ffmpeg_exe,
    _run_ffmpeg,
    _source_segment_for_task,
    _video_file_info,
)


DEFAULT_SEGMENTS = json.dumps(
    [
        {"start_frame": 0, "end_frame": 195},
        {"start_frame": 196, "end_frame": 483},
    ],
    ensure_ascii=False,
    indent=2,
)

ASSET_UPLOAD_FPS = Fraction(24, 1)


@dataclass(frozen=True)
class SeedanceSegment:
    index: int
    start_frame: int
    end_frame: int


@dataclass
class _SegmentJob:
    source_path: str
    job_dir: Path
    force_rerun: bool = False


def _source_path(video: Any, work_dir: Path) -> str:
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    trim_start = 0.0
    trim_duration = 0.0
    if hasattr(video, "get_active_trim_window"):
        trim_start, trim_duration = video.get_active_trim_window()
    if isinstance(source, str) and Path(source).is_file() and abs(float(trim_start)) < 1e-9 and float(trim_duration) <= 0:
        return source
    path = work_dir / "source_video.mp4"
    video.save_to(str(path))
    return str(path)


def _parse_segments(value: str, *, frame_count: int) -> list[SeedanceSegment]:
    try:
        payload = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Seedance 分段不是有效 JSON：{exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("Seedance 分段必须是非空数组。")

    segments: list[SeedanceSegment] = []
    expected_start = 0
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个 Seedance 分段必须是对象。")
        start = int(item.get("start_frame", -1))
        end = int(item.get("end_frame", -1))
        if start != expected_start or end < start:
            raise ValueError(
                f"第 {index} 个 Seedance 分段必须从第 {expected_start} 帧连续开始，当前为 {start}-{end}。"
            )
        segments.append(SeedanceSegment(index=index, start_frame=start, end_frame=end))
        expected_start = end + 1
    if expected_start != frame_count:
        raise ValueError(f"Seedance 分段必须完整覆盖 0-{frame_count - 1} 帧，当前结束于 {expected_start - 1}。")
    return segments


def _task_for_segment(segment: SeedanceSegment, *, fps: float) -> dict[str, Any]:
    source_duration = (segment.end_frame - segment.start_frame + 1) / fps
    request_duration = int(math.ceil(source_duration - 1e-6))
    if not 4 <= request_duration <= 15:
        raise ValueError(
            f"第 {segment.index} 段时长 {source_duration:.3f} 秒，向上取整后为 {request_duration} 秒；"
            "Seedance 资产参考视频要求 4-15 秒。"
        )
    return {
        "index": segment.index,
        "start": segment.start_frame / fps,
        "source_start": segment.start_frame / fps,
        "source_duration": source_duration,
        "duration": source_duration,
        "request_duration": request_duration,
        "padding_start": 0.0,
        "padding_end": 0.0,
    }


def _asset_report(asset_id: str, report_text: str, reused: bool) -> dict[str, Any]:
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError:
        report = {"report": report_text}
    return {"asset_id": asset_id, "cache_reused": bool(reused), "details": report}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _job_signature(
    *,
    source_sha256: str,
    character_asset_ids: dict[str, str],
    segments: list[SeedanceSegment],
    prompt: str,
    model: str,
    resolution: str,
    ratio: str,
    watermark: bool,
    seed: int,
) -> str:
    payload = {
        "source_sha256": source_sha256,
        "character_assets": character_asset_ids,
        "segments": [segment.__dict__ for segment in segments],
        "prompt": str(prompt),
        "model": str(model),
        "resolution": str(resolution),
        "ratio": str(ratio),
        "watermark": bool(watermark),
        "seed": int(seed),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_segment_frames(
    source_path: Path,
    output_path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: Fraction,
) -> None:
    if frame_count <= 0:
        raise ValueError("分段目标帧数必须大于 0。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps_text = f"{fps.numerator}/{fps.denominator}"
    target_duration = frame_count / float(fps)
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={fps_text},"
        f"tpad=stop_mode=clone:stop_duration={target_duration + 1.0:.6f},"
        f"trim=start_frame=0:end_frame={frame_count},"
        "setpts=N/(FRAME_RATE*TB)"
    )
    _run_ffmpeg(
        [
            _ffmpeg_exe(),
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
        error_prefix="Seedance 分段按帧规范化失败",
    )


def _prepare_asset_upload_video(
    source_video: Any,
    output_path: Path,
    *,
    force_rerun: bool = False,
) -> tuple[Any, dict[str, Any]]:
    source_path = source_video.get_stream_source() if hasattr(source_video, "get_stream_source") else None
    if not isinstance(source_path, str) or not Path(source_path).is_file():
        raise ValueError("Seedance 资产上传前无法取得本地参考视频文件。")

    if force_rerun or not output_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps_text = f"{ASSET_UPLOAD_FPS.numerator}/{ASSET_UPLOAD_FPS.denominator}"
        _run_ffmpeg(
            [
                _ffmpeg_exe(),
                "-y",
                "-i",
                source_path,
                "-an",
                "-vf",
                f"fps={fps_text},setpts=N/(24*TB)",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            error_prefix="Seedance 资产上传视频转为 24 FPS 失败",
        )

    try:
        with av.open(str(output_path), mode="r") as container:
            stream = container.streams.video[0]
            average_fps = float(stream.average_rate) if stream.average_rate else 0.0
            nominal_fps = float(stream.base_rate) if stream.base_rate else average_fps
            frames = int(stream.frames or 0)
            duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
            width, height = int(stream.width), int(stream.height)
    except (av.FFmpegError, IndexError, OSError) as exc:
        raise RuntimeError(f"Seedance 资产上传视频校验失败：{exc}") from exc
    if not 23.8 <= average_fps <= 60.0:
        raise RuntimeError(
            f"Seedance 资产上传视频平均帧率为 {average_fps:.6f} FPS，"
            "不在平台要求的 23.8-60 FPS 范围内。"
        )

    report = {
        "path": str(output_path),
        "fps": round(nominal_fps, 6),
        "average_fps": round(average_fps, 6),
        "frames": frames,
        "duration": round(duration, 6),
        "width": width,
        "height": height,
    }
    return InputImpl.VideoFromFile(str(output_path)), report


def generate_three_person_seedance_video(
    video: Any,
    image_a: Any,
    image_b: Any,
    image_c: Any,
    *,
    segments_json: str,
    prompt: str,
    model: str,
    resolution: str,
    ratio: str,
    reuse_cached_assets: bool,
    force_rerun_segments: bool,
    watermark: bool,
    seed: int,
) -> tuple[Any, str, str]:
    output_root = Path(folder_paths.get_output_directory()) / "company_remote" / "three_person_seedance"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="company_three_person_seedance_") as temporary:
        work_dir = Path(temporary)
        source_path = _source_path(video, work_dir)
        total_duration, fps, frame_count = _video_file_info(source_path, float(video.get_duration()))
        with av.open(source_path, mode="r") as container:
            stream = container.streams.video[0]
            width, height = int(stream.width), int(stream.height)
            fps_rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(24, 1)
        segments = _parse_segments(segments_json, frame_count=frame_count)

        character_assets: dict[str, tuple[str, str, bool]] = {}
        for label, image in (("A", image_a), ("B", image_b), ("C", image_c)):
            character_assets[label] = create_seedance_image_asset(
                image,
                character_label=f"人物 {label}",
                reuse_cached=bool(reuse_cached_assets),
            )

        character_asset_ids = {label: values[0] for label, values in character_assets.items()}
        signature = _job_signature(
            source_sha256=_file_sha256(source_path),
            character_asset_ids=character_asset_ids,
            segments=segments,
            prompt=prompt,
            model=model,
            resolution=resolution,
            ratio=ratio,
            watermark=watermark,
            seed=seed,
        )
        job_dir = output_root / signature[:20]
        job_dir.mkdir(parents=True, exist_ok=True)
        job = _SegmentJob(source_path=source_path, job_dir=job_dir, force_rerun=bool(force_rerun_segments))
        manifest_path = job_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file() and not force_rerun_segments:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        existing_tasks = {
            int(item.get("index")): item
            for item in manifest.get("segments", [])
            if isinstance(item, dict) and item.get("index") is not None
        }
        manifest = {
            "version": 1,
            "signature": signature,
            "status": "running",
            "source": {
                "path": source_path,
                "duration": round(total_duration, 6),
                "fps": round(fps, 6),
                "frames": frame_count,
                "width": width,
                "height": height,
            },
            "character_assets": character_asset_ids,
            "segments": [],
        }
        normalized_paths: list[Path] = []

        for segment in segments:
            task = _task_for_segment(segment, fps=fps)
            normalized_path = job_dir / "segments" / f"segment_{segment.index:04d}.mp4"
            segment_report = existing_tasks.get(segment.index, {}) if not force_rerun_segments else {}
            if segment_report.get("status") == "success" and normalized_path.is_file():
                manifest["segments"].append(segment_report)
                normalized_paths.append(normalized_path)
                continue
            source_segment = _source_segment_for_task(job, task)
            asset_upload_video, asset_upload_report = _prepare_asset_upload_video(
                source_segment,
                job_dir / "asset_uploads" / f"segment_{segment.index:04d}_24fps.mp4",
                force_rerun=bool(force_rerun_segments),
            )
            source_asset_id, source_report_text, source_reused = create_seedance_video_asset(
                asset_upload_video,
                video_label=f"原视频第 {segment.index} 段",
                reuse_cached=bool(reuse_cached_assets),
            )
            segment_report = {
                "index": segment.index,
                "status": "asset_ready",
                "frames": [segment.start_frame, segment.end_frame],
                "source_start": round(float(task["source_start"]), 6),
                "source_duration": round(float(task["source_duration"]), 6),
                "request_duration": int(task["request_duration"]),
                "asset_upload": asset_upload_report,
                "source_asset": _asset_report(source_asset_id, source_report_text, source_reused),
                "task_id": str(segment_report.get("task_id") or ""),
                "normalized_path": str(normalized_path),
            }
            manifest["segments"].append(segment_report)
            _atomic_write_json(manifest_path, manifest)

            def record_submission(task_id: str, *, report: dict[str, Any] = segment_report) -> None:
                report["task_id"] = str(task_id)
                report["status"] = "submitted"
                _atomic_write_json(manifest_path, manifest)

            generated_video, generated_path, task_id, generation_report_text = generate_three_person_asset_video(
                source_video_asset_id=source_asset_id,
                character_a_asset_id=character_assets["A"][0],
                character_b_asset_id=character_assets["B"][0],
                character_c_asset_id=character_assets["C"][0],
                prompt=str(prompt),
                model=str(model),
                resolution=str(resolution),
                ratio=str(ratio),
                duration=int(task["request_duration"]),
                generate_audio=False,
                watermark=bool(watermark),
                seed=int(seed) + segment.index - 1,
                resume_task_id=str(segment_report.get("task_id") or "") if not force_rerun_segments else "",
                submitted_callback=record_submission,
            )
            del generated_video
            _normalize_segment_frames(
                Path(generated_path),
                normalized_path,
                frame_count=segment.end_frame - segment.start_frame + 1,
                width=width,
                height=height,
                fps=fps_rate,
            )
            normalized_paths.append(normalized_path)
            segment_report.update(
                {
                    "status": "success",
                    "task_id": task_id,
                    "generation": json.loads(generation_report_text),
                }
            )
            _atomic_write_json(manifest_path, manifest)

        final_path = job_dir / "final" / "three_person_seedance_final.mp4"
        _concat_and_mux(normalized_paths, source_path, final_path, total_duration, work_dir)

        manifest["status"] = "success"
        manifest["final_path"] = str(final_path)
        _atomic_write_json(manifest_path, manifest)

    report = {
        "status": "success",
        "source": {
            "path": source_path,
            "duration": round(total_duration, 6),
            "fps": round(fps, 6),
            "frames": frame_count,
            "width": width,
            "height": height,
        },
        "characters": {
            label: _asset_report(*values)
            for label, values in character_assets.items()
        },
        "segments": manifest["segments"],
        "final_path": str(final_path),
        "manifest_path": str(manifest_path),
        "audio": "restored_from_original",
    }
    return InputImpl.VideoFromFile(str(final_path)), str(final_path), json.dumps(report, ensure_ascii=False, indent=2)


class CompanyThreePersonSeedanceVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyThreePersonSeedanceVideo",
            display_name="三人物整头造型 Seedance 分段转换",
            category="company-remote/video/Seedance",
            description="按连续镜头段注册私有视频资产，让 Seedance 用 A/B/C 参考图替换整头造型，并恢复原音频。",
            search_aliases=["Three Person Seedance Head Replacement", "三人物整头替换", "三人物视频重绘"],
            inputs=[
                IO.Video.Input("video", display_name="原视频"),
                IO.Image.Input("image_a", display_name="人物 A 整头参考图"),
                IO.Image.Input("image_b", display_name="人物 B 整头参考图"),
                IO.Image.Input("image_c", display_name="人物 C 整头参考图"),
                IO.String.Input(
                    "segments_json",
                    display_name="连续分段 JSON",
                    multiline=True,
                    default=DEFAULT_SEGMENTS,
                    tooltip="每段必须连续覆盖全部帧；每段时长向上取整后须在 4-15 秒。",
                ),
                IO.String.Input("prompt", display_name="三人物整头映射提示词", multiline=True, default=""),
                IO.Combo.Input("model", display_name="模型", options=MODEL_OPTIONS, default=MODEL_OPTIONS[0]),
                IO.Combo.Input("resolution", display_name="分辨率", options=["480p", "720p", "1080p"], default="720p"),
                IO.Combo.Input("ratio", display_name="比例", options=["adaptive", "16:9", "9:16", "1:1"], default="adaptive"),
                IO.Boolean.Input("reuse_cached_assets", display_name="复用相同素材资产 ID", default=True),
                IO.Boolean.Input(
                    "force_rerun_segments",
                    display_name="强制重新提交全部分段",
                    default=False,
                    advanced=True,
                    tooltip="关闭时复用成功分段或继续轮询已有任务 ID，避免重复付费。",
                ),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input("seed", display_name="首段种子", default=0, min=0, max=2147483646, step=1),
            ],
            outputs=[
                IO.Video.Output(display_name="最终视频"),
                IO.String.Output(display_name="最终视频路径"),
                IO.String.Output(display_name="处理报告 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        image_a: Any,
        image_b: Any,
        image_c: Any,
        segments_json: str,
        prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        reuse_cached_assets: bool,
        force_rerun_segments: bool,
        watermark: bool,
        seed: int,
    ):
        result = generate_three_person_seedance_video(
            video,
            image_a,
            image_b,
            image_c,
            segments_json=segments_json,
            prompt=prompt,
            model=model,
            resolution=resolution,
            ratio=ratio,
            reuse_cached_assets=reuse_cached_assets,
            force_rerun_segments=force_rerun_segments,
            watermark=watermark,
            seed=seed,
        )
        return IO.NodeOutput(*result, ui={"text": (result[2],)})
