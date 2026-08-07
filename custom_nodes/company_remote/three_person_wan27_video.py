from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
from comfy_api.latest import IO, InputImpl

import folder_paths

from .client import _image_to_bytes, generate_dashscope_video
from .config_store import get_config
from .long_video import _concat_and_mux, _ffmpeg_exe, _run_ffmpeg, _source_segment_for_task, _video_file_info
from .three_person_seedance_video import _prepare_asset_upload_video, _source_path


DEFAULT_WAN27_SEGMENTS = json.dumps(
    [
        {"start_frame": 0, "end_frame": 195},
        {"start_frame": 196, "end_frame": 373},
        {"start_frame": 374, "end_frame": 483},
    ],
    ensure_ascii=False,
    indent=2,
)


@dataclass(frozen=True)
class Wan27Segment:
    index: int
    start_frame: int
    end_frame: int


@dataclass
class _SegmentJob:
    source_path: str
    job_dir: Path
    force_rerun: bool = False


def _parse_segments(value: str, *, frame_count: int, fps: float) -> list[Wan27Segment]:
    try:
        payload = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Wan2.7 分段不是有效 JSON：{exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("Wan2.7 分段必须是非空数组。")

    segments: list[Wan27Segment] = []
    expected_start = 0
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个 Wan2.7 分段必须是对象。")
        start = int(item.get("start_frame", -1))
        end = int(item.get("end_frame", -1))
        if start != expected_start or end < start:
            raise ValueError(f"第 {index} 个 Wan2.7 分段必须从第 {expected_start} 帧连续开始，当前为 {start}-{end}。")
        duration = (end - start + 1) / fps
        if not 2.0 <= duration <= 10.0:
            raise ValueError(
                f"第 {index} 个 Wan2.7 分段为 {duration:.3f} 秒；"
                "wan2.7-videoedit 每次输入必须在 2-10 秒内。"
            )
        segments.append(Wan27Segment(index=index, start_frame=start, end_frame=end))
        expected_start = end + 1
    if expected_start != frame_count:
        raise ValueError(f"Wan2.7 分段必须完整覆盖 0-{frame_count - 1} 帧，当前结束于 {expected_start - 1}。")
    return segments


def _task_for_segment(segment: Wan27Segment, *, fps: float) -> dict[str, Any]:
    frame_count = segment.end_frame - segment.start_frame + 1
    duration = frame_count / fps
    return {
        "index": segment.index,
        "start": segment.start_frame / fps,
        "source_start": segment.start_frame / fps,
        "source_duration": duration,
        "duration": duration,
        "padding_start": 0.0,
        "padding_end": 0.0,
    }


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _image_sha256(image: Any) -> str:
    content, _mime, _extension = _image_to_bytes(image, "png")
    return hashlib.sha256(content).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _job_signature(
    *,
    source_sha256: str,
    image_sha256: dict[str, str],
    segments: list[Wan27Segment],
    prompt: str,
    resolution: str,
    prompt_extend: bool,
    watermark: bool,
    seed: int,
    negative_prompt: str,
) -> str:
    payload = {
        "source_sha256": source_sha256,
        "images": image_sha256,
        "segments": [segment.__dict__ for segment in segments],
        "prompt": str(prompt),
        "model": "wan2.7-videoedit",
        "resolution": str(resolution),
        "prompt_extend": bool(prompt_extend),
        "watermark": bool(watermark),
        "seed": int(seed),
        "negative_prompt": str(negative_prompt),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generate_wan27_segment(
    video: Any,
    image_a: Any,
    image_b: Any,
    image_c: Any,
    *,
    prompt: str,
    resolution: str,
    prompt_extend: bool,
    watermark: bool,
    seed: int,
    negative_prompt: str,
) -> tuple[Any, str]:
    config = copy.copy(get_config("aliyun_dashscope_video_direct"))
    config.tos_enabled = False
    config.media_delivery = "base64"
    config.extra_headers = {
        **config.extra_headers,
        "X-DashScope-Async": "enable",
        "X-DashScope-OssResourceResolve": "enable",
    }
    return generate_dashscope_video(
        config,
        operation="dashscope_video_edit",
        model="wan2.7-videoedit",
        prompt=prompt,
        resolution=resolution,
        duration=0,
        negative_prompt=negative_prompt,
        prompt_extend=prompt_extend,
        watermark=watermark,
        seed=seed,
        edit_video=video,
        reference_images=[image_a, image_b, image_c],
        audio_setting="origin",
    )


def _normalize_segment_frames(
    source_path: Path,
    output_path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: Fraction,
) -> None:
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
        error_prefix="Wan2.7 分段按原帧数规范化失败",
    )


def _merge_exact_frames_and_restore_audio(
    segment_paths: list[Path],
    source_path: str,
    output_path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: Fraction,
    work_dir: Path,
) -> None:
    merged_with_audio = work_dir / "merged_with_audio.mp4"
    exact_silent = work_dir / "merged_exact_frames.mp4"
    _concat_and_mux(segment_paths, source_path, merged_with_audio, frame_count / float(fps), work_dir)
    _normalize_segment_frames(
        merged_with_audio,
        exact_silent,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            _ffmpeg_exe(),
            "-y",
            "-i",
            str(exact_silent),
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
            str(output_path),
        ],
        error_prefix="Wan2.7 完整视频恢复原音频失败",
    )
    with av.open(str(output_path), mode="r") as container:
        actual_frames = int(container.streams.video[0].frames or 0)
    if actual_frames != frame_count:
        raise RuntimeError(f"Wan2.7 完整视频合并后应为 {frame_count} 帧，实际为 {actual_frames} 帧。")


def _local_video_path(video: Any, target_path: Path) -> str:
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else None
    trim_start = 0.0
    trim_duration = 0.0
    if hasattr(video, "get_active_trim_window"):
        trim_start, trim_duration = video.get_active_trim_window()
    if (
        isinstance(source, str)
        and Path(source).is_file()
        and abs(float(trim_start)) < 1e-9
        and float(trim_duration) <= 0
    ):
        return source
    target_path.parent.mkdir(parents=True, exist_ok=True)
    video.save_to(str(target_path))
    return str(target_path)


def split_three_person_wan27_full_video(
    video: Any,
    *,
    segments_json: str,
    force_resplit: bool = False,
) -> tuple[Any, Any, Any, str]:
    """Expose the three paid-request inputs as real workflow outputs."""
    output_root = Path(folder_paths.get_output_directory()) / "company_remote" / "three_person_wan27_visible"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="company_wan27_visible_split_") as temporary:
        work_dir = Path(temporary)
        source_path = _source_path(video, work_dir)
        fallback_duration = float(video.get_duration()) if hasattr(video, "get_duration") else 0.0
        total_duration, fps, frame_count = _video_file_info(source_path, fallback_duration)
        segments = _parse_segments(segments_json, frame_count=frame_count, fps=fps)
        if len(segments) != 3:
            raise ValueError(f"可视化 Wan2.7 工作流固定显示 3 个生成分支，当前分段数为 {len(segments)}。")

        signature_payload = {
            "source_sha256": _file_sha256(source_path),
            "segments": [segment.__dict__ for segment in segments],
            "upload_fps": 24,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        job_dir = output_root / signature[:20] / "split"
        job = _SegmentJob(source_path=source_path, job_dir=job_dir, force_rerun=bool(force_resplit))
        outputs: list[Any] = []
        report_segments: list[dict[str, Any]] = []
        for segment in segments:
            task = _task_for_segment(segment, fps=fps)
            source_segment = _source_segment_for_task(job, task)
            upload_video, upload_report = _prepare_asset_upload_video(
                source_segment,
                job_dir / "uploads" / f"segment_{segment.index:04d}_24fps.mp4",
                force_rerun=bool(force_resplit),
            )
            outputs.append(upload_video)
            report_segments.append(
                {
                    "index": segment.index,
                    "frames": [segment.start_frame, segment.end_frame],
                    "source_start": round(float(task["source_start"]), 6),
                    "source_duration": round(float(task["source_duration"]), 6),
                    "wan_input": upload_report,
                }
            )

    report = {
        "status": "success",
        "stage": "split_only_no_remote_request",
        "source": {
            "path": source_path,
            "duration": round(total_duration, 6),
            "fps": round(fps, 6),
            "frames": frame_count,
        },
        "segments": report_segments,
    }
    return outputs[0], outputs[1], outputs[2], json.dumps(report, ensure_ascii=False, indent=2)


def merge_three_person_wan27_segments(
    original_video: Any,
    segment_1: Any,
    segment_2: Any,
    segment_3: Any,
    *,
    segments_json: str,
    force_remerge: bool = False,
) -> tuple[Any, str, str]:
    """Normalize three visible Wan outputs, merge exact frames and restore source audio."""
    output_root = Path(folder_paths.get_output_directory()) / "company_remote" / "three_person_wan27_visible"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="company_wan27_visible_merge_") as temporary:
        work_dir = Path(temporary)
        source_path = _local_video_path(original_video, work_dir / "original.mp4")
        generated_paths = [
            _local_video_path(segment_1, work_dir / "generated_1.mp4"),
            _local_video_path(segment_2, work_dir / "generated_2.mp4"),
            _local_video_path(segment_3, work_dir / "generated_3.mp4"),
        ]
        fallback_duration = float(original_video.get_duration()) if hasattr(original_video, "get_duration") else 0.0
        total_duration, fps, frame_count = _video_file_info(source_path, fallback_duration)
        with av.open(source_path, mode="r") as container:
            stream = container.streams.video[0]
            width, height = int(stream.width), int(stream.height)
            fps_rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(24, 1)
        segments = _parse_segments(segments_json, frame_count=frame_count, fps=fps)
        if len(segments) != 3:
            raise ValueError(f"可视化 Wan2.7 合并节点固定接收 3 个生成结果，当前分段数为 {len(segments)}。")

        signature_payload = {
            "source_sha256": _file_sha256(source_path),
            "generated_sha256": [_file_sha256(path) for path in generated_paths],
            "segments": [segment.__dict__ for segment in segments],
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        job_dir = output_root / signature[:20] / "merge"
        normalized_paths: list[Path] = []
        for segment, generated_path in zip(segments, generated_paths):
            normalized_path = job_dir / "normalized" / f"segment_{segment.index:04d}.mp4"
            if force_remerge or not normalized_path.is_file():
                _normalize_segment_frames(
                    Path(generated_path),
                    normalized_path,
                    frame_count=segment.end_frame - segment.start_frame + 1,
                    width=width,
                    height=height,
                    fps=fps_rate,
                )
            normalized_paths.append(normalized_path)

        final_path = job_dir / "final" / "three_person_wan27_full.mp4"
        if force_remerge or not final_path.is_file():
            _merge_exact_frames_and_restore_audio(
                normalized_paths,
                source_path,
                final_path,
                frame_count=frame_count,
                width=width,
                height=height,
                fps=fps_rate,
                work_dir=work_dir,
            )

    report = {
        "status": "success",
        "stage": "normalize_merge_restore_audio",
        "source": {
            "duration": round(total_duration, 6),
            "fps": round(fps, 6),
            "frames": frame_count,
            "width": width,
            "height": height,
        },
        "segments": [
            {
                "index": segment.index,
                "frames": [segment.start_frame, segment.end_frame],
                "generated_path": generated_path,
                "normalized_path": str(normalized_path),
            }
            for segment, generated_path, normalized_path in zip(segments, generated_paths, normalized_paths)
        ],
        "final_path": str(final_path),
        "audio": "restored_from_original",
    }
    return InputImpl.VideoFromFile(str(final_path)), str(final_path), json.dumps(report, ensure_ascii=False, indent=2)


def generate_three_person_wan27_full_video(
    video: Any,
    image_a: Any,
    image_b: Any,
    image_c: Any,
    *,
    segments_json: str,
    prompt: str,
    resolution: str,
    reuse_completed_segments: bool,
    force_rerun_segments: bool,
    prompt_extend: bool,
    watermark: bool,
    seed: int,
    negative_prompt: str,
) -> tuple[Any, str, str]:
    output_root = Path(folder_paths.get_output_directory()) / "company_remote" / "three_person_wan27"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="company_three_person_wan27_") as temporary:
        work_dir = Path(temporary)
        source_path = _source_path(video, work_dir)
        total_duration, fps, frame_count = _video_file_info(source_path, float(video.get_duration()))
        with av.open(source_path, mode="r") as container:
            stream = container.streams.video[0]
            width, height = int(stream.width), int(stream.height)
            fps_rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(24, 1)
        segments = _parse_segments(segments_json, frame_count=frame_count, fps=fps)
        image_hashes = {
            "A": _image_sha256(image_a),
            "B": _image_sha256(image_b),
            "C": _image_sha256(image_c),
        }
        signature = _job_signature(
            source_sha256=_file_sha256(source_path),
            image_sha256=image_hashes,
            segments=segments,
            prompt=prompt,
            resolution=resolution,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            negative_prompt=negative_prompt,
        )
        job_dir = output_root / signature[:20]
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "manifest.json"
        existing: dict[str, Any] = {}
        if reuse_completed_segments and not force_rerun_segments and manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        existing_segments = {
            int(item["index"]): item
            for item in existing.get("segments", [])
            if isinstance(item, dict) and item.get("index") is not None
        }
        manifest: dict[str, Any] = {
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
            "image_sha256": image_hashes,
            "segments": [],
        }
        job = _SegmentJob(source_path=source_path, job_dir=job_dir, force_rerun=bool(force_rerun_segments))
        normalized_paths: list[Path] = []

        for segment in segments:
            normalized_path = job_dir / "segments" / f"segment_{segment.index:04d}.mp4"
            previous = existing_segments.get(segment.index, {})
            if reuse_completed_segments and previous.get("status") == "success" and normalized_path.is_file():
                manifest["segments"].append(previous)
                normalized_paths.append(normalized_path)
                continue

            task = _task_for_segment(segment, fps=fps)
            source_segment = _source_segment_for_task(job, task)
            upload_video, upload_report = _prepare_asset_upload_video(
                source_segment,
                job_dir / "uploads" / f"segment_{segment.index:04d}_24fps.mp4",
                force_rerun=bool(force_rerun_segments),
            )
            report = {
                "index": segment.index,
                "status": "submitting",
                "frames": [segment.start_frame, segment.end_frame],
                "source_start": round(float(task["source_start"]), 6),
                "source_duration": round(float(task["source_duration"]), 6),
                "request_path": str(source_segment.get_stream_source()),
                "upload": upload_report,
                "normalized_path": str(normalized_path),
            }
            manifest["segments"].append(report)
            _atomic_write_json(manifest_path, manifest)
            generated_video, generated_path = _generate_wan27_segment(
                upload_video,
                image_a,
                image_b,
                image_c,
                prompt=(
                    f"{str(prompt).strip()}\n\n"
                    f"这是完整视频的第 {segment.index}/{len(segments)} 段，人物 A/B/C 映射必须与其他分段完全一致。"
                ),
                resolution=resolution,
                prompt_extend=prompt_extend,
                watermark=watermark,
                seed=int(seed),
                negative_prompt=negative_prompt,
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
            report.update({"status": "success", "remote_result_path": str(generated_path)})
            normalized_paths.append(normalized_path)
            _atomic_write_json(manifest_path, manifest)

        final_path = job_dir / "final" / "three_person_wan27_full.mp4"
        _merge_exact_frames_and_restore_audio(
            normalized_paths,
            source_path,
            final_path,
            frame_count=frame_count,
            width=width,
            height=height,
            fps=fps_rate,
            work_dir=work_dir,
        )
        manifest.update({"status": "success", "final_path": str(final_path)})
        _atomic_write_json(manifest_path, manifest)

    report = {
        "status": "success",
        "source": manifest["source"],
        "segments": manifest["segments"],
        "final_path": str(final_path),
        "manifest_path": str(manifest_path),
        "audio": "restored_from_original",
    }
    return InputImpl.VideoFromFile(str(final_path)), str(final_path), json.dumps(report, ensure_ascii=False, indent=2)


class CompanyWan27SplitThreeSegments(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyWan27SplitThreeSegments",
            display_name="Wan 2.7 完整视频拆成三段",
            category="company-remote/video/Alibaba Cloud",
            description="只做本地切分和 24 FPS 上传副本转换，不会提交远程付费任务。",
            inputs=[
                IO.Video.Input("video", display_name="完整原视频"),
                IO.String.Input(
                    "segments_json",
                    display_name="三段帧范围 JSON",
                    multiline=True,
                    default=DEFAULT_WAN27_SEGMENTS,
                    tooltip="必须由三个连续分段完整覆盖原视频，每段须为 2-10 秒。",
                ),
                IO.Boolean.Input("force_resplit", display_name="强制重新切分", default=False, advanced=True),
            ],
            outputs=[
                IO.Video.Output(display_name="第1段：0-195帧"),
                IO.Video.Output(display_name="第2段：196-373帧"),
                IO.Video.Output(display_name="第3段：374-483帧"),
                IO.String.Output(display_name="切分报告 JSON"),
            ],
        )

    @classmethod
    def execute(cls, video: Any, segments_json: str, force_resplit: bool = False):
        return split_three_person_wan27_full_video(
            video,
            segments_json=segments_json,
            force_resplit=force_resplit,
        )


class CompanyWan27MergeThreeSegments(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyWan27MergeThreeSegments",
            display_name="Wan 2.7 三段精确合并并恢复原音频",
            category="company-remote/video/Alibaba Cloud",
            description="将三个可见的 Wan2.7 结果按原帧数规范化、合并，并恢复完整原视频音频。",
            inputs=[
                IO.Video.Input("original_video", display_name="完整原视频（用于帧率和音频）"),
                IO.Video.Input("segment_1", display_name="第1段 Wan 结果"),
                IO.Video.Input("segment_2", display_name="第2段 Wan 结果"),
                IO.Video.Input("segment_3", display_name="第3段 Wan 结果"),
                IO.String.Input(
                    "segments_json",
                    display_name="三段帧范围 JSON",
                    multiline=True,
                    default=DEFAULT_WAN27_SEGMENTS,
                ),
                IO.Boolean.Input("force_remerge", display_name="强制重新合并", default=False, advanced=True),
            ],
            outputs=[
                IO.Video.Output(display_name="完整合并视频"),
                IO.String.Output(display_name="完整视频路径"),
                IO.String.Output(display_name="合并报告 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        original_video: Any,
        segment_1: Any,
        segment_2: Any,
        segment_3: Any,
        segments_json: str,
        force_remerge: bool = False,
    ):
        return merge_three_person_wan27_segments(
            original_video,
            segment_1,
            segment_2,
            segment_3,
            segments_json=segments_json,
            force_remerge=force_remerge,
        )


class CompanyWan27ThreePersonFullVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyWan27ThreePersonFullVideo",
            display_name="Wan 2.7 三人物完整视频分段替换",
            category="company-remote/video/Alibaba Cloud",
            description=(
                "将超过 10 秒的完整原视频按镜头边界切成合法分段，逐段执行 wan2.7-videoedit，"
                "再按原帧数合并并恢复完整原音频。"
            ),
            inputs=[
                IO.Video.Input("video", display_name="完整原视频"),
                IO.Image.Input("image_a", display_name="图1：替换原视频人物 A"),
                IO.Image.Input("image_b", display_name="图2：替换原视频人物 B"),
                IO.Image.Input("image_c", display_name="图3：替换原视频人物 C"),
                IO.String.Input(
                    "segments_json",
                    display_name="完整视频分段 JSON",
                    multiline=True,
                    default=DEFAULT_WAN27_SEGMENTS,
                    tooltip="必须连续覆盖完整视频；每段须为 2-10 秒。",
                ),
                IO.String.Input("prompt", display_name="人物替换指令", multiline=True, default=""),
                IO.Combo.Input("resolution", display_name="分辨率", options=["720P", "1080P"], default="720P"),
                IO.Boolean.Input("reuse_completed_segments", display_name="复用已成功分段", default=True),
                IO.Boolean.Input(
                    "force_rerun_segments",
                    display_name="强制重新提交全部分段",
                    default=False,
                    advanced=True,
                ),
                IO.Boolean.Input("prompt_extend", display_name="智能改写", default=True, advanced=True),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input("seed", display_name="首段种子", default=0, min=0, max=2147483645, step=1),
                IO.String.Input(
                    "negative_prompt",
                    display_name="负面提示词",
                    multiline=True,
                    default="",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Video.Output(display_name="完整替换视频"),
                IO.String.Output(display_name="完整视频路径"),
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
        resolution: str,
        reuse_completed_segments: bool = True,
        force_rerun_segments: bool = False,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: int = 0,
        negative_prompt: str = "",
    ):
        return generate_three_person_wan27_full_video(
            video,
            image_a,
            image_b,
            image_c,
            segments_json=segments_json,
            prompt=prompt,
            resolution=resolution,
            reuse_completed_segments=reuse_completed_segments,
            force_rerun_segments=force_rerun_segments,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            negative_prompt=negative_prompt,
        )
