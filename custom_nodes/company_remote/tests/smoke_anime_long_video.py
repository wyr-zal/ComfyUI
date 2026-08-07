from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
from PIL import Image, ImageDraw

import folder_paths
import server


class _Routes:
    def __getattr__(self, _name):
        def route(*_args, **_kwargs):
            return lambda function: function

        return route


if not hasattr(server.PromptServer, "instance"):
    server.PromptServer.instance = type("PromptServerInstance", (), {"routes": _Routes()})()

from comfy_api.latest import InputImpl
from custom_nodes.company_remote import long_video
from custom_nodes.company_remote.nodes import (
    ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
    ANIME_LONG_VIDEO_PROMPT,
)


def _first_frame(path: Path) -> Image.Image:
    with av.open(str(path), mode="r") as container:
        frame = next(container.decode(video=0))
    image = frame.to_image().convert("RGB")
    image.thumbnail((320, 180), Image.Resampling.LANCZOS)
    return image


def create_contact_sheet(output: Path) -> None:
    input_root = Path(folder_paths.get_input_directory())
    videos = sorted(input_root.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"没有找到测试视频：{input_root}")

    tile_width, tile_height = 320, 220
    columns = 3
    rows = (len(videos) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(videos):
        image = _first_frame(path)
        x = index % columns * tile_width
        y = index // columns * tile_height
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + 184), path.name[:46], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(output)


def create_video_sheet(source: Path, output: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    with av.open(str(source), mode="r") as container:
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        video_stream = container.streams.video[0]
        width = int(video_stream.width)
        height = int(video_stream.height)
        fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        audio_streams = len(container.streams.audio)
    if duration <= 0:
        raise RuntimeError(f"无法读取视频时长：{source}")

    timestamps = [0.05, duration / 2.0, max(0.05, duration - 0.05)]
    images: list[Image.Image] = []
    for timestamp in timestamps:
        with av.open(str(source), mode="r") as container:
            container.seek(int(timestamp * av.time_base), backward=True)
            frame = None
            for candidate in container.decode(video=0):
                frame = candidate
                if candidate.time is not None and float(candidate.time) >= timestamp - 0.02:
                    break
            if frame is None:
                raise RuntimeError(f"无法读取 {timestamp:.3f} 秒的视频帧：{source}")
        image = frame.to_image().convert("RGB")
        image.thumbnail((480, 270), Image.Resampling.LANCZOS)
        images.append(image)

    sheet = Image.new("RGB", (480 * len(images), 310), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (timestamp, image) in enumerate(zip(timestamps, images, strict=True)):
        x = index * 480
        sheet.paste(image, (x, 0))
        draw.text((x + 4, 276), f"{timestamp:.2f}s", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(
        json.dumps(
            {
                "source": str(source),
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "audio_streams": audio_streams,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


def _contains_video_reference(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"video_url", "video", "reference_video", "reference_videos"} and item:
                return True
            if _contains_video_reference(item):
                return True
    elif isinstance(value, list):
        return any(_contains_video_reference(item) for item in value)
    return False


def _generate_and_verify(
    job: long_video.LongVideoJob,
    *,
    source: Path,
    detection: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    options = job.manifest.get("auto_asset_options", {})
    assert options.get("visual_style") == "anime_2d"
    assert options.get("send_source_video") is False
    for task in job.manifest["tasks"]:
        items = task.get("reference_package", {}).get("items", [])
        assert items, f"第 {task['index']} 段没有动漫参考图"

    job = long_video.generate_long_video_segments(job)
    _, result_report, _ = long_video.collect_long_video_results(job)
    _, final_path, manifest_path, _ = long_video.merge_long_video_job(job)

    debug_root = Path(folder_paths.get_user_directory()) / "default" / "company_remote" / "debug"
    payload_path = debug_root / "seedance2_last_payload.json"
    media_path = debug_root / "seedance2_last_media.json"
    payload_debug = json.loads(payload_path.read_text(encoding="utf-8"))
    media_debug = json.loads(media_path.read_text(encoding="utf-8"))
    if _contains_video_reference(payload_debug.get("payload")):
        raise AssertionError(f"Seedance 请求中仍包含参考视频：{payload_path}")
    video_media = [
        item
        for item in media_debug.get("media", [])
        if item.get("kind") == "video" or item.get("media_kind") == "video"
    ]
    if video_media:
        raise AssertionError(f"Seedance 媒体清单中仍包含视频：{video_media}")

    first_task = job.manifest["tasks"][0]
    evidence = {
        "source": str(source),
        "detection": detection,
        "selection": selection,
        "job_id": job.manifest["job_id"],
        "visual_style": options["visual_style"],
        "send_source_video": options["send_source_video"],
        "reference_image_count": len(first_task.get("reference_package", {}).get("items", [])),
        "source_video_sent": first_task.get("source_video_sent"),
        "seedance_video_media_count": len(video_media),
        "result": json.loads(result_report),
        "final_path": final_path,
        "manifest_path": manifest_path,
        "seedance_payload_debug": str(payload_path),
        "seedance_media_debug": str(media_path),
    }
    evidence_path = job.job_dir / "anime_seedance_smoke_evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


def resume_smoke(manifest_path: Path) -> None:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    options = manifest.get("auto_asset_options", {})
    settings = manifest.get("settings", {})
    source = Path(manifest["source_path"])
    assets = long_video.LongVideoAssets(
        manifest=manifest.get("assets", {}),
        people={},
        backgrounds={},
    )
    job = long_video.LongVideoJob(
        video=InputImpl.VideoFromFile(str(source)),
        assets=assets,
        prompt=str(settings.get("prompt") or ANIME_LONG_VIDEO_PROMPT),
        engine=str(manifest.get("engine") or "seedance"),
        model=str(manifest.get("model") or "Seedance 2.0 Fast"),
        segment_duration=int(manifest.get("segment_duration") or 10),
        ai_model=str(settings.get("ai_model") or "gpt-5.6-terra"),
        max_retries=1,
        resume=False,
        force_rerun=True,
        negative_prompt=str(settings.get("negative_prompt") or ANIME_LONG_VIDEO_NEGATIVE_PROMPT),
        total_duration=float(manifest["source_duration"]),
        source_path=str(source),
        job_dir=manifest_path.parent,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    assert options.get("send_source_video") is False
    segmentation = manifest.get("segmentation", {})
    _generate_and_verify(
        job,
        source=source,
        detection=segmentation,
        selection=segmentation.get("config", {}).get("continuity_test_selection", {}),
    )


def run_smoke(source: Path, *, start_shot: int, fixed_mode: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)

    video = InputImpl.VideoFromFile(str(source))
    plan, detection_json, _ = long_video.detect_long_video_shots(
        video=video,
        mode="固定时长" if fixed_mode else "镜头优先（推荐）",
        fixed_duration=10,
        sensitivity="标准",
        use_audio_silence=True,
        auto_fallback=True,
    )
    selected_plan, _, selection_json = long_video.select_continuous_shot_range(
        plan,
        start_shot=start_shot,
        shot_count=1,
    )
    job = long_video.plan_long_video_auto_asset_job(
        shot_plan=selected_plan,
        prompt=ANIME_LONG_VIDEO_PROMPT,
        engine="seedance",
        model="Seedance 2.0 Fast",
        ai_model="gpt-5.6-terra",
        image_model="gpt-image-2",
        image_quality="medium",
        reuse_threshold=0.92,
        max_retries=1,
        resume=False,
        force_rerun=True,
        force_rerun_assets=True,
        negative_prompt=ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
        visual_style="anime_2d",
        send_source_video=False,
    )
    job, asset_report, _ = long_video.build_long_video_auto_assets(job)
    if job.manifest.get("status") != "auto_assets_ready":
        raise RuntimeError(f"动漫资产未全部生成：{asset_report}")
    job, reference_report, _ = long_video.pack_long_video_auto_references(job)
    if job.manifest.get("status") != "auto_references_packed":
        raise RuntimeError(f"动漫参考包未准备完成：{reference_report}")

    _generate_and_verify(
        job,
        source=source,
        detection=json.loads(detection_json),
        selection=json.loads(selection_json),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    contact = subparsers.add_parser("contact-sheet")
    contact.add_argument("--output", type=Path, required=True)
    video_sheet = subparsers.add_parser("video-sheet")
    video_sheet.add_argument("--source", type=Path, required=True)
    video_sheet.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--start-shot", type=int, default=1)
    run.add_argument("--fixed-mode", action="store_true")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "contact-sheet":
        create_contact_sheet(args.output)
    elif args.command == "video-sheet":
        create_video_sheet(args.source, args.output)
    elif args.command == "run":
        run_smoke(args.source, start_shot=args.start_shot, fixed_mode=args.fixed_mode)
    else:
        resume_smoke(args.manifest)


if __name__ == "__main__":
    main()
