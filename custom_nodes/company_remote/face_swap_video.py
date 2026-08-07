from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image

import folder_paths
from comfy.utils import ProgressBar
from comfy_api.latest import IO, InputImpl


DEFAULT_SCENE_MAPPING = json.dumps(
    [
        {"start_frame": 0, "end_frame": 20, "characters": {"A": 0, "B": 1}},
        {"start_frame": 21, "end_frame": 109, "characters": {"C": 0}},
        {"start_frame": 110, "end_frame": 195, "characters": {"A": 0, "B": 1}},
        {"start_frame": 196, "end_frame": 329, "characters": {"C": 0}},
        {"start_frame": 330, "end_frame": 373, "characters": {"A": 0, "B": 1}},
        {"start_frame": 374, "end_frame": 483, "characters": {"C": 0}},
    ],
    ensure_ascii=False,
    indent=2,
)


@dataclass(frozen=True)
class SceneRule:
    start_frame: int
    end_frame: int
    characters: dict[str, int]


def _reactor_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "ComfyUI-ReActor"
    if not path.is_dir():
        raise RuntimeError("未安装 ComfyUI-ReActor，请先安装官方 Gourieff/ComfyUI-ReActor 节点。")
    return path


def _reactor_api():
    path = str(_reactor_path())
    if path not in sys.path:
        sys.path.insert(0, path)
    from scripts import reactor_swapper

    reactor_swapper.providers = ["CPUExecutionProvider"]
    return reactor_swapper


def _parse_scene_mapping(value: str) -> list[SceneRule]:
    try:
        payload = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"镜头人物映射不是有效 JSON：{exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("镜头人物映射必须是非空数组。")

    rules: list[SceneRule] = []
    previous_end = -1
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条镜头映射必须是对象。")
        start = int(item.get("start_frame", -1))
        end = int(item.get("end_frame", -1))
        raw_characters = item.get("characters")
        if start < 0 or end < start:
            raise ValueError(f"第 {index} 条镜头映射的帧范围无效：{start}-{end}。")
        if start <= previous_end:
            raise ValueError(f"第 {index} 条镜头映射与前一条重叠或未按顺序排列。")
        if not isinstance(raw_characters, dict) or not raw_characters:
            raise ValueError(f"第 {index} 条镜头映射没有 characters。")
        characters: dict[str, int] = {}
        for label, face_index in raw_characters.items():
            normalized = str(label).strip().upper()
            if normalized not in {"A", "B", "C"}:
                raise ValueError(f"第 {index} 条镜头映射包含未知人物 {label}。")
            value_index = int(face_index)
            if value_index < 0:
                raise ValueError(f"第 {index} 条镜头映射的人脸序号不能为负数。")
            characters[normalized] = value_index
        rules.append(SceneRule(start, end, characters))
        previous_end = end
    return rules


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


def _tensor_to_pil(image: Any) -> Image.Image:
    frame = image[0] if getattr(image, "ndim", 0) == 4 else image
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"人物参考图必须是 RGB IMAGE，当前形状为 {array.shape}。")
    array = np.clip(array[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _find_source_face(reactor_swapper: Any, image: Image.Image, label: str):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    faces = reactor_swapper.analyze_faces(bgr)
    if not faces:
        raise ValueError(f"人物 {label} 参考图没有检测到清晰人脸。请使用单人、正脸、无遮挡图片。")
    return reactor_swapper.sort_by_order(faces, "large-small")[0]


def _rule_for_frame(rules: list[SceneRule], frame_index: int) -> SceneRule | None:
    for rule in rules:
        if rule.start_frame <= frame_index <= rule.end_frame:
            return rule
    return None


def _normalized_embedding(face: Any) -> np.ndarray | None:
    value = getattr(face, "normed_embedding", None)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else None


def _build_identity_anchors(
    reactor_swapper: Any,
    source_path: str,
    rules: list[SceneRule],
    *,
    sample_every: int = 6,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    samples: dict[str, list[np.ndarray]] = {"A": [], "B": [], "C": []}
    with av.open(source_path, mode="r") as source:
        for frame_index, frame in enumerate(source.decode(video=0)):
            if frame_index % max(1, int(sample_every)) != 0:
                continue
            rule = _rule_for_frame(rules, frame_index)
            if rule is None:
                continue
            faces = reactor_swapper.sort_by_order(
                reactor_swapper.analyze_faces(frame.to_ndarray(format="bgr24")),
                "left-right",
            )
            required_count = max(rule.characters.values()) + 1
            if len(faces) < required_count:
                continue
            for label, face_index in rule.characters.items():
                embedding = _normalized_embedding(faces[face_index])
                if embedding is not None:
                    samples[label].append(embedding)

    anchors: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    required_labels = {label for rule in rules for label in rule.characters}
    for label in sorted(required_labels):
        values = samples[label]
        if not values:
            raise ValueError(f"无法从镜头映射建立原人物 {label} 的身份特征，请调整映射或使用更清晰的视频。")
        anchor = np.mean(np.stack(values), axis=0)
        anchor /= max(float(np.linalg.norm(anchor)), 1e-8)
        anchors[label] = anchor
        counts[label] = len(values)
    return anchors, counts


def _assign_identity_faces(
    faces: list[Any],
    labels: list[str],
    anchors: dict[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    candidates = [(face, _normalized_embedding(face)) for face in faces]
    candidates = [(face, embedding) for face, embedding in candidates if embedding is not None]
    if not candidates or not labels:
        return {}

    best_score = float("-inf")
    best_assignment: dict[str, Any] = {}
    maximum = min(len(labels), len(candidates))
    for count in range(1, maximum + 1):
        for selected_labels in combinations(labels, count):
            for selected_faces in permutations(range(len(candidates)), count):
                scores = [
                    float(np.dot(anchors[label], candidates[face_index][1]))
                    for label, face_index in zip(selected_labels, selected_faces, strict=True)
                ]
                if any(score < threshold for score in scores):
                    continue
                score = sum(scores) + count
                if score > best_score:
                    best_score = score
                    best_assignment = {
                        label: candidates[face_index][0]
                        for label, face_index in zip(selected_labels, selected_faces, strict=True)
                    }
    return best_assignment


def _swap_frame(
    reactor_swapper: Any,
    frame_bgr: np.ndarray,
    rule: SceneRule | None,
    source_faces: dict[str, Any],
    identity_anchors: dict[str, np.ndarray],
    swapper: Any,
    identity_threshold: float,
) -> tuple[np.ndarray, list[str]]:
    if rule is None:
        return frame_bgr, []
    target_faces = reactor_swapper.analyze_faces(frame_bgr)
    matched = _assign_identity_faces(
        target_faces,
        list(rule.characters),
        identity_anchors,
        threshold=identity_threshold,
    )
    result = frame_bgr
    swapped: list[str] = []
    for label in rule.characters:
        target_face = matched.get(label)
        if target_face is None:
            continue
        result = swapper.get(result, target_face, source_faces[label])
        swapped.append(label)
    return result, swapped


def _encode_frames(source_path: str, silent_path: Path, process_frame) -> dict[str, Any]:
    frames_processed = 0
    frames_with_swaps = 0
    swaps_by_character = {"A": 0, "B": 0, "C": 0}
    with av.open(source_path, mode="r") as source:
        input_stream = source.streams.video[0]
        fps = input_stream.average_rate or input_stream.base_rate
        if not fps:
            raise ValueError("源视频没有可用帧率。")
        frame_time_base = Fraction(fps.denominator, fps.numerator)
        total_frames = int(input_stream.frames or 0)
        progress = ProgressBar(max(1, total_frames))
        with av.open(str(silent_path), mode="w") as output:
            output_stream = output.add_stream("libx264", rate=fps)
            output_stream.width = int(input_stream.width)
            output_stream.height = int(input_stream.height)
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": "18", "preset": "veryfast", "threads": "2"}
            for frame_index, frame in enumerate(source.decode(video=0)):
                result_bgr, swapped = process_frame(frame_index, frame.to_ndarray(format="bgr24"))
                out_frame = av.VideoFrame.from_ndarray(result_bgr, format="bgr24")
                # The source stream time base does not necessarily match the new
                # CFR encoder. Reusing source PTS can round multiple frames to the
                # same encoder timestamp and make MP4 muxing fail with duplicate
                # DTS values. Generate a monotonic CFR timeline for this stream.
                out_frame.pts = frame_index
                out_frame.time_base = frame_time_base
                for packet in output_stream.encode(out_frame):
                    output.mux(packet)
                frames_processed += 1
                if swapped:
                    frames_with_swaps += 1
                for label in swapped:
                    swaps_by_character[label] += 1
                progress.update(1)
            for packet in output_stream.encode():
                output.mux(packet)
    return {
        "frames_processed": frames_processed,
        "frames_with_swaps": frames_with_swaps,
        "swaps_by_character": swaps_by_character,
        "fps": float(fps),
    }


def _ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _mux_original_audio(silent_path: Path, source_path: str, output_path: Path) -> None:
    with av.open(str(silent_path), mode="r") as silent:
        video_stream = silent.streams.video[0]
        if video_stream.duration is not None:
            video_duration = float(video_stream.duration * video_stream.time_base)
        elif silent.duration is not None:
            video_duration = float(silent.duration / av.time_base)
        else:
            raise ValueError("换脸后的无声视频没有可用时长。")
    with av.open(source_path, mode="r") as source:
        has_audio = bool(source.streams.audio)

    command = [
        _ffmpeg_executable(),
        "-y",
        "-i",
        str(silent_path),
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
    ]
    if has_audio:
        command.extend(
            [
                "-map",
                "1:a:0",
                "-af",
                "apad",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
            ]
        )
    command.extend(
        [
            "-t",
            f"{video_duration:.9f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
        raise RuntimeError(f"恢复原视频音频失败：{detail}")


def swap_video_faces_streaming(
    video: Any,
    face_a: Any,
    face_b: Any,
    face_c: Any,
    *,
    scene_mapping: str = DEFAULT_SCENE_MAPPING,
    identity_threshold: float = 0.25,
) -> tuple[Any, str, str]:
    rules = _parse_scene_mapping(scene_mapping)
    output_root = Path(folder_paths.get_output_directory()) / "face_swap"
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    output_path = output_root / f"three_person_face_swap_{run_id}.mp4"

    reactor_swapper = _reactor_api()
    reactor_swapper.unload_all_models()
    model_path = Path(folder_paths.models_dir) / "insightface" / "inswapper_128.onnx"
    if not model_path.is_file():
        raise RuntimeError(f"缺少 ReActor 换脸模型：{model_path}")

    source_faces = {
        "A": _find_source_face(reactor_swapper, _tensor_to_pil(face_a), "A"),
        "B": _find_source_face(reactor_swapper, _tensor_to_pil(face_b), "B"),
        "C": _find_source_face(reactor_swapper, _tensor_to_pil(face_c), "C"),
    }
    swapper = reactor_swapper.getFaceSwapModel(str(model_path))

    with tempfile.TemporaryDirectory(prefix="comfy_face_swap_") as directory:
        work_dir = Path(directory)
        source_path = _source_path(video, work_dir)
        identity_anchors, anchor_sample_counts = _build_identity_anchors(reactor_swapper, source_path, rules)
        silent_path = work_dir / "swapped_silent.mp4"
        stats = _encode_frames(
            source_path,
            silent_path,
            lambda frame_index, frame_bgr: _swap_frame(
                reactor_swapper,
                frame_bgr,
                _rule_for_frame(rules, frame_index),
                source_faces,
                identity_anchors,
                swapper,
                float(identity_threshold),
            ),
        )
        _mux_original_audio(silent_path, source_path, output_path)

    reactor_swapper.unload_all_models()
    report = {
        "status": "success",
        "output_path": str(output_path),
        "provider": "CPUExecutionProvider",
        "model": model_path.name,
        "scene_count": len(rules),
        "identity_threshold": float(identity_threshold),
        "identity_anchor_samples": anchor_sample_counts,
        **stats,
    }
    return InputImpl.VideoFromFile(str(output_path)), str(output_path), json.dumps(report, ensure_ascii=False, indent=2)


class CompanyThreePersonFaceSwapVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyThreePersonFaceSwapVideo",
            display_name="三人物视频流式换脸（CPU）",
            category="company-local/video/face-swap",
            description="按镜头映射逐帧替换人物 A/B/C 的脸，限制内存占用并保留原视频音频。",
            inputs=[
                IO.Video.Input("video", display_name="原视频"),
                IO.Image.Input("face_a", display_name="人物 A 新脸"),
                IO.Image.Input("face_b", display_name="人物 B 新脸"),
                IO.Image.Input("face_c", display_name="人物 C 新脸"),
                IO.String.Input(
                    "scene_mapping",
                    display_name="镜头人物映射 JSON",
                    multiline=True,
                    default=DEFAULT_SCENE_MAPPING,
                    tooltip="每段填写起止帧，以及人物 A/B/C 对应的目标人脸序号；目标脸按从左到右编号。",
                ),
                IO.Float.Input(
                    "identity_threshold",
                    display_name="身份相似度下限",
                    default=0.25,
                    min=0.0,
                    max=0.8,
                    step=0.01,
                    tooltip="低于该相似度的脸不会替换，可减少人物缺席或遮挡时的串脸。",
                ),
            ],
            outputs=[
                IO.Video.Output(display_name="换脸视频"),
                IO.String.Output(display_name="输出路径"),
                IO.String.Output(display_name="处理报告 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        face_a: Any,
        face_b: Any,
        face_c: Any,
        scene_mapping: str,
        identity_threshold: float = 0.25,
    ):
        result = swap_video_faces_streaming(
            video,
            face_a,
            face_b,
            face_c,
            scene_mapping=scene_mapping,
            identity_threshold=identity_threshold,
        )
        return IO.NodeOutput(*result, ui={"text": (result[2],)})
