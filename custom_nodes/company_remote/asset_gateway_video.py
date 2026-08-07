from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import requests

import folder_paths
from comfy_api.latest import IO, InputImpl

from .asset_gateway import ASSET_GATEWAY_CONFIG_NAME, _request_headers, _safe_error_text
from .client import CompanyRemoteAPIError
from .config_store import RemoteMediaConfig, get_config


MODEL_OPTIONS = [
    "doubao-seedance-2-0-fast-260128",
    "doubao-seedance-2-0-mini-260615",
    "doubao-seedance-2-0-260128",
]
SUCCESS_STATUSES = {"completed", "succeeded", "success", "done"}
FAILURE_STATUSES = {"failed", "error", "cancelled", "canceled"}


def _video_endpoint(config: RemoteMediaConfig, task_id: str = "") -> str:
    path = "/v1/videos"
    if task_id:
        path += "/" + urllib.parse.quote(task_id, safe="")
    return urllib.parse.urljoin(config.base_url.rstrip("/") + "/", path.lstrip("/"))


def _asset_id(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise CompanyRemoteAPIError(f"{label}资产 ID 不能为空。")
    if not normalized.startswith("asset-"):
        raise CompanyRemoteAPIError(f"{label}资产 ID 格式无效：应以 asset- 开头。")
    return normalized


def _content_item(asset_id: str, asset_type: str, role: str) -> dict[str, Any]:
    field = "video_url" if asset_type == "video" else "image_url"
    return {
        "type": field,
        field: {"url": f"asset://{asset_id}"},
        "role": role,
    }


def build_three_person_asset_content(
    source_video_asset_id: str,
    character_a_asset_id: str,
    character_b_asset_id: str,
    character_c_asset_id: str,
) -> list[dict[str, Any]]:
    return [
        _content_item(_asset_id(source_video_asset_id, "源视频"), "video", "reference_video"),
        _content_item(_asset_id(character_a_asset_id, "人物 A"), "image", "reference_image"),
        _content_item(_asset_id(character_b_asset_id, "人物 B"), "image", "reference_image"),
        _content_item(_asset_id(character_c_asset_id, "人物 C"), "image", "reference_image"),
    ]


def build_three_person_video_payload(
    *,
    source_video_asset_id: str,
    character_a_asset_id: str,
    character_b_asset_id: str,
    character_c_asset_id: str,
    prompt: str,
    model: str,
    resolution: str,
    ratio: str,
    duration: int,
    generate_audio: bool,
    watermark: bool,
    seed: int,
) -> dict[str, Any]:
    normalized_prompt = str(prompt or "").strip()
    if not normalized_prompt:
        raise CompanyRemoteAPIError("视频生成提示词不能为空。")
    seconds = int(duration)
    if not 4 <= seconds <= 15:
        raise CompanyRemoteAPIError("Seedance 参考视频时长必须在 4 到 15 秒之间。")

    content = build_three_person_asset_content(
        source_video_asset_id,
        character_a_asset_id,
        character_b_asset_id,
        character_c_asset_id,
    )
    return {
        "model": str(model or MODEL_OPTIONS[0]).strip(),
        "prompt": normalized_prompt,
        "duration": seconds,
        "seconds": str(seconds),
        "metadata": {
            "content": content,
            "resolution": str(resolution),
            "ratio": str(ratio),
            "duration": seconds,
            "generate_audio": bool(generate_audio),
            "watermark": bool(watermark),
            "seed": int(seed),
        },
    }


def _json_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    config: RemoteMediaConfig,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = session.request(
            method,
            url,
            headers=_request_headers(config),
            json=json_body,
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise CompanyRemoteAPIError(f"连接 Seedance 资产网关失败：{exc}") from exc
    if not response.ok:
        raise CompanyRemoteAPIError(
            f"Seedance 资产网关请求失败（HTTP {response.status_code}）：{_safe_error_text(response)}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise CompanyRemoteAPIError("Seedance 资产网关返回的不是有效 JSON。") from exc
    if not isinstance(payload, dict):
        raise CompanyRemoteAPIError("Seedance 资产网关返回的 JSON 不是对象。")
    return payload


def _task_id(payload: dict[str, Any]) -> str:
    for key in ("task_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        return _task_id(data)
    raise CompanyRemoteAPIError("视频提交成功，但响应中没有 task_id。")


def _result_url(payload: dict[str, Any]) -> str:
    candidates: list[Any] = [payload.get("url"), payload.get("video_url")]
    for parent_name in ("metadata", "content", "data", "output"):
        parent = payload.get(parent_name)
        if isinstance(parent, dict):
            candidates.extend([parent.get("url"), parent.get("video_url")])
            content = parent.get("content")
            if isinstance(content, dict):
                candidates.extend([content.get("url"), content.get("video_url")])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _task_error(payload: dict[str, Any]) -> str:
    value = payload.get("error") or payload.get("reason") or payload.get("message")
    if isinstance(value, dict):
        value = value.get("message") or value.get("code") or json.dumps(value, ensure_ascii=False)
    return str(value or "远端未返回失败详情")[:1600]


def _poll_task(
    session: requests.Session,
    config: RemoteMediaConfig,
    task_id: str,
) -> tuple[str, dict[str, Any], int]:
    attempts = max(1, int(config.max_poll_attempts or 120))
    interval = max(1.0, float(config.poll_interval_seconds or 5))
    last_payload: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            time.sleep(interval)
        last_payload = _json_request(
            session,
            "GET",
            _video_endpoint(config, task_id),
            config=config,
        )
        status = str(last_payload.get("status") or "").strip().lower()
        result_url = _result_url(last_payload)
        if status in FAILURE_STATUSES:
            raise CompanyRemoteAPIError(f"Seedance 视频任务 {task_id} 失败：{_task_error(last_payload)}")
        if result_url and (not status or status in SUCCESS_STATUSES):
            return result_url, last_payload, attempt
    status = str(last_payload.get("status") or "unknown")
    raise CompanyRemoteAPIError(
        f"Seedance 视频任务 {task_id} 在轮询上限内未完成（当前状态：{status}）。"
        "可把该 ID 填入‘继续轮询已有任务 ID’后再次运行，节点不会重复提交。"
    )


def _download_video(
    session: requests.Session,
    config: RemoteMediaConfig,
    url: str,
    task_id: str,
) -> str:
    headers = _request_headers(config) if url.startswith(config.base_url.rstrip("/") + "/") else {}
    headers.pop("Content-Type", None)
    try:
        response = session.get(url, headers=headers, timeout=config.timeout_seconds, stream=True)
    except requests.RequestException as exc:
        raise CompanyRemoteAPIError(f"下载 Seedance 结果视频失败：{exc}") from exc
    if not response.ok:
        raise CompanyRemoteAPIError(f"下载 Seedance 结果视频失败（HTTP {response.status_code}）。")
    output_dir = Path(folder_paths.get_output_directory()) / "company_remote" / "seedance_asset_gateway"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = "".join(ch for ch in task_id if ch.isalnum() or ch in {"-", "_"})[:100] or "task"
    output_path = output_dir / f"three_person_{safe_task_id}.mp4"
    temp_path = output_path.with_suffix(".mp4.tmp")
    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    if temp_path.stat().st_size <= 0:
        raise CompanyRemoteAPIError("Seedance 结果视频为空文件。")
    os.replace(temp_path, output_path)
    return str(output_path)


def generate_three_person_asset_video(
    *,
    source_video_asset_id: str,
    character_a_asset_id: str,
    character_b_asset_id: str,
    character_c_asset_id: str,
    prompt: str,
    model: str,
    resolution: str,
    ratio: str,
    duration: int,
    generate_audio: bool,
    watermark: bool,
    seed: int,
    resume_task_id: str = "",
    submitted_callback: Callable[[str], None] | None = None,
) -> tuple[Any, str, str, str]:
    config = get_config(ASSET_GATEWAY_CONFIG_NAME)
    session = requests.Session()
    session.trust_env = False
    resumed = bool(str(resume_task_id or "").strip())
    build_three_person_asset_content(
        source_video_asset_id,
        character_a_asset_id,
        character_b_asset_id,
        character_c_asset_id,
    )
    if resumed:
        task_id = str(resume_task_id).strip()
        payload: dict[str, Any] = {}
    else:
        payload = build_three_person_video_payload(
            source_video_asset_id=source_video_asset_id,
            character_a_asset_id=character_a_asset_id,
            character_b_asset_id=character_b_asset_id,
            character_c_asset_id=character_c_asset_id,
            prompt=prompt,
            model=model,
            resolution=resolution,
            ratio=ratio,
            duration=duration,
            generate_audio=generate_audio,
            watermark=watermark,
            seed=seed,
        )
        response = _json_request(
            session,
            "POST",
            _video_endpoint(config),
            config=config,
            json_body=payload,
        )
        task_id = _task_id(response)

    if submitted_callback is not None and not resumed:
        submitted_callback(task_id)

    result_url, result_payload, poll_attempts = _poll_task(session, config, task_id)
    output_path = _download_video(session, config, result_url, task_id)
    report = {
        "status": str(result_payload.get("status") or "completed"),
        "task_id": task_id,
        "resumed": resumed,
        "model": str(model),
        "duration": int(duration),
        "resolution": str(resolution),
        "ratio": str(ratio),
        "assets": {
            "source_video": str(source_video_asset_id),
            "character_a": str(character_a_asset_id),
            "character_b": str(character_b_asset_id),
            "character_c": str(character_c_asset_id),
        },
        "request_contract": "metadata.content media URLs use asset:// asset URIs",
        "poll_attempts": poll_attempts,
        "output_path": output_path,
    }
    return InputImpl.VideoFromFile(output_path), output_path, task_id, json.dumps(report, ensure_ascii=False, indent=2)


class CompanySeedanceAssetGatewayThreePersonVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedanceAssetGatewayThreePersonVideo",
            display_name="Seedance 资产 ID 三人物视频转换",
            category="company-remote/video/Seedance",
            description="把同一资产网关创建的源视频与 A/B/C 人物资产 ID 交给 /v1/videos 生成，并轮询下载结果。",
            search_aliases=["Seedance Asset Three Person Video", "三人物资产视频转换"],
            inputs=[
                IO.String.Input("source_video_asset_id", display_name="源视频资产 ID"),
                IO.String.Input("character_a_asset_id", display_name="人物 A 资产 ID（参考图 1）"),
                IO.String.Input("character_b_asset_id", display_name="人物 B 资产 ID（参考图 2）"),
                IO.String.Input("character_c_asset_id", display_name="人物 C 资产 ID（参考图 3）"),
                IO.String.Input("prompt", display_name="三人物映射提示词", multiline=True, default=""),
                IO.Combo.Input("model", display_name="模型", options=MODEL_OPTIONS, default=MODEL_OPTIONS[0]),
                IO.Combo.Input("resolution", display_name="分辨率", options=["480p", "720p", "1080p"], default="480p"),
                IO.Combo.Input("ratio", display_name="比例", options=["adaptive", "16:9", "9:16", "1:1"], default="adaptive"),
                IO.Int.Input("duration", display_name="输出时长（秒）", default=10, min=4, max=15, step=1),
                IO.Boolean.Input("generate_audio", display_name="生成音频", default=False),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input("seed", display_name="种子", default=0, min=0, max=2147483647, step=1),
                IO.String.Input(
                    "resume_task_id",
                    display_name="继续轮询已有任务 ID",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="填入 task_... 后只轮询并下载，不会再次提交付费生成任务。",
                ),
            ],
            outputs=[
                IO.Video.Output(display_name="生成视频"),
                IO.String.Output(display_name="输出路径"),
                IO.String.Output(display_name="任务 ID"),
                IO.String.Output(display_name="报告 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        source_video_asset_id: str,
        character_a_asset_id: str,
        character_b_asset_id: str,
        character_c_asset_id: str,
        prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        duration: int,
        generate_audio: bool,
        watermark: bool,
        seed: int,
        resume_task_id: str = "",
    ):
        result = generate_three_person_asset_video(
            source_video_asset_id=source_video_asset_id,
            character_a_asset_id=character_a_asset_id,
            character_b_asset_id=character_b_asset_id,
            character_c_asset_id=character_c_asset_id,
            prompt=prompt,
            model=model,
            resolution=resolution,
            ratio=ratio,
            duration=duration,
            generate_audio=generate_audio,
            watermark=watermark,
            seed=seed,
            resume_task_id=resume_task_id,
        )
        return IO.NodeOutput(*result, ui={"text": (result[2], result[3])})
