from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

import folder_paths
from comfy_api.latest import IO

from .client import CompanyRemoteAPIError, _image_to_bytes, _upload_media_to_tos, _video_to_bytes
from .config_store import RemoteMediaConfig, get_config


ASSET_GATEWAY_CONFIG_NAME = "seedance_asset_gateway"
TOS_CONFIG_NAME = "seedance2"
ASSET_CACHE_FILE_NAME = "asset_gateway_cache.json"
_CACHE_LOCK = threading.Lock()


def _cache_path() -> Path:
    return Path(folder_paths.get_user_directory()) / "default" / "company_remote" / ASSET_CACHE_FILE_NAME


def _load_cache() -> dict[str, Any]:
    path = _cache_path()
    if not path.is_file():
        return {"version": 1, "assets": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "assets": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), dict):
        return {"version": 1, "assets": {}}
    return payload


def _save_cache(payload: dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _image_dimensions(content: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(content)) as image:
        return int(image.width), int(image.height)


def _request_headers(config: RemoteMediaConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json", **config.extra_headers}
    api_key = config.get_api_key().strip()
    if not api_key:
        source = f"环境变量 {config.api_key_env}" if config.api_key_env else f"配置 {config.name}"
        raise CompanyRemoteAPIError(f"资产平台注册凭据为空，请检查{source}。")
    prefix = config.auth_prefix.strip()
    headers[config.auth_header] = f"{prefix} {api_key}".strip()
    return headers


def _asset_endpoint(config: RemoteMediaConfig) -> str:
    return urllib.parse.urljoin(config.base_url.rstrip("/") + "/", config.submit_path.lstrip("/"))


def _asset_detail_endpoint(config: RemoteMediaConfig, asset_id: str) -> str:
    return _asset_endpoint(config).rstrip("/") + "/" + urllib.parse.quote(asset_id, safe="")


def _safe_error_text(response: requests.Response) -> str:
    text = (response.text or "").strip()[:1600]
    if not text:
        return "远端未返回错误详情"
    return urllib.parse.urlsplit(text)._replace(query="").geturl() if text.startswith(("http://", "https://")) else text


def _extract_asset_id(payload: Any) -> str:
    candidates: list[Any] = []
    if isinstance(payload, dict):
        candidates.extend([payload.get("asset_id"), payload.get("id")])
        data = payload.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("asset_id"), data.get("id")])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise CompanyRemoteAPIError("资产平台响应成功，但没有返回 asset_id。")


def _register_asset(config: RemoteMediaConfig, *, image_url: str, asset_type: str) -> tuple[str, dict[str, Any]]:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            _asset_endpoint(config),
            headers=_request_headers(config),
            json={"url": image_url, "asset_type": asset_type},
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise CompanyRemoteAPIError(f"连接资产平台失败：{exc}") from exc
    if not response.ok:
        raise CompanyRemoteAPIError(f"资产平台注册失败（HTTP {response.status_code}）：{_safe_error_text(response)}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CompanyRemoteAPIError("资产平台返回的不是有效 JSON。") from exc
    return _extract_asset_id(payload), payload


def _asset_result(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("Result", "result", "data"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _asset_error(payload: dict[str, Any]) -> tuple[str, str]:
    result = _asset_result(payload)
    value = result.get("Error") or result.get("error") or {}
    if isinstance(value, dict):
        code = str(value.get("Code") or value.get("code") or "").strip()
        message = str(value.get("Message") or value.get("message") or "").strip()
        return code, message
    return "", str(value or "").strip()


def _get_asset(config: RemoteMediaConfig, asset_id: str) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            _asset_detail_endpoint(config, asset_id),
            headers=_request_headers(config),
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise CompanyRemoteAPIError(f"查询素材资产 {asset_id} 状态失败：{exc}") from exc
    if not response.ok:
        raise CompanyRemoteAPIError(
            f"查询素材资产 {asset_id} 状态失败（HTTP {response.status_code}）：{_safe_error_text(response)}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise CompanyRemoteAPIError(f"查询素材资产 {asset_id} 时返回的不是有效 JSON。") from exc
    if not isinstance(payload, dict):
        raise CompanyRemoteAPIError(f"查询素材资产 {asset_id} 时返回的 JSON 不是对象。")
    return payload


def _wait_for_asset_active(config: RemoteMediaConfig, asset_id: str) -> tuple[dict[str, Any], int]:
    attempts = max(1, int(getattr(config, "max_poll_attempts", 120) or 120))
    interval = max(0.1, float(getattr(config, "poll_interval_seconds", 5) or 5))
    last_status = ""
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            time.sleep(interval)
        payload = _get_asset(config, asset_id)
        result = _asset_result(payload)
        last_status = str(result.get("Status") or result.get("status") or "").strip()
        normalized_status = last_status.lower()
        if normalized_status == "active":
            return result, attempt
        if normalized_status == "failed":
            code, message = _asset_error(payload)
            detail = "：".join(item for item in (code, message) if item) or "远端未返回失败详情"
            raise CompanyRemoteAPIError(f"素材资产 {asset_id} 处理失败：{detail}")
    raise CompanyRemoteAPIError(
        f"素材资产 {asset_id} 在轮询上限内未完成（当前状态：{last_status or 'unknown'}）。"
    )


def _evict_cached_asset(digest: str, asset_id: str) -> None:
    with _CACHE_LOCK:
        cache = _load_cache()
        cached = cache["assets"].get(digest)
        if isinstance(cached, dict) and str(cached.get("asset_id") or "") == asset_id:
            cache["assets"].pop(digest, None)
            _save_cache(cache)


def create_seedance_image_asset(
    image: Any,
    *,
    character_label: str,
    reuse_cached: bool = True,
    gateway_config_name: str = ASSET_GATEWAY_CONFIG_NAME,
    tos_config_name: str = TOS_CONFIG_NAME,
) -> tuple[str, str, bool]:
    label = str(character_label or "人物").strip()[:40] or "人物"
    gateway_config = get_config(gateway_config_name)
    tos_config = get_config(tos_config_name)
    content, mime, extension = _image_to_bytes(image, "png")
    width, height = _image_dimensions(content)
    if not 300 <= width <= 6000 or not 300 <= height <= 6000:
        raise CompanyRemoteAPIError(
            f"{label} 图片尺寸为 {width}x{height}；资产平台要求宽高均在 300 到 6000 像素之间。"
        )

    digest = hashlib.sha256(content).hexdigest()
    cached_asset_id = ""
    cached_registered_at = ""
    with _CACHE_LOCK:
        cache = _load_cache()
        cached = cache["assets"].get(digest)
        if (
            reuse_cached
            and isinstance(cached, dict)
            and cached.get("asset_type") == "Image"
            and str(cached.get("asset_id") or "").strip()
        ):
            cached_asset_id = str(cached["asset_id"])
            cached_registered_at = str(cached.get("registered_at") or "")

    if cached_asset_id:
        try:
            _, status_attempts = _wait_for_asset_active(gateway_config, cached_asset_id)
        except Exception:
            _evict_cached_asset(digest, cached_asset_id)
            raise
        report = {
            "character": label,
            "asset_id": cached_asset_id,
            "asset_type": "Image",
            "asset_status": "Active",
            "status_poll_attempts": status_attempts,
            "image": {"width": width, "height": height, "sha256": digest},
            "cache_reused": True,
            "registered_at": cached_registered_at,
        }
        return cached_asset_id, json.dumps(report, ensure_ascii=False, indent=2), True

    signed_url, object_key = _upload_media_to_tos(
        tos_config,
        content=content,
        mime=mime,
        extension=extension,
        role=f"asset_image_{digest[:12]}",
    )
    asset_id, _ = _register_asset(gateway_config, image_url=signed_url, asset_type="Image")
    _, status_attempts = _wait_for_asset_active(gateway_config, asset_id)
    registered_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cache_entry = {
        "asset_id": asset_id,
        "asset_type": "Image",
        "asset_status": "Active",
        "width": width,
        "height": height,
        "registered_at": registered_at,
        "tos_object_key": object_key,
    }
    with _CACHE_LOCK:
        cache = _load_cache()
        cache["assets"][digest] = cache_entry
        _save_cache(cache)

    report = {
        "character": label,
        "asset_id": asset_id,
        "asset_type": "Image",
        "asset_status": "Active",
        "status_poll_attempts": status_attempts,
        "image": {"width": width, "height": height, "sha256": digest},
        "cache_reused": False,
        "registered_at": registered_at,
        "tos_object_key": object_key,
    }
    return asset_id, json.dumps(report, ensure_ascii=False, indent=2), False


def create_seedance_video_asset(
    video: Any,
    *,
    video_label: str = "原视频",
    reuse_cached: bool = True,
    gateway_config_name: str = ASSET_GATEWAY_CONFIG_NAME,
    tos_config_name: str = TOS_CONFIG_NAME,
) -> tuple[str, str, bool]:
    label = str(video_label or "原视频").strip()[:40] or "原视频"
    gateway_config = get_config(gateway_config_name)
    tos_config = get_config(tos_config_name)
    content, mime, extension, source_info = _video_to_bytes(video)
    digest = hashlib.sha256(content).hexdigest()
    cached_asset_id = ""
    cached_registered_at = ""
    with _CACHE_LOCK:
        cache = _load_cache()
        cached = cache["assets"].get(digest)
        if (
            reuse_cached
            and isinstance(cached, dict)
            and cached.get("asset_type") == "Video"
            and str(cached.get("asset_id") or "").strip()
        ):
            cached_asset_id = str(cached["asset_id"])
            cached_registered_at = str(cached.get("registered_at") or "")

    if cached_asset_id:
        try:
            _, status_attempts = _wait_for_asset_active(gateway_config, cached_asset_id)
        except Exception:
            _evict_cached_asset(digest, cached_asset_id)
            raise
        report = {
            "video": label,
            "asset_id": cached_asset_id,
            "asset_type": "Video",
            "asset_status": "Active",
            "status_poll_attempts": status_attempts,
            "source": source_info,
            "sha256": digest,
            "bytes": len(content),
            "cache_reused": True,
            "registered_at": cached_registered_at,
        }
        return cached_asset_id, json.dumps(report, ensure_ascii=False, indent=2), True

    signed_url, object_key = _upload_media_to_tos(
        tos_config,
        content=content,
        mime=mime,
        extension=extension,
        role=f"asset_video_{digest[:12]}",
    )
    asset_id, _ = _register_asset(gateway_config, image_url=signed_url, asset_type="Video")
    _, status_attempts = _wait_for_asset_active(gateway_config, asset_id)
    registered_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cache_entry = {
        "asset_id": asset_id,
        "asset_type": "Video",
        "asset_status": "Active",
        "bytes": len(content),
        "registered_at": registered_at,
        "tos_object_key": object_key,
    }
    with _CACHE_LOCK:
        cache = _load_cache()
        cache["assets"][digest] = cache_entry
        _save_cache(cache)

    report = {
        "video": label,
        "asset_id": asset_id,
        "asset_type": "Video",
        "asset_status": "Active",
        "status_poll_attempts": status_attempts,
        "source": source_info,
        "sha256": digest,
        "bytes": len(content),
        "cache_reused": False,
        "registered_at": registered_at,
        "tos_object_key": object_key,
    }
    return asset_id, json.dumps(report, ensure_ascii=False, indent=2), False


def create_seedance_abc_assets(
    image_a: Any,
    image_b: Any,
    image_c: Any,
    *,
    reuse_cached: bool = True,
) -> tuple[str, str, str, str]:
    tasks = [
        ("人物 A", image_a),
        ("人物 B", image_b),
        ("人物 C", image_c),
    ]

    def register(task: tuple[str, Any]) -> tuple[str, str, bool]:
        label, image = task
        try:
            return create_seedance_image_asset(
                image,
                character_label=label,
                reuse_cached=reuse_cached,
            )
        except Exception as exc:
            raise CompanyRemoteAPIError(f"{label} 注册失败：{exc}") from exc

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="seedance-asset") as executor:
        results = list(executor.map(register, tasks))

    reports = [json.loads(result[1]) for result in results]
    summary = {
        "asset_type": "Image",
        "parallel_workers": 3,
        "assets": reports,
    }
    return results[0][0], results[1][0], results[2][0], json.dumps(summary, ensure_ascii=False, indent=2)


def create_seedance_video_abc_assets(
    video: Any,
    image_a: Any,
    image_b: Any,
    image_c: Any,
    *,
    reuse_cached: bool = True,
) -> tuple[str, str, str, str, str]:
    tasks = [
        ("原视频", "Video", video),
        ("人物 A", "Image", image_a),
        ("人物 B", "Image", image_b),
        ("人物 C", "Image", image_c),
    ]

    def register(task: tuple[str, str, Any]) -> tuple[str, str, bool]:
        label, asset_type, media = task
        try:
            if asset_type == "Video":
                return create_seedance_video_asset(
                    media,
                    video_label=label,
                    reuse_cached=reuse_cached,
                )
            return create_seedance_image_asset(
                media,
                character_label=label,
                reuse_cached=reuse_cached,
            )
        except Exception as exc:
            raise CompanyRemoteAPIError(f"{label} 注册失败：{exc}") from exc

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="seedance-media-asset") as executor:
        results = list(executor.map(register, tasks))

    summary = {
        "parallel_workers": 4,
        "source_video": json.loads(results[0][1]),
        "characters": [json.loads(result[1]) for result in results[1:]],
    }
    return (
        results[0][0],
        results[1][0],
        results[2][0],
        results[3][0],
        json.dumps(summary, ensure_ascii=False, indent=2),
    )


class CompanySeedanceImageAssetCreate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedanceImageAssetCreate",
            display_name="Seedance 真人图片注册为资产",
            category="company-remote/video/Seedance",
            description="先把本地人物图上传到 TOS，再注册为 Seedance 平台 Image 资产并返回 asset_id。",
            search_aliases=["Seedance Asset Create", "真人资产注册", "Create Image Asset"],
            inputs=[
                IO.Image.Input("image", display_name="人物参考图"),
                IO.String.Input(
                    "character_label",
                    display_name="人物标签",
                    default="人物 A",
                    tooltip="只用于报告和 TOS 文件名，例如人物 A、人物 B、人物 C。",
                ),
                IO.Boolean.Input(
                    "reuse_cached",
                    display_name="复用相同图片的资产 ID",
                    default=True,
                    tooltip="开启后，相同图片不会重复上传和注册。关闭可强制创建新资产。",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="资产 ID"),
                IO.String.Output(display_name="注册报告 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image: Any, character_label: str = "人物 A", reuse_cached: bool = True):
        asset_id, report, cache_reused = create_seedance_image_asset(
            image,
            character_label=character_label,
            reuse_cached=reuse_cached,
        )
        status = "已复用" if cache_reused else "已新建"
        return IO.NodeOutput(asset_id, report, ui={"text": (f"{character_label}：{status}资产 {asset_id}", report)})


class CompanySeedanceABCAssetCreate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedanceABCAssetCreate",
            display_name="Seedance 人物 A/B/C 并行注册资产",
            category="company-remote/video/Seedance",
            description="同时上传 3 张真人参考图到 TOS，并注册成 3 个 Seedance Image 资产。",
            search_aliases=["Seedance ABC Assets", "三人物资产注册", "Parallel Image Assets"],
            inputs=[
                IO.Image.Input("image_a", display_name="人物 A 参考图"),
                IO.Image.Input("image_b", display_name="人物 B 参考图"),
                IO.Image.Input("image_c", display_name="人物 C 参考图"),
                IO.Boolean.Input(
                    "reuse_cached",
                    display_name="复用相同图片的资产 ID",
                    default=True,
                    tooltip="开启后，相同图片不会重复上传和注册。关闭可强制创建新资产。",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="人物 A 资产 ID"),
                IO.String.Output(display_name="人物 B 资产 ID"),
                IO.String.Output(display_name="人物 C 资产 ID"),
                IO.String.Output(display_name="汇总报告 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image_a: Any, image_b: Any, image_c: Any, reuse_cached: bool = True):
        asset_a, asset_b, asset_c, report = create_seedance_abc_assets(
            image_a,
            image_b,
            image_c,
            reuse_cached=reuse_cached,
        )
        return IO.NodeOutput(
            asset_a,
            asset_b,
            asset_c,
            report,
            ui={"text": (f"人物 A：{asset_a}\n人物 B：{asset_b}\n人物 C：{asset_c}", report)},
        )


class CompanySeedanceVideoABCAssetCreate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedanceVideoABCAssetCreate",
            display_name="Seedance 原视频 + 人物 A/B/C 注册资产",
            category="company-remote/video/Seedance",
            description="把源视频注册为 Video 资产，同时把 A/B/C 三张人物图注册为 Image 资产。",
            search_aliases=["Seedance Video ABC Assets", "三人物视频资产注册", "Video and Character Assets"],
            inputs=[
                IO.Video.Input("video", display_name="源视频（建议先裁到 4-15 秒）"),
                IO.Image.Input("image_a", display_name="人物 A 参考图"),
                IO.Image.Input("image_b", display_name="人物 B 参考图"),
                IO.Image.Input("image_c", display_name="人物 C 参考图"),
                IO.Boolean.Input(
                    "reuse_cached",
                    display_name="复用相同素材的资产 ID",
                    default=True,
                    tooltip="开启后，内容 Hash 相同的视频或图片不会重复上传和注册。",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="源视频资产 ID"),
                IO.String.Output(display_name="人物 A 资产 ID"),
                IO.String.Output(display_name="人物 B 资产 ID"),
                IO.String.Output(display_name="人物 C 资产 ID"),
                IO.String.Output(display_name="汇总报告 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Any, image_a: Any, image_b: Any, image_c: Any, reuse_cached: bool = True):
        source_asset, asset_a, asset_b, asset_c, report = create_seedance_video_abc_assets(
            video,
            image_a,
            image_b,
            image_c,
            reuse_cached=reuse_cached,
        )
        return IO.NodeOutput(
            source_asset,
            asset_a,
            asset_b,
            asset_c,
            report,
            ui={
                "text": (
                    f"源视频：{source_asset}\n人物 A：{asset_a}\n人物 B：{asset_b}\n人物 C：{asset_c}",
                    report,
                )
            },
        )
