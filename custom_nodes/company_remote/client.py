from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image

import folder_paths
from comfy_api.latest import InputImpl

from .config_store import RemoteMediaConfig, get_config_dir


class CompanyRemoteAPIError(RuntimeError):
    pass


_THREAD_LOCAL = threading.local()
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_FILE_NAME = "gpttext_models_cache.json"
DEFAULT_OPENAI_TEXT_MODEL = "gpt-5.4"


def generate_image(
    config: RemoteMediaConfig,
    *,
    operation: str,
    prompt: str,
    model: str = "",
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    max_images: int = 1,
    reference_images: list[Any] | None = None,
    extra_values: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    values = _base_values(
        operation=operation,
        output_type="image",
        prompt=prompt,
        model=model,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        max_images=max_images,
        **(extra_values or {}),
    )
    payload = _build_payload(config, values=values, images={"reference_images": reference_images or []})
    response_data = _submit(config, payload)
    url = _resolve_result_url(config, response_data, output_type="image")
    path = _download_file(url, config=config, output_type="image")
    return _load_image_tensor(path), path


def generate_openai_image(
    config: RemoteMediaConfig,
    *,
    prompt: str,
    model: str,
    size: str,
    quality: str,
    background: str,
    n: int,
    files: list[tuple[str, tuple[str, BytesIO, str]]] | None = None,
):
    prompt = _normalize_prompt_text(prompt)
    image_count = int(n)
    if image_count != 1:
        raise CompanyRemoteAPIError("AI-Zero-Token currently supports exactly one image per request (n=1).")
    values = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "background": background,
        "n": image_count,
        "size": size,
        "response_format": "b64_json",
        "moderation": "low",
    }
    if files:
        payload = _build_payload(config, values=values)
        image_items, mask_item = _openai_files_to_json_media(files)
        payload["images"] = image_items
        if mask_item is not None:
            payload["mask"] = mask_item
        response_data = _submit_openai_json(config, path="/images/edits", payload=payload)
    else:
        response_data = _submit(config, _build_payload(config, values=values))
    return _openai_image_response_to_tensor(response_data, config=config)


def generate_openai_chat_text(
    config: RemoteMediaConfig,
    *,
    skill: str,
    user_prompt: str,
    model: str = "gpt-5.4",
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    skill = _normalize_required_text(skill, field_name="skill")
    user_prompt = _normalize_required_text(user_prompt, field_name="user_prompt")
    payload = _build_payload(
        config,
        values={
            "model": model,
            "skill": skill,
            "user_prompt": user_prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
    )
    if "messages" not in payload:
        payload["messages"] = [
            {"role": "system", "content": skill},
            {"role": "user", "content": user_prompt},
        ]
    payload.pop("skill", None)
    payload.pop("user_prompt", None)
    response_data = _submit(config, payload)
    text = _extract_openai_text(response_data, "choices.0.message.content")
    if not text:
        text = _extract_openai_text(response_data, "output_text")
    if not text:
        text = _extract_openai_text(response_data, "output.0.content.0.text")
    if not text:
        raise CompanyRemoteAPIError("Response did not contain text path 'choices.0.message.content' or 'output_text'.")
    return _normalize_prompt_text(text)


def generate_openai_image_prompt_text(
    config: RemoteMediaConfig,
    *,
    skill: str,
    modification_target: str,
    image: Any,
    model: str = "gpt-5.4",
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    skill = _normalize_required_text(skill, field_name="skill")
    modification_target = _normalize_required_text(modification_target, field_name="modification_target")
    image_urls = _image_batch_to_data_urls(image)

    payload = _build_payload(
        config,
        values={
            "model": model,
            "skill": skill,
            "user_prompt": modification_target,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
    )
    payload["messages"] = [
        {"role": "system", "content": skill},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": modification_target},
                *[
                    {"type": "image_url", "image_url": {"url": image_url}}
                    for image_url in image_urls
                ],
            ],
        },
    ]
    payload.pop("skill", None)
    payload.pop("user_prompt", None)

    response_data = _submit(config, payload)
    text = _extract_openai_text(response_data, "choices.0.message.content")
    if not text:
        text = _extract_openai_text(response_data, "output_text")
    if not text:
        text = _extract_openai_text(response_data, "output.0.content.0.text")
    if not text:
        raise CompanyRemoteAPIError("Response did not contain text path 'choices.0.message.content' or 'output_text'.")
    return _normalize_prompt_text(text)


def generate_video(
    config: RemoteMediaConfig,
    *,
    operation: str,
    prompt: str,
    model: str = "",
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    resolution: str = "720p",
    ratio: str = "16:9",
    duration: float = 5.0,
    fps: int = 24,
    seed: int = 0,
    image: Any = None,
    first_frame: Any = None,
    last_frame: Any = None,
    reference_images: list[Any] | None = None,
    reference_video: Any = None,
    reference_videos: list[Any] | None = None,
    reference_video_url: str = "",
    extra_values: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    values = _base_values(
        operation=operation,
        output_type="video",
        prompt=prompt,
        model=model,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        resolution=resolution,
        ratio=ratio,
        duration=duration,
        fps=fps,
        seed=seed,
        reference_video_url=reference_video_url,
        **(extra_values or {}),
    )
    images = {
        "image": image,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "reference_images": reference_images or [],
    }
    videos = {"reference_video": reference_video, "reference_videos": reference_videos or []}
    payload, media_debug = _build_payload_with_debug(config, values=values, images=images, videos=videos)
    _write_request_debug(config, operation=operation, values=values, payload=payload, media_debug=media_debug)
    response_data = _submit(config, payload)
    url = _resolve_result_url(config, response_data, output_type="video")
    path = _download_file(url, config=config, output_type="video")
    return InputImpl.VideoFromFile(path), path


def test_connection(config: RemoteMediaConfig) -> dict[str, Any]:
    path = config.test_path or config.submit_path
    url = _join_url(config.base_url, path)
    headers = _build_headers(config)
    response = _request_raw("GET", url, headers=headers, timeout=min(config.timeout_seconds, 30))
    return {"ok": response.ok, "status": response.status_code, "body": _safe_response_text(response)}


def get_cached_openai_model_ids() -> list[str]:
    try:
        with _MODEL_CACHE_LOCK:
            with open(_model_cache_path(), "r", encoding="utf-8") as handle:
                data = json.load(handle)
        models = data.get("models", []) if isinstance(data, dict) else data
        normalized = _normalize_model_ids(models)
        if normalized:
            return _prioritize_default_model(normalized)
    except (OSError, ValueError, TypeError):
        pass
    return [DEFAULT_OPENAI_TEXT_MODEL]


def get_openai_model_ids(config: RemoteMediaConfig) -> list[str]:
    path = config.test_path or "/models"
    url = _join_url(config.base_url, path)
    headers = _build_headers(config)
    headers.pop("Content-Type", None)
    try:
        response_data = _request_json(
            "GET",
            url,
            headers=headers,
            timeout=min(config.timeout_seconds, 5),
        )
        models = _extract_model_ids(response_data)
        if not models:
            raise CompanyRemoteAPIError("Model list response did not contain any model IDs.")
    except (CompanyRemoteAPIError, ValueError, TypeError):
        return get_cached_openai_model_ids()

    models = _prioritize_default_model(models)
    try:
        _write_model_cache(models)
    except OSError:
        pass
    return models


def _extract_model_ids(response_data: Any) -> list[str]:
    if isinstance(response_data, dict):
        candidates = response_data.get("data")
        if candidates is None:
            candidates = response_data.get("models")
    else:
        candidates = response_data
    return _normalize_model_ids(candidates)


def _normalize_model_ids(candidates: Any) -> list[str]:
    if not isinstance(candidates, list):
        return []
    models: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if isinstance(item, str):
            model_id = item.strip()
        elif isinstance(item, dict):
            model_id = str(item.get("id") or item.get("name") or "").strip()
        else:
            model_id = ""
        if model_id and model_id not in seen:
            models.append(model_id)
            seen.add(model_id)
    return models


def _prioritize_default_model(models: list[str]) -> list[str]:
    if DEFAULT_OPENAI_TEXT_MODEL not in models:
        return models
    return [DEFAULT_OPENAI_TEXT_MODEL, *[model for model in models if model != DEFAULT_OPENAI_TEXT_MODEL]]


def _model_cache_path() -> str:
    return os.path.join(get_config_dir(), _MODEL_CACHE_FILE_NAME)


def _write_model_cache(models: list[str]) -> None:
    cache_path = _model_cache_path()
    temp_path = f"{cache_path}.{os.getpid()}.tmp"
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }
    with _MODEL_CACHE_LOCK:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, cache_path)


def _base_values(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _submit(config: RemoteMediaConfig, payload: dict[str, Any]) -> Any:
    headers = _build_headers(config)
    submit_url = _join_url(config.base_url, config.submit_path)
    if config.method == "GET":
        return _request_json("GET", submit_url, headers=headers, params=payload, timeout=config.timeout_seconds)
    return _request_json("POST", submit_url, headers=headers, json_body=payload, timeout=config.timeout_seconds)


def _resolve_result_url(config: RemoteMediaConfig, response_data: Any, *, output_type: str) -> str:
    url_paths = [config.response_image_url_path if output_type == "image" else config.response_video_url_path]
    url_paths.append(config.response_result_url_path)
    for path in url_paths:
        result_url = _extract_first_string(response_data, path)
        if result_url:
            return result_url

    if config.poll_enabled:
        task_id = _extract_first_string(response_data, config.response_task_id_path)
        if not task_id:
            raise CompanyRemoteAPIError(
                f"Response did not contain result url paths {url_paths} or task id path '{config.response_task_id_path}'."
            )
        return _poll_for_result(config, task_id=task_id, output_type=output_type)

    raise CompanyRemoteAPIError(f"Response did not contain result url paths {url_paths}.")


def _build_payload(
    config: RemoteMediaConfig,
    *,
    values: dict[str, Any],
    images: dict[str, Any] | None = None,
    videos: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload, _ = _build_payload_with_debug(config, values=values, images=images, videos=videos)
    return payload


def _build_payload_with_debug(
    config: RemoteMediaConfig,
    *,
    values: dict[str, Any],
    images: dict[str, Any] | None = None,
    videos: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    media_debug: list[dict[str, Any]] = []
    payload = _deep_copy_json(config.request_template)
    payload = _render_template(payload, values)

    for key, value in values.items():
        if value not in (None, "") and key not in payload:
            payload[key] = value

    images = images or {}
    single_image = images.get("image")
    if single_image is not None and config.image_field and config.image_field not in payload:
        payload[config.image_field] = _image_to_url(single_image, config, role="image", media_debug=media_debug)

    first_frame = images.get("first_frame")
    if first_frame is not None:
        encoded_first_frame = _image_to_url(first_frame, config, role="first_frame", media_debug=media_debug)
        if _uses_ark_content_generation(config):
            _append_ark_content_urls(payload, "image_url", [encoded_first_frame], role="first_frame")
        elif config.first_frame_field and config.first_frame_field not in payload:
            payload[config.first_frame_field] = encoded_first_frame

    last_frame = images.get("last_frame")
    if last_frame is not None:
        encoded_last_frame = _image_to_url(last_frame, config, role="last_frame", media_debug=media_debug)
        if _uses_ark_content_generation(config):
            _append_ark_content_urls(payload, "image_url", [encoded_last_frame], role="last_frame")
        elif config.last_frame_field and config.last_frame_field not in payload:
            payload[config.last_frame_field] = encoded_last_frame

    reference_images = [img for img in images.get("reference_images", []) if img is not None]
    if reference_images:
        if _uses_ark_content_generation(config) and str(values.get("operation", "")).startswith("seedance2_"):
            prepared_images = _prepare_seedance_reference_images(reference_images)
            encoded_images = [
                _image_to_url(
                    item["image"],
                    config,
                    role=item["role"],
                    media_debug=media_debug,
                    source_info=item["source"],
                )
                for item in prepared_images
            ]
        else:
            encoded_images = [
                _image_to_url(img, config, role=f"reference_image_{index + 1}", media_debug=media_debug)
                for index, img in enumerate(reference_images)
            ]
        if _uses_ark_content_generation(config):
            _append_ark_content_urls(payload, "image_url", encoded_images, role="reference_image")
        else:
            if config.reference_images_field and config.reference_images_field not in payload:
                payload[config.reference_images_field] = encoded_images
            if config.images_field and config.images_field not in payload:
                payload[config.images_field] = encoded_images

    videos = videos or {}
    reference_video = videos.get("reference_video")
    if reference_video is not None and not _uses_ark_content_generation(config):
        encoded_video = _video_to_url(reference_video, config, role="reference_video", media_debug=media_debug)
        if config.video_field and config.video_field not in payload:
            payload[config.video_field] = encoded_video
        if config.reference_videos_field and config.reference_videos_field not in payload:
            payload[config.reference_videos_field] = [encoded_video]

    reference_videos = [vid for vid in videos.get("reference_videos", []) if vid is not None]
    if reference_videos:
        encoded_videos = [
            _video_to_url(vid, config, role=f"reference_video_{index + 1}", media_debug=media_debug)
            for index, vid in enumerate(reference_videos)
        ]
        if _uses_ark_content_generation(config):
            _append_ark_content_urls(payload, "video_url", encoded_videos, role="reference_video")
        else:
            if config.video_field and config.video_field not in payload:
                payload[config.video_field] = encoded_videos[0]
            if config.reference_videos_field and config.reference_videos_field not in payload:
                payload[config.reference_videos_field] = encoded_videos

    reference_video_url = values.get("reference_video_url")
    if reference_video_url:
        reference_video_url = str(reference_video_url).strip()
        if reference_video_url:
            _record_media_debug(
                media_debug,
                role="reference_video_url",
                media_kind="video",
                delivery="external_url",
                mime="",
                extension="",
                content=b"",
                url=reference_video_url,
                object_key="",
                source={"kind": "url"},
            )
            if _uses_ark_content_generation(config):
                _append_ark_content_urls(payload, "video_url", [reference_video_url], role="reference_video_url")
            elif config.reference_videos_field and config.reference_videos_field not in payload:
                payload[config.reference_videos_field] = [reference_video_url]

    return payload, media_debug


def _uses_ark_content_generation(config: RemoteMediaConfig) -> bool:
    return "/api/v3/contents/generations/tasks" in (config.submit_path or "") and "ark" in (config.base_url or "")


def _append_ark_content_urls(payload: dict[str, Any], content_type: str, urls: list[str], *, role: str) -> None:
    content = payload.setdefault("content", [])
    if not isinstance(content, list):
        content = [content]
        payload["content"] = content
    for url in urls:
        content.append({
            "type": content_type,
            content_type: {"url": url},
            "role": role,
        })


def _build_headers(config: RemoteMediaConfig) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if config.method != "GET":
        headers["Content-Type"] = "application/json"
    headers.update({str(k): str(v) for k, v in config.extra_headers.items() if str(k).strip()})
    api_key = config.get_api_key()
    if api_key and config.auth_header:
        value = api_key
        if config.auth_prefix:
            value = f"{config.auth_prefix.strip()} {api_key}"
        headers[config.auth_header] = value
    return headers


def _poll_for_result(config: RemoteMediaConfig, *, task_id: str, output_type: str) -> str:
    headers = _build_headers(config)
    url_paths = [config.response_image_url_path if output_type == "image" else config.response_video_url_path]
    url_paths.append(config.response_result_url_path)
    for _ in range(config.max_poll_attempts):
        time.sleep(config.poll_interval_seconds)
        path = config.poll_path_template.format(task_id=urllib.parse.quote(task_id, safe=""))
        response_data = _request_json("GET", _join_url(config.base_url, path), headers=headers, timeout=config.timeout_seconds)
        status = _extract_first_string(response_data, config.response_status_path).lower()
        if status in config.failure_statuses:
            raise CompanyRemoteAPIError(f"Remote task failed with status '{status}': {response_data}")
        for url_path in url_paths:
            result_url = _extract_first_string(response_data, url_path)
            if result_url and (not status or status in config.success_statuses):
                return result_url
    raise CompanyRemoteAPIError("Remote task timed out while polling.")


def _request_json(method: str, url: str, *, headers: dict[str, str], timeout: int, params=None, json_body=None) -> Any:
    try:
        response = _request_raw(method, url, headers=headers, params=params, json=json_body, timeout=timeout)
    except requests.RequestException as exc:
        raise CompanyRemoteAPIError(f"Remote request failed: {exc}") from exc
    if response.status_code == 413:
        raise CompanyRemoteAPIError(
            "Remote request body exceeded the AI-Zero-Token gateway limit. "
            "Reduce the number or resolution of input images, or increase AZT_BODY_LIMIT_MB."
        )
    if not response.ok:
        raise CompanyRemoteAPIError(f"Remote request failed with HTTP {response.status_code}: {_safe_response_text(response)}")
    try:
        return response.json()
    except ValueError as exc:
        raise CompanyRemoteAPIError(f"Remote response is not JSON: {_safe_response_text(response)}") from exc


def _request_raw(method: str, url: str, **kwargs: Any) -> requests.Response:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _THREAD_LOCAL.session = session
    return session.request(method, url, **kwargs)


def _openai_files_to_json_media(
    files: list[tuple[str, tuple[str, BytesIO, str]]],
) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    images: list[dict[str, str]] = []
    mask: dict[str, str] | None = None

    for field_name, (filename, content, mime_type) in files:
        raw = content.getvalue()
        resolved_mime = mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        item = {
            "image_url": f"data:{resolved_mime};base64,{base64.b64encode(raw).decode('ascii')}"
        }
        if field_name == "mask":
            mask = item
        elif field_name in {"image", "image[]"}:
            images.append(item)

    if not images:
        raise CompanyRemoteAPIError("OpenAI image edit request did not contain an input image.")
    return images, mask


def _submit_openai_json(
    config: RemoteMediaConfig,
    *,
    path: str,
    payload: dict[str, Any],
) -> Any:
    headers = _build_headers(config)
    return _request_json(
        "POST",
        _join_url(config.base_url, path),
        headers=headers,
        json_body=payload,
        timeout=config.timeout_seconds,
    )


def _openai_image_response_to_tensor(response_data: Any, *, config: RemoteMediaConfig):
    data = _extract_value(response_data, "data")
    image_payloads: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                value = item.get("b64_json") or item.get("url")
                if value:
                    image_payloads.append(str(value))
            elif item not in (None, ""):
                image_payloads.append(str(item))

    if not image_payloads:
        for path in (config.response_image_url_path, "data.0.b64_json", "data.0.url", config.response_result_url_path):
            value = _extract_first_string(response_data, path)
            if value:
                image_payloads.append(value)
                break

    if not image_payloads:
        raise CompanyRemoteAPIError("Response did not contain OpenAI image data at 'data[].b64_json' or 'data[].url'.")

    image_tensors = [_image_payload_to_tensor(payload, config=config) for payload in image_payloads]
    ref_h, ref_w = image_tensors[0].shape[:2]
    for i, tensor in enumerate(image_tensors):
        if tensor.shape[:2] != (ref_h, ref_w):
            from comfy.utils import common_upscale

            samples = tensor.unsqueeze(0).movedim(-1, 1)
            samples = common_upscale(samples, ref_w, ref_h, "bilinear", "center")
            image_tensors[i] = samples.movedim(1, -1).squeeze(0)

    try:
        import torch
    except ImportError as exc:
        raise CompanyRemoteAPIError("torch is required to return IMAGE tensors") from exc
    return torch.stack(image_tensors, dim=0)


def _image_payload_to_tensor(payload: str, *, config: RemoteMediaConfig):
    content = _image_payload_to_bytes(payload, config=config)
    with Image.open(BytesIO(content)) as img:
        img = img.convert("RGBA")
        array = np.asarray(img).astype(np.float32) / 255.0
    try:
        import torch
    except ImportError as exc:
        raise CompanyRemoteAPIError("torch is required to return IMAGE tensors") from exc
    return torch.from_numpy(array)


def _image_payload_to_bytes(payload: str, *, config: RemoteMediaConfig) -> bytes:
    value = payload.strip()
    if value.startswith("data:"):
        _, encoded = value.split(",", 1)
        return base64.b64decode(encoded)
    if value.startswith(("http://", "https://")):
        url = value if value.startswith(("http://", "https://")) else _join_url(config.base_url, value)
        request_headers = _build_headers(config) if url.startswith(config.base_url) else {}
        try:
            response = _request_raw("GET", url, headers=request_headers, timeout=config.timeout_seconds)
        except requests.RequestException as exc:
            raise CompanyRemoteAPIError(f"Result download failed: {exc}") from exc
        if not response.ok:
            raise CompanyRemoteAPIError(f"Result download failed with HTTP {response.status_code}: {_safe_response_text(response)}")
        return response.content
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as base64_exc:
        if value.startswith("/"):
            url = _join_url(config.base_url, value)
            request_headers = _build_headers(config)
            try:
                response = _request_raw("GET", url, headers=request_headers, timeout=config.timeout_seconds)
            except requests.RequestException as exc:
                raise CompanyRemoteAPIError(f"Result download failed: {exc}") from exc
            if not response.ok:
                raise CompanyRemoteAPIError(f"Result download failed with HTTP {response.status_code}: {_safe_response_text(response)}")
            return response.content
        raise CompanyRemoteAPIError("Image response was neither a valid URL nor valid base64 data.") from base64_exc


def _download_file(url_or_data_uri: str, *, config: RemoteMediaConfig, output_type: str) -> str:
    output_root = folder_paths.get_output_directory()
    output_dir = os.path.join(output_root, "company_remote")
    os.makedirs(output_dir, exist_ok=True)

    if url_or_data_uri.startswith("data:"):
        header, encoded = url_or_data_uri.split(",", 1)
        content = base64.b64decode(encoded)
        content_type = header.split(";", 1)[0].removeprefix("data:")
        extension = _extension_from_content(output_type, "", content_type)
        path = os.path.join(output_dir, f"company_remote_{output_type}_{int(time.time() * 1000)}{extension}")
        with open(path, "wb") as f:
            f.write(content)
        return path

    url = url_or_data_uri if url_or_data_uri.startswith(("http://", "https://")) else _join_url(config.base_url, url_or_data_uri)
    request_headers = _build_headers(config) if url.startswith(config.base_url) else {}
    try:
        response = _request_raw("GET", url, headers=request_headers, timeout=config.timeout_seconds, stream=True)
    except requests.RequestException as exc:
        raise CompanyRemoteAPIError(f"Result download failed: {exc}") from exc
    if not response.ok:
        raise CompanyRemoteAPIError(f"Result download failed with HTTP {response.status_code}: {_safe_response_text(response)}")

    extension = _extension_from_content(output_type, url_or_data_uri, response.headers.get("Content-Type", ""))
    path = os.path.join(output_dir, f"company_remote_{output_type}_{int(time.time() * 1000)}{extension}")
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return path


def _load_image_tensor(path: str):
    with Image.open(path) as img:
        img = img.convert("RGB")
        array = np.asarray(img).astype(np.float32) / 255.0
    try:
        import torch
        return torch.from_numpy(array)[None,]
    except ImportError as exc:
        raise CompanyRemoteAPIError("torch is required to return IMAGE tensors") from exc


def _image_to_url(
    image: Any,
    config: RemoteMediaConfig,
    *,
    role: str,
    media_debug: list[dict[str, Any]] | None = None,
    source_info: dict[str, Any] | None = None,
) -> str:
    content, mime, extension = _image_to_bytes(image, config.image_format)
    source = {"kind": type(image).__name__, **(source_info or {})}
    if config.tos_enabled or config.media_delivery == "tos_presigned":
        url, key = _upload_media_to_tos(config, content=content, mime=mime, extension=extension, role=role)
        _record_media_debug(
            media_debug,
            role=role,
            media_kind="image",
            delivery="tos_presigned",
            mime=mime,
            extension=extension,
            content=content,
            url=url,
            object_key=key,
            source=source,
        )
        return url
    encoded = base64.b64encode(content).decode("ascii")
    data_uri = f"data:{mime};base64,{encoded}"
    _record_media_debug(
        media_debug,
        role=role,
        media_kind="image",
        delivery="base64",
        mime=mime,
        extension=extension,
        content=content,
        url=data_uri,
        object_key="",
        source=source,
    )
    return data_uri


def _image_to_data_uri(image: Any, image_format: str) -> str:
    content, mime, _ = _image_to_bytes(image, image_format)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _prepare_seedance_reference_images(
    reference_images: list[Any],
    *,
    max_images: int = 9,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for input_index, image in enumerate(reference_images, start=1):
        if getattr(image, "ndim", 0) == 4:
            frame_count = int(image.shape[0])
            frames = [image[index] for index in range(frame_count)]
        else:
            frames = [image]
        if frames:
            sources.append({"input_index": input_index, "frames": frames})

    if not sources:
        return []
    if len(sources) > max_images:
        raise CompanyRemoteAPIError(
            f"Seedance 2.0 supports at most {max_images} reference image inputs, got {len(sources)}."
        )

    frame_counts = [len(source["frames"]) for source in sources]
    if sum(frame_counts) <= max_images:
        slot_counts = list(frame_counts)
    else:
        slot_counts = [1 for _ in sources]
        remaining = max_images - len(sources)
        while remaining > 0:
            candidates = [index for index, count in enumerate(frame_counts) if slot_counts[index] < count]
            if not candidates:
                break
            selected = max(candidates, key=lambda index: (frame_counts[index] / slot_counts[index], -index))
            slot_counts[selected] += 1
            remaining -= 1

    prepared: list[dict[str, Any]] = []
    for source, slot_count in zip(sources, slot_counts):
        input_index = source["input_index"]
        frames = source["frames"]
        frame_count = len(frames)
        for slot_index in range(slot_count):
            start = slot_index * frame_count // slot_count
            end = (slot_index + 1) * frame_count // slot_count
            chunk = frames[start:end]
            is_sheet = len(chunk) > 1
            prepared.append({
                "image": _make_reference_contact_sheet(chunk) if is_sheet else chunk[0],
                "role": (
                    f"reference_image_{input_index}_sheet_{slot_index + 1}_of_{slot_count}"
                    if is_sheet
                    else f"reference_image_{input_index}_frame_{start + 1}"
                ),
                "source": {
                    "input_index": input_index,
                    "input_batch_frames": frame_count,
                    "delivery_mode": "contact_sheet" if is_sheet else "single_frame",
                    "sheet_index": slot_index + 1,
                    "sheet_count": slot_count,
                    "frame_start": start + 1,
                    "frame_end": end,
                    "frames_in_image": len(chunk),
                },
            })
    return prepared


def _make_reference_contact_sheet(frames: list[Any]) -> np.ndarray:
    if not frames:
        raise CompanyRemoteAPIError("Cannot build a Seedance reference contact sheet from an empty frame list.")

    arrays = [_image_frame_to_uint8(frame) for frame in frames]
    frame_height, frame_width = arrays[0].shape[:2]
    count = len(arrays)
    target_aspect = 16 / 9
    best_columns = min(
        range(1, count + 1),
        key=lambda columns: (
            abs(math.log(((columns * frame_width) / (math.ceil(count / columns) * frame_height)) / target_aspect)),
            columns * math.ceil(count / columns) - count,
        ),
    )
    columns = best_columns
    rows = math.ceil(count / columns)

    raw_width = columns * frame_width
    raw_height = rows * frame_height
    max_edge = 6000
    max_pixels = 8_295_044
    scale = min(
        1.0,
        max_edge / max(raw_width, raw_height),
        math.sqrt(max_pixels / (raw_width * raw_height)),
    )
    thumb_width = max(1, int(frame_width * scale))
    thumb_height = max(1, int(frame_height * scale))
    canvas_width = columns * thumb_width
    canvas_height = rows * thumb_height
    background = 255 if float(np.mean(arrays[0])) >= 127.5 else 0
    canvas = Image.new("RGB", (canvas_width, canvas_height), (background, background, background))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

    for index, array in enumerate(arrays):
        frame = Image.fromarray(array)
        if frame.size != (thumb_width, thumb_height):
            frame = frame.resize((thumb_width, thumb_height), resample=resampling)
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        canvas.paste(frame, (x, y))

    return np.asarray(canvas).astype(np.float32) / 255.0


def _image_frame_to_uint8(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    array = np.clip(255.0 * np.asarray(frame), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 3:
        array = array[:, :, :3]
    else:
        raise CompanyRemoteAPIError(f"Unsupported IMAGE frame shape for Seedance reference: {array.shape}")
    return array


def _image_to_bytes(image: Any, image_format: str) -> tuple[bytes, str, str]:
    frame = image[0] if getattr(image, "ndim", 0) == 4 else image
    array = _image_frame_to_uint8(frame)
    pil_image = Image.fromarray(array)
    fmt = (image_format or "png").lower()
    if fmt not in {"png", "jpeg", "jpg", "webp"}:
        fmt = "png"
    save_format = "JPEG" if fmt in {"jpg", "jpeg"} else fmt.upper()
    if save_format == "JPEG" and pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    buf = BytesIO()
    pil_image.save(buf, format=save_format)
    mime = "image/jpeg" if save_format == "JPEG" else f"image/{fmt}"
    extension = ".jpg" if save_format == "JPEG" else f".{fmt}"
    return buf.getvalue(), mime, extension


def _image_batch_to_data_urls(image: Any) -> list[str]:
    if image is None:
        raise CompanyRemoteAPIError("Field 'image' cannot be empty.")

    ndim = getattr(image, "ndim", 0)
    if ndim == 4:
        if image.shape[0] == 0:
            raise CompanyRemoteAPIError("Field 'image' cannot contain an empty batch.")
        frames = [image[index] for index in range(image.shape[0])]
    elif ndim == 3:
        frames = [image]
    else:
        raise CompanyRemoteAPIError(f"Field 'image' must be an IMAGE tensor, got shape {getattr(image, 'shape', None)}.")

    data_urls: list[str] = []
    for frame in frames:
        content, mime, _ = _image_to_bytes(frame, "png")
        encoded = base64.b64encode(content).decode("ascii")
        data_urls.append(f"data:{mime};base64,{encoded}")
    return data_urls


def _upload_media_to_tos(
    config: RemoteMediaConfig,
    *,
    content: bytes,
    mime: str,
    extension: str,
    role: str,
) -> tuple[str, str]:
    try:
        import tos
    except ImportError as exc:
        raise CompanyRemoteAPIError("TOS delivery requires the 'tos' package. Install it with the Windows portable Python: ..\\python_embeded\\python.exe -m pip install tos") from exc

    ak = config.get_tos_access_key()
    sk = config.get_tos_secret_key()
    if not ak:
        raise CompanyRemoteAPIError(f"TOS access key environment variable '{config.tos_access_key_env}' is not set")
    if not sk:
        raise CompanyRemoteAPIError(f"TOS secret key environment variable '{config.tos_secret_key_env}' is not set")

    key = _build_tos_object_key(config, role=role, extension=extension)
    try:
        client = tos.TosClientV2(ak, sk, config.tos_endpoint, config.tos_region)
        client.put_object(bucket=config.tos_bucket, key=key, content=content, content_type=mime)
        signed = client.pre_signed_url(
            _tos_http_get_method(tos),
            bucket=config.tos_bucket,
            key=key,
            expires=config.tos_url_expires_seconds,
        )
    except Exception as exc:
        raise CompanyRemoteAPIError(f"TOS upload or signing failed for object '{key}': {exc}") from exc
    return _normalize_presigned_url(signed), key


def _upload_image_to_tos(config: RemoteMediaConfig, *, content: bytes, mime: str, extension: str, role: str) -> str:
    url, _ = _upload_media_to_tos(config, content=content, mime=mime, extension=extension, role=role)
    return url


def _build_tos_object_key(config: RemoteMediaConfig, *, role: str, extension: str) -> str:
    safe_role = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(role or "image")).strip("_") or "image"
    date_part = time.strftime("%Y%m%d")
    unique = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}"
    return f"{config.tos_key_prefix}{date_part}/{safe_role}_{unique}{extension}"


def _normalize_presigned_url(value: Any) -> str:
    url = getattr(value, "signed_url", None) or getattr(value, "url", None) or value
    if not isinstance(url, str):
        raise CompanyRemoteAPIError(f"TOS pre_signed_url returned an unsupported value: {type(value).__name__}")
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url.lstrip("/")
    return url


def _tos_http_get_method(tos_module: Any) -> Any:
    method_type = getattr(tos_module, "HttpMethodType", None)
    if method_type is None:
        raise CompanyRemoteAPIError("TOS SDK does not expose HttpMethodType")
    for name in ("HTTP_METHOD_GET", "Http_Method_Get"):
        value = getattr(method_type, name, None)
        if value is not None:
            return value
    raise CompanyRemoteAPIError("TOS SDK does not expose a GET pre-signed URL method")


def _video_to_url(
    video: Any,
    config: RemoteMediaConfig,
    *,
    role: str,
    media_debug: list[dict[str, Any]] | None = None,
) -> str:
    content, mime, extension, source_info = _video_to_bytes(video)
    if config.tos_enabled or config.media_delivery == "tos_presigned":
        url, key = _upload_media_to_tos(config, content=content, mime=mime, extension=extension, role=role)
        _record_media_debug(
            media_debug,
            role=role,
            media_kind="video",
            delivery="tos_presigned",
            mime=mime,
            extension=extension,
            content=content,
            url=url,
            object_key=key,
            source=source_info,
        )
        return url
    data_uri = f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"
    _record_media_debug(
        media_debug,
        role=role,
        media_kind="video",
        delivery="base64",
        mime=mime,
        extension=extension,
        content=content,
        url=data_uri,
        object_key="",
        source=source_info,
    )
    return data_uri


def _video_to_bytes(video: Any) -> tuple[bytes, str, str, dict[str, Any]]:
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else video
    if isinstance(source, BytesIO):
        source.seek(0)
        content = source.read()
        source.seek(0)
        mime = "video/mp4"
        extension = ".mp4"
        source_info = {"kind": "BytesIO"}
    elif isinstance(source, str):
        with open(source, "rb") as f:
            content = f.read()
        mime = mimetypes.guess_type(source)[0] or "video/mp4"
        extension = _extension_from_content("video", source, mime)
        source_info = {"kind": "file", "basename": os.path.basename(source)}
    else:
        raise CompanyRemoteAPIError("reference_video must be a Comfy VIDEO input or a file-like object")
    return content, mime, extension, source_info


def _video_to_data_uri(video: Any) -> str:
    content, mime, _, _ = _video_to_bytes(video)
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def _record_media_debug(
    media_debug: list[dict[str, Any]] | None,
    *,
    role: str,
    media_kind: str,
    delivery: str,
    mime: str,
    extension: str,
    content: bytes,
    url: str,
    object_key: str,
    source: dict[str, Any],
) -> None:
    if media_debug is None:
        return
    item: dict[str, Any] = {
        "index": len(media_debug) + 1,
        "role": role,
        "media_kind": media_kind,
        "delivery": delivery,
        "mime": mime,
        "extension": extension,
        "bytes": len(content),
        "sha256_16": hashlib.sha256(content).hexdigest()[:16] if content else "",
        "object_key": object_key,
        "url": url if url.startswith(("http://", "https://")) else _redact_url_or_data_uri(url),
        "url_redacted": _redact_url_or_data_uri(url),
        "source": source,
    }
    expires_at = _presigned_url_expires_at(url)
    if expires_at:
        item["url_expires_at"] = expires_at
    if media_kind == "image" and content:
        dimensions = _image_dimensions_from_bytes(content)
        if dimensions:
            item["dimensions"] = dimensions
    media_debug.append(item)


def _write_request_debug(
    config: RemoteMediaConfig,
    *,
    operation: str,
    values: dict[str, Any],
    payload: dict[str, Any],
    media_debug: list[dict[str, Any]],
) -> None:
    debug_dir = os.path.join(folder_paths.get_user_directory(), "default", "company_remote", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in config.name).strip("_") or "default"
    payload_debug = {
        "created_at": created_at,
        "config_name": config.name,
        "operation": operation,
        "endpoint": {
            "base_url": config.base_url,
            "submit_path": config.submit_path,
            "method": config.method,
        },
        "delivery": {
            "media_delivery": config.media_delivery,
            "tos_enabled": config.tos_enabled,
            "tos_bucket": config.tos_bucket,
            "tos_endpoint": config.tos_endpoint,
            "tos_region": config.tos_region,
            "tos_key_prefix": config.tos_key_prefix,
            "tos_url_expires_seconds": config.tos_url_expires_seconds,
        },
        "values": _sanitize_for_debug(values),
        "payload": _sanitize_for_debug(
            payload,
            redact_urls=not (config.name == "seedance2" and config.tos_enabled),
        ),
        "media_count": len(media_debug),
    }
    media_payload = {
        "created_at": created_at,
        "config_name": config.name,
        "operation": operation,
        "media": media_debug,
    }
    _write_json_atomic(os.path.join(debug_dir, f"{safe_name}_last_payload.json"), payload_debug)
    _write_json_atomic(os.path.join(debug_dir, f"{safe_name}_last_media.json"), media_payload)


def _write_json_atomic(path: str, data: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _sanitize_for_debug(value: Any, *, redact_urls: bool = True) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"api_key", "authorization", "secret", "access_key", "secret_key"}:
                out[key] = "<redacted>"
            else:
                out[key] = _sanitize_for_debug(item, redact_urls=redact_urls)
        return out
    if isinstance(value, list):
        return [_sanitize_for_debug(item, redact_urls=redact_urls) for item in value]
    if isinstance(value, str):
        return _redact_url_or_data_uri(value) if redact_urls else value
    return value


def _redact_url_or_data_uri(value: str) -> str:
    if value.startswith("data:"):
        header = value.split(",", 1)[0]
        return f"{header},<base64 redacted>"
    if value.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(value)
        redacted = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        if parsed.query:
            return redacted + "?<query redacted>"
        return redacted
    return value


def _presigned_url_expires_at(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        return ""
    query = urllib.parse.parse_qs(urllib.parse.urlparse(value).query)
    query_lower = {key.lower(): values for key, values in query.items()}
    date_values = query_lower.get("x-tos-date")
    expires_values = query_lower.get("x-tos-expires")
    if not date_values or not expires_values:
        return ""
    try:
        issued_at = datetime.strptime(date_values[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        expires_at = issued_at + timedelta(seconds=int(expires_values[0]))
    except (ValueError, TypeError):
        return ""
    return expires_at.astimezone().isoformat(timespec="seconds")


def _image_dimensions_from_bytes(content: bytes) -> dict[str, int] | None:
    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
        return {"width": int(width), "height": int(height)}
    except Exception:
        return None


def _extract_first_string(data: Any, path: str) -> str:
    value = _extract_value(data, path)
    if isinstance(value, list):
        for item in value:
            if item not in (None, ""):
                return str(item)
        return ""
    if value is None:
        return ""
    return str(value)


def _extract_openai_text(data: Any, path: str) -> str:
    value = _extract_value(data, path)
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)


def _normalize_required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise CompanyRemoteAPIError(f"Field '{field_name}' must be a string.")
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("\ufeff"):
        text = text.removeprefix("\ufeff").lstrip()
    if not text:
        raise CompanyRemoteAPIError(f"Field '{field_name}' cannot be empty.")
    return text


def _normalize_prompt_text(value: Any) -> str:
    text = _normalize_required_text(value, field_name="prompt")
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1]).strip()
    if not text:
        raise CompanyRemoteAPIError("Field 'prompt' cannot be empty after removing the Markdown code fence.")
    return text


def _extract_value(data: Any, path: str) -> Any:
    if not path:
        return None
    value = data
    for part in path.split("."):
        if value is None:
            return None
        if isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _render_template(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            if value == "{" + key + "}":
                return "" if replacement is None else replacement
        out = value
        for key, replacement in variables.items():
            if isinstance(replacement, (dict, list)):
                continue
            out = out.replace("{" + key + "}", "" if replacement is None else str(replacement))
        return out
    if isinstance(value, list):
        return [_render_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _render_template(item, variables) for key, item in value.items()}
    return value


def _deep_copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value or {}))


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _extension_from_content(output_type: str, url: str, content_type: str) -> str:
    path_ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
    if output_type == "image":
        if path_ext in {".png", ".jpg", ".jpeg", ".webp"}:
            return path_ext
        if "jpeg" in content_type:
            return ".jpg"
        if "webp" in content_type:
            return ".webp"
        return ".png"
    if path_ext in {".mp4", ".webm", ".mov", ".mkv"}:
        return path_ext
    if "webm" in content_type:
        return ".webm"
    if "quicktime" in content_type:
        return ".mov"
    return ".mp4"


def _safe_response_text(response: requests.Response) -> str:
    text = response.text or ""
    return text[:2000]
