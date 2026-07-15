from __future__ import annotations

import json
import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import folder_paths


PLUGIN_DIR_NAME = "company_remote"
CONFIGS_FILE_NAME = "configs.json"
DEFAULT_CONFIG_NAME = "default"


class ConfigError(ValueError):
    pass


@dataclass
class RemoteMediaConfig:
    name: str
    base_url: str = ""
    submit_path: str = "/generate"
    method: str = "POST"
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"
    api_key: str = ""
    api_key_env: str = ""
    timeout_seconds: int = 600
    poll_enabled: bool = True
    poll_path_template: str = "/tasks/{task_id}"
    poll_interval_seconds: float = 5.0
    max_poll_attempts: int = 120
    test_path: str = ""
    request_template: dict[str, Any] = field(default_factory=dict)
    image_field: str = "image"
    images_field: str = "images"
    first_frame_field: str = "first_frame"
    last_frame_field: str = "last_frame"
    reference_images_field: str = "reference_images"
    video_field: str = "video"
    reference_videos_field: str = "reference_videos"
    image_format: str = "png"
    media_delivery: str = "base64"
    tos_enabled: bool = False
    tos_bucket: str = ""
    tos_endpoint: str = ""
    tos_region: str = ""
    tos_key_prefix: str = "comfyui/seedance/"
    tos_access_key_env: str = "VOLC_ACCESS_KEY"
    tos_secret_key_env: str = "VOLC_SECRET_KEY"
    tos_url_expires_seconds: int = 7200
    response_image_url_path: str = "image_url"
    response_video_url_path: str = "video_url"
    response_result_url_path: str = "url"
    response_task_id_path: str = "task_id"
    response_status_path: str = "status"
    success_statuses: list[str] = field(default_factory=lambda: ["succeeded", "success", "completed", "done"])
    failure_statuses: list[str] = field(default_factory=lambda: ["failed", "error", "cancelled", "canceled"])
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteMediaConfig":
        if not isinstance(data, dict):
            raise ConfigError("config must be an object")
        merged = cls(name=str(data.get("name") or "")).to_dict(include_secret=True)
        merged.update(data)
        cfg = cls(**{key: merged[key] for key in cls.__dataclass_fields__.keys() if key in merged})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.name = _normalize_name(self.name)
        self.method = self.method.upper().strip() or "POST"
        if self.method not in {"GET", "POST"}:
            raise ConfigError("method must be GET or POST")
        if not self.base_url.strip():
            raise ConfigError("base_url is required")
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigError("base_url must start with http:// or https://")
        self.submit_path = _normalize_path(self.submit_path or "/generate")
        self.poll_path_template = _normalize_path(self.poll_path_template or "/tasks/{task_id}")
        if self.test_path:
            self.test_path = _normalize_path(self.test_path)
        self.timeout_seconds = _bounded_int(self.timeout_seconds, "timeout_seconds", 1, 7200)
        self.max_poll_attempts = _bounded_int(self.max_poll_attempts, "max_poll_attempts", 1, 10000)
        self.poll_interval_seconds = _bounded_float(self.poll_interval_seconds, "poll_interval_seconds", 0.1, 300.0)
        if not isinstance(self.request_template, dict):
            raise ConfigError("request_template must be an object")
        if not isinstance(self.extra_headers, dict):
            raise ConfigError("extra_headers must be an object")
        self.success_statuses = _normalize_string_list(self.success_statuses)
        self.failure_statuses = _normalize_string_list(self.failure_statuses)
        self.media_delivery = str(self.media_delivery or "base64").strip().lower()
        if self.media_delivery not in {"base64", "tos_presigned"}:
            raise ConfigError("media_delivery must be base64 or tos_presigned")
        self.tos_enabled = _to_bool(self.tos_enabled)
        if self.tos_enabled:
            self.media_delivery = "tos_presigned"
        elif self.media_delivery == "tos_presigned":
            self.tos_enabled = True
        self.tos_bucket = str(self.tos_bucket or "").strip()
        self.tos_endpoint = _normalize_host(self.tos_endpoint)
        self.tos_region = str(self.tos_region or "").strip()
        self.tos_key_prefix = _normalize_key_prefix(self.tos_key_prefix)
        self.tos_access_key_env = str(self.tos_access_key_env or "").strip()
        self.tos_secret_key_env = str(self.tos_secret_key_env or "").strip()
        self.tos_url_expires_seconds = _bounded_int(self.tos_url_expires_seconds, "tos_url_expires_seconds", 60, 604800)
        if self.tos_enabled:
            if not self.tos_bucket:
                raise ConfigError("tos_bucket is required when TOS delivery is enabled")
            if not self.tos_endpoint:
                raise ConfigError("tos_endpoint is required when TOS delivery is enabled")
            if not self.tos_region:
                raise ConfigError("tos_region is required when TOS delivery is enabled")
            if not self.tos_access_key_env or not self.tos_secret_key_env:
                raise ConfigError("TOS access key and secret key environment variable names are required")

    def to_dict(self, include_secret: bool = False) -> dict[str, Any]:
        out = {
            "name": self.name,
            "base_url": self.base_url,
            "submit_path": self.submit_path,
            "method": self.method,
            "auth_header": self.auth_header,
            "auth_prefix": self.auth_prefix,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "poll_enabled": self.poll_enabled,
            "poll_path_template": self.poll_path_template,
            "poll_interval_seconds": self.poll_interval_seconds,
            "max_poll_attempts": self.max_poll_attempts,
            "test_path": self.test_path,
            "request_template": self.request_template,
            "image_field": self.image_field,
            "images_field": self.images_field,
            "first_frame_field": self.first_frame_field,
            "last_frame_field": self.last_frame_field,
            "reference_images_field": self.reference_images_field,
            "video_field": self.video_field,
            "reference_videos_field": self.reference_videos_field,
            "image_format": self.image_format,
            "media_delivery": self.media_delivery,
            "tos_enabled": self.tos_enabled,
            "tos_bucket": self.tos_bucket,
            "tos_endpoint": self.tos_endpoint,
            "tos_region": self.tos_region,
            "tos_key_prefix": self.tos_key_prefix,
            "tos_access_key_env": self.tos_access_key_env,
            "tos_secret_key_env": self.tos_secret_key_env,
            "tos_url_expires_seconds": self.tos_url_expires_seconds,
            "response_image_url_path": self.response_image_url_path,
            "response_video_url_path": self.response_video_url_path,
            "response_result_url_path": self.response_result_url_path,
            "response_task_id_path": self.response_task_id_path,
            "response_status_path": self.response_status_path,
            "success_statuses": self.success_statuses,
            "failure_statuses": self.failure_statuses,
            "extra_headers": self.extra_headers,
        }
        if include_secret:
            out["api_key"] = self.api_key
        else:
            out["has_api_key"] = bool(self.api_key)
        return out

    def get_api_key(self) -> str:
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return self.api_key

    def get_tos_access_key(self) -> str:
        return os.environ.get(self.tos_access_key_env, "") if self.tos_access_key_env else ""

    def get_tos_secret_key(self) -> str:
        return os.environ.get(self.tos_secret_key_env, "") if self.tos_secret_key_env else ""


def get_config_dir() -> str:
    return os.path.join(folder_paths.get_user_directory(), "default", PLUGIN_DIR_NAME)


def get_config_path() -> str:
    return os.path.join(get_config_dir(), CONFIGS_FILE_NAME)


def load_configs(include_secret: bool = False) -> list[dict[str, Any]]:
    return [cfg.to_dict(include_secret=include_secret) for cfg in _load_config_objects()]


def get_config_names() -> list[str]:
    names = [cfg.name for cfg in _load_config_objects()]
    return names or [DEFAULT_CONFIG_NAME]


def get_config(name: str) -> RemoteMediaConfig:
    normalized = _normalize_name(name)
    for cfg in _load_config_objects():
        if cfg.name == normalized:
            return cfg
    raise ConfigError(f"company remote config not found: {normalized}")


def upsert_config(data: dict[str, Any], original_name: str | None = None) -> dict[str, Any]:
    configs = _load_config_objects()
    target = _normalize_name(original_name) if original_name else _normalize_name(str(data.get("name") or ""))
    existing = next((item for item in configs if item.name == target), None)
    if existing is not None and not data.get("api_key"):
        data = dict(data)
        data["api_key"] = existing.api_key

    cfg = RemoteMediaConfig.from_dict(data)
    replaced = False
    for index, item in enumerate(configs):
        if item.name == target:
            configs[index] = cfg
            replaced = True
            break
    if not replaced:
        configs.append(cfg)
    _save_config_objects(configs)
    return cfg.to_dict(include_secret=False)


def delete_config(name: str) -> None:
    normalized = _normalize_name(name)
    configs = [cfg for cfg in _load_config_objects() if cfg.name != normalized]
    _save_config_objects(configs)


def _load_config_objects() -> list[RemoteMediaConfig]:
    path = get_config_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_configs = raw.get("configs", []) if isinstance(raw, dict) else raw
    if not isinstance(raw_configs, list):
        raise ConfigError("configs file must contain a list")
    configs = []
    for item in raw_configs:
        try:
            configs.append(RemoteMediaConfig.from_dict(item))
        except ConfigError:
            continue
    return configs


def _save_config_objects(configs: list[RemoteMediaConfig]) -> None:
    os.makedirs(get_config_dir(), exist_ok=True)
    path = get_config_path()
    payload = {"configs": [cfg.to_dict(include_secret=True) for cfg in sorted(configs, key=lambda c: c.name)]}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def _normalize_name(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ConfigError("name is required")
    if any(ch in value for ch in "\\/\0"):
        raise ConfigError("name cannot contain path separators")
    return value[:80]


def _normalize_path(path: str) -> str:
    value = str(path or "").strip()
    if not value.startswith("/"):
        value = "/" + value
    return value


def _bounded_int(value: Any, field_name: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ConfigError(f"{field_name} must be between {minimum} and {maximum}")
    return result


def _bounded_float(value: Any, field_name: str, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number") from exc
    if result < minimum or result > maximum:
        raise ConfigError(f"{field_name} must be between {minimum} and {maximum}")
    return result


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_host(value: Any) -> str:
    host = str(value or "").strip()
    if host.startswith("[") and "](" in host and host.endswith(")"):
        host = host[host.index("](") + 2:-1].strip()
    if "://" in host:
        parsed = urllib.parse.urlparse(host)
        host = parsed.netloc or parsed.path
    host = host.strip().strip("/")
    return host.split("/", 1)[0]


def _normalize_key_prefix(value: Any) -> str:
    prefix = str(value or "").strip().replace("\\", "/")
    while "//" in prefix:
        prefix = prefix.replace("//", "/")
    prefix = prefix.strip("/")
    return f"{prefix}/" if prefix else ""


def _normalize_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",")]
    if not isinstance(values, list):
        return []
    return [str(value).strip().lower() for value in values if str(value).strip()]
