from __future__ import annotations

import asyncio

from aiohttp import web

from server import PromptServer

from .client import CompanyRemoteAPIError, get_cached_openai_model_ids, get_openai_model_ids, test_connection
from .config_store import ConfigError, RemoteMediaConfig, delete_config, get_config, load_configs, upsert_config
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


WEB_DIRECTORY = "./web"


def _json_error(status: int, code: str, message: str):
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


@PromptServer.instance.routes.get("/api/company_remote/configs")
@PromptServer.instance.routes.get("/company_remote/configs")
async def list_company_remote_configs(request):
    return web.json_response({"configs": load_configs(include_secret=False)})


@PromptServer.instance.routes.get("/api/company_remote/models")
@PromptServer.instance.routes.get("/company_remote/models")
async def list_company_remote_models(request):
    config_name = str(request.query.get("config") or "gpttext").strip()
    try:
        models = await asyncio.to_thread(get_openai_model_ids, get_config(config_name))
    except ConfigError:
        models = get_cached_openai_model_ids()
    return web.json_response(models)


@PromptServer.instance.routes.post("/api/company_remote/configs")
@PromptServer.instance.routes.post("/company_remote/configs")
async def create_company_remote_config(request):
    try:
        body = await request.json()
        return web.json_response({"config": upsert_config(body)}, status=201)
    except ConfigError as exc:
        return _json_error(400, "INVALID_CONFIG", str(exc))


@PromptServer.instance.routes.put("/api/company_remote/configs/{name}")
@PromptServer.instance.routes.put("/company_remote/configs/{name}")
async def update_company_remote_config(request):
    try:
        body = await request.json()
        name = request.match_info["name"]
        return web.json_response({"config": upsert_config(body, original_name=name)})
    except ConfigError as exc:
        return _json_error(400, "INVALID_CONFIG", str(exc))


@PromptServer.instance.routes.delete("/api/company_remote/configs/{name}")
@PromptServer.instance.routes.delete("/company_remote/configs/{name}")
async def delete_company_remote_config(request):
    try:
        delete_config(request.match_info["name"])
        return web.json_response({"ok": True})
    except ConfigError as exc:
        return _json_error(400, "INVALID_CONFIG", str(exc))


@PromptServer.instance.routes.post("/api/company_remote/test")
@PromptServer.instance.routes.post("/company_remote/test")
async def test_company_remote_config(request):
    try:
        body = await request.json()
        config = RemoteMediaConfig.from_dict(body)
        return web.json_response(test_connection(config))
    except ConfigError as exc:
        return _json_error(400, "INVALID_CONFIG", str(exc))
    except CompanyRemoteAPIError as exc:
        return _json_error(502, "REMOTE_API_ERROR", str(exc))
    except Exception as exc:
        return _json_error(502, "REMOTE_TEST_FAILED", str(exc))


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
