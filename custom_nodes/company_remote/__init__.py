from __future__ import annotations

import asyncio

from aiohttp import web

from server import PromptServer

from .client import CompanyRemoteAPIError, get_cached_openai_model_ids, get_openai_model_ids, test_connection
from .config_store import ConfigError, RemoteMediaConfig, delete_config, get_config, load_configs, upsert_config
from .asset_gateway import (
    CompanySeedanceABCAssetCreate,
    CompanySeedanceImageAssetCreate,
    CompanySeedanceVideoABCAssetCreate,
)
from .asset_gateway_video import CompanySeedanceAssetGatewayThreePersonVideo
from .face_swap_video import CompanyThreePersonFaceSwapVideo
from .three_person_seedance_video import CompanyThreePersonSeedanceVideo
from .three_person_wan27_video import (
    CompanyWan27MergeThreeSegments,
    CompanyWan27SplitThreeSegments,
    CompanyWan27ThreePersonFullVideo,
)
from .long_video import load_identity_mapping_record, save_identity_mapping_record
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS["CompanySeedanceImageAssetCreate"] = CompanySeedanceImageAssetCreate
NODE_DISPLAY_NAME_MAPPINGS["CompanySeedanceImageAssetCreate"] = "Seedance 真人图片注册为资产"
NODE_CLASS_MAPPINGS["CompanySeedanceABCAssetCreate"] = CompanySeedanceABCAssetCreate
NODE_DISPLAY_NAME_MAPPINGS["CompanySeedanceABCAssetCreate"] = "Seedance 人物 A/B/C 并行注册资产"
NODE_CLASS_MAPPINGS["CompanySeedanceVideoABCAssetCreate"] = CompanySeedanceVideoABCAssetCreate
NODE_DISPLAY_NAME_MAPPINGS["CompanySeedanceVideoABCAssetCreate"] = "Seedance 原视频 + 人物 A/B/C 注册资产"
NODE_CLASS_MAPPINGS["CompanySeedanceAssetGatewayThreePersonVideo"] = CompanySeedanceAssetGatewayThreePersonVideo
NODE_DISPLAY_NAME_MAPPINGS["CompanySeedanceAssetGatewayThreePersonVideo"] = "Seedance 资产 ID 三人物视频转换"
NODE_CLASS_MAPPINGS["CompanyThreePersonFaceSwapVideo"] = CompanyThreePersonFaceSwapVideo
NODE_DISPLAY_NAME_MAPPINGS["CompanyThreePersonFaceSwapVideo"] = "三人物视频流式换脸（CPU）"
NODE_CLASS_MAPPINGS["CompanyThreePersonSeedanceVideo"] = CompanyThreePersonSeedanceVideo
NODE_DISPLAY_NAME_MAPPINGS["CompanyThreePersonSeedanceVideo"] = "三人物整头造型 Seedance 分段转换"
NODE_CLASS_MAPPINGS["CompanyWan27ThreePersonFullVideo"] = CompanyWan27ThreePersonFullVideo
NODE_DISPLAY_NAME_MAPPINGS["CompanyWan27ThreePersonFullVideo"] = "Wan 2.7 三人物完整视频分段替换"
NODE_CLASS_MAPPINGS["CompanyWan27SplitThreeSegments"] = CompanyWan27SplitThreeSegments
NODE_DISPLAY_NAME_MAPPINGS["CompanyWan27SplitThreeSegments"] = "Wan 2.7 完整视频拆成三段"
NODE_CLASS_MAPPINGS["CompanyWan27MergeThreeSegments"] = CompanyWan27MergeThreeSegments
NODE_DISPLAY_NAME_MAPPINGS["CompanyWan27MergeThreeSegments"] = "Wan 2.7 三段精确合并并恢复原音频"


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


@PromptServer.instance.routes.get("/api/company_remote/identity_mappings/{series_id}")
@PromptServer.instance.routes.get("/company_remote/identity_mappings/{series_id}")
async def get_company_remote_identity_mapping(request):
    try:
        return web.json_response(load_identity_mapping_record(request.match_info["series_id"]))
    except ValueError as exc:
        return _json_error(400, "INVALID_IDENTITY_MAPPING", str(exc))


@PromptServer.instance.routes.put("/api/company_remote/identity_mappings/{series_id}")
@PromptServer.instance.routes.put("/company_remote/identity_mappings/{series_id}")
@PromptServer.instance.routes.post("/api/company_remote/identity_mappings/{series_id}")
@PromptServer.instance.routes.post("/company_remote/identity_mappings/{series_id}")
async def save_company_remote_identity_mapping(request):
    try:
        body = await request.json()
    except Exception:
        return _json_error(400, "INVALID_IDENTITY_MAPPING", "请求体必须是 JSON 对象。")
    try:
        return web.json_response(save_identity_mapping_record(request.match_info["series_id"], body))
    except ValueError as exc:
        return _json_error(400, "INVALID_IDENTITY_MAPPING", str(exc))


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
