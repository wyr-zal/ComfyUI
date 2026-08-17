from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = ROOT / "user" / "default" / "workflows"
SOURCE = WORKFLOW_DIR / "视频欧美转绘_持久化提示词.json"
ASSET_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_长视频_参考素材准备.json"
RESTYLE_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_长视频_分段转绘.json"
VISUAL_RESTYLE_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_长视频_可视化分阶段转绘.json"
SHOT_AWARE_RESTYLE_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_长视频_镜头感知分阶段转绘.json"
SHOT_TEST_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_长视频_分镜检测测试.json"
CONTINUITY_TEST_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_长视频_连续分镜生成测试.json"
AUTO_ASSET_TEST_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_按镜头自动资产测试.json"
AUTO_ASSET_RESTYLE_OUTPUT = WORKFLOW_DIR / "视频欧美转绘_长视频_按镜头自动资产转绘.json"
ANIME_V2_SOURCE = WORKFLOW_DIR / "人物视频动漫化_长视频_按镜头自动资产转绘_Seedance版_素材库复用v2.json"
ANIME_V2_LIMITED_OUTPUT = WORKFLOW_DIR / "人物视频多风格转绘_长视频_按镜头自动资产转绘_Seedance版_素材库复用v2_限时生成.json"
ANIME_V3_LIMITED_OUTPUT = WORKFLOW_DIR / "人物视频多风格转绘_长视频_按镜头自动资产转绘_Seedance版_素材库复用v3_短镜头合并_音频可选_限时生成.json"
ANIME_V3_MANUAL_BATCH_OUTPUT = WORKFLOW_DIR / "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅.json"
ANIME_V3_MANUAL_BATCH_PIPELINE_OUTPUT = WORKFLOW_DIR / "人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_流水线.json"
ANIME_V2_PARALLEL_OUTPUT = WORKFLOW_DIR / "人物视频动漫化_长视频_按镜头自动资产转绘_Seedance版_素材库复用v2_并行生成.json"

PERSON_IDS = ("A", "B", "C")
BACKGROUND_IDS = tuple(f"BG{index:02d}" for index in range(1, 9))
ANIME_LONG_VIDEO_NEGATIVE_PROMPT = (
    "真人，真实人脸，真实皮肤，照片，摄影，写实，半真人，真人背景，皮肤毛孔，镜头噪点，"
    "写实3D，风格漂移，真人闪回，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印"
)
WESTERN_LONG_VIDEO_PROMPT = (
    "把本段重新演绎为完整、鲜明且统一的欧美化视频。参考图片是最终人物身份、服装、道具和环境美术的唯一视觉依据；"
    "人物必须整体重设计为符合当代欧美审美的外国人物，而不是只更换面孔；环境必须彻底重构为真实可信、地域统一的"
    "欧美国家场景，而不是轻微调色。严格保持实际人物数量、人物关系、主要剧情顺序、镜头构图和场景功能，"
    "从第一帧到最后一帧保持人物身份、整体造型、欧美环境、媒介和光影稳定一致。"
)
WESTERN_LONG_VIDEO_NEGATIVE_PROMPT = (
    "只换脸，亚洲面孔残留，本土东方造型，中式古装，中式建筑，中式家具，中文招牌，本土化道路设施，"
    "原人物残留，原服装残留，原背景残留，轻微调色，少量道具替换，地域混乱，媒介变化，风格漂移，"
    "人物复制，身份变化，多余人物，肢体畸形，额外手指，背景跳变，字幕，文字，Logo，水印"
)


def clone(nodes, node_id: int, new_id: int, *, title: str, pos: list[float]) -> dict:
    node = copy.deepcopy(nodes[node_id])
    node["id"] = new_id
    node["title"] = title
    node["pos"] = pos
    node["order"] = 0
    node["mode"] = 0
    for item in node.get("inputs", []):
        item["link"] = None
    for output in node.get("outputs", []):
        output["links"] = None
    return node


def set_widget(node: dict, values: list) -> None:
    node["widgets_values"] = values


def set_named_widget_values(node: dict, updates: dict[str, object]) -> None:
    names = [item.get("name") for item in node.get("inputs", []) if item.get("widget")]
    current = dict(zip(names, node.get("widgets_values", []), strict=False))
    current.update(updates)
    node["widgets_values"] = [current.get(name, "") for name in names]


def find_input(node: dict, name: str) -> tuple[int, dict]:
    for index, item in enumerate(node.get("inputs", [])):
        if item.get("name") == name:
            return index, item
    raise KeyError(f"{node['type']} input {name} not found")


def add_link(workflow: dict, source: dict, source_slot: int, target: dict, target_name: str, link_id: int, data_type: str) -> None:
    target_slot, target_input = find_input(target, target_name)
    target_input["link"] = link_id
    source_output = source["outputs"][source_slot]
    source_output.setdefault("links", [])
    if source_output["links"] is None:
        source_output["links"] = []
    source_output["links"].append(link_id)
    workflow["links"].append([link_id, source["id"], source_slot, target["id"], target_slot, data_type])


def make_manifest_node(node_id: int, pos: list[float]) -> dict:
    inputs = [
        {"name": "asset_name", "localized_name": "资产名称", "type": "STRING", "widget": {"name": "asset_name"}, "link": None},
        {"name": "mapping_json", "localized_name": "人物/背景映射 JSON", "type": "STRING", "widget": {"name": "mapping_json"}, "link": None},
    ]
    for person_id in PERSON_IDS:
        inputs.append({"name": f"person_{person_id}", "label": f"person_{person_id}", "localized_name": f"欧美化人物 {person_id}", "type": "IMAGE", "link": None})
    for background_id in BACKGROUND_IDS:
        inputs.append({"name": background_id, "label": background_id, "localized_name": f"欧美化背景 {background_id}", "type": "IMAGE", "link": None})
    return {
        "id": node_id,
        "type": "CompanyLongVideoAssetManifest",
        "pos": pos,
        "size": [460, 760],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": [
            {"localized_name": "资产清单 JSON", "name": "资产清单 JSON", "type": "STRING", "links": []},
            {"localized_name": "manifest 路径", "name": "manifest 路径", "type": "STRING", "links": []},
        ],
        "title": "保存已确认的欧美化资产清单",
        "properties": {"Node name for S&R": "CompanyLongVideoAssetManifest"},
        "widgets_values": [
            "long_video_assets",
            '{\n'
            '  "people": {"A": {"source": "原人物 A", "identity": ""}, '
            '"B": {"source": "原人物 B", "identity": ""}, '
            '"C": {"source": "原人物 C", "identity": ""}},\n'
            '  "backgrounds": {"BG01": {"source": "原背景 BG01", "description": ""}},\n'
            '  "mapping": {\n'
            '    "people": {"A": "原人物 A -> 欧美化人物 A"},\n'
            '    "backgrounds": {"BG01": "原背景 BG01 -> 欧美化背景 BG01"}\n'
            '  }\n'
            '}',
        ],
        "color": "#234",
        "bgcolor": "#345",
    }


def make_restyle_node(node_id: int, pos: list[float]) -> dict:
    inputs = [
        {"name": "video", "localized_name": "长视频", "type": "VIDEO", "link": None},
        {"name": "assets_manifest", "localized_name": "资产清单 JSON 或路径", "type": "STRING", "widget": {"name": "assets_manifest"}, "link": None},
        {"name": "prompt", "localized_name": "视频提示词", "type": "STRING", "widget": {"name": "prompt"}, "link": None},
        {"name": "engine", "localized_name": "视频引擎", "type": "COMBO", "widget": {"name": "engine"}, "link": None},
        {"name": "model", "localized_name": "模型", "type": "COMBO", "widget": {"name": "model"}, "link": None},
        {"name": "segment_duration", "localized_name": "目标分段时长（秒）", "type": "COMBO", "widget": {"name": "segment_duration"}, "link": None},
        {"name": "analysis_model", "localized_name": "分段分析模型", "type": "COMBO", "widget": {"name": "analysis_model"}, "link": None},
        {"name": "max_retries", "localized_name": "每段最大重试次数", "type": "INT", "widget": {"name": "max_retries"}, "link": None},
        {"name": "resume", "localized_name": "复用已完成分段", "type": "BOOLEAN", "widget": {"name": "resume"}, "link": None},
        {"name": "force_rerun", "localized_name": "强制重跑全部分段", "type": "BOOLEAN", "widget": {"name": "force_rerun"}, "link": None},
        {"name": "negative_prompt", "localized_name": "负面提示词", "type": "STRING", "widget": {"name": "negative_prompt"}, "link": None},
    ]
    for person_id in PERSON_IDS:
        inputs.append({"name": f"person_{person_id}", "label": f"person_{person_id}", "localized_name": f"欧美化人物 {person_id}", "type": "IMAGE", "link": None})
    for background_id in BACKGROUND_IDS:
        inputs.append({"name": background_id, "label": background_id, "localized_name": f"欧美化背景 {background_id}", "type": "IMAGE", "link": None})
    return {
        "id": node_id,
        "type": "CompanyLongVideoRestyle",
        "pos": pos,
        "size": [620, 930],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": [
            {"localized_name": "最终视频", "name": "最终视频", "type": "VIDEO", "links": []},
            {"localized_name": "最终视频路径", "name": "最终视频路径", "type": "STRING", "links": []},
            {"localized_name": "任务 manifest 路径", "name": "任务 manifest 路径", "type": "STRING", "links": []},
            {"localized_name": "任务状态 JSON", "name": "任务状态 JSON", "type": "STRING", "links": []},
        ],
        "title": "长视频分段欧美化转绘（顺序执行，可断点恢复）",
        "properties": {"Node name for S&R": "CompanyLongVideoRestyle"},
        "widgets_values": [
            "",
            "",
            "参考视频只规定原剧情、动作、镜头和时间过程。参考人物图片和参考背景图片是最终视觉身份标准。保持实际人物数量、人物对应关系、场景连续性和真实欧美影视审美，不新增、删除、复制或融合人物。",
            "Seedance 2.0",
            "Seedance 2.0 Fast",
            10,
            "gpt-5.4",
            2,
            True,
            False,
            "",
        ],
        "color": "#432",
        "bgcolor": "#653",
    }


def make_stage_node(
    node_id: int,
    node_type: str,
    title: str,
    pos: list[float],
    size: list[float],
    inputs: list[dict],
    outputs: list[dict],
    widgets_values: list | None = None,
) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "title": title,
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets_values or [],
        "color": "#234",
        "bgcolor": "#345",
    }


def stage_input(name: str, data_type: str, localized_name: str, *, widget: bool = False) -> dict:
    value = {"name": name, "localized_name": localized_name, "type": data_type, "link": None}
    if widget:
        value["widget"] = {"name": name}
    return value


def stage_output(name: str, data_type: str) -> dict:
    return {"name": name, "localized_name": name, "type": data_type, "links": []}


def make_mapping_analyzer(node_id: int, pos: list[float]) -> dict:
    return {
        "id": node_id,
        "type": "CompanyLongVideoMappingAnalyzer",
        "pos": pos,
        "size": [560, 900],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "person_A", "localized_name": "欧美化人物 A", "type": "IMAGE", "link": None},
            {"name": "person_B", "localized_name": "欧美化人物 B", "type": "IMAGE", "link": None},
            {"name": "person_C", "localized_name": "欧美化人物 C", "type": "IMAGE", "link": None},
            *[
                {"name": background_id, "localized_name": f"欧美化背景 {background_id}", "type": "IMAGE", "link": None}
                for background_id in BACKGROUND_IDS
            ],
            {"name": "analysis_model", "localized_name": "分析模型", "type": "COMBO", "widget": {"name": "analysis_model"}, "link": None},
        ],
        "outputs": [{"localized_name": "人物与背景映射 JSON", "name": "人物与背景映射 JSON", "type": "STRING", "links": []}],
        "title": "AI 初步识别人物与背景映射（人工确认后填写资产清单）",
        "properties": {"Node name for S&R": "CompanyLongVideoMappingAnalyzer"},
        "widgets_values": ["gpt-5.4", "刷新模型列表"],
        "color": "#223",
        "bgcolor": "#335",
    }


def build_asset_workflow(source: dict) -> dict:
    templates = {node["id"]: node for node in source["nodes"]}
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, ASSET_OUTPUT.name)),
        "revision": 0,
        "last_node_id": 0,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "workflow_note": "新建副本：长视频欧美化参考素材准备。原工作流保持不变。先确认人物/背景图片，再把 manifest 路径复制到长视频分段转绘工作流。",
            "long_video": {"asset_stage": True, "max_people": 3, "max_backgrounds": 8},
        },
        "version": 0.4,
    }
    nodes = {}
    next_id = 1
    next_link = 1
    x_people = 40
    for index, person_id in enumerate(PERSON_IDS):
        load = clone(templates, 65, next_id, title=f"输入原人物 {person_id}", pos=[x_people, index * 900])
        if index > 0:
            load["mode"] = 4
        set_widget(load, ["", "image"])
        workflow["nodes"].append(load)
        nodes[next_id] = load
        load_id = next_id
        next_id += 1
        enhancer = clone(templates, 220, next_id, title=f"人物 {person_id}：图片提示词优化", pos=[420, index * 900])
        if index > 0:
            enhancer["mode"] = 4
        set_widget(enhancer, [
            "你是 GPT Image 2 多模态图像编辑提示词专家。根据输入图像输出可直接用于图生图的中文提示词，不要解释。",
            "只处理当前人物，将其转为符合当代欧美审美的外国人物；保持视觉媒介、性别、年龄段和核心身份特征，允许大幅重新设计发型、妆容、服装、动作和气质；输出完整单人图，置于均匀高对比色键背景。",
            "gpt-5.4", 0.2, 1400, "刷新模型列表",
        ])
        workflow["nodes"].append(enhancer)
        nodes[next_id] = enhancer
        enhancer_id = next_id
        next_id += 1
        image = clone(templates, 221, next_id, title=f"人物 {person_id}：GPT Image 2 欧美化", pos=[900, index * 900])
        if index > 0:
            image["mode"] = 4
        set_widget(image, ["", "gpt-image-2", "auto", 1024, 1024, "opaque", "high", 1, 100000 + index, "fixed"])
        workflow["nodes"].append(image)
        nodes[next_id] = image
        image_id = next_id
        next_id += 1
        preview = clone(templates, 222, next_id, title=f"预览：欧美化人物 {person_id}", pos=[1450, index * 900])
        if index > 0:
            preview["mode"] = 4
        preview["inputs"] = [{"name": "images", "localized_name": "图像", "type": "IMAGE", "link": None}]
        workflow["nodes"].append(preview)
        nodes[next_id] = preview
        preview_id = next_id
        next_id += 1
        add_link(workflow, nodes[load_id], 0, nodes[enhancer_id], "image", next_link, "IMAGE"); next_link += 1
        add_link(workflow, nodes[enhancer_id], 0, nodes[image_id], "prompt", next_link, "STRING"); next_link += 1
        add_link(workflow, nodes[load_id], 0, nodes[image_id], "model.images.image_1", next_link, "IMAGE"); next_link += 1
        add_link(workflow, nodes[image_id], 0, nodes[preview_id], "images", next_link, "IMAGE"); next_link += 1
    background_start = 2800
    for index, background_id in enumerate(BACKGROUND_IDS):
        load = clone(templates, 65, next_id, title=f"输入原背景 {background_id}", pos=[x_people, background_start + index * 650])
        if index > 0:
            load["mode"] = 4
        set_widget(load, ["", "image"])
        workflow["nodes"].append(load); nodes[next_id] = load; load_id = next_id; next_id += 1
        enhancer = clone(templates, 223, next_id, title=f"背景 {background_id}：图片提示词优化", pos=[420, background_start + index * 650])
        if index > 0:
            enhancer["mode"] = 4
        set_widget(enhancer, [
            "你是 GPT Image 2 多模态图像编辑提示词专家。根据输入图像输出可直接用于图生图的中文提示词，不要解释。",
            "移除全部人物、人体局部、人物倒影和人物阴影，保持原镜头的空间结构和透视，补全遮挡区域并转换为符合欧美电影审美的真实场景。",
            "gpt-5.4", 0.2, 1400, "刷新模型列表",
        ])
        workflow["nodes"].append(enhancer); nodes[next_id] = enhancer; enhancer_id = next_id; next_id += 1
        image = clone(templates, 224, next_id, title=f"背景 {background_id}：GPT Image 2 欧美化", pos=[900, background_start + index * 650])
        if index > 0:
            image["mode"] = 4
        set_widget(image, ["", "gpt-image-2", "auto", 1024, 1024, "opaque", "high", 1, 200000 + index, "fixed"])
        workflow["nodes"].append(image); nodes[next_id] = image; image_id = next_id; next_id += 1
        preview = clone(templates, 225, next_id, title=f"预览：欧美化背景 {background_id}", pos=[1450, background_start + index * 650])
        if index > 0:
            preview["mode"] = 4
        preview["inputs"] = [{"name": "images", "localized_name": "图像", "type": "IMAGE", "link": None}]
        workflow["nodes"].append(preview); nodes[next_id] = preview; preview_id = next_id; next_id += 1
        add_link(workflow, nodes[load_id], 0, nodes[enhancer_id], "image", next_link, "IMAGE"); next_link += 1
        add_link(workflow, nodes[enhancer_id], 0, nodes[image_id], "prompt", next_link, "STRING"); next_link += 1
        add_link(workflow, nodes[load_id], 0, nodes[image_id], "model.images.image_1", next_link, "IMAGE"); next_link += 1
        add_link(workflow, nodes[image_id], 0, nodes[preview_id], "images", next_link, "IMAGE"); next_link += 1
    manifest = make_manifest_node(next_id, [2050, 500])
    workflow["nodes"].append(manifest); nodes[next_id] = manifest; manifest_id = next_id; next_id += 1
    for index, person_id in enumerate(PERSON_IDS):
        image_node = next(node for node in workflow["nodes"] if node["title"] == f"人物 {person_id}：GPT Image 2 欧美化")
        add_link(workflow, image_node, 0, manifest, f"person_{person_id}", next_link, "IMAGE"); next_link += 1
    for background_id in BACKGROUND_IDS:
        image_node = next(node for node in workflow["nodes"] if node["title"] == f"背景 {background_id}：GPT Image 2 欧美化")
        add_link(workflow, image_node, 0, manifest, background_id, next_link, "IMAGE"); next_link += 1
    mapping_analyzer = make_mapping_analyzer(next_id, [2050, 2300])
    workflow["nodes"].append(mapping_analyzer); nodes[next_id] = mapping_analyzer; next_id += 1
    mapping_sources = [
        next(node for node in workflow["nodes"] if node["title"] == f"人物 {person_id}：GPT Image 2 欧美化")
        for person_id in PERSON_IDS
    ]
    for source_node in mapping_sources:
        target_name = f"person_{PERSON_IDS[mapping_sources.index(source_node)]}"
        add_link(workflow, source_node, 0, mapping_analyzer, target_name, next_link, "IMAGE"); next_link += 1
    for background_id in BACKGROUND_IDS:
        source_node = next(node for node in workflow["nodes"] if node["title"] == f"背景 {background_id}：GPT Image 2 欧美化")
        add_link(workflow, source_node, 0, mapping_analyzer, background_id, next_link, "IMAGE"); next_link += 1
    workflow["last_node_id"] = next_id - 1
    workflow["last_link_id"] = next_link - 1
    workflow["groups"] = [{"id": 1, "title": "人物欧美化参考图", "bounding": [-30, -50, 1800, 2700], "color": "#6b5b95", "flags": {}}, {"id": 2, "title": "背景欧美化参考图", "bounding": [-30, 2700, 1800, 5600], "color": "#3f789e", "flags": {}}, {"id": 3, "title": "确认并保存资产清单", "bounding": [1980, 350, 650, 1600], "color": "#4f7f69", "flags": {}}]
    return workflow


def build_restyle_workflow(source: dict) -> dict:
    templates = {node["id"]: node for node in source["nodes"]}
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, RESTYLE_OUTPUT.name)),
        "revision": 0,
        "last_node_id": 0,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {"workflow_note": "新建副本：读取已确认的欧美化资产 manifest，长视频按 10/15 秒分段转绘并恢复原音频。原工作流保持不变。", "long_video": {"restyle_stage": True}},
        "version": 0.4,
    }
    load_video_template_id = next(node["id"] for node in source["nodes"] if node.get("type") == "LoadVideo")
    video = clone(templates, load_video_template_id, 1, title="输入长视频", pos=[0, 0])
    set_widget(video, ["", "image"])
    workflow["nodes"].append(video)
    restyle = make_restyle_node(2, [500, 0])
    workflow["nodes"].append(restyle)
    save = clone(templates, 255, 3, title="保存最终长视频（含原音频）", pos=[1300, 180])
    save["inputs"] = [{"name": "video", "localized_name": "视频", "type": "VIDEO", "link": None}, {"name": "filename_prefix", "localized_name": "文件名前缀", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None}, {"name": "format", "localized_name": "格式", "type": "COMBO", "widget": {"name": "format"}, "link": None}, {"name": "codec", "localized_name": "编码器", "type": "COMBO", "widget": {"name": "codec"}, "link": None}]
    set_widget(save, ["video/LongVideoRestyle", "auto", "auto"])
    workflow["nodes"].append(save)
    next_link = 1
    add_link(workflow, video, 0, restyle, "video", next_link, "VIDEO"); next_link += 1
    add_link(workflow, restyle, 0, save, "video", next_link, "VIDEO"); next_link += 1
    workflow["last_node_id"] = 3
    workflow["last_link_id"] = next_link - 1
    workflow["groups"] = [{"id": 1, "title": "长视频输入与分段处理", "bounding": [-40, -80, 1200, 1180], "color": "#3f789e", "flags": {}}, {"id": 2, "title": "最终输出", "bounding": [1240, -40, 450, 620], "color": "#4f7f69", "flags": {}}]
    return workflow


def build_visual_restyle_workflow(source: dict) -> dict:
    templates = {node["id"]: node for node in source["nodes"]}
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, VISUAL_RESTYLE_OUTPUT.name)),
        "revision": 0,
        "last_node_id": 10,
        "last_link_id": 10,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "workflow_note": "新建副本：长视频欧美化可视化分阶段转绘。原有工作流保持不变。每个阶段独立展示状态并共享断点 manifest。",
            "long_video": {"restyle_stage": True, "visible_stages": True},
        },
        "version": 0.4,
    }

    load_video_template_id = next(node["id"] for node in source["nodes"] if node.get("type") == "LoadVideo")
    video = clone(templates, load_video_template_id, 1, title="输入长视频", pos=[0, 420])
    set_widget(video, ["", "image"])
    workflow["nodes"].append(video)

    asset_inputs = [stage_input("assets_manifest", "STRING", "资产清单 JSON 或路径", widget=True)]
    asset_inputs.extend(stage_input(f"person_{person_id}", "IMAGE", f"欧美化人物 {person_id}") for person_id in PERSON_IDS)
    asset_inputs.extend(stage_input(background_id, "IMAGE", f"欧美化背景 {background_id}") for background_id in BACKGROUND_IDS)
    asset_loader = make_stage_node(
        2,
        "CompanyLongVideoAssetLoader",
        "1. 读取并校验资产清单",
        [0, -320],
        [480, 650],
        asset_inputs,
        [stage_output("已加载资产", "长视频资产清单"), stage_output("资产摘要 JSON", "STRING")],
        [""],
    )
    workflow["nodes"].append(asset_loader)

    planner = make_stage_node(
        3,
        "CompanyLongVideoSegmentPlanner",
        "2. 视频切分与任务规划",
        [620, 0],
        [580, 720],
        [
            stage_input("video", "VIDEO", "长视频"),
            stage_input("assets", "长视频资产清单", "已加载资产"),
            stage_input("prompt", "STRING", "视频提示词", widget=True),
            stage_input("engine", "COMBO", "视频引擎", widget=True),
            stage_input("model", "COMBO", "模型", widget=True),
            stage_input("segment_duration", "COMBO", "目标分段时长（秒）", widget=True),
            stage_input("analysis_model", "COMBO", "分段分析模型", widget=True),
            stage_input("max_retries", "INT", "每段最大重试次数", widget=True),
            stage_input("resume", "BOOLEAN", "复用已完成分段", widget=True),
            stage_input("force_rerun", "BOOLEAN", "强制重跑全部分段", widget=True),
            stage_input("negative_prompt", "STRING", "负面提示词", widget=True),
        ],
        [
            stage_output("长视频任务", "长视频分段任务"),
            stage_output("切分计划 JSON", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
        ],
        [
            "参考视频只规定原剧情、动作、镜头和时间过程。参考人物图片和参考背景图片是最终视觉身份标准。保持实际人物数量、人物对应关系、场景连续性和真实欧美影视审美，不新增、删除、复制或融合人物。",
            "Seedance 2.0",
            "Seedance 2.0 Fast",
            10,
            "gpt-5.4",
            2,
            True,
            False,
            "",
        ],
    )
    workflow["nodes"].append(planner)

    analyzer = make_stage_node(
        4,
        "CompanyLongVideoSegmentAnalyzer",
        "3. GPT 分析各段人物与背景",
        [1350, 0],
        [430, 280],
        [stage_input("job", "长视频分段任务", "长视频任务")],
        [stage_output("已分析任务", "长视频分段任务"), stage_output("分析结果 JSON", "STRING")],
    )
    matcher = make_stage_node(
        5,
        "CompanyLongVideoReferenceMatcher",
        "4. 为每段匹配参考素材",
        [1950, 0],
        [430, 280],
        [stage_input("job", "长视频分段任务", "已分析任务")],
        [stage_output("已匹配任务", "长视频分段任务"), stage_output("匹配结果 JSON", "STRING")],
    )
    generator = make_stage_node(
        6,
        "CompanyLongVideoSegmentGenerator",
        "5. 顺序生成全部视频分段",
        [2550, 0],
        [460, 300],
        [stage_input("job", "长视频分段任务", "已匹配任务")],
        [stage_output("已生成任务", "长视频分段任务"), stage_output("生成状态 JSON", "STRING")],
    )
    collector = make_stage_node(
        7,
        "CompanyLongVideoResultCollector",
        "6. 核对分段结果并预览末帧",
        [3180, 0],
        [470, 320],
        [stage_input("job", "长视频分段任务", "已生成任务")],
        [
            stage_output("可合并任务", "长视频分段任务"),
            stage_output("分段结果 JSON", "STRING"),
            stage_output("各段末帧预览", "IMAGE"),
        ],
    )
    merger = make_stage_node(
        8,
        "CompanyLongVideoFinalMerger",
        "7. 合并分段并恢复原音频",
        [3810, 0],
        [470, 340],
        [stage_input("job", "长视频分段任务", "可合并任务")],
        [
            stage_output("最终视频", "VIDEO"),
            stage_output("最终视频路径", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
            stage_output("任务状态 JSON", "STRING"),
        ],
    )
    workflow["nodes"].extend([analyzer, matcher, generator, collector, merger])

    preview = clone(templates, 222, 9, title="预览：每个分段的末帧", pos=[3180, 500])
    preview["inputs"] = [{"name": "images", "localized_name": "图像", "type": "IMAGE", "link": None}]
    workflow["nodes"].append(preview)
    save = clone(templates, 255, 10, title="保存最终长视频（含原音频）", pos=[4480, 0])
    save["inputs"] = [
        stage_input("video", "VIDEO", "视频"),
        stage_input("filename_prefix", "STRING", "文件名前缀", widget=True),
        stage_input("format", "COMBO", "格式", widget=True),
        stage_input("codec", "COMBO", "编码器", widget=True),
    ]
    set_widget(save, ["video/LongVideoRestyle", "auto", "auto"])
    workflow["nodes"].append(save)

    next_link = 1
    add_link(workflow, video, 0, planner, "video", next_link, "VIDEO"); next_link += 1
    add_link(workflow, asset_loader, 0, planner, "assets", next_link, "长视频资产清单"); next_link += 1
    add_link(workflow, planner, 0, analyzer, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, analyzer, 0, matcher, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, matcher, 0, generator, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, generator, 0, collector, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, collector, 0, merger, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, collector, 2, preview, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, merger, 0, save, "video", next_link, "VIDEO"); next_link += 1
    workflow["last_link_id"] = next_link - 1
    workflow["groups"] = [
        {"id": 1, "title": "输入与资源", "bounding": [-60, -380, 560, 1180], "color": "#6b5b95", "flags": {}},
        {"id": 2, "title": "可视化分段处理", "bounding": [560, -80, 3760, 900], "color": "#3f789e", "flags": {}},
        {"id": 3, "title": "最终输出", "bounding": [4420, -80, 420, 620], "color": "#4f7f69", "flags": {}},
    ]
    return workflow


def build_shot_aware_restyle_workflow(source: dict) -> dict:
    templates = {node["id"]: node for node in source["nodes"]}
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, SHOT_AWARE_RESTYLE_OUTPUT.name)),
        "revision": 0,
        "last_node_id": 12,
        "last_link_id": 11,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "workflow_note": "新建副本：镜头感知长视频欧美化分阶段转绘。先按原剪辑检测硬切和淡入淡出，再按模型时长限制适配；检测失败自动回退固定切分。原有工作流保持不变。",
            "long_video": {"restyle_stage": True, "visible_stages": True, "shot_aware": True, "manifest_version": 3},
        },
        "version": 0.4,
    }

    load_video_template_id = next(node["id"] for node in source["nodes"] if node.get("type") == "LoadVideo")
    video = clone(templates, load_video_template_id, 1, title="输入长视频", pos=[0, 430])
    set_widget(video, ["", "image"])
    workflow["nodes"].append(video)

    asset_inputs = [stage_input("assets_manifest", "STRING", "资产清单 JSON 或路径", widget=True)]
    asset_inputs.extend(stage_input(f"person_{person_id}", "IMAGE", f"欧美化人物 {person_id}") for person_id in PERSON_IDS)
    asset_inputs.extend(stage_input(background_id, "IMAGE", f"欧美化背景 {background_id}") for background_id in BACKGROUND_IDS)
    asset_loader = make_stage_node(
        2,
        "CompanyLongVideoAssetLoader",
        "1. 读取并校验人物/背景资产",
        [0, -340],
        [500, 650],
        asset_inputs,
        [stage_output("已加载资产", "长视频资产清单"), stage_output("资产摘要 JSON", "STRING")],
        [""],
    )

    detector = make_stage_node(
        3,
        "CompanyLongVideoShotDetector",
        "2. 检测原视频镜头和转场",
        [650, 360],
        [520, 420],
        [
            stage_input("video", "VIDEO", "长视频"),
            stage_input("mode", "COMBO", "切分模式", widget=True),
            stage_input("fixed_duration", "COMBO", "固定分段时长（秒）", widget=True),
            stage_input("sensitivity", "COMBO", "镜头检测灵敏度", widget=True),
            stage_input("use_audio_silence", "BOOLEAN", "用音频停顿辅助长镜头切分", widget=True),
            stage_input("auto_fallback", "BOOLEAN", "检测失败自动改用固定切分", widget=True),
        ],
        [
            stage_output("镜头计划", "长视频镜头计划"),
            stage_output("镜头检测 JSON", "STRING"),
            stage_output("镜头起始帧预览", "IMAGE"),
        ],
        ["镜头优先（推荐）", "10", "标准", True, True],
    )

    cut_preview = clone(templates, 222, 4, title="预览：各逻辑镜头起始画面", pos=[1260, 650])
    cut_preview["inputs"] = [{"name": "images", "localized_name": "图像", "type": "IMAGE", "link": None}]

    adapter = make_stage_node(
        5,
        "CompanyLongVideoDurationAdapter",
        "3. 按模型时长适配并建立任务",
        [1260, -40],
        [610, 620],
        [
            stage_input("shot_plan", "长视频镜头计划", "镜头计划"),
            stage_input("assets", "长视频资产清单", "已加载资产"),
            stage_input("prompt", "STRING", "视频提示词", widget=True),
            stage_input("engine", "COMBO", "视频引擎", widget=True),
            stage_input("model", "COMBO", "模型", widget=True),
            stage_input("analysis_model", "COMBO", "分段分析模型", widget=True),
            stage_input("max_retries", "INT", "每段最大重试次数", widget=True),
            stage_input("resume", "BOOLEAN", "复用已完成分段", widget=True),
            stage_input("force_rerun", "BOOLEAN", "强制重跑全部分段", widget=True),
            stage_input("negative_prompt", "STRING", "负面提示词", widget=True),
        ],
        [
            stage_output("长视频任务", "长视频分段任务"),
            stage_output("时长适配计划 JSON", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
        ],
        [
            "参考视频规定原剧情、动作、镜头和时间过程。参考人物与参考背景是最终视觉身份标准。保持实际人物数量、对应关系、场景连续性和真实欧美影视审美，不新增、删除、复制或融合人物。",
            "Seedance 2.0",
            "Seedance 2.0 Fast",
            "gpt-5.4",
            2,
            True,
            False,
            "",
        ],
    )

    analyzer = make_stage_node(
        6, "CompanyLongVideoSegmentAnalyzer", "4. GPT 分析各请求片段人物与背景", [2040, 0], [440, 280],
        [stage_input("job", "长视频分段任务", "长视频任务")],
        [stage_output("已分析任务", "长视频分段任务"), stage_output("分析结果 JSON", "STRING")],
    )
    matcher = make_stage_node(
        7, "CompanyLongVideoReferenceMatcher", "5. 为每段匹配人物与背景参考", [2640, 0], [440, 280],
        [stage_input("job", "长视频分段任务", "已分析任务")],
        [stage_output("已匹配任务", "长视频分段任务"), stage_output("匹配结果 JSON", "STRING")],
    )
    generator = make_stage_node(
        8, "CompanyLongVideoSegmentGenerator", "6. 顺序生成全部视频分段", [3240, 0], [460, 300],
        [stage_input("job", "长视频分段任务", "已匹配任务")],
        [stage_output("已生成任务", "长视频分段任务"), stage_output("生成状态 JSON", "STRING")],
    )
    collector = make_stage_node(
        9, "CompanyLongVideoResultCollector", "7. 核对分段结果并预览末帧", [3860, 0], [470, 320],
        [stage_input("job", "长视频分段任务", "已生成任务")],
        [
            stage_output("可合并任务", "长视频分段任务"),
            stage_output("分段结果 JSON", "STRING"),
            stage_output("各段末帧预览", "IMAGE"),
        ],
    )
    result_preview = clone(templates, 222, 10, title="预览：每个生成分段的末帧", pos=[3860, 500])
    result_preview["inputs"] = [{"name": "images", "localized_name": "图像", "type": "IMAGE", "link": None}]
    merger = make_stage_node(
        11, "CompanyLongVideoFinalMerger", "8. 裁回短镜头、合并并恢复原音频", [4490, 0], [490, 350],
        [stage_input("job", "长视频分段任务", "可合并任务")],
        [
            stage_output("最终视频", "VIDEO"),
            stage_output("最终视频路径", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
            stage_output("任务状态 JSON", "STRING"),
        ],
    )
    save = clone(templates, 255, 12, title="保存最终长视频（含原音频）", pos=[5150, 0])
    save["inputs"] = [
        stage_input("video", "VIDEO", "视频"),
        stage_input("filename_prefix", "STRING", "文件名前缀", widget=True),
        stage_input("format", "COMBO", "格式", widget=True),
        stage_input("codec", "COMBO", "编码器", widget=True),
    ]
    set_widget(save, ["video/LongVideoShotAwareRestyle", "auto", "auto"])
    workflow["nodes"].extend([asset_loader, detector, cut_preview, adapter, analyzer, matcher, generator, collector, result_preview, merger, save])

    next_link = 1
    add_link(workflow, video, 0, detector, "video", next_link, "VIDEO"); next_link += 1
    add_link(workflow, detector, 0, adapter, "shot_plan", next_link, "长视频镜头计划"); next_link += 1
    add_link(workflow, detector, 2, cut_preview, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, asset_loader, 0, adapter, "assets", next_link, "长视频资产清单"); next_link += 1
    add_link(workflow, adapter, 0, analyzer, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, analyzer, 0, matcher, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, matcher, 0, generator, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, generator, 0, collector, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, collector, 0, merger, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, collector, 2, result_preview, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, merger, 0, save, "video", next_link, "VIDEO"); next_link += 1
    workflow["last_link_id"] = next_link - 1
    workflow["groups"] = [
        {"id": 1, "title": "输入与已确认参考资产", "bounding": [-60, -400, 580, 1250], "color": "#6b5b95", "flags": {}},
        {"id": 2, "title": "镜头检测与时长适配", "bounding": [580, -100, 1360, 1000], "color": "#81683e", "flags": {}},
        {"id": 3, "title": "逐段分析、匹配与生成", "bounding": [1980, -80, 2440, 900], "color": "#3f789e", "flags": {}},
        {"id": 4, "title": "合并与最终输出", "bounding": [4430, -80, 1120, 620], "color": "#4f7f69", "flags": {}},
    ]
    return workflow


def build_shot_test_workflow(source: dict) -> dict:
    templates = {node["id"]: node for node in source["nodes"]}
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, SHOT_TEST_OUTPUT.name)),
        "revision": 0,
        "last_node_id": 8,
        "last_link_id": 7,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "workflow_note": "分镜检测本地测试工作流：只检测和预览，不连接 GPT、Seedance 或 Wan。先观察镜头起始帧与切点前后帧，再通过镜头序号逐个播放检查。",
            "long_video": {"shot_detection_test": True, "remote_requests": False},
        },
        "version": 0.4,
    }

    load_video_template_id = next(node["id"] for node in source["nodes"] if node.get("type") == "LoadVideo")
    preview_template_id = next(node["id"] for node in source["nodes"] if node.get("type") == "PreviewImage")
    video = clone(templates, load_video_template_id, 1, title="1. 输入待检测的长视频", pos=[0, 160])
    set_widget(video, ["", "image"])

    detector = make_stage_node(
        2,
        "CompanyLongVideoShotDetector",
        "2. 检测硬切、淡入淡出和逻辑镜头",
        [520, 100],
        [560, 430],
        [
            stage_input("video", "VIDEO", "长视频"),
            stage_input("mode", "COMBO", "切分模式", widget=True),
            stage_input("fixed_duration", "COMBO", "固定分段时长（秒）", widget=True),
            stage_input("sensitivity", "COMBO", "镜头检测灵敏度", widget=True),
            stage_input("use_audio_silence", "BOOLEAN", "用音频停顿辅助长镜头切分", widget=True),
            stage_input("auto_fallback", "BOOLEAN", "检测失败自动改用固定切分", widget=True),
        ],
        [
            stage_output("镜头计划", "长视频镜头计划"),
            stage_output("镜头检测 JSON", "STRING"),
            stage_output("镜头起始帧预览", "IMAGE"),
        ],
        ["镜头优先（推荐）", "10", "标准", True, True],
    )

    inspector = make_stage_node(
        3,
        "CompanyLongVideoShotInspector",
        "3. 选择镜头并播放检查（本地，不调用远端）",
        [1220, 80],
        [600, 390],
        [
            stage_input("shot_plan", "长视频镜头计划", "镜头计划"),
            stage_input("shot_index", "INT", "要检查的镜头序号", widget=True),
            stage_input("export_all_shots", "BOOLEAN", "导出全部检测镜头", widget=True),
        ],
        [
            stage_output("选中镜头视频", "VIDEO"),
            stage_output("选中镜头首中尾帧", "IMAGE"),
            stage_output("切点前后帧对比", "IMAGE"),
            stage_output("分镜检查报告 JSON", "STRING"),
            stage_output("全部镜头导出目录", "STRING"),
        ],
        [1, False],
    )

    shot_starts = clone(templates, preview_template_id, 4, title="验证 A：所有逻辑镜头起始画面", pos=[520, 620])
    shot_starts["inputs"] = [{"name": "images", "localized_name": "图像", "type": "IMAGE", "link": None}]
    selected_frames = make_stage_node(
        5,
        "CompanyFixedColumnImagePreview",
        "验证 B：每行固定 3 张（开始 / 中间 / 结束）",
        [1940, 0],
        [1020, 520],
        [
            stage_input("images", "IMAGE", "图像"),
            stage_input("columns", "COMBO", "每行图片数量", widget=True),
            stage_input("gap", "INT", "图片间距", widget=True),
        ],
        [stage_output("原始图片", "IMAGE")],
        ["3", 8],
    )
    boundary_frames = make_stage_node(
        6,
        "CompanyFixedColumnImagePreview",
        "验证 C：每行固定 2 张（切点前 / 切点后）",
        [1940, 620],
        [700, 920],
        [
            stage_input("images", "IMAGE", "图像"),
            stage_input("columns", "COMBO", "每行图片数量", widget=True),
            stage_input("gap", "INT", "图片间距", widget=True),
        ],
        [stage_output("原始图片", "IMAGE")],
        ["2", 8],
    )

    detector_report = make_stage_node(
        7,
        "CompanyPersistentPromptDisplay",
        "检测报告：镜头数量、切点时间与降级原因",
        [520, 1120],
        [560, 420],
        [stage_input("text", "STRING", "检测报告")],
        [stage_output("检测报告", "STRING")],
    )
    inspector_report = make_stage_node(
        8,
        "CompanyPersistentPromptDisplay",
        "检查报告：当前镜头时间范围与切点预览顺序",
        [1220, 620],
        [600, 520],
        [stage_input("text", "STRING", "检查报告")],
        [stage_output("检查报告", "STRING")],
    )
    workflow["nodes"].extend(
        [video, detector, inspector, shot_starts, selected_frames, boundary_frames, detector_report, inspector_report]
    )

    next_link = 1
    add_link(workflow, video, 0, detector, "video", next_link, "VIDEO"); next_link += 1
    add_link(workflow, detector, 0, inspector, "shot_plan", next_link, "长视频镜头计划"); next_link += 1
    add_link(workflow, detector, 2, shot_starts, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, inspector, 1, selected_frames, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, inspector, 2, boundary_frames, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, detector, 1, detector_report, "text", next_link, "STRING"); next_link += 1
    add_link(workflow, inspector, 3, inspector_report, "text", next_link, "STRING"); next_link += 1
    workflow["last_link_id"] = next_link - 1
    workflow["groups"] = [
        {"id": 1, "title": "本地分镜检测", "bounding": [-60, 20, 1160, 980], "color": "#3f789e", "flags": {}},
        {"id": 2, "title": "逐镜头播放与切点核对", "bounding": [1150, -60, 1870, 1660], "color": "#81683e", "flags": {}},
        {"id": 3, "title": "检测明细", "bounding": [460, 1060, 640, 540], "color": "#6b5b95", "flags": {}},
    ]
    return workflow


def build_continuity_test_workflow(source: dict) -> dict:
    templates = {node["id"]: node for node in source["nodes"]}
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, CONTINUITY_TEST_OUTPUT.name)),
        "revision": 0,
        "last_node_id": 15,
        "last_link_id": 14,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "workflow_note": (
                "连续分镜远端生成工作流：先检测分镜，再选择任意数量的连续镜头生成。"
                "从第 2 个生成分段开始自动加入上一段末帧作为软连续性参考；每段视频、末帧和合并结果均可检查。"
            ),
            "long_video": {
                "continuity_generation_test": True,
                "remote_requests": True,
                "continuity": "soft_previous_end_frame",
            },
        },
        "version": 0.4,
    }

    load_video_template_id = next(node["id"] for node in source["nodes"] if node.get("type") == "LoadVideo")
    video = clone(templates, load_video_template_id, 1, title="1. 输入测试视频", pos=[0, 320])
    set_widget(video, ["", "image"])

    detector = make_stage_node(
        2,
        "CompanyLongVideoShotDetector",
        "2. 检测完整视频分镜",
        [500, 240],
        [560, 430],
        [
            stage_input("video", "VIDEO", "长视频"),
            stage_input("mode", "COMBO", "切分模式", widget=True),
            stage_input("fixed_duration", "COMBO", "固定分段时长（秒）", widget=True),
            stage_input("sensitivity", "COMBO", "镜头检测灵敏度", widget=True),
            stage_input("use_audio_silence", "BOOLEAN", "用音频停顿辅助长镜头切分", widget=True),
            stage_input("auto_fallback", "BOOLEAN", "检测失败自动改用固定切分", widget=True),
        ],
        [
            stage_output("镜头计划", "长视频镜头计划"),
            stage_output("镜头检测 JSON", "STRING"),
            stage_output("镜头起始帧预览", "IMAGE"),
        ],
        ["镜头优先（推荐）", "10", "标准", True, True],
    )
    shot_starts = make_stage_node(
        3,
        "CompanyFixedColumnImagePreview",
        "按顺序确认：分镜起始画面（左到右、上到下，从 1 开始）",
        [500, 760],
        [920, 520],
        [
            stage_input("images", "IMAGE", "图像"),
            stage_input("columns", "COMBO", "每行图片数量", widget=True),
            stage_input("gap", "INT", "图片间距", widget=True),
        ],
        [stage_output("原始图片", "IMAGE")],
        ["3", 8],
    )
    selector = make_stage_node(
        4,
        "CompanyLongVideoContinuityRangeSelector",
        "3. 选择任意数量的连续镜头",
        [1500, 240],
        [560, 360],
        [
            stage_input("shot_plan", "长视频镜头计划", "镜头计划"),
            stage_input("start_shot", "INT", "起始镜头序号", widget=True),
            stage_input("shot_count", "INT", "连续镜头数量（0=全部剩余）", widget=True),
        ],
        [
            stage_output("选中范围镜头计划", "长视频镜头计划"),
            stage_output("选中范围原视频", "VIDEO"),
            stage_output("范围选择报告 JSON", "STRING"),
        ],
        [1, 0],
    )
    selection_report = make_stage_node(
        5,
        "CompanyPersistentPromptDisplay",
        "检查：选中的原镜头编号和时间范围",
        [1500, 700],
        [560, 500],
        [stage_input("text", "STRING", "范围选择报告")],
        [stage_output("范围选择报告", "STRING")],
    )

    asset_inputs = [stage_input("assets_manifest", "STRING", "资产清单 JSON 或路径", widget=True)]
    asset_inputs.extend(stage_input(f"person_{person_id}", "IMAGE", f"欧美化人物 {person_id}") for person_id in PERSON_IDS)
    asset_inputs.extend(stage_input(background_id, "IMAGE", f"欧美化背景 {background_id}") for background_id in BACKGROUND_IDS)
    asset_loader = make_stage_node(
        6,
        "CompanyLongVideoAssetLoader",
        "4. 读取第一阶段确认的欧美化资产",
        [1500, -560],
        [560, 680],
        asset_inputs,
        [stage_output("已加载资产", "长视频资产清单"), stage_output("资产摘要 JSON", "STRING")],
        [""],
    )

    adapter = make_stage_node(
        7,
        "CompanyLongVideoDurationAdapter",
        "5. 只为选中范围建立生成任务",
        [2200, -60],
        [630, 650],
        [
            stage_input("shot_plan", "长视频镜头计划", "选中范围镜头计划"),
            stage_input("assets", "长视频资产清单", "已加载资产"),
            stage_input("prompt", "STRING", "视频提示词", widget=True),
            stage_input("engine", "COMBO", "视频引擎", widget=True),
            stage_input("model", "COMBO", "模型", widget=True),
            stage_input("analysis_model", "COMBO", "分段分析模型", widget=True),
            stage_input("max_retries", "INT", "每段最大重试次数", widget=True),
            stage_input("resume", "BOOLEAN", "复用已完成分段", widget=True),
            stage_input("force_rerun", "BOOLEAN", "强制重跑全部分段", widget=True),
            stage_input("negative_prompt", "STRING", "负面提示词", widget=True),
        ],
        [
            stage_output("长视频任务", "长视频分段任务"),
            stage_output("时长适配计划 JSON", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
        ],
        [
            "按照选中连续分镜的原剧情、动作、镜头和节奏重新演绎。参考人物与背景是最终外观标准；保持人物数量、身份对应和场景关系。从第二段开始参考上一生成分段末帧，保持动作方向、空间位置和视觉风格连续。",
            "Seedance 2.0",
            "Seedance 2.0 Fast",
            "gpt-5.4",
            1,
            True,
            False,
            "人物肢体结构错误，身份交换，人物增减，背景跳变，镜头方向无原因突变",
        ],
    )
    plan_report = make_stage_node(
        8,
        "CompanyPersistentPromptDisplay",
        "检查：实际将发送多少个模型请求",
        [2200, 720],
        [630, 500],
        [stage_input("text", "STRING", "时长适配计划")],
        [stage_output("时长适配计划", "STRING")],
    )
    analyzer = make_stage_node(
        9,
        "CompanyLongVideoSegmentAnalyzer",
        "6. GPT 识别每段人物、背景和动作",
        [3000, 0],
        [470, 300],
        [stage_input("job", "长视频分段任务", "长视频任务")],
        [stage_output("已分析任务", "长视频分段任务"), stage_output("分析结果 JSON", "STRING")],
    )
    matcher = make_stage_node(
        10,
        "CompanyLongVideoReferenceMatcher",
        "7. 为每段匹配人物、背景和连续性参考",
        [3600, 0],
        [500, 320],
        [stage_input("job", "长视频分段任务", "已分析任务")],
        [stage_output("已匹配任务", "长视频分段任务"), stage_output("匹配结果 JSON", "STRING")],
    )
    generator = make_stage_node(
        11,
        "CompanyLongVideoSegmentGenerator",
        "8. 顺序生成选中的连续分镜",
        [4250, 0],
        [500, 330],
        [stage_input("job", "长视频分段任务", "已匹配任务")],
        [stage_output("已生成任务", "长视频分段任务"), stage_output("生成状态 JSON", "STRING")],
    )
    continuity_preview = make_stage_node(
        12,
        "CompanyLongVideoContinuityPreview",
        "9. 逐段播放并检查连续性",
        [4900, 0],
        [720, 620],
        [stage_input("job", "长视频分段任务", "已生成任务")],
        [
            stage_output("可合并任务", "长视频分段任务"),
            stage_output("连续性检查 JSON", "STRING"),
            stage_output("各段末帧", "IMAGE"),
        ],
    )
    end_frames = make_stage_node(
        13,
        "CompanyFixedColumnImagePreview",
        "对比：每个生成分段的末帧（每行 3 张）",
        [4900, 760],
        [900, 500],
        [
            stage_input("images", "IMAGE", "图像"),
            stage_input("columns", "COMBO", "每行图片数量", widget=True),
            stage_input("gap", "INT", "图片间距", widget=True),
        ],
        [stage_output("原始图片", "IMAGE")],
        ["3", 8],
    )
    merger = make_stage_node(
        14,
        "CompanyLongVideoFinalMerger",
        "10. 合并本次测试分镜并恢复对应原音频",
        [5800, 0],
        [520, 380],
        [stage_input("job", "长视频分段任务", "可合并任务")],
        [
            stage_output("最终视频", "VIDEO"),
            stage_output("最终视频路径", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
            stage_output("任务状态 JSON", "STRING"),
        ],
    )
    save = clone(templates, 255, 15, title="保存连续分镜测试视频", pos=[6470, 0])
    save["inputs"] = [
        stage_input("video", "VIDEO", "视频"),
        stage_input("filename_prefix", "STRING", "文件名前缀", widget=True),
        stage_input("format", "COMBO", "格式", widget=True),
        stage_input("codec", "COMBO", "编码器", widget=True),
    ]
    set_widget(save, ["video/LongVideoContinuityTest", "auto", "auto"])

    workflow["nodes"].extend(
        [
            video,
            detector,
            shot_starts,
            selector,
            selection_report,
            asset_loader,
            adapter,
            plan_report,
            analyzer,
            matcher,
            generator,
            continuity_preview,
            end_frames,
            merger,
            save,
        ]
    )
    next_link = 1
    add_link(workflow, video, 0, detector, "video", next_link, "VIDEO"); next_link += 1
    add_link(workflow, detector, 2, shot_starts, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, detector, 0, selector, "shot_plan", next_link, "长视频镜头计划"); next_link += 1
    add_link(workflow, selector, 2, selection_report, "text", next_link, "STRING"); next_link += 1
    add_link(workflow, selector, 0, adapter, "shot_plan", next_link, "长视频镜头计划"); next_link += 1
    add_link(workflow, asset_loader, 0, adapter, "assets", next_link, "长视频资产清单"); next_link += 1
    add_link(workflow, adapter, 1, plan_report, "text", next_link, "STRING"); next_link += 1
    add_link(workflow, adapter, 0, analyzer, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, analyzer, 0, matcher, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, matcher, 0, generator, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, generator, 0, continuity_preview, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, continuity_preview, 2, end_frames, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, continuity_preview, 0, merger, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, merger, 0, save, "video", next_link, "VIDEO"); next_link += 1
    workflow["last_link_id"] = next_link - 1
    workflow["groups"] = [
        {"id": 1, "title": "A. 检测并选择连续分镜", "bounding": [-60, 160, 2120, 1160], "color": "#3f789e", "flags": {}},
        {"id": 2, "title": "B. 加载已确认欧美化资产", "bounding": [1440, -620, 660, 740], "color": "#6b5b95", "flags": {}},
        {"id": 3, "title": "C. 逐步分析、匹配和顺序生成", "bounding": [2140, -120, 3520, 1400], "color": "#81683e", "flags": {}},
        {"id": 4, "title": "D. 合并并保存本次测试结果", "bounding": [5740, -100, 1260, 620], "color": "#4f7f69", "flags": {}},
    ]
    return workflow


def _auto_asset_planner_node(node_id: int, pos: list[float]) -> dict:
    return make_stage_node(
        node_id,
        "CompanyLongVideoAutoAssetPlanner",
        "3. 按镜头建立自动人物/背景资产任务",
        pos,
        [600, 760],
        [
            stage_input("shot_plan", "长视频镜头计划", "镜头计划"),
            stage_input("prompt", "STRING", "视频提示词", widget=True),
            stage_input("engine", "COMBO", "视频引擎", widget=True),
            stage_input("model", "COMBO", "模型", widget=True),
            stage_input("analysis_model", "COMBO", "镜头分析模型", widget=True),
            stage_input("image_model", "COMBO", "自动资产图片模型", widget=True),
            stage_input("image_quality", "COMBO", "自动资产图片质量", widget=True),
            stage_input("reuse_threshold", "FLOAT", "跨镜头复用置信度阈值", widget=True),
            stage_input("max_retries", "INT", "每段最大重试次数", widget=True),
            stage_input("resume", "BOOLEAN", "复用已完成任务和资产", widget=True),
            stage_input("force_rerun", "BOOLEAN", "强制重跑视频分段", widget=True),
            stage_input("force_rerun_assets", "BOOLEAN", "强制重建镜头资产", widget=True),
            stage_input("negative_prompt", "STRING", "负面提示词", widget=True),
            stage_input("image_provider", "COMBO", "自动资产图片服务", widget=True),
        ],
        [
            stage_output("自动资产长视频任务", "长视频分段任务"),
            stage_output("任务计划 JSON", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
        ],
        [
            "原视频只定义剧情、动作、镜头和时间过程。自动生成的人物与场景参考必须呈现鲜明的欧美影视、欧美动画或欧美漫画审美，保持原始媒介和叙事功能，不是轻微调色。",
            "Seedance 2.0",
            "Seedance 2.0 Fast",
            "gpt-5.4",
            "gpt-image-2",
            "medium",
            0.92,
            2,
            True,
            False,
            False,
            "",
            "WisArt",
        ],
    )


def build_auto_asset_test_workflow(source: dict) -> dict:
    templates = {node["id"]: node for node in source["nodes"]}
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, AUTO_ASSET_TEST_OUTPUT.name)),
        "revision": 0,
        "last_node_id": 8,
        "last_link_id": 7,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "workflow_note": "按镜头自动资产工作流：对选中的任意数量连续镜头导出首中尾帧、自动生成欧美化人物和首尾场景、打包参考图；默认不调用 Seedance 或 Wan。",
            "long_video": {"auto_shot_assets": True, "test_only": True, "remote_video_requests": False},
        },
        "version": 0.4,
    }
    load_video_template_id = next(node["id"] for node in source["nodes"] if node.get("type") == "LoadVideo")
    video = clone(templates, load_video_template_id, 1, title="1. 输入长视频", pos=[0, 260])
    set_widget(video, ["", "image"])
    detector = make_stage_node(
        2,
        "CompanyLongVideoShotDetector",
        "2. 检测镜头与转场",
        [480, 210],
        [540, 430],
        [
            stage_input("video", "VIDEO", "长视频"),
            stage_input("mode", "COMBO", "切分模式", widget=True),
            stage_input("fixed_duration", "COMBO", "固定分段时长（秒）", widget=True),
            stage_input("sensitivity", "COMBO", "镜头检测灵敏度", widget=True),
            stage_input("use_audio_silence", "BOOLEAN", "用音频停顿辅助长镜头切分", widget=True),
            stage_input("auto_fallback", "BOOLEAN", "检测失败自动改用固定切分", widget=True),
        ],
        [stage_output("镜头计划", "长视频镜头计划"), stage_output("镜头检测 JSON", "STRING"), stage_output("镜头起始帧预览", "IMAGE")],
        ["镜头优先（推荐）", "10", "标准", True, True],
    )
    selector = make_stage_node(
        3,
        "CompanyLongVideoContinuityRangeSelector",
        "3. 选择任意数量的连续镜头",
        [1140, 220],
        [500, 330],
        [
            stage_input("shot_plan", "长视频镜头计划", "镜头计划"),
            stage_input("start_shot", "INT", "起始镜头序号", widget=True),
            stage_input("shot_count", "INT", "连续镜头数量（0=全部剩余）", widget=True),
        ],
        [
            stage_output("选中范围镜头计划", "长视频镜头计划"),
            stage_output("选中连续视频", "VIDEO"),
            stage_output("选择报告 JSON", "STRING"),
        ],
        [1, 0],
    )
    planner = _auto_asset_planner_node(4, [1780, -80])
    builder = make_stage_node(
        5,
        "CompanyLongVideoAutoAssetBuilder",
        "4. 提取首中尾帧并自动生成欧美化人物/场景",
        [2480, 0],
        [480, 320],
        [stage_input("job", "长视频分段任务", "自动资产长视频任务")],
        [
            stage_output("已生成镜头资产的任务", "长视频分段任务"),
            stage_output("镜头资产报告 JSON", "STRING"),
            stage_output("源帧与自动资产预览", "IMAGE"),
        ],
    )
    asset_preview = make_stage_node(
        6,
        "CompanyFixedColumnImagePreview",
        "检查：首中尾帧、人物与首尾场景资产",
        [3060, 0],
        [760, 520],
        [stage_input("images", "IMAGE", "图像"), stage_input("columns", "COMBO", "每行图片数量", widget=True), stage_input("gap", "INT", "图片间距", widget=True)],
        [stage_output("原始图片", "IMAGE")],
        ["3", 8],
    )
    packer = make_stage_node(
        7,
        "CompanyLongVideoAutoReferencePacker",
        "5. 按视频模型上限打包自动参考图",
        [3960, 0],
        [460, 300],
        [stage_input("job", "长视频分段任务", "已生成镜头资产的任务")],
        [
            stage_output("已打包参考素材的任务", "长视频分段任务"),
            stage_output("参考素材报告 JSON", "STRING"),
            stage_output("自动参考包预览", "IMAGE"),
        ],
    )
    reference_preview = make_stage_node(
        8,
        "CompanyFixedColumnImagePreview",
        "检查：将交给视频模型的自动参考包",
        [4540, 0],
        [720, 470],
        [stage_input("images", "IMAGE", "图像"), stage_input("columns", "COMBO", "每行图片数量", widget=True), stage_input("gap", "INT", "图片间距", widget=True)],
        [stage_output("原始图片", "IMAGE")],
        ["3", 8],
    )
    workflow["nodes"].extend([video, detector, selector, planner, builder, asset_preview, packer, reference_preview])
    next_link = 1
    add_link(workflow, video, 0, detector, "video", next_link, "VIDEO"); next_link += 1
    add_link(workflow, detector, 0, selector, "shot_plan", next_link, "长视频镜头计划"); next_link += 1
    add_link(workflow, selector, 0, planner, "shot_plan", next_link, "长视频镜头计划"); next_link += 1
    add_link(workflow, planner, 0, builder, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, builder, 2, asset_preview, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, builder, 0, packer, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, packer, 2, reference_preview, "images", next_link, "IMAGE"); next_link += 1
    workflow["last_link_id"] = next_link - 1
    workflow["groups"] = [
        {"id": 1, "title": "A. 检测与选择连续镜头范围", "bounding": [-60, 140, 1740, 700], "color": "#3f789e", "flags": {}},
        {"id": 2, "title": "B. 自动生成并检查人物/背景资产", "bounding": [1720, -140, 2140, 780], "color": "#81683e", "flags": {}},
        {"id": 3, "title": "C. 打包并检查视频参考图", "bounding": [3900, -100, 1440, 700], "color": "#4f7f69", "flags": {}},
    ]
    return workflow


def build_auto_asset_restyle_workflow(source: dict) -> dict:
    workflow = build_auto_asset_test_workflow(source)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, AUTO_ASSET_RESTYLE_OUTPUT.name))
    workflow["extra"] = {
        "workflow_note": "新建副本：按镜头自动生成欧美化人物与背景资产，再顺序生成全部视频分段。先用自动资产测试工作流验收人物和场景参考图。",
        "long_video": {"auto_shot_assets": True, "remote_video_requests": True, "manifest_version": 4},
    }
    workflow["last_node_id"] = 13
    workflow["last_link_id"] = 12
    templates = {node["id"]: node for node in source["nodes"]}
    generator = make_stage_node(
        9,
        "CompanyLongVideoSegmentGenerator",
        "6. 顺序生成全部视频分段",
        [5460, 0],
        [480, 300],
        [stage_input("job", "长视频分段任务", "已打包参考素材的任务")],
        [stage_output("已生成任务", "长视频分段任务"), stage_output("生成状态 JSON", "STRING")],
    )
    continuity_preview = make_stage_node(
        10,
        "CompanyLongVideoContinuityPreview",
        "7. 逐段播放检查生成结果与连续性",
        [6080, 0],
        [510, 340],
        [stage_input("job", "长视频分段任务", "已生成任务")],
        [
            stage_output("可合并任务", "长视频分段任务"),
            stage_output("连续性报告 JSON", "STRING"),
            stage_output("各段末帧预览", "IMAGE"),
        ],
    )
    ends_preview = make_stage_node(
        11,
        "CompanyFixedColumnImagePreview",
        "预览：每段生成视频的末帧",
        [6080, 480],
        [650, 440],
        [stage_input("images", "IMAGE", "图像"), stage_input("columns", "COMBO", "每行图片数量", widget=True), stage_input("gap", "INT", "图片间距", widget=True)],
        [stage_output("原始图片", "IMAGE")],
        ["3", 8],
    )
    merger = make_stage_node(
        12,
        "CompanyLongVideoFinalMerger",
        "8. 合并分段并恢复原音频",
        [6780, 0],
        [510, 350],
        [stage_input("job", "长视频分段任务", "可合并任务")],
        [
            stage_output("最终视频", "VIDEO"),
            stage_output("最终视频路径", "STRING"),
            stage_output("任务 manifest 路径", "STRING"),
            stage_output("任务状态 JSON", "STRING"),
        ],
    )
    save = clone(templates, 255, 13, title="保存按镜头自动资产转绘视频", pos=[7460, 0])
    save["inputs"] = [
        stage_input("video", "VIDEO", "视频"),
        stage_input("filename_prefix", "STRING", "文件名前缀", widget=True),
        stage_input("format", "COMBO", "格式", widget=True),
        stage_input("codec", "COMBO", "编码器", widget=True),
    ]
    set_widget(save, ["video/LongVideoAutoShotAssets", "auto", "auto"])
    workflow["nodes"].extend([generator, continuity_preview, ends_preview, merger, save])
    next_link = max((link[0] for link in workflow["links"]), default=0) + 1
    node_by_id = {node["id"]: node for node in workflow["nodes"]}
    add_link(workflow, node_by_id[7], 0, generator, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, generator, 0, continuity_preview, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, continuity_preview, 2, ends_preview, "images", next_link, "IMAGE"); next_link += 1
    add_link(workflow, continuity_preview, 0, merger, "job", next_link, "长视频分段任务"); next_link += 1
    add_link(workflow, merger, 0, save, "video", next_link, "VIDEO"); next_link += 1
    workflow["last_link_id"] = next_link - 1
    workflow["groups"].append({"id": 4, "title": "D. 顺序生成、检查、合并与保存", "bounding": [5400, -100, 2600, 1100], "color": "#6b5b95", "flags": {}})
    return workflow


def build_anime_v2_limited_workflow(source: dict) -> dict:
    workflow = copy.deepcopy(source)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, ANIME_V2_LIMITED_OUTPUT.name))
    workflow["revision"] = 0
    workflow.setdefault("extra", {})
    workflow["extra"]["workflow_note"] = (
        "人物视频多风格转绘限时生成工作流：支持欧美化、真人影视、二维动漫、3D 游戏 CG、漫画插画和自定义视觉方向；"
        "按分钟、百分比、镜头数量或全部剩余选择生成范围，目标落在镜头中间时保留完整镜头。"
    )
    workflow["extra"].setdefault("long_video", {})
    workflow["extra"]["long_video"]["range_limit"] = {
        "enabled": True,
        "default_mode": "按分钟",
        "default_minutes": 2.0,
        "whole_shot_policy": True,
    }
    workflow["extra"]["long_video"]["selectable_visual_styles"] = [
        "western",
        "photoreal",
        "anime_2d",
        "cg_3d",
        "comic_illustration",
        "custom",
    ]
    multi_style_titles = {
        4: "3. 建立人物/背景多风格资产任务（Seedance 不上传原视频）",
        5: "4. 提取首中尾帧并生成目标人物/场景",
        6: "检查：原始首中尾帧与转换后的目标人物/场景",
        7: "5. 打包仅含目标人物/场景图片的 Seedance 参考素材",
        8: "检查：将交给 Seedance 的目标人物/场景参考图",
        9: "6. Seedance 仅参考目标人物/场景图生成分段（不上传原视频）",
        13: "保存多风格人物视频结果（Seedance）",
    }
    for node in workflow["nodes"]:
        if node.get("id") in multi_style_titles:
            node["title"] = multi_style_titles[node["id"]]
    planner = next(node for node in workflow["nodes"] if node.get("id") == 4)
    planner_widget_names = [item["name"] for item in planner.get("inputs", []) if item.get("widget")]
    if "negative_prompt" in planner_widget_names:
        negative_index = planner_widget_names.index("negative_prompt")
        planner["widgets_values"][negative_index] = ANIME_LONG_VIDEO_NEGATIVE_PROMPT
    if not any(item.get("name") == "target_resource_type" for item in planner.get("inputs", [])):
        planner.setdefault("inputs", []).append(stage_input("target_resource_type", "COMBO", "目标资源类型", widget=True))
        planner.setdefault("widgets_values", []).append("二维动漫资源")
    selector = next(node for node in workflow["nodes"] if node.get("id") == 3)
    old_input_link = None
    if selector.get("inputs"):
        old_input_link = selector["inputs"][0].get("link")
    selector.update(
        {
            "type": "CompanyLongVideoLengthRangeSelector",
            "size": [580, 470],
            "title": "3. 按时长/百分比选择生成范围",
            "properties": {"Node name for S&R": "CompanyLongVideoLengthRangeSelector"},
            "inputs": [
                stage_input("shot_plan", "长视频镜头计划", "镜头计划"),
                stage_input("start_shot", "INT", "起始镜头序号", widget=True),
                stage_input("limit_mode", "COMBO", "生成范围控制方式", widget=True),
                stage_input("limit_minutes", "FLOAT", "生成时长（分钟，0=全部剩余）", widget=True),
                stage_input("limit_percent", "FLOAT", "占原视频总长百分比（0=全部剩余）", widget=True),
                stage_input("shot_count", "INT", "镜头数量（0=全部剩余）", widget=True),
            ],
            "widgets_values": [1, "按分钟", 2.0, 30.0, 0],
        }
    )
    selector["inputs"][0]["link"] = old_input_link
    for group in workflow.get("groups", []):
        if group.get("id") == 1:
            group["title"] = "A. 检测并限制生成范围"
            group["bounding"] = [-60, 140, 1800, 760]
        elif group.get("id") == 2:
            group["title"] = "B. 自动生成并检查目标人物/背景资产"
        elif group.get("id") == 3:
            group["title"] = "C. 打包并检查 Seedance 目标参考图"
    return workflow


def build_anime_v3_limited_workflow(source: dict) -> dict:
    workflow = build_anime_v2_limited_workflow(source)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, ANIME_V3_LIMITED_OUTPUT.name))
    workflow["revision"] = 0
    workflow["extra"]["workflow_note"] = (
        "v3 独立工作流：逐个原逻辑镜头分析并生成资产；不足 4 秒的相邻镜头确定性合并后发送给 Seedance，"
        "同组镜头只发送通过质量筛选的完整转绘画面，并按图片序号标注对应时间范围。默认由 Seedance 同时生成音频；"
        "开启使用原视频音频后关闭生成音频，并按请求组恢复原音轨。远端生成视频不裁剪，短于请求时仅补最后一帧。"
    )
    workflow["extra"]["long_video"].update(
        {
            "processing_contract_version": 3,
            "short_shot_grouping": "adjacent-short-shots-v1",
            "reference_package_version": "integrated-frames-v2",
            "use_original_audio_default": False,
            "preserve_generated_video_content": True,
        }
    )
    node_by_id = {node["id"]: node for node in workflow["nodes"]}
    planner = node_by_id[4]
    planner.update(
        {
            "type": "CompanyLongVideoAnimeAssetPlannerV3",
            "title": "3. 建立 v3 多风格资产任务（短镜头合并 / 音频可选）",
            "properties": {"Node name for S&R": "CompanyLongVideoAnimeAssetPlannerV3"},
        }
    )
    if not any(item.get("name") == "use_original_audio" for item in planner.get("inputs", [])):
        planner.setdefault("inputs", []).append(
            stage_input("use_original_audio", "BOOLEAN", "使用原视频音频", widget=True)
        )
        planner.setdefault("widgets_values", []).append(False)
    node_by_id[5]["title"] = "4. 完整画面转绘并自动淘汰弱转换"
    node_by_id[7]["title"] = "5. 只打包合格整帧并标注对应时间"
    node_by_id[9]["title"] = "6. Seedance 顺序生成完整请求组（默认同时生成音频）"
    node_by_id[10]["title"] = "7. 逐组播放检查完整画面与连续性"
    node_by_id[12]["title"] = "8. 合并完整请求组并按开关处理音频"
    node_by_id[13]["title"] = "保存 v3 多风格人物视频结果（Seedance）"
    return workflow


def build_anime_v3_manual_batch_workflow(source: dict) -> dict:
    workflow = build_anime_v3_limited_workflow(source)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, ANIME_V3_MANUAL_BATCH_OUTPUT.name))
    workflow["revision"] = 0
    workflow.setdefault("extra", {})
    workflow["extra"]["workflow_note"] = (
        "手动批次审阅版：每次只生成一个约 1 分钟批次，完成后暂停等待人工检查。"
        "继续下一批按原视频时间线推进，上一批已提交的最终帧只注入下一批第一个 Seedance 请求；"
        "生成结果多几秒或少几秒不会改变源视频游标。重试当前批保持同一源范围和批次编号。"
    )
    workflow["extra"].setdefault("long_video", {})
    workflow["extra"]["long_video"].update(
        {
            "manual_batch": True,
            "pause_after_each_batch": True,
            "remote_video_requests": True,
            "processing_contract_version": 4,
            "batch_contract": "seedance_v3_manual_batch_v1",
            "default_batch_minutes": 1.0,
            "batch_minutes_range": [0.5, 5.0],
            "boundary_tolerance_seconds": 10.0,
            "source_cursor_rule": "source_timeline_authoritative",
            "cross_batch_reference_role": "cross_batch_final_frame",
        }
    )
    node_by_id = {node["id"]: node for node in workflow["nodes"]}
    selector = node_by_id[3]
    old_input_link = next((item.get("link") for item in selector.get("inputs", []) if item.get("name") == "shot_plan"), None)
    selector.update(
        {
            "type": "CompanyLongVideoManualBatchRangeSelector",
            "size": [610, 520],
            "title": "3. 控制本次生成批次（完成后人工检查）",
            "properties": {"Node name for S&R": "CompanyLongVideoManualBatchRangeSelector"},
            "inputs": [
                stage_input("shot_plan", "长视频镜头计划", "完整镜头计划"),
                stage_input("action", "COMBO", "批次动作", widget=True),
                stage_input("series_id", "STRING", "系列 ID（新建可留空）", widget=True),
                stage_input("batch_minutes", "FLOAT", "每批目标时长（分钟）", widget=True),
                stage_input("boundary_tolerance", "FLOAT", "镜头边界容差（秒）", widget=True),
            ],
            "outputs": [
                stage_output("当前批次镜头计划", "长视频镜头计划"),
                stage_output("当前批次原视频", "VIDEO"),
                stage_output("批次范围报告 JSON", "STRING"),
                stage_output("系列状态 JSON", "STRING"),
            ],
            "widgets_values": ["新建系列", "", 1.0, 10.0],
        }
    )
    selector["inputs"][0]["link"] = old_input_link
    planner = node_by_id[4]
    planner.update(
        {
            "type": "CompanyLongVideoManualBatchPlannerV1",
            "title": "4. 建立当前批次资产任务（contract=4）",
            "properties": {"Node name for S&R": "CompanyLongVideoManualBatchPlannerV1"},
        }
    )
    planner_inputs = [item for item in planner.get("inputs", []) if item.get("name") != "batch_action" and item.get("name") != "batch_series_id"]
    planner["inputs"] = planner_inputs
    if not any(item.get("name") == "use_original_audio" for item in planner["inputs"]):
        planner["inputs"].append(stage_input("use_original_audio", "BOOLEAN", "使用原视频音频", widget=True))
    set_named_widget_values(
        planner,
        {
            "prompt": WESTERN_LONG_VIDEO_PROMPT,
            "negative_prompt": WESTERN_LONG_VIDEO_NEGATIVE_PROMPT,
            "target_resource_type": "欧美化资源",
            "use_original_audio": False,
        },
    )
    node_by_id[5]["title"] = "5. 整帧欧美化并自动淘汰弱转换"
    node_by_id[7]["title"] = "6. 只打包合格整帧并标注时间"
    node_by_id[9]["title"] = "7. Seedance 生成当前批次（仅同镜头拆分时续接）"
    node_by_id[10]["title"] = "8. 检查当前批次结果，满意后再继续"
    node_by_id[12].update(
        {
            "type": "CompanyLongVideoManualBatchFinalizerV1",
            "title": "9. 合并并提交当前批次，然后暂停审阅",
            "properties": {"Node name for S&R": "CompanyLongVideoManualBatchFinalizerV1"},
            "inputs": [
                stage_input("job", "长视频分段任务", "可合并当前批次任务"),
                stage_input("series_state_json", "STRING", "系列状态 JSON"),
            ],
            "outputs": [
                stage_output("当前批次视频", "VIDEO"),
                stage_output("当前批次视频路径", "STRING"),
                stage_output("已提交系列状态 JSON", "STRING"),
                stage_output("人工审阅状态 JSON", "STRING"),
            ],
            "widgets_values": [""],
        }
    )
    node_by_id[13]["title"] = "保存当前已审阅批次（满意后再运行下一批）"
    existing_job_link = next(
        (link[0] for link in workflow.get("links", []) if link[1] == 10 and link[2] == 0 and link[3] == 12),
        None,
    )
    if existing_job_link is None:
        existing_job_link = max((link[0] for link in workflow.get("links", [])), default=0) + 1
        add_link(workflow, node_by_id[10], 0, node_by_id[12], "job", existing_job_link, "长视频分段任务")
    else:
        node_by_id[12]["inputs"][0]["link"] = existing_job_link
    next_link = max((link[0] for link in workflow.get("links", [])), default=0) + 1
    add_link(workflow, selector, 3, node_by_id[12], "series_state_json", next_link, "STRING")
    workflow["last_link_id"] = max(int(workflow.get("last_link_id", 0)), next_link)
    for link in workflow.get("links", []):
        if link[3] == 13 and link[2] == 0:
            link[4] = 0
            link[5] = "VIDEO"
    workflow["groups"] = [
        {**group, "title": ("A. 检测并控制当前批次" if group.get("id") == 1 else group.get("title"))}
        for group in workflow.get("groups", [])
    ]
    workflow["groups"].append({"id": 5, "title": "E. 当前批次提交后暂停，人工检查后继续", "bounding": [6600, -120, 1700, 900], "color": "#8a6d3b", "flags": {}})
    return workflow


def build_anime_v3_manual_batch_pipeline_workflow(source: dict) -> dict:
    """Create the real-time preview variant with global integrated-frame calibration."""
    workflow = copy.deepcopy(source)
    existing: dict | None = None
    if ANIME_V3_MANUAL_BATCH_PIPELINE_OUTPUT.is_file():
        try:
            existing = json.loads(ANIME_V3_MANUAL_BATCH_PIPELINE_OUTPUT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, ANIME_V3_MANUAL_BATCH_PIPELINE_OUTPUT.name))
    workflow["revision"] = 0
    workflow.setdefault("extra", {})
    workflow["extra"]["workflow_note"] = (
        "实时预览版：先为当前批次生成完整画面的目标风格转绘，并自动淘汰转换太弱或构图失真的结果；"
        "同一场景和强风格参考会复用。全部镜头完成质量比较后，只把合格整帧按时间范围交给 Seedance。"
        "不同场景不传上一组末帧，保留结果检查、合并和每批人工审阅。"
    )
    workflow["extra"].setdefault("long_video", {})
    workflow["extra"]["long_video"].update(
        {
            "asset_video_pipeline": True,
            "asset_video_overlap": False,
            "global_integrated_frame_calibration": True,
            "pipeline_video_order": "sequential_with_same_logical_shot_continuity_only",
            "pipeline_asset_order": "logical_member_order",
            "intermediate_reference_review": False,
        }
    )

    node_by_id = {node["id"]: node for node in workflow["nodes"]}
    pipeline = make_stage_node(
        5,
        "CompanyLongVideoPipelineAssetVideoGenerator",
        "5. 实时预览整帧转绘，筛选后生成 Seedance",
        [2480, 0],
        [720, 700],
        [
            stage_input("job", "长视频分段任务", "当前批次自动资产任务"),
            stage_input("image_concurrency", "INT", "图片并发数（0=无上限）", widget=True),
        ],
        [
            stage_output("已生成当前批次任务", "长视频分段任务"),
            stage_output("流水线状态 JSON", "STRING"),
        ],
        [5],
    )
    node_by_id[5] = pipeline

    if isinstance(existing, dict):
        existing_nodes = {node.get("id"): node for node in existing.get("nodes", []) if isinstance(node, dict)}
        for node_id in (1, 4, 13):
            previous = existing_nodes.get(node_id)
            current = node_by_id.get(node_id)
            if isinstance(previous, dict) and isinstance(current, dict):
                if isinstance(previous.get("widgets_values"), list):
                    current["widgets_values"] = copy.deepcopy(previous["widgets_values"])
                if isinstance(previous.get("size"), list):
                    current["size"] = copy.deepcopy(previous["size"])
        if isinstance(existing.get("ds"), dict):
            workflow["ds"] = copy.deepcopy(existing["ds"])

    node_by_id[10]["pos"] = [3200, 0]
    node_by_id[10]["title"] = "6. 检查当前批次结果，满意后再继续"
    node_by_id[11]["pos"] = [3200, 540]
    node_by_id[12]["pos"] = [4050, 0]
    node_by_id[12]["title"] = "7. 合并并提交当前批次，然后暂停审阅"
    node_by_id[13]["pos"] = [4800, 0]
    node_by_id[13]["title"] = "保存当前已审阅批次（满意后再运行下一批）"

    removed_node_ids = {6, 7, 8, 9}
    workflow["nodes"] = [
        pipeline if node["id"] == 5 else node
        for node in workflow["nodes"]
        if node["id"] not in removed_node_ids
    ]
    workflow["links"] = [
        link
        for link in workflow.get("links", [])
        if link[1] not in removed_node_ids and link[3] not in removed_node_ids
    ]

    next_link = max((int(link[0]) for link in workflow["links"]), default=0) + 1
    workflow["links"].append([next_link, pipeline["id"], 0, node_by_id[10]["id"], 0, "长视频分段任务"])
    workflow["last_link_id"] = max(int(workflow.get("last_link_id", 0)), next_link)

    # Rebuild the LiteGraph endpoint bookkeeping after removing the intermediate nodes.
    active_nodes = {node["id"]: node for node in workflow["nodes"]}
    for node in active_nodes.values():
        for item in node.get("inputs", []):
            item["link"] = None
        for output in node.get("outputs", []):
            output["links"] = []
    for link_id, source_id, source_slot, target_id, target_slot, _data_type in workflow["links"]:
        source = active_nodes[source_id]
        target = active_nodes[target_id]
        source["outputs"][source_slot]["links"].append(link_id)
        target["inputs"][target_slot]["link"] = link_id

    workflow["groups"] = [
        {
            "id": 1,
            "title": "A. 检测并控制当前批次",
            "bounding": [-60, 140, 1780, 760],
            "color": "#3f789e",
            "flags": {},
        },
        {
            "id": 2,
            "title": "B. 资产与 Seedance 流水线（资源就绪即生成视频）",
            "bounding": [1740, -140, 1420, 860],
            "color": "#81683e",
            "flags": {},
        },
        {
            "id": 4,
            "title": "C. 检查当前批次结果",
            "bounding": [3180, -120, 840, 1180],
            "color": "#6b5b95",
            "flags": {},
        },
        {
            "id": 5,
            "title": "D. 当前批次提交后暂停，人工检查后继续",
            "bounding": [4040, -120, 1400, 900],
            "color": "#8a6d3b",
            "flags": {},
        },
    ]
    return workflow


def build_anime_v2_parallel_workflow(source: dict) -> dict:
    """Create an independent parallel-generation copy of the anime v2 workflow."""
    workflow = copy.deepcopy(source)
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, ANIME_V2_PARALLEL_OUTPUT.name))
    workflow["revision"] = 0
    workflow.setdefault("extra", {})
    workflow["extra"]["workflow_note"] = (
        "D 方案新建副本：按镜头自动资产和 Seedance 参考图准备完成后，默认并发 3 个独立视频分段；"
        "每完成一个分段就通过节点内实时预览显示。并行段不等待上一段末帧，因此速度更快但连续性需要在全部完成后复核；"
        "失败会保留 manifest 和 parallel_video_progress.json，重新执行并开启复用已完成分段即可从断点继续，原串行工作流不修改。"
    )
    workflow["extra"].setdefault("long_video", {})
    workflow["extra"]["long_video"].update(
        {
            "parallel_generation": True,
            "default_concurrency": 3,
            "live_segment_preview": True,
            "resume_progress_file": "parallel_video_progress.json",
            "continuity_tradeoff": "parallel_segments_do_not_use_previous_segment_end_frame",
        }
    )
    node_by_id = {node["id"]: node for node in workflow["nodes"]}
    generator = node_by_id[9]
    existing_job_link = next(
        (item.get("link") for item in generator.get("inputs", []) if item.get("name") == "job"),
        None,
    )
    generator.update(
        {
            "type": "CompanyLongVideoParallelSegmentGenerator",
            "title": "6. Seedance 并行生成分段（每段完成立即显示）",
            "size": [560, 520],
            "properties": {"Node name for S&R": "CompanyLongVideoParallelSegmentGenerator"},
        }
    )
    generator["inputs"] = [
        stage_input("job", "长视频分段任务", "已打包参考素材的任务"),
        stage_input("concurrency", "INT", "并发分段数", widget=True),
    ]
    generator["inputs"][0]["link"] = existing_job_link
    generator["widgets_values"] = [3]
    generator["outputs"] = [
        stage_output("已生成任务", "长视频分段任务"),
        stage_output("生成状态 JSON", "STRING"),
    ]
    continuity = node_by_id[10]
    continuity["title"] = "7. 全部完成后检查生成结果与连续性"
    workflow["groups"] = [
        {
            **group,
            "title": "D. 并行生成、实时预览、复核与合并"
            if group.get("id") == 4
            else group.get("title"),
        }
        for group in workflow.get("groups", [])
    ]
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-only", action="store_true", help="只生成参考素材准备工作流")
    parser.add_argument("--visual-only", action="store_true", help="只生成可视化分阶段工作流副本")
    parser.add_argument("--shot-aware-only", action="store_true", help="只生成镜头感知分阶段工作流副本")
    parser.add_argument("--shot-test-only", action="store_true", help="只生成本地分镜检测测试工作流")
    parser.add_argument("--continuity-test-only", action="store_true", help="只生成连续分镜远端生成测试工作流")
    parser.add_argument("--auto-asset-test-only", action="store_true", help="只生成按镜头自动资产测试工作流")
    parser.add_argument("--auto-asset-restyle-only", action="store_true", help="只生成按镜头自动资产完整转绘工作流")
    parser.add_argument("--anime-v2-limited-only", action="store_true", help="只生成多风格素材库复用 v2 限时生成工作流")
    parser.add_argument("--anime-v3-limited-only", action="store_true", help="只生成 v3 短镜头合并和音频可选限时工作流")
    parser.add_argument("--anime-v3-manual-batch-only", action="store_true", help="只生成 v3 手动批次审阅工作流")
    parser.add_argument("--anime-v3-manual-batch-pipeline-only", action="store_true", help="只生成 v3 手动批次资产视频流水线工作流")
    parser.add_argument("--anime-v2-parallel-only", action="store_true", help="只生成动漫化素材库复用 v2 并行生成工作流")
    args = parser.parse_args()
    if args.anime_v2_limited_only:
        anime_source = json.loads(ANIME_V2_SOURCE.read_text(encoding="utf-8"))
        ANIME_V2_LIMITED_OUTPUT.write_text(
            json.dumps(build_anime_v2_limited_workflow(anime_source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(ANIME_V2_LIMITED_OUTPUT)
        return
    if args.anime_v3_limited_only:
        anime_source = json.loads(ANIME_V2_SOURCE.read_text(encoding="utf-8"))
        ANIME_V3_LIMITED_OUTPUT.write_text(
            json.dumps(build_anime_v3_limited_workflow(anime_source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(ANIME_V3_LIMITED_OUTPUT)
        return
    if args.anime_v3_manual_batch_only:
        source_path = ANIME_V3_LIMITED_OUTPUT if ANIME_V3_LIMITED_OUTPUT.is_file() else ANIME_V2_SOURCE
        anime_source = json.loads(source_path.read_text(encoding="utf-8"))
        ANIME_V3_MANUAL_BATCH_OUTPUT.write_text(
            json.dumps(build_anime_v3_manual_batch_workflow(anime_source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(ANIME_V3_MANUAL_BATCH_OUTPUT)
        return
    if args.anime_v3_manual_batch_pipeline_only:
        source_path = ANIME_V3_MANUAL_BATCH_OUTPUT if ANIME_V3_MANUAL_BATCH_OUTPUT.is_file() else ANIME_V2_SOURCE
        anime_source = json.loads(source_path.read_text(encoding="utf-8"))
        ANIME_V3_MANUAL_BATCH_PIPELINE_OUTPUT.write_text(
            json.dumps(build_anime_v3_manual_batch_pipeline_workflow(anime_source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(ANIME_V3_MANUAL_BATCH_PIPELINE_OUTPUT)
        return
    if args.anime_v2_parallel_only:
        anime_source = json.loads(ANIME_V2_SOURCE.read_text(encoding="utf-8"))
        ANIME_V2_PARALLEL_OUTPUT.write_text(
            json.dumps(build_anime_v2_parallel_workflow(anime_source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(ANIME_V2_PARALLEL_OUTPUT)
        return
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if args.asset_only:
        ASSET_OUTPUT.write_text(json.dumps(build_asset_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(ASSET_OUTPUT)
        return
    if args.visual_only:
        VISUAL_RESTYLE_OUTPUT.write_text(json.dumps(build_visual_restyle_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(VISUAL_RESTYLE_OUTPUT)
        return
    if args.shot_aware_only:
        SHOT_AWARE_RESTYLE_OUTPUT.write_text(json.dumps(build_shot_aware_restyle_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(SHOT_AWARE_RESTYLE_OUTPUT)
        return
    if args.shot_test_only:
        SHOT_TEST_OUTPUT.write_text(json.dumps(build_shot_test_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(SHOT_TEST_OUTPUT)
        return
    if args.continuity_test_only:
        CONTINUITY_TEST_OUTPUT.write_text(
            json.dumps(build_continuity_test_workflow(source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(CONTINUITY_TEST_OUTPUT)
        return
    if args.auto_asset_test_only:
        AUTO_ASSET_TEST_OUTPUT.write_text(
            json.dumps(build_auto_asset_test_workflow(source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(AUTO_ASSET_TEST_OUTPUT)
        return
    if args.auto_asset_restyle_only:
        AUTO_ASSET_RESTYLE_OUTPUT.write_text(
            json.dumps(build_auto_asset_restyle_workflow(source), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(AUTO_ASSET_RESTYLE_OUTPUT)
        return
    ASSET_OUTPUT.write_text(json.dumps(build_asset_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RESTYLE_OUTPUT.write_text(json.dumps(build_restyle_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    VISUAL_RESTYLE_OUTPUT.write_text(json.dumps(build_visual_restyle_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SHOT_AWARE_RESTYLE_OUTPUT.write_text(json.dumps(build_shot_aware_restyle_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SHOT_TEST_OUTPUT.write_text(json.dumps(build_shot_test_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CONTINUITY_TEST_OUTPUT.write_text(json.dumps(build_continuity_test_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUTO_ASSET_TEST_OUTPUT.write_text(json.dumps(build_auto_asset_test_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUTO_ASSET_RESTYLE_OUTPUT.write_text(json.dumps(build_auto_asset_restyle_workflow(source), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ASSET_OUTPUT)
    print(RESTYLE_OUTPUT)
    print(VISUAL_RESTYLE_OUTPUT)
    print(SHOT_AWARE_RESTYLE_OUTPUT)
    print(SHOT_TEST_OUTPUT)
    print(CONTINUITY_TEST_OUTPUT)
    print(AUTO_ASSET_TEST_OUTPUT)
    print(AUTO_ASSET_RESTYLE_OUTPUT)


if __name__ == "__main__":
    main()
