from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parents[3] / "user" / "default" / "workflows"
SOURCE_PATH = WORKFLOW_DIR / "动漫角色转真人_纯远端多人ABC自动分流GPTImage2版.json"
OUTPUT_PATH = WORKFLOW_DIR / "动漫角色转真人_纯远端多人ABC自动分流_人物背景精修高清版.json"

PERSON_REFINE_PROMPT = """对输入的人物单独进行高质量真人精修。严格锁定该人物的身份、年龄感、性别、脸型、五官关系、肤色、发型发色、体型、服装配色与结构、配饰、道具及角色气质，不与其他人物交换或混合特征。

重点修复眼睛、眉毛、鼻子、嘴唇、牙齿、耳朵、发际线、发丝、皮肤毛孔、手指、关节、肢体结构、服装边缘、布料纹理、饰品和道具细节；消除动漫残留、塑料皮肤、假发感、五官不对称、错误手指、肢体扭曲、糊边、色键溢色与生成伪影。

允许在不改变身份与核心造型的前提下，微调不自然的姿态、手势、视线、表情、身体重心和服装褶皱，使人物更像真实演员并具有自然表现力。保持完整单人、清晰轮廓和便于后续合成的均匀纯色色键背景；不得增加其他人物、文字、Logo 或水印。"""

BACKGROUND_REFINE_PROMPT = """对输入的无人真人化背景进行第二次电影级优化。严格保持当前场景的地点身份、画幅、镜头方向、透视、空间布局、建筑与家具位置、道路和地面结构、主要道具以及光源方向，不改变为其他场景。

彻底清除所有人物、人体局部、倒影、人物阴影、人物专属遮挡物、色键残留和生成伪影；按照周围透视、材质、纹理、照明与空间关系自然补全被遮挡区域。增强建筑、墙面、门窗、地面、家具、植被、远景、材质纹理、反射、阴影、空气透视和景深层次，使背景真实、连续、干净并具有电影质感。

背景必须保持完全无人，不得重新生成任何人物、人体、脸、手、倒影或人形剪影；不得出现空间结构漂移、重复物件、弯曲直线、乱码、文字、Logo 或水印。"""

FINAL_REFINE_PROMPT = """将输入的多人真人化合成结果制作成高分辨率、电影级写实的最终成片。严格保持现有实际人数、每个人的独立身份、脸型五官、肤色、发型发色、体型、服装、配饰、道具、人物对应关系、场景身份、剧情关系、主要动作、人物站位、镜头构图和空间布局，不得交换、融合、删除、复制或新增人物。

统一精修所有人物的眼睛、皮肤、发丝、牙齿、手部、关节、肢体、服装、饰品和道具细节；修正合成边缘、色键残留、比例、透视、脚底接触、人物遮挡、接触阴影、反射、环境光、色温、景深和运动关系，使所有人物自然处于同一真实场景。增强背景材质、光影层次、边缘清晰度和电影调色，同时保持画面自然，不要过度磨皮、锐化或产生 HDR 塑料感。

禁止身份漂移、人物串脸、服装互换、人数变化、脸部闪烁感、额外手指、肢体畸形、穿模、漂浮、人物与背景割裂、重复物体、背景结构改变、乱码、字幕、文字、Logo 和水印。"""


def _node_by_id(workflow: dict[str, Any], node_id: int) -> dict[str, Any]:
    return next(node for node in workflow["nodes"] if node["id"] == node_id)


def _set_node_link(node: dict[str, Any], input_name: str, link_id: int) -> None:
    target = next(item for item in node["inputs"] if item["name"] == input_name)
    target["link"] = link_id


def _make_gpt_image_node(
    template: dict[str, Any],
    *,
    node_id: int,
    title: str,
    prompt: str,
    pos: list[float],
    seed: int,
    size: str = "auto",
    custom_width: int = 1024,
    custom_height: int = 1536,
) -> dict[str, Any]:
    node = copy.deepcopy(template)
    node["id"] = node_id
    node["title"] = title
    node["pos"] = pos
    node["order"] = 0
    node["mode"] = 0
    for item in node["inputs"]:
        item["link"] = None
    node["outputs"][0]["links"] = []
    node["widgets_values"] = [
        prompt,
        "gpt-image-2",
        size,
        custom_width,
        custom_height,
        "opaque",
        "high",
        1,
        seed,
        "randomize",
    ]
    return node


def _make_preview_node(template: dict[str, Any], *, node_id: int, title: str, pos: list[float]) -> dict[str, Any]:
    node = copy.deepcopy(template)
    node["id"] = node_id
    node["title"] = title
    node["pos"] = pos
    node["order"] = 0
    node["inputs"][0]["link"] = None
    return node


def _make_link(
    links: list[list[Any]],
    nodes: dict[int, dict[str, Any]],
    *,
    link_id: int,
    source_id: int,
    source_slot: int,
    target_id: int,
    target_slot: int,
    data_type: str,
) -> None:
    links.append([link_id, source_id, source_slot, target_id, target_slot, data_type])
    source_links = nodes[source_id]["outputs"][source_slot].setdefault("links", [])
    if link_id not in source_links:
        source_links.append(link_id)
    nodes[target_id]["inputs"][target_slot]["link"] = link_id


def build() -> dict[str, Any]:
    with SOURCE_PATH.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)

    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"company-remote:{OUTPUT_PATH.name}"))
    workflow["revision"] = 0
    existing = {node["id"]: node for node in workflow["nodes"]}
    gpt_template = existing[10]
    preview_template = existing[11]

    additions = [
        _make_gpt_image_node(
            gpt_template,
            node_id=27,
            title="人物 A：独立高质量精修",
            prompt=PERSON_REFINE_PROMPT,
            pos=[1680, 0],
            seed=100001,
        ),
        _make_gpt_image_node(
            gpt_template,
            node_id=28,
            title="人物 B：独立高质量精修",
            prompt=PERSON_REFINE_PROMPT,
            pos=[1680, 700],
            seed=100002,
        ),
        _make_gpt_image_node(
            gpt_template,
            node_id=29,
            title="人物 C：独立高质量精修",
            prompt=PERSON_REFINE_PROMPT,
            pos=[1680, 1400],
            seed=100003,
        ),
        _make_gpt_image_node(
            gpt_template,
            node_id=30,
            title="背景：无人场景二次优化",
            prompt=BACKGROUND_REFINE_PROMPT,
            pos=[1680, 2100],
            seed=100004,
        ),
        _make_gpt_image_node(
            gpt_template,
            node_id=31,
            title="最终成片：多人全图高清精修",
            prompt=FINAL_REFINE_PROMPT,
            pos=[4300, 760],
            seed=100005,
            size="Custom",
            custom_width=2048,
            custom_height=3072,
        ),
        _make_preview_node(
            preview_template,
            node_id=32,
            title="预览：人物 A 精修结果",
            pos=[2260, 40],
        ),
        _make_preview_node(
            preview_template,
            node_id=33,
            title="预览：背景二次优化结果",
            pos=[2260, 2140],
        ),
        _make_preview_node(
            preview_template,
            node_id=34,
            title="预览：最终多人高清精修结果",
            pos=[4880, 700],
        ),
    ]
    workflow["nodes"].extend(additions)
    nodes = {node["id"]: node for node in workflow["nodes"]}

    # Move the existing first-pass nodes into a readable two-column pipeline.
    positions = {
        10: [1100, 0],
        12: [1100, 700],
        15: [1100, 1400],
        18: [1100, 2100],
        13: [2260, 760],
        14: [2600, 700],
        16: [2260, 1460],
        17: [2600, 1400],
        20: [3000, 0],
        21: [3000, 700],
        22: [3000, 1400],
        23: [3580, 1050],
        24: [3940, 650],
        25: [4880, 260],
        26: [4880, 1120],
    }
    for node_id, pos in positions.items():
        nodes[node_id]["pos"] = pos

    # Remove old previews and direct SaveImage path that are superseded by refined outputs.
    removed_node_ids = {11, 19}
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in removed_node_ids]
    removed_link_ids = {14, 23, 26, 27, 30, 31, 34, 35, 36, 37, 38, 45, 46}
    workflow["links"] = [link for link in workflow["links"] if link[0] not in removed_link_ids]
    for link in workflow["links"]:
        if link[0] == 16:
            link[1] = 28  # Refined person B -> existing lazy preview switch.
        elif link[0] == 20:
            link[1] = 29  # Refined person C -> existing lazy preview switch.
    nodes = {node["id"]: node for node in workflow["nodes"]}

    # Reset links that are replaced below.
    for node_id in (10, 12, 15, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34):
        for output in nodes[node_id].get("outputs", []):
            if output.get("type") in {"IMAGE", "*"}:
                output["links"] = [
                    link_id
                    for link_id in (output.get("links") or [])
                    if link_id not in removed_link_ids
                ]

    links = workflow["links"]
    new_links = [
        (47, 10, 0, 27, 7, "IMAGE"),
        (48, 27, 0, 32, 0, "IMAGE"),
        (49, 12, 0, 28, 7, "IMAGE"),
        (50, 15, 0, 29, 7, "IMAGE"),
        (51, 18, 0, 30, 7, "IMAGE"),
        (52, 30, 0, 33, 0, "IMAGE"),
        (53, 30, 0, 20, 8, "IMAGE"),
        (54, 27, 0, 20, 9, "IMAGE"),
        (55, 30, 0, 21, 8, "IMAGE"),
        (56, 27, 0, 21, 9, "IMAGE"),
        (57, 28, 0, 21, 10, "IMAGE"),
        (58, 30, 0, 22, 8, "IMAGE"),
        (59, 27, 0, 22, 9, "IMAGE"),
        (60, 28, 0, 22, 10, "IMAGE"),
        (61, 29, 0, 22, 11, "IMAGE"),
        (62, 24, 0, 31, 7, "IMAGE"),
        (63, 31, 0, 34, 0, "IMAGE"),
        (64, 31, 0, 25, 0, "IMAGE"),
        (65, 31, 0, 26, 0, "IMAGE"),
    ]
    for link_id, source_id, source_slot, target_id, target_slot, data_type in new_links:
        _make_link(
            links,
            nodes,
            link_id=link_id,
            source_id=source_id,
            source_slot=source_slot,
            target_id=target_id,
            target_slot=target_slot,
            data_type=data_type,
        )

    # Ensure all target sockets use the new connections.
    replacements = {
        (20, "model.images.image_2"): 53,
        (20, "model.images.image_3"): 54,
        (21, "model.images.image_2"): 55,
        (21, "model.images.image_3"): 56,
        (21, "model.images.image_4"): 57,
        (22, "model.images.image_2"): 58,
        (22, "model.images.image_3"): 59,
        (22, "model.images.image_4"): 60,
        (22, "model.images.image_5"): 61,
    }
    for (node_id, input_name), link_id in replacements.items():
        _set_node_link(nodes[node_id], input_name, link_id)

    # Recompute output link lists and execution order from the final links.
    for node in workflow["nodes"]:
        for output in node.get("outputs", []):
            output["links"] = None
    for link in links:
        link_id, source_id, source_slot, _, _, _ = link
        output = nodes[source_id]["outputs"][source_slot]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(link_id)

    workflow["last_node_id"] = 34
    workflow["last_link_id"] = 65
    for order, node in enumerate(workflow["nodes"]):
        node["order"] = order

    workflow["groups"] = [
        {"id": 1, "title": "输入与统一人物识别", "bounding": [-40, -70, 1080, 1220], "color": "#3f789e", "flags": {}},
        {"id": 2, "title": "人物 A/B/C：独立真人化与独立精修", "bounding": [1060, -70, 1860, 2080], "color": "#6b5b95", "flags": {}},
        {"id": 3, "title": "背景：移除人物、真人化与二次优化", "bounding": [1060, 2030, 1530, 620], "color": "#4f7f69", "flags": {}},
        {"id": 4, "title": "按实际人数 lazy 合成：只执行一个组合", "bounding": [2940, -70, 1340, 1920], "color": "#8f6f4e", "flags": {}},
        {"id": 5, "title": "最终全图高清精修、预览与保存", "bounding": [4240, 170, 990, 1380], "color": "#3f789e", "flags": {}},
    ]
    workflow.setdefault("extra", {})["workflow_note"] = (
        "自动识别 1-3 人；人物 A/B/C 分别真人化并独立精修，背景移除人物后进行二次优化，"
        "再按实际人数 lazy 合成，最后进行一次全图高清精修。"
    )
    workflow["extra"]["codex_note"] = {
        "source_workflow": SOURCE_PATH.name,
        "person_order": "画面从左到右；横向位置接近时从前到后",
        "pipeline": [
            "人物 A/B/C 独立真人化 -> 各自高质量精修",
            "背景移除全部人物并真人化 -> 无人场景二次优化",
            "按实际人数 lazy 选择单人/双人/三人合成",
            "合成结果 -> 最终多人全图高清精修 -> 预览与保存",
        ],
        "lazy_execution": "单人不执行 B/C 及其精修，双人不执行 C 及其精修，三人执行全部人物分支；三种合成仅执行实际人数对应分支",
        "final_reference_order": ["原始图片", "精修背景", "精修人物 A", "精修人物 B（如有）", "精修人物 C（如有）"],
        "saving": "仅最终高清精修结果连接 SaveImage；人物 A 精修、背景优化和最终精修均有 PreviewImage",
    }
    workflow.setdefault("extra", {}).setdefault("ds", {})["scale"] = 0.22
    workflow["extra"]["ds"]["offset"] = [600, 720]
    return workflow


def main() -> None:
    workflow = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(workflow, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    print(os.fspath(OUTPUT_PATH))


if __name__ == "__main__":
    main()
