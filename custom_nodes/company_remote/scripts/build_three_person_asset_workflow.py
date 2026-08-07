from __future__ import annotations

import json
import uuid
from pathlib import Path


WORKFLOW_NAME = "三人物ABC_火山资源ID_Seedance转换_10秒验证.json"
PROMPT = """严格参照源视频资产保留原有镜头、剪辑顺序、机位、动作、身体姿态、服装、车辆、道路、背景、光线和节奏，只转换三位人物的脸部身份与头部造型。
参考图 1 是人物 A：源视频车旁双人镜头中画面左侧人物，转换为酒红官服人物。
参考图 2 是人物 B：源视频车旁双人镜头中画面右侧、低头人物，转换为浅绿官服人物。
参考图 3 是人物 C：源视频灰西装单人镜头中的人物，转换为黑甲武将。
A、B、C 身份必须固定且互不串脸；同一人物跨镜头保持一致。不得改变人物数量，不新增人物，不删除人物，不交换左右位置，不改变原背景与剧情，不参考三张人物图的背景。画面真实、自然、稳定。"""


def input_node(node_id: int, node_type: str, title: str, pos: list[int], filename: str, *, video: bool = False):
    input_name = "file" if video else "image"
    output_type = "VIDEO" if video else "IMAGE"
    node = {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": [400, 420],
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": [
            {"localized_name": "文件" if video else "图像", "name": input_name, "type": "COMBO", "widget": {"name": input_name}, "link": None},
            {"localized_name": "选择要上传的文件" if video else "选择文件上传", "name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None},
        ],
        "outputs": [{"localized_name": "视频" if video else "图像", "name": output_type, "type": output_type, "slot_index": 0, "links": [node_id]}],
        "title": title,
        "properties": {"Node name for S&R": node_type},
        "widgets_values": [filename, "image"],
    }
    if not video:
        node["outputs"].append({"localized_name": "遮罩", "name": "MASK", "type": "MASK", "links": None})
    return node


def build_workflow() -> dict:
    nodes = [
        input_node(1, "LoadVideo", "1. 十秒验证视频（固定 24 FPS，已包含 A/B/C）", [40, 520], "three_person_asset_gateway_test_10s_24fps.mp4", video=True),
        input_node(2, "LoadImage", "2A. 人物 A：车旁左侧 → 酒红官服人物", [520, 20], "three_person_asset_A_red_official.png"),
        input_node(3, "LoadImage", "2B. 人物 B：车旁右侧 → 浅绿官服人物", [520, 520], "three_person_asset_B_green_official.png"),
        input_node(4, "LoadImage", "2C. 人物 C：灰西装单人 → 黑甲武将", [520, 1020], "three_person_asset_C_black_warrior.png"),
        {
            "id": 5,
            "type": "CompanySeedanceVideoABCAssetCreate",
            "pos": [1040, 430],
            "size": [610, 560],
            "flags": {},
            "order": 4,
            "mode": 4,
            "inputs": [
                {"localized_name": "源视频（建议先裁到 4-15 秒）", "name": "video", "type": "VIDEO", "link": 1},
                {"localized_name": "人物 A 参考图", "name": "image_a", "type": "IMAGE", "link": 2},
                {"localized_name": "人物 B 参考图", "name": "image_b", "type": "IMAGE", "link": 3},
                {"localized_name": "人物 C 参考图", "name": "image_c", "type": "IMAGE", "link": 4},
                {"localized_name": "复用相同素材的资产 ID", "name": "reuse_cached", "type": "BOOLEAN", "widget": {"name": "reuse_cached"}, "link": None},
            ],
            "outputs": [
                {"localized_name": "源视频资产 ID", "name": "源视频资产 ID", "type": "STRING", "slot_index": 0, "links": [5]},
                {"localized_name": "人物 A 资产 ID", "name": "人物 A 资产 ID", "type": "STRING", "slot_index": 1, "links": [6]},
                {"localized_name": "人物 B 资产 ID", "name": "人物 B 资产 ID", "type": "STRING", "slot_index": 2, "links": [7]},
                {"localized_name": "人物 C 资产 ID", "name": "人物 C 资产 ID", "type": "STRING", "slot_index": 3, "links": [8]},
                {"localized_name": "汇总报告 JSON", "name": "汇总报告 JSON", "type": "STRING", "links": []},
            ],
            "title": "3. 先 TOS，再并行注册 Video + A/B/C 四个资产（默认停用）",
            "properties": {"Node name for S&R": "CompanySeedanceVideoABCAssetCreate"},
            "widgets_values": [True],
            "color": "#6b5338",
            "bgcolor": "#896c49",
        },
        {
            "id": 6,
            "type": "CompanySeedanceAssetGatewayThreePersonVideo",
            "pos": [1780, 330],
            "size": [760, 900],
            "flags": {},
            "order": 5,
            "mode": 4,
            "inputs": [
                {"localized_name": "源视频资产 ID", "name": "source_video_asset_id", "type": "STRING", "link": 5},
                {"localized_name": "人物 A 资产 ID（参考图 1）", "name": "character_a_asset_id", "type": "STRING", "link": 6},
                {"localized_name": "人物 B 资产 ID（参考图 2）", "name": "character_b_asset_id", "type": "STRING", "link": 7},
                {"localized_name": "人物 C 资产 ID（参考图 3）", "name": "character_c_asset_id", "type": "STRING", "link": 8},
                {"localized_name": "三人物映射提示词", "name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": None},
                {"localized_name": "模型", "name": "model", "type": "COMBO", "widget": {"name": "model"}, "link": None},
                {"localized_name": "分辨率", "name": "resolution", "type": "COMBO", "widget": {"name": "resolution"}, "link": None},
                {"localized_name": "比例", "name": "ratio", "type": "COMBO", "widget": {"name": "ratio"}, "link": None},
                {"localized_name": "输出时长（秒）", "name": "duration", "type": "INT", "widget": {"name": "duration"}, "link": None},
                {"localized_name": "生成音频", "name": "generate_audio", "type": "BOOLEAN", "widget": {"name": "generate_audio"}, "link": None},
                {"localized_name": "水印", "name": "watermark", "type": "BOOLEAN", "widget": {"name": "watermark"}, "link": None},
                {"localized_name": "种子", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                {"localized_name": "继续轮询已有任务 ID", "name": "resume_task_id", "type": "STRING", "widget": {"name": "resume_task_id"}, "link": None},
            ],
            "outputs": [
                {"localized_name": "生成视频", "name": "生成视频", "type": "VIDEO", "slot_index": 0, "links": [9]},
                {"localized_name": "输出路径", "name": "输出路径", "type": "STRING", "links": []},
                {"localized_name": "任务 ID", "name": "任务 ID", "type": "STRING", "links": []},
                {"localized_name": "报告 JSON", "name": "报告 JSON", "type": "STRING", "links": []},
            ],
            "title": "4. 用同一网关的资产 ID 生成三人物转换视频（默认停用）",
            "properties": {"Node name for S&R": "CompanySeedanceAssetGatewayThreePersonVideo"},
            "widgets_values": [
                "",
                "",
                "",
                "",
                PROMPT,
                "doubao-seedance-2-0-fast-260128",
                "480p",
                "adaptive",
                10,
                False,
                False,
                0,
                "",
            ],
            "color": "#35534a",
            "bgcolor": "#446b5f",
        },
        {
            "id": 7,
            "type": "SaveVideo",
            "pos": [2700, 520],
            "size": [390, 520],
            "flags": {},
            "order": 6,
            "mode": 4,
            "inputs": [
                {"localized_name": "视频", "name": "video", "type": "VIDEO", "link": 9},
                {"localized_name": "文件名前缀", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None},
                {"localized_name": "格式", "name": "format", "type": "COMBO", "widget": {"name": "format"}, "link": None},
                {"localized_name": "编码器", "name": "codec", "type": "COMBO", "widget": {"name": "codec"}, "link": None},
            ],
            "outputs": [{"localized_name": "video", "name": "video", "type": "VIDEO", "links": None}],
            "title": "5. 保存验证结果（默认停用）",
            "properties": {"Node name for S&R": "SaveVideo"},
            "widgets_values": ["video/three_person_asset_gateway_test", "auto", "auto"],
        },
    ]
    links = [
        [1, 1, 0, 5, 0, "VIDEO"], [2, 2, 0, 5, 1, "IMAGE"], [3, 3, 0, 5, 2, "IMAGE"], [4, 4, 0, 5, 3, "IMAGE"],
        [5, 5, 0, 6, 0, "STRING"], [6, 5, 1, 6, 1, "STRING"], [7, 5, 2, 6, 2, "STRING"], [8, 5, 3, 6, 3, "STRING"],
        [9, 6, 0, 7, 0, "VIDEO"],
    ]
    return {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": 7,
        "last_link_id": 9,
        "nodes": nodes,
        "links": links,
        "groups": [
            {"id": 1, "title": "A. 已准备的 10 秒源视频与三张人物参考图", "bounding": [-10, -40, 960, 1540], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 2, "title": "B. 远程资产注册：会上传素材，先启用节点 3", "bounding": [980, 350, 730, 720], "color": "#8f6f4e", "font_size": 24, "flags": {}},
            {"id": 3, "title": "C. Seedance 10 秒验证：会产生费用，再启用节点 4/5", "bounding": [1720, 250, 1430, 1060], "color": "#4f7f69", "font_size": 24, "flags": {}},
        ],
        "config": {},
        "extra": {"ds": {"scale": 0.58, "offset": [80, 60]}, "frontendVersion": "1.25.9", "workflowRendererVersion": "LG"},
        "version": 0.4,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = root / "user" / "default" / "workflows" / WORKFLOW_NAME
    output.write_text(json.dumps(build_workflow(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
