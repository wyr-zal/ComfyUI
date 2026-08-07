from __future__ import annotations

import json
import uuid
from pathlib import Path


WORKFLOW_NAME = "Wan2.7_三人物参考图直传_无TOS无火山_10秒验证.json"
FULL_WORKFLOW_NAME = "Wan2.7_三人物参考图直传_无TOS无火山_20秒完整处理.json"
PROMPT = """严格保留原视频的时序、人物动作、表演、镜头运动、构图、景别、场景、光线和剪辑节奏，只替换三名指定人物的身份与外观。

人物映射必须全程固定：将原视频中车旁左侧人物A替换为图1人物；将原视频中车旁右侧人物B替换为图2人物；将原视频中灰西装单人C替换为图3人物。图1、图2、图3仅作为对应人物的脸型、五官、发型、年龄感、体型、服装和配饰参考。

三个人物从头到尾不得串脸、身份互换、面部融合、复制或消失。保持原视频人物的身体位置、朝向、动作幅度、遮挡关系和口型时序；保持原视频背景及所有非人物元素不变。人物必须自然融入原视频光照、阴影、透视和运动模糊。

禁止新增人物，禁止改变剧情，禁止改变镜头，禁止改变场景，禁止字幕、文字、Logo、水印、肢体畸形、额外手指、脸部闪烁、服装跳变或背景跳变。"""


def image_node(node_id: int, title: str, position: list[int], filename: str, link_id: int) -> dict:
    return {
        "id": node_id,
        "type": "LoadImage",
        "pos": position,
        "size": [380, 430],
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": [
            {"localized_name": "图像", "name": "image", "type": "COMBO", "widget": {"name": "image"}, "link": None},
            {"localized_name": "选择文件上传", "name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "图像", "name": "IMAGE", "type": "IMAGE", "slot_index": 0, "links": [link_id]},
            {"localized_name": "遮罩", "name": "MASK", "type": "MASK", "links": None},
        ],
        "title": title,
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": [filename, "image"],
    }


def build_workflow() -> dict:
    nodes = [
        {
            "id": 1,
            "type": "LoadVideo",
            "pos": [30, 0],
            "size": [430, 350],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [
                {"localized_name": "文件", "name": "file", "type": "COMBO", "widget": {"name": "file"}, "link": None}
            ],
            "outputs": [{"localized_name": "视频", "name": "VIDEO", "type": "VIDEO", "slot_index": 0, "links": [1]}],
            "title": "1. 原视频（2-10秒，自动上传百炼临时OSS）",
            "properties": {"Node name for S&R": "LoadVideo"},
            "widgets_values": ["three_person_asset_gateway_test_10s_24fps.mp4"],
        },
        image_node(2, "2A. 图1：替换车旁左侧人物 A", [40, 430], "three_person_asset_A_red_official.png", 2),
        image_node(3, "2B. 图2：替换车旁右侧人物 B", [40, 930], "three_person_asset_B_green_official.png", 3),
        image_node(4, "2C. 图3：替换灰西装单人人物 C", [40, 1430], "three_person_asset_C_black_warrior.png", 4),
        {
            "id": 5,
            "type": "CompanyWan27ThreePersonVideoEdit",
            "pos": [650, 580],
            "size": [850, 930],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
                {"localized_name": "原视频（2-10 秒）", "name": "video", "type": "VIDEO", "link": 1},
                {"localized_name": "图1：替换原视频人物 A", "name": "image_a", "type": "IMAGE", "link": 2},
                {"localized_name": "图2：替换原视频人物 B", "name": "image_b", "type": "IMAGE", "link": 3},
                {"localized_name": "图3：替换原视频人物 C", "name": "image_c", "type": "IMAGE", "link": 4},
                {"localized_name": "人物替换指令", "name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": None},
                {"localized_name": "模型", "name": "model", "type": "COMBO", "widget": {"name": "model"}, "link": None},
                {"localized_name": "分辨率", "name": "resolution", "type": "COMBO", "widget": {"name": "resolution"}, "link": None},
                {"localized_name": "截断时长（0=原视频）", "name": "duration", "type": "INT", "widget": {"name": "duration"}, "link": None},
                {"localized_name": "声音", "name": "audio_setting", "type": "COMBO", "widget": {"name": "audio_setting"}, "link": None},
                {"localized_name": "智能改写", "name": "prompt_extend", "type": "BOOLEAN", "widget": {"name": "prompt_extend"}, "link": None},
                {"localized_name": "水印", "name": "watermark", "type": "BOOLEAN", "widget": {"name": "watermark"}, "link": None},
                {"localized_name": "种子", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                {"localized_name": "负面提示词", "name": "negative_prompt", "type": "STRING", "widget": {"name": "negative_prompt"}, "link": None},
            ],
            "outputs": [
                {"localized_name": "视频", "name": "视频", "type": "VIDEO", "slot_index": 0, "links": [5]},
                {"localized_name": "视频路径", "name": "视频路径", "type": "STRING", "links": None},
            ],
            "title": "3. Wan 2.7 原视频三人物替换",
            "properties": {"Node name for S&R": "CompanyWan27ThreePersonVideoEdit"},
            "widgets_values": [
                PROMPT,
                "wan2.7-videoedit",
                "720P",
                0,
                "origin",
                True,
                False,
                0,
                "身份互换，串脸，面部融合，人物复制，人物消失，肢体畸形，额外手指，字幕，文字，Logo，水印",
            ],
            "color": "#35534a",
            "bgcolor": "#446b5f",
        },
        {
            "id": 6,
            "type": "SaveVideo",
            "pos": [1660, 800],
            "size": [390, 520],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
                {"localized_name": "视频", "name": "video", "type": "VIDEO", "link": 5},
                {"localized_name": "文件名前缀", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None},
                {"localized_name": "格式", "name": "format", "type": "COMBO", "widget": {"name": "format"}, "link": None},
                {"localized_name": "编码器", "name": "codec", "type": "COMBO", "widget": {"name": "codec"}, "link": None},
            ],
            "outputs": [{"localized_name": "video", "name": "video", "type": "VIDEO", "links": None}],
            "title": "4. 保存 Wan 2.7 三人物替换结果",
            "properties": {"Node name for S&R": "SaveVideo"},
            "widgets_values": ["video/Wan27_videoedit_three_people", "auto", "auto"],
        },
    ]
    return {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": 6,
        "last_link_id": 5,
        "nodes": nodes,
        "links": [
            [1, 1, 0, 5, 0, "VIDEO"],
            [2, 2, 0, 5, 1, "IMAGE"],
            [3, 3, 0, 5, 2, "IMAGE"],
            [4, 4, 0, 5, 3, "IMAGE"],
            [5, 5, 0, 6, 0, "VIDEO"],
        ],
        "groups": [
            {"id": 1, "title": "A. 原视频 + 三张人物参考图", "bounding": [-10, -50, 500, 1970], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 2, "title": "B. Wan 2.7 VideoEdit：视频→百炼临时OSS，图片→Base64", "bounding": [580, 500, 980, 1080], "color": "#4f7f69", "font_size": 24, "flags": {}},
            {"id": 3, "title": "C. 保存结果（点击执行会提交付费任务）", "bounding": [1600, 720, 520, 700], "color": "#8f6f4e", "font_size": 24, "flags": {}},
        ],
        "config": {},
        "extra": {
            "workflow_note": "原视频自动上传百炼48小时临时OSS，三张参考图Base64直传；不使用TOS或火山资产。输入视频须为2-10秒。",
            "ds": {"scale": 0.52, "offset": [110, 70]},
            "frontendVersion": "1.45.20",
        },
        "version": 0.4,
    }


def build_full_workflow() -> dict:
    segments_json = "[\n  {\n    \"start_frame\": 0,\n    \"end_frame\": 195\n  },\n  {\n    \"start_frame\": 196,\n    \"end_frame\": 373\n  },\n  {\n    \"start_frame\": 374,\n    \"end_frame\": 483\n  }\n]"
    negative_prompt = "身份互换，串脸，面部融合，人物复制，人物消失，肢体畸形，额外手指，字幕，文字，Logo，水印"

    def wan_node(node_id: int, title: str, position: list[int], links: list[int], output_links: list[int], segment_note: str) -> dict:
        return {
            "id": node_id,
            "type": "CompanyWan27ThreePersonVideoEdit",
            "pos": position,
            "size": [850, 850],
            "flags": {},
            "order": node_id - 1,
            "mode": 0,
            "inputs": [
                {"localized_name": "原视频（2-10 秒）", "name": "video", "type": "VIDEO", "link": links[0]},
                {"localized_name": "图1：替换原视频人物 A", "name": "image_a", "type": "IMAGE", "link": links[1]},
                {"localized_name": "图2：替换原视频人物 B", "name": "image_b", "type": "IMAGE", "link": links[2]},
                {"localized_name": "图3：替换原视频人物 C", "name": "image_c", "type": "IMAGE", "link": links[3]},
                {"localized_name": "人物替换指令", "name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": None},
                {"localized_name": "模型", "name": "model", "type": "COMBO", "widget": {"name": "model"}, "link": None},
                {"localized_name": "分辨率", "name": "resolution", "type": "COMBO", "widget": {"name": "resolution"}, "link": None},
                {"localized_name": "截断时长（0=原视频）", "name": "duration", "type": "INT", "widget": {"name": "duration"}, "link": None},
                {"localized_name": "声音", "name": "audio_setting", "type": "COMBO", "widget": {"name": "audio_setting"}, "link": None},
                {"localized_name": "智能改写", "name": "prompt_extend", "type": "BOOLEAN", "widget": {"name": "prompt_extend"}, "link": None},
                {"localized_name": "水印", "name": "watermark", "type": "BOOLEAN", "widget": {"name": "watermark"}, "link": None},
                {"localized_name": "种子", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                {"localized_name": "负面提示词", "name": "negative_prompt", "type": "STRING", "widget": {"name": "negative_prompt"}, "link": None},
            ],
            "outputs": [
                {"localized_name": "视频", "name": "视频", "type": "VIDEO", "slot_index": 0, "links": output_links},
                {"localized_name": "视频路径", "name": "视频路径", "type": "STRING", "links": None},
            ],
            "title": title,
            "properties": {"Node name for S&R": "CompanyWan27ThreePersonVideoEdit"},
            "widgets_values": [
                f"{PROMPT}\n\n{segment_note} 人物 A/B/C 映射必须与另外两段完全一致。",
                "wan2.7-videoedit",
                "720P",
                0,
                "origin",
                True,
                False,
                0,
                negative_prompt,
            ],
            "color": "#35534a",
            "bgcolor": "#446b5f",
        }

    def save_node(node_id: int, title: str, position: list[int], input_link: int, prefix: str) -> dict:
        return {
            "id": node_id,
            "type": "SaveVideo",
            "pos": position,
            "size": [390, 480],
            "flags": {},
            "order": node_id - 1,
            "mode": 0,
            "inputs": [
                {"localized_name": "视频", "name": "video", "type": "VIDEO", "link": input_link},
                {"localized_name": "文件名前缀", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None},
                {"localized_name": "格式", "name": "format", "type": "COMBO", "widget": {"name": "format"}, "link": None},
                {"localized_name": "编码器", "name": "codec", "type": "COMBO", "widget": {"name": "codec"}, "link": None},
            ],
            "outputs": [{"localized_name": "video", "name": "video", "type": "VIDEO", "links": None}],
            "title": title,
            "properties": {"Node name for S&R": "SaveVideo"},
            "widgets_values": [prefix, "auto", "auto"],
        }

    source = {
        "id": 1,
        "type": "LoadVideo",
        "pos": [30, 0],
        "size": [430, 350],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"localized_name": "文件", "name": "file", "type": "COMBO", "widget": {"name": "file"}, "link": None}],
        "outputs": [{"localized_name": "视频", "name": "VIDEO", "type": "VIDEO", "slot_index": 0, "links": [1, 17]}],
        "title": "1. 完整原视频（20.5秒 / 484帧）",
        "properties": {"Node name for S&R": "LoadVideo"},
        "widgets_values": ["three_person_face_swap_source.mp4"],
    }
    images = [
        image_node(2, "2A. 图1：固定替换人物 A", [40, 430], "three_person_asset_A_red_official.png", 5),
        image_node(3, "2B. 图2：固定替换人物 B", [40, 930], "three_person_asset_B_green_official.png", 6),
        image_node(4, "2C. 图3：固定替换人物 C", [40, 1430], "three_person_asset_C_black_warrior.png", 7),
    ]
    images[0]["outputs"][0]["links"] = [5, 8, 11]
    images[1]["outputs"][0]["links"] = [6, 9, 12]
    images[2]["outputs"][0]["links"] = [7, 10, 13]
    split = {
        "id": 5,
        "type": "CompanyWan27SplitThreeSegments",
        "pos": [620, 600],
        "size": [660, 560],
        "flags": {},
        "order": 4,
        "mode": 0,
        "inputs": [
            {"localized_name": "完整原视频", "name": "video", "type": "VIDEO", "link": 1},
            {"localized_name": "三段帧范围 JSON", "name": "segments_json", "type": "STRING", "widget": {"name": "segments_json"}, "link": None},
            {"localized_name": "强制重新切分", "name": "force_resplit", "type": "BOOLEAN", "widget": {"name": "force_resplit"}, "link": None},
        ],
        "outputs": [
            {"localized_name": "第1段：0-195帧", "name": "第1段：0-195帧", "type": "VIDEO", "slot_index": 0, "links": [2]},
            {"localized_name": "第2段：196-373帧", "name": "第2段：196-373帧", "type": "VIDEO", "slot_index": 1, "links": [3]},
            {"localized_name": "第3段：374-483帧", "name": "第3段：374-483帧", "type": "VIDEO", "slot_index": 2, "links": [4]},
            {"localized_name": "切分报告 JSON", "name": "切分报告 JSON", "type": "STRING", "links": None},
        ],
        "title": "3. 本地拆分：输出三个真实视频分支（不付费）",
        "properties": {"Node name for S&R": "CompanyWan27SplitThreeSegments"},
        "widgets_values": [segments_json, False],
        "color": "#37474f",
        "bgcolor": "#52656f",
    }
    nodes = [
        source,
        *images,
        split,
        wan_node(6, "4A. 第1段 Wan 2.7 生成（0-195帧 / 约8.32秒）", [1430, -100], [2, 5, 6, 7], [14, 18], "这是完整视频第 1/3 段（0-195 帧）。"),
        wan_node(7, "4B. 第2段 Wan 2.7 生成（196-373帧 / 约7.55秒）", [1430, 900], [3, 8, 9, 10], [15, 19], "这是完整视频第 2/3 段（196-373 帧）。"),
        wan_node(8, "4C. 第3段 Wan 2.7 生成（374-483帧 / 约4.67秒）", [1430, 1900], [4, 11, 12, 13], [16, 20], "这是完整视频第 3/3 段（374-483 帧）。"),
        save_node(9, "5A. 检查并保存第1段生成结果", [2380, 100], 14, "video/Wan27_full20s_segment_01"),
        save_node(10, "5B. 检查并保存第2段生成结果", [2380, 1100], 15, "video/Wan27_full20s_segment_02"),
        save_node(11, "5C. 检查并保存第3段生成结果", [2380, 2100], 16, "video/Wan27_full20s_segment_03"),
        {
            "id": 12,
            "type": "CompanyWan27MergeThreeSegments",
            "pos": [2980, 850],
            "size": [790, 720],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [
                {"localized_name": "完整原视频（用于帧率和音频）", "name": "original_video", "type": "VIDEO", "link": 17},
                {"localized_name": "第1段 Wan 结果", "name": "segment_1", "type": "VIDEO", "link": 18},
                {"localized_name": "第2段 Wan 结果", "name": "segment_2", "type": "VIDEO", "link": 19},
                {"localized_name": "第3段 Wan 结果", "name": "segment_3", "type": "VIDEO", "link": 20},
                {"localized_name": "三段帧范围 JSON", "name": "segments_json", "type": "STRING", "widget": {"name": "segments_json"}, "link": None},
                {"localized_name": "强制重新合并", "name": "force_remerge", "type": "BOOLEAN", "widget": {"name": "force_remerge"}, "link": None},
            ],
            "outputs": [
                {"localized_name": "完整合并视频", "name": "完整合并视频", "type": "VIDEO", "slot_index": 0, "links": [21]},
                {"localized_name": "完整视频路径", "name": "完整视频路径", "type": "STRING", "links": None},
                {"localized_name": "合并报告 JSON", "name": "合并报告 JSON", "type": "STRING", "links": None},
            ],
            "title": "6. 三段按原帧数合并 + 恢复完整原音频",
            "properties": {"Node name for S&R": "CompanyWan27MergeThreeSegments"},
            "widgets_values": [segments_json, False],
            "color": "#5b4732",
            "bgcolor": "#806348",
        },
        save_node(13, "7. 保存最终 20.5 秒 / 484帧结果", [3980, 970], 21, "video/Wan27_videoedit_three_people_full_20s"),
    ]
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, FULL_WORKFLOW_NAME)),
        "revision": 1,
        "last_node_id": 13,
        "last_link_id": 21,
        "nodes": nodes,
        "links": [
            [1, 1, 0, 5, 0, "VIDEO"],
            [2, 5, 0, 6, 0, "VIDEO"],
            [3, 5, 1, 7, 0, "VIDEO"],
            [4, 5, 2, 8, 0, "VIDEO"],
            [5, 2, 0, 6, 1, "IMAGE"],
            [6, 3, 0, 6, 2, "IMAGE"],
            [7, 4, 0, 6, 3, "IMAGE"],
            [8, 2, 0, 7, 1, "IMAGE"],
            [9, 3, 0, 7, 2, "IMAGE"],
            [10, 4, 0, 7, 3, "IMAGE"],
            [11, 2, 0, 8, 1, "IMAGE"],
            [12, 3, 0, 8, 2, "IMAGE"],
            [13, 4, 0, 8, 3, "IMAGE"],
            [14, 6, 0, 9, 0, "VIDEO"],
            [15, 7, 0, 10, 0, "VIDEO"],
            [16, 8, 0, 11, 0, "VIDEO"],
            [17, 1, 0, 12, 0, "VIDEO"],
            [18, 6, 0, 12, 1, "VIDEO"],
            [19, 7, 0, 12, 2, "VIDEO"],
            [20, 8, 0, 12, 3, "VIDEO"],
            [21, 12, 0, 13, 0, "VIDEO"],
        ],
        "groups": [
            {"id": 1, "title": "A. 完整原视频 + 三张固定人物参考图", "bounding": [-10, -60, 500, 1980], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 2, "title": "B. 本地切分：0-195 / 196-373 / 374-483 帧", "bounding": [550, 500, 800, 800], "color": "#5d7280", "font_size": 24, "flags": {}},
            {"id": 3, "title": "C1. 第1段：独立付费生成 + 中间结果", "bounding": [1360, -180, 1470, 980], "color": "#4f7f69", "font_size": 24, "flags": {}},
            {"id": 4, "title": "C2. 第2段：独立付费生成 + 中间结果", "bounding": [1360, 820, 1470, 980], "color": "#4f7f69", "font_size": 24, "flags": {}},
            {"id": 5, "title": "C3. 第3段：独立付费生成 + 中间结果", "bounding": [1360, 1820, 1470, 980], "color": "#4f7f69", "font_size": 24, "flags": {}},
            {"id": 6, "title": "D. 精确合并 484 帧 + 恢复原音频 + 最终保存", "bounding": [2900, 760, 1570, 1100], "color": "#8f6f4e", "font_size": 24, "flags": {}},
        ],
        "config": {},
        "extra": {
            "workflow_note": (
                "可视化完整 20.5 秒版本：本地拆分节点明确输出三个合法 Wan2.7 输入；"
                "三个独立生成节点分别提交付费任务，并各自保存中间结果；最后统一规范化到原帧数、合并为 484 帧并恢复原音频。"
            ),
            "ds": {"scale": 0.28, "offset": [80, 70]},
            "frontendVersion": "1.45.20",
        },
        "version": 0.4,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = root / "user" / "default" / "workflows" / WORKFLOW_NAME
    output.write_text(json.dumps(build_workflow(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    full_output = root / "user" / "default" / "workflows" / FULL_WORKFLOW_NAME
    full_output.write_text(json.dumps(build_full_workflow(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(full_output)


if __name__ == "__main__":
    main()
