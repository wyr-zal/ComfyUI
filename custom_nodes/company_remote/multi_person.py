from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


PERSON_LABELS = ("A", "B", "C")

MULTI_PERSON_ANALYZER_SKILL = """你是多人图像分析与 GPT Image 2 编辑提示词编排专家。
你必须先识别输入图片中的主要人物数量，再建立唯一且稳定的人物身份表，并为后续独立处理与重新合成生成提示词。

人物编号规则：
1. 按人物在画面中的中心位置从左到右编号为 A、B、C。
2. 横向位置接近时，按从前景到后景编号。
3. 不把海报、照片、屏幕画面、雕像、倒影或纯背景路人重复计为主要人物。
4. 每个人必须记录发型、发色、年龄感、脸部气质、体型、服装、配饰、道具和输入图的视觉媒介。
5. 所有提示词必须保持人物与特征一一对应，不得交换、融合、删除、复制或新增人物。

输出必须是单个 JSON 对象，不要 Markdown、代码围栏、解释或标题。person_count 必须填写实际识别数量，即使为 0 或超过 3 也不得篡改。"""

MULTI_PERSON_REPAIR_SKILL = """你只负责把一段多人图像分析结果修复成指定 JSON 结构。
不得重新分析图片，不得改变人物数量、人物编号、身份特征或原有处理意图。
只输出单个合法 JSON 对象，不要 Markdown、代码围栏、解释或标题。"""


class MultiPersonAnalysisError(ValueError):
    """Base error for invalid multi-person analysis responses."""


class MultiPersonFormatError(MultiPersonAnalysisError):
    """Raised when the response cannot be parsed or misses required fields."""


class MultiPersonCountError(MultiPersonAnalysisError):
    """Raised when the detected main-person count is outside the supported range."""


@dataclass(frozen=True)
class MultiPersonAnalysis:
    person_count: int
    identity_manifest: str
    person_a_prompt: str
    person_b_prompt: str
    person_c_prompt: str
    background_prompt: str
    final_prompt: str

    def node_outputs(self) -> tuple[int, str, str, str, str, str, str]:
        return (
            self.person_count,
            self.identity_manifest,
            self.person_a_prompt,
            self.person_b_prompt,
            self.person_c_prompt,
            self.background_prompt,
            self.final_prompt,
        )


def build_analysis_request(modification_target: str) -> str:
    target = str(modification_target or "").strip()
    if not target:
        raise ValueError("转换目标不能为空。")

    return f"""转换目标：
{target}

请分析参考图片并返回以下完整 JSON 结构：
{{
  "person_count": 2,
  "identity_manifest": "统一身份表，先说明编号顺序，再逐一描述人物 A/B/C 的稳定身份特征",
  "person_a_prompt": "人物 A 的独立 GPT Image 2 编辑提示词",
  "person_b_prompt": "人物 B 的独立 GPT Image 2 编辑提示词；不存在时为空字符串",
  "person_c_prompt": "人物 C 的独立 GPT Image 2 编辑提示词；不存在时为空字符串",
  "background_prompt": "移除全部人物并按转换目标处理背景的 GPT Image 2 编辑提示词",
  "final_prompt": "按原图人物关系重新合成背景与处理后人物的 GPT Image 2 编辑提示词"
}}

提示词编写要求：
- 每个人物提示词必须明确人物编号及可见身份特征，只处理该人物，排除其他人物。
- 人物处理结果应为完整、清晰、便于重新合成的单人图；复杂遮挡部分按身份和身体结构合理补全，并置于与该人物颜色区别明显的均匀纯色色键背景。
- 保持每个人的核心身份、服装、配饰和道具对应关系，但允许根据转换目标创造性调整动作、姿态、表情和镜头表现。
- background_prompt 必须移除全部人物、人体局部、倒影和人物阴影，并合理补全遮挡区域。
- final_prompt 必须说明参考图顺序：图 1 是原始构图，图 2 是处理后背景，后续图片依次是人物 A、B、C；只引用实际存在的人物。
- final_prompt 必须保持实际人数与身份一一对应，但允许重新设计更自然且能展示人物特点的动作、表情和互动。
- 不得在任何提示词中把 A/B/C 的身份特征互换。
"""


def build_repair_request(raw_text: str, error: Exception) -> str:
    return f"""目标 JSON 字段固定为：
person_count, identity_manifest, person_a_prompt, person_b_prompt, person_c_prompt, background_prompt, final_prompt。

约束：
- person_count 必须是实际识别到的整数，不得为了通过校验而改成人为数量。
- 实际存在的人物提示词不能为空；不存在的人物提示词使用空字符串。
- 其余文本字段必须是字符串。

原始返回：
{raw_text}

解析错误：
{error}
"""


def parse_multi_person_analysis(raw_text: str) -> MultiPersonAnalysis:
    data = _decode_json_object(raw_text)
    person_count = data.get("person_count")
    if isinstance(person_count, bool) or not isinstance(person_count, int):
        raise MultiPersonFormatError("person_count 必须是整数。")
    if person_count < 1:
        raise MultiPersonCountError(f"未识别到主要人物，实际数量为 {person_count}；当前工作流要求 1-3 人。")
    if person_count > 3:
        raise MultiPersonCountError(f"识别到 {person_count} 个主要人物；当前工作流最多支持 3 人。")

    identity_manifest = _required_string(data, "identity_manifest")
    prompts = {
        label: _optional_string(data, f"person_{label.lower()}_prompt")
        for label in PERSON_LABELS
    }
    for index, label in enumerate(PERSON_LABELS, start=1):
        if index <= person_count and not prompts[label]:
            raise MultiPersonFormatError(f"已识别人物 {label}，但 person_{label.lower()}_prompt 为空。")
        if index > person_count:
            prompts[label] = ""

    return MultiPersonAnalysis(
        person_count=person_count,
        identity_manifest=identity_manifest,
        person_a_prompt=prompts["A"],
        person_b_prompt=prompts["B"],
        person_c_prompt=prompts["C"],
        background_prompt=_required_string(data, "background_prompt"),
        final_prompt=_required_string(data, "final_prompt"),
    )


def _decode_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise MultiPersonFormatError("模型返回为空。")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _decode_embedded_json(text)
    if not isinstance(value, dict):
        raise MultiPersonFormatError("模型返回的 JSON 根节点必须是对象。")
    return value


def _decode_embedded_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise MultiPersonFormatError("模型返回中没有可解析的 JSON 对象。")


def _required_string(data: dict[str, Any], key: str) -> str:
    value = _optional_string(data, key)
    if not value:
        raise MultiPersonFormatError(f"字段 {key} 不能为空。")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise MultiPersonFormatError(f"字段 {key} 必须是字符串。")
    return value.strip()
