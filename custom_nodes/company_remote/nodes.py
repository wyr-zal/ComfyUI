from __future__ import annotations

import asyncio
import copy
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import IO, InputImpl, UI
from comfy_api_nodes.util import downscale_image_tensor, get_number_of_images, validate_string

from .client import (
    DEFAULT_OPENAI_TEXT_MODEL,
    generate_dashscope_image,
    generate_dashscope_video,
    generate_image,
    generate_openai_chat_text,
    generate_openai_image,
    generate_openai_image_prompt_text,
    generate_video,
    get_cached_openai_model_ids,
)
from .config_store import ConfigError, get_config, get_gpt_image_provider_config, load_configs
from .multi_person import (
    MULTI_PERSON_ANALYZER_SKILL,
    MULTI_PERSON_REPAIR_SKILL,
    MultiPersonCountError,
    MultiPersonFormatError,
    build_analysis_request,
    build_repair_request,
    parse_multi_person_analysis,
)
from .long_video import (
    AUTO_ASSET_STYLE_WESTERN,
    AUTO_ASSET_STYLE_ANIME,
    AUTO_ASSET_STYLE_CG_3D,
    AUTO_ASSET_STYLE_COMIC,
    AUTO_ASSET_STYLE_CUSTOM,
    AUTO_ASSET_STYLE_PHOTOREAL,
    BACKGROUND_IDS,
    PERSON_IDS,
    asset_image_inputs,
    analyze_long_video_job,
    analyze_asset_mapping,
    collect_long_video_results,
    connected_asset_dict,
    create_asset_manifest,
    detect_long_video_shots,
    generate_long_video_segments_parallel,
    build_long_video_auto_assets,
    build_asset_library_view,
    generate_long_video_pipeline,
    generate_long_video_segments,
    inspect_long_video_shots,
    load_long_video_assets,
    match_long_video_references,
    merge_long_video_job,
    plan_long_video_job,
    process_long_video,
    pack_long_video_auto_references,
    plan_long_video_auto_asset_job,
    MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
    MANUAL_BATCH_CONTRACT,
    _manual_batch_read_state,
    select_manual_batch_range,
    select_continuous_shot_range,
    select_long_video_length_range,
)


LongVideoAssetsType = IO.Custom("长视频资产清单")
LongVideoShotPlanType = IO.Custom("长视频镜头计划")
LongVideoJobType = IO.Custom("长视频分段任务")


SEEDREAM_MODEL_OPTIONS = [
    "seedream 5.0 lite",
    "seedream-4-5-251128",
    "seedream-4-0-250828",
]

SEEDREAM_PRESETS = [
    ("2048x2048 (1:1)", 2048, 2048),
    ("2304x1728 (4:3)", 2304, 1728),
    ("1728x2304 (3:4)", 1728, 2304),
    ("2560x1440 (16:9)", 2560, 1440),
    ("1440x2560 (9:16)", 1440, 2560),
    ("2496x1664 (3:2)", 2496, 1664),
    ("1664x2496 (2:3)", 1664, 2496),
    ("3024x1296 (21:9)", 3024, 1296),
    ("3072x3072 (1:1)", 3072, 3072),
    ("4096x4096 (1:1)", 4096, 4096),
    ("Custom", None, None),
]

SEEDANCE_MODEL_OPTIONS = [
    ("Seedance 2.0", ["480p", "720p", "1080p"]),
    ("Seedance 2.0 Fast", ["480p", "720p"]),
]

RATIO_OPTIONS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]
KLING_MODEL_OPTIONS = [
    "kling-v2-master",
    "kling-v2-5-turbo",
    "kling-v2-1-master",
    "kling-v2-1",
    "kling-v1-6",
]
KLING_MODE_OPTIONS = ["std", "pro"]
KLING_DURATION_OPTIONS = ["5", "10"]
KLING_ASPECT_RATIO_OPTIONS = ["16:9", "9:16", "1:1"]
ALIYUN_QWEN_IMAGE_MODELS = [
    "qwen-image-2.0-pro-2026-06-22",
    "qwen-image-2.0-pro-2026-04-22",
]
ALIYUN_TEXT_TO_VIDEO_MODELS = [
    "wan2.7-t2v-2026-06-12",
    "wan2.7-t2v-2026-04-25",
    "happyhorse-1.1-t2v",
    "happyhorse-1.0-t2v",
]
ALIYUN_IMAGE_TO_VIDEO_MODELS = [
    "wan2.7-i2v-2026-04-25",
    "happyhorse-1.1-i2v",
    "happyhorse-1.0-i2v",
]
ALIYUN_REFERENCE_TO_VIDEO_MODELS = [
    "wan2.7-r2v-2026-06-12",
    "happyhorse-1.1-r2v",
    "happyhorse-1.0-r2v",
]
ALIYUN_VIDEO_EDIT_MODELS = ["wan2.7-videoedit", "happyhorse-1.0-video-edit"]
ALIYUN_VIDEO_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]

PROVIDER_ALIASES = {
    "gptimage2": ["gptimage2", "gpt_image2", "gpt-image-2", "gpt image 2", "ai_zero_token"],
    "gpttext": ["gpttext", "gpt_text", "gpt-text", "openai_chat", "prompt_enhancer", "ai_zero_token_text"],
    "seedream": ["seedream", "seedream_image", "bytedance_seedream"],
    "seedance2": ["seedance2", "seedance", "seedance_2", "seedance 2.0", "bytedance_seedance"],
    "kling": ["kling", "kling_image_to_video"],
    "vidu": ["vidu", "viduq3", "vidu_q3"],
    "minimax_hailuo": ["minimax_hailuo", "minimax", "hailuo", "minimax-hailuo"],
    "aliyun_dashscope_image": ["aliyun_dashscope_image", "dashscope_image", "aliyun_image"],
    "aliyun_dashscope_video": ["aliyun_dashscope_video", "dashscope_video", "aliyun_video"],
}

GPT_IMAGE_PROVIDER_WISART = "WisArt"
GPT_IMAGE_PROVIDER_AI_ZERO_TOKEN = "AI-Zero-Token"
GPT_IMAGE_PROVIDER_OPTIONS = [GPT_IMAGE_PROVIDER_WISART, GPT_IMAGE_PROVIDER_AI_ZERO_TOKEN]


def _available_config_names() -> list[str]:
    return [str(item.get("name") or "") for item in load_configs(include_secret=False) if item.get("name")]


def _load_provider_config(provider: str):
    names = _available_config_names()
    lowered = {name.lower(): name for name in names}
    candidates = [*PROVIDER_ALIASES.get(provider, [provider]), "default"]
    for candidate in candidates:
        actual_name = lowered.get(candidate.lower())
        if actual_name:
            try:
                return get_config(actual_name)
            except ConfigError:
                break
    if len(names) == 1:
        return get_config(names[0])
    expected = "、".join(candidates)
    raise ValueError(
        f"Company Remote 未找到 {provider} 配置。请在 Company Remote 配置面板中创建 “{expected}” 之一，"
        "或只保留一个通用配置供所有公司节点使用。"
    )


def _parse_seedream_size(size_preset: str, width: int, height: int) -> tuple[int, int]:
    for label, preset_width, preset_height in SEEDREAM_PRESETS:
        if label == size_preset and preset_width and preset_height:
            return int(preset_width), int(preset_height)
    return int(width), int(height)


def _seedance2_text_inputs(resolutions: list[str], default_ratio: str = "16:9") -> list[Any]:
    return [
        IO.String.Input(
            "prompt",
            display_name="提示词",
            multiline=True,
            default="",
            tooltip="用于视频生成的提示词。",
        ),
        IO.Combo.Input(
            "resolution",
            options=resolutions,
            display_name="分辨率",
            tooltip="输出视频分辨率。",
        ),
        IO.Combo.Input(
            "ratio",
            options=RATIO_OPTIONS,
            display_name="比例",
            default=default_ratio,
            tooltip="输出视频比例。",
        ),
        IO.Int.Input(
            "duration",
            display_name="时长",
            default=7,
            min=4,
            max=15,
            step=1,
            tooltip="输出视频时长，单位秒（4-15）。",
            display_mode=IO.NumberDisplay.slider,
        ),
        IO.Boolean.Input(
            "generate_audio",
            display_name="生成音频",
            default=True,
            tooltip="为输出视频生成音频。",
        ),
    ]


def _seedance2_reference_inputs(resolutions: list[str], default_ratio: str = "adaptive") -> list[Any]:
    return [
        *_seedance2_text_inputs(resolutions, default_ratio=default_ratio),
        IO.Autogrow.Input(
            "reference_images",
            display_name="参考图片",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("reference_image", display_name="参考图"),
                names=[f"image_{index}" for index in range(1, 10)],
                min=0,
            ),
        ),
        IO.Autogrow.Input(
            "reference_videos",
            display_name="参考视频",
            template=IO.Autogrow.TemplateNames(
                IO.Video.Input("reference_video", display_name="参考视频"),
                names=[f"video_{index}" for index in range(1, 4)],
                min=0,
            ),
        ),
        IO.Boolean.Input(
            "auto_downscale",
            display_name="自动降采样",
            default=True,
            optional=True,
            tooltip="传递给支持参考视频自动降采样的平台。",
        ),
        IO.Boolean.Input(
            "auto_upscale",
            display_name="自动升采样",
            default=False,
            optional=True,
            advanced=True,
            tooltip="传递给支持参考视频自动升采样的平台。",
        ),
        IO.Autogrow.Input(
            "reference_assets",
            display_name="资源 ID",
            template=IO.Autogrow.TemplateNames(
                IO.String.Input("reference_asset", display_name="资源 ID"),
                names=[f"asset_{index}" for index in range(1, 10)],
                min=0,
            ),
        ),
    ]


def _dynamic_seedance_options(input_factory, *, default_ratio: str = "16:9") -> list[IO.DynamicCombo.Option]:
    return [
        IO.DynamicCombo.Option(label, input_factory(resolutions, default_ratio=default_ratio))
        for label, resolutions in SEEDANCE_MODEL_OPTIONS
    ]


def _dict_values(data: Any) -> list[Any]:
    if isinstance(data, dict):
        return [value for value in data.values() if value not in (None, "")]
    if isinstance(data, list):
        return [value for value in data if value not in (None, "")]
    return []


def _gpt_image_shared_inputs():
    return [
        IO.Combo.Input(
            "quality",
            default="auto",
            options=["auto", "low", "medium", "high"],
            tooltip="Image quality, affects cost and generation time.",
        ),
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, 17)],
                min=0,
            ),
            tooltip="Optional reference image(s) for image editing. Up to 16 images.",
        ),
        IO.Mask.Input(
            "mask",
            optional=True,
            tooltip="Optional mask for inpainting (white areas will be replaced). Requires exactly one reference image.",
        ),
    ]


def _gpt_image_legacy_model_inputs():
    return [
        IO.Combo.Input(
            "size",
            default="auto",
            options=["auto", "1024x1024", "1024x1536", "1536x1024"],
            tooltip="Image size.",
        ),
        IO.Combo.Input(
            "background",
            default="auto",
            options=["auto", "opaque", "transparent"],
            tooltip="Return image with or without background.",
        ),
        *_gpt_image_shared_inputs(),
    ]


def _gpt_text_model_input(input_name: str = "model", *, display_name: str = "模型"):
    models = get_cached_openai_model_ids()
    default = DEFAULT_OPENAI_TEXT_MODEL if DEFAULT_OPENAI_TEXT_MODEL in models else models[0]
    return IO.Combo.Input(
        input_name,
        options=models,
        default=default,
        display_name=display_name,
        tooltip="从 AI-Zero-Token /v1/models 动态加载；连接失败时使用最后一次成功缓存。",
    )


def _validate_gpt_text_model(model: str):
    if not isinstance(model, str) or not model.strip():
        return "模型不能为空。"
    return True


class CompanyPromptEnhancer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyPromptEnhancer",
            display_name="公司提示词优化",
            category="company-remote/text/OpenAI",
            description="通过 AI-Zero-Token 本地 OpenAI 兼容接口，把普通提示词优化成高级提示词。",
            search_aliases=["Company Prompt Enhancer", "AI Zero Token Text", "OpenAI ChatGPT", "prompt enhance"],
            inputs=[
                IO.String.Input(
                    "skill",
                    display_name="Skill",
                    multiline=True,
                    default="",
                    tooltip="提示词优化规则，作为 system message 发送给本地文生文模型。",
                ),
                IO.String.Input(
                    "user_prompt",
                    display_name="用户提示词",
                    multiline=True,
                    default="",
                    tooltip="用户输入的普通提示词，作为 user message 发送给本地文生文模型。",
                ),
                _gpt_text_model_input(),
                IO.Float.Input(
                    "temperature",
                    display_name="温度",
                    default=0.2,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="控制输出随机性。提示词优化建议保持较低。",
                ),
                IO.Int.Input(
                    "max_tokens",
                    display_name="最大输出 Tokens",
                    default=1000,
                    min=16,
                    max=16384,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="限制优化后提示词的最大输出长度。",
                ),
            ],
            outputs=[IO.String.Output(display_name="优化提示词")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, model: str):
        return _validate_gpt_text_model(model)

    @classmethod
    async def execute(
        cls,
        skill: str,
        user_prompt: str,
        model: str = DEFAULT_OPENAI_TEXT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ):
        text = await asyncio.to_thread(
            generate_openai_chat_text,
            _load_provider_config("gpttext"),
            skill=skill,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return IO.NodeOutput(text, ui={"text": (text,)})


class CompanyImagePromptEnhancer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyImagePromptEnhancer",
            display_name="图片提示词优化节点",
            category="company-remote/text/OpenAI",
            description="分析输入图片和修改目标，通过 AI-Zero-Token 生成针对当前画面的 GPT Image 2 提示词。",
            search_aliases=["Image Prompt Enhancer", "Vision Prompt Enhancer", "AI Zero Token Vision"],
            inputs=[
                IO.String.Input(
                    "skill",
                    display_name="Skill",
                    multiline=True,
                    default="",
                    tooltip="看图分析和提示词编写规则，作为 system message 发送。",
                ),
                IO.String.Input(
                    "modification_target",
                    display_name="修改目标",
                    multiline=True,
                    default="",
                    tooltip="说明需要基于当前图片完成的修改目标。",
                ),
                IO.Image.Input(
                    "image",
                    display_name="参考图片",
                    tooltip="必填。IMAGE batch 中的全部图片都会按原尺寸发送给模型分析。",
                ),
                _gpt_text_model_input(),
                IO.Float.Input(
                    "temperature",
                    display_name="温度",
                    default=0.2,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="控制输出随机性。",
                ),
                IO.Int.Input(
                    "max_tokens",
                    display_name="最大输出 Tokens",
                    default=1000,
                    min=16,
                    max=16384,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="限制定制提示词的最大输出长度。",
                ),
            ],
            outputs=[IO.String.Output(display_name="优化提示词")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, model: str):
        return _validate_gpt_text_model(model)

    @classmethod
    async def execute(
        cls,
        skill: str,
        modification_target: str,
        image: Any,
        model: str = DEFAULT_OPENAI_TEXT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ):
        text = await asyncio.to_thread(
            generate_openai_image_prompt_text,
            _load_provider_config("gpttext"),
            skill=skill,
            modification_target=modification_target,
            image=image,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return IO.NodeOutput(text, ui={"text": (text,)})


class CompanyPersistentPromptDisplay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyPersistentPromptDisplay",
            display_name="持久化提示词显示",
            category="company-remote/text/utilities",
            description="显示并原样传递 STRING；执行结果随工作流保存并在重新打开时恢复。",
            search_aliases=["Persistent Prompt Display", "Persistent Text", "保存提示词"],
            inputs=[
                IO.String.Input(
                    "text",
                    display_name="提示词",
                    multiline=True,
                    force_input=True,
                    tooltip="连接提示词优化节点的 STRING 输出。",
                ),
            ],
            outputs=[IO.String.Output(display_name="提示词")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, text: str):
        value = str(text or "")
        return IO.NodeOutput(value, ui={"text": (value,)})


class CompanyFixedColumnImagePreview(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyFixedColumnImagePreview",
            display_name="固定列数图片预览",
            category="company-remote/image/utilities",
            description="在节点内按固定列数铺满宽度显示图片，图片较多时使用纵向滚动。",
            inputs=[
                IO.Image.Input("images", display_name="图像"),
                IO.Combo.Input("columns", options=["2", "3"], default="3", display_name="每行图片数量"),
                IO.Int.Input("gap", display_name="图片间距", default=8, min=0, max=64, step=1),
            ],
            outputs=[IO.Image.Output("images", display_name="原始图片")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, columns: str, gap: int = 8):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1:
            raise ValueError("固定列数预览需要至少一张 IMAGE。")
        preview = UI.PreviewImage(images, cls=cls)
        return IO.NodeOutput(
            images,
            ui={
                "fixed_grid_images": preview.values,
                "fixed_grid_columns": (int(columns),),
                "fixed_grid_gap": (int(gap),),
            },
        )


class CompanyMultiPersonPromptAnalyzer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyMultiPersonPromptAnalyzer",
            display_name="多人角色识别与提示词拆分",
            category="company-remote/text/OpenAI",
            description="一次识别图片中的 1-3 个主要人物，统一编号并输出人物、背景和合成提示词。",
            search_aliases=[
                "Multi Person Prompt Analyzer",
                "Character A B C Analyzer",
                "多人识别",
                "人物 A B C",
            ],
            inputs=[
                IO.String.Input(
                    "modification_target",
                    display_name="转换目标",
                    multiline=True,
                    default=(
                        "将输入画面转换为具有真实摄影质感的真人场景。保持每个人物可辨识的发型、"
                        "服装、配饰、道具和角色气质，允许重新设计自然动作、表情、镜头和环境互动。"
                    ),
                    tooltip="说明人物与背景需要完成的整体转换目标。",
                ),
                IO.Image.Input(
                    "image",
                    display_name="原始图片",
                    tooltip="必填。一次分析并统一识别人物 A、B、C。",
                ),
                _gpt_text_model_input(),
                IO.Float.Input(
                    "temperature",
                    display_name="温度",
                    default=0.2,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="人物识别建议使用较低温度。",
                ),
                IO.Int.Input(
                    "max_tokens",
                    display_name="最大输出 Tokens",
                    default=3000,
                    min=512,
                    max=16384,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="用于完整输出身份表及全部分支提示词。",
                ),
            ],
            outputs=[
                IO.Int.Output(display_name="人物数量"),
                IO.String.Output(display_name="统一身份表"),
                IO.String.Output(display_name="人物 A 提示词"),
                IO.String.Output(display_name="人物 B 提示词"),
                IO.String.Output(display_name="人物 C 提示词"),
                IO.String.Output(display_name="背景处理提示词"),
                IO.String.Output(display_name="最终合成提示词"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, model: str):
        return _validate_gpt_text_model(model)

    @classmethod
    async def execute(
        cls,
        modification_target: str,
        image: Any,
        model: str = DEFAULT_OPENAI_TEXT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ):
        config = _load_provider_config("gpttext")
        raw_text = await asyncio.to_thread(
            generate_openai_image_prompt_text,
            config,
            skill=MULTI_PERSON_ANALYZER_SKILL,
            modification_target=build_analysis_request(modification_target),
            image=image,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            analysis = parse_multi_person_analysis(raw_text)
        except MultiPersonCountError:
            raise
        except MultiPersonFormatError as first_error:
            repaired_text = await asyncio.to_thread(
                generate_openai_chat_text,
                config,
                skill=MULTI_PERSON_REPAIR_SKILL,
                user_prompt=build_repair_request(raw_text, first_error),
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            try:
                analysis = parse_multi_person_analysis(repaired_text)
            except MultiPersonFormatError as second_error:
                raise ValueError(
                    f"多人识别返回格式无效，自动修复后仍无法解析：{second_error}"
                ) from second_error

        outputs = analysis.node_outputs()
        return IO.NodeOutput(
            *outputs,
            ui={
                "person_count": (analysis.person_count,),
                "identity_manifest": (analysis.identity_manifest,),
                "person_a_prompt": (analysis.person_a_prompt,),
                "person_b_prompt": (analysis.person_b_prompt,),
                "person_c_prompt": (analysis.person_c_prompt,),
                "background_prompt": (analysis.background_prompt,),
                "final_prompt": (analysis.final_prompt,),
            },
        )


class CompanyLongVideoAssetManifest(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoAssetManifest",
            display_name="长视频欧美化资产清单",
            category="company-remote/video/long video",
            description="保存已确认的人物与背景欧美化参考图，并生成可供长视频批处理复用的 manifest。",
            search_aliases=["Long Video Asset Manifest", "长视频参考素材", "人物背景资产清单"],
            inputs=[
                IO.String.Input("asset_name", display_name="资产名称", default="long_video_assets"),
                IO.String.Input(
                    "mapping_json",
                    display_name="人物/背景映射 JSON",
                    multiline=True,
                    default=(
                        '{\n'
                        '  "people": {"A": {"source": "原人物 A", "identity": ""}},\n'
                        '  "backgrounds": {"BG01": {"source": "原背景 BG01", "description": ""}},\n'
                        '  "mapping": "原人物 A -> 欧美化人物 A"\n'
                        '}'
                    ),
                    tooltip="可由 AI 生成后人工修正；只保存说明和映射，不保存原始输入媒体。",
                ),
                *asset_image_inputs(),
            ],
            outputs=[
                IO.String.Output(display_name="资产清单 JSON"),
                IO.String.Output(display_name="manifest 路径"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, asset_name: str, mapping_json: str, **kwargs):
        people = connected_asset_dict(kwargs, "person_", PERSON_IDS)
        backgrounds = connected_asset_dict(kwargs, "", BACKGROUND_IDS)
        manifest, path = create_asset_manifest(
            asset_name=asset_name,
            mapping_json=mapping_json,
            people=people,
            backgrounds=backgrounds,
        )
        return IO.NodeOutput(manifest, path, ui={"text": (manifest,), "path": (path,)})


class CompanyLongVideoMappingAnalyzer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoMappingAnalyzer",
            display_name="长视频人物与背景映射分析",
            category="company-remote/video/long video",
            description="分析已确认的欧美化人物 A/B/C 和背景 BG01-BG08，生成可人工修改的完整资产映射 JSON。",
            search_aliases=["Long Video Asset Mapping", "人物背景映射确认"],
            inputs=[
                IO.Image.Input("person_A", display_name="欧美化人物 A"),
                IO.Image.Input("person_B", display_name="欧美化人物 B", optional=True),
                IO.Image.Input("person_C", display_name="欧美化人物 C", optional=True),
                *[IO.Image.Input(background_id, display_name=f"欧美化背景 {background_id}", optional=True) for background_id in BACKGROUND_IDS],
                _gpt_text_model_input("analysis_model", display_name="分析模型"),
            ],
            outputs=[IO.String.Output(display_name="人物与背景映射 JSON")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, analysis_model: str):
        return _validate_gpt_text_model(analysis_model)

    @classmethod
    async def execute(
        cls,
        person_A: Any,
        person_B: Any = None,
        person_C: Any = None,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        **kwargs,
    ):
        mapping = await asyncio.to_thread(
            analyze_asset_mapping,
            people={"A": person_A, "B": person_B, "C": person_C},
            backgrounds=connected_asset_dict(kwargs, "", BACKGROUND_IDS),
            model=analysis_model,
        )
        return IO.NodeOutput(mapping, ui={"text": (mapping,)})


def _normalize_long_video_engine_model(engine: str, model: str) -> tuple[str, str]:
    normalized_engine = "wan" if str(engine).lower().startswith("wan") else "seedance"
    normalized_model = str(model).strip()
    if normalized_engine == "wan":
        normalized_model = "wan2.7-r2v-2026-06-12"
    elif not normalized_model.lower().startswith("seedance"):
        normalized_model = "Seedance 2.0 Fast"
    return normalized_engine, normalized_model


class CompanyLongVideoAssetLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoAssetLoader",
            display_name="读取长视频资产清单",
            category="company-remote/video/long video/stages",
            description="读取并校验第一阶段保存的人物、背景资产清单。",
            inputs=[
                IO.String.Input(
                    "assets_manifest",
                    display_name="资产清单 JSON 或路径",
                    multiline=True,
                    default="",
                ),
                *asset_image_inputs(),
            ],
            outputs=[
                LongVideoAssetsType.Output("assets", display_name="已加载资产"),
                IO.String.Output("summary", display_name="资产摘要 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, assets_manifest: str, **kwargs):
        assets, summary = load_long_video_assets(
            assets_manifest,
            people=connected_asset_dict(kwargs, "person_", PERSON_IDS),
            backgrounds=connected_asset_dict(kwargs, "", BACKGROUND_IDS),
        )
        return IO.NodeOutput(assets, summary, ui={"text": (summary,)})


class CompanyLongVideoShotDetector(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoShotDetector",
            display_name="长视频镜头检测",
            category="company-remote/video/long video/stages",
            description="识别硬切和淡入淡出，输出逻辑镜头、检测记录及镜头起始帧预览。",
            inputs=[
                IO.Video.Input("video", display_name="长视频"),
                IO.Combo.Input(
                    "mode",
                    options=["镜头优先（推荐）", "固定时长"],
                    default="镜头优先（推荐）",
                    display_name="切分模式",
                ),
                IO.Combo.Input("fixed_duration", options=["10", "15"], default="10", display_name="固定分段时长（秒）"),
                IO.Combo.Input("sensitivity", options=["低", "标准", "高"], default="标准", display_name="镜头检测灵敏度"),
                IO.Boolean.Input("use_audio_silence", display_name="用音频停顿辅助长镜头切分", default=True),
                IO.Boolean.Input("auto_fallback", display_name="检测失败自动改用固定切分", default=True),
            ],
            outputs=[
                LongVideoShotPlanType.Output("shot_plan", display_name="镜头计划"),
                IO.String.Output("shots_json", display_name="镜头检测 JSON"),
                IO.Image.Output("previews", display_name="镜头起始帧预览"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        mode: str,
        fixed_duration: str,
        sensitivity: str,
        use_audio_silence: bool = True,
        auto_fallback: bool = True,
    ):
        plan, status, previews = detect_long_video_shots(
            video=video,
            mode=mode,
            fixed_duration=int(fixed_duration),
            sensitivity=sensitivity,
            use_audio_silence=bool(use_audio_silence),
            auto_fallback=bool(auto_fallback),
        )
        return IO.NodeOutput(plan, status, previews, ui={"text": (status,)})


class CompanyLongVideoShotInspector(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoShotInspector",
            display_name="分镜检测结果检查",
            category="company-remote/video/long video/testing",
            description="选择并播放一个逻辑镜头，同时显示首中尾帧和所有切点前后帧；只做本地检查，不调用远端模型。",
            inputs=[
                LongVideoShotPlanType.Input("shot_plan", display_name="镜头计划"),
                IO.Int.Input("shot_index", display_name="要检查的镜头序号", default=1, min=1, max=9999, step=1),
                IO.Boolean.Input("export_all_shots", display_name="导出全部检测镜头", default=False),
            ],
            outputs=[
                IO.Video.Output("selected_video", display_name="选中镜头视频"),
                IO.Image.Output("selected_frames", display_name="选中镜头首中尾帧"),
                IO.Image.Output("boundary_frames", display_name="切点前后帧对比"),
                IO.String.Output("report_json", display_name="分镜检查报告 JSON"),
                IO.String.Output("export_directory", display_name="全部镜头导出目录"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, shot_plan: Any, shot_index: int, export_all_shots: bool = False):
        selected_video, selected_frames, boundary_frames, report, export_directory = inspect_long_video_shots(
            shot_plan,
            shot_index=int(shot_index),
            export_all_shots=bool(export_all_shots),
        )
        subfolder = "company_remote_shot_tests"
        preview_directory = Path(folder_paths.get_temp_directory()) / subfolder
        preview_directory.mkdir(parents=True, exist_ok=True)
        filename = f"shot_preview_{int(shot_index):04d}_{time.time_ns()}.mp4"
        selected_video.save_to(str(preview_directory / filename))
        preview = UI.PreviewVideo([UI.SavedResult(filename, subfolder, IO.FolderType.temp)])
        return IO.NodeOutput(
            selected_video,
            selected_frames,
            boundary_frames,
            report,
            export_directory,
            ui=preview,
        )


class CompanyLongVideoContinuityRangeSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoContinuityRangeSelector",
            display_name="连续分镜范围选择",
            category="company-remote/video/long video/testing",
            description="从检测结果中选择任意数量的连续镜头，重置时间轴后交给后续视频生成链路；不保存完整范围预览，避免长视频内存峰值。",
            inputs=[
                LongVideoShotPlanType.Input("shot_plan", display_name="镜头计划"),
                IO.Int.Input("start_shot", display_name="起始镜头序号", default=1, min=1, max=9999, step=1),
                IO.Int.Input("shot_count", display_name="连续镜头数量（0=全部剩余）", default=0, min=0, step=1),
            ],
            outputs=[
                LongVideoShotPlanType.Output("shot_plan", display_name="选中范围镜头计划"),
                IO.Video.Output("selected_video", display_name="选中范围原视频"),
                IO.String.Output("report_json", display_name="范围选择报告 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, shot_plan: Any, start_shot: int, shot_count: int):
        selected_plan, selected_video, report = select_continuous_shot_range(
            shot_plan,
            start_shot=int(start_shot),
            shot_count=int(shot_count),
        )
        return IO.NodeOutput(selected_plan, selected_video, report, ui={"text": (report,)})


class CompanyLongVideoLengthRangeSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoLengthRangeSelector",
            display_name="按时长/百分比选择生成范围",
            category="company-remote/video/long video/testing",
            description="按全部剩余、分钟、总长百分比或镜头数量选择连续镜头范围；目标落在镜头中间时保留完整镜头。",
            inputs=[
                LongVideoShotPlanType.Input("shot_plan", display_name="镜头计划"),
                IO.Int.Input("start_shot", display_name="起始镜头序号", default=1, min=1, max=9999, step=1),
                IO.Combo.Input(
                    "limit_mode",
                    options=["全部剩余", "按分钟", "按总长百分比", "按镜头数量"],
                    default="按分钟",
                    display_name="生成范围控制方式",
                ),
                IO.Float.Input("limit_minutes", display_name="生成时长（分钟，0=全部剩余）", default=3.0, min=0.0, max=9999.0, step=0.1),
                IO.Float.Input("limit_percent", display_name="占原视频总长百分比（0=全部剩余）", default=30.0, min=0.0, max=100.0, step=1.0),
                IO.Int.Input("shot_count", display_name="镜头数量（0=全部剩余）", default=0, min=0, step=1),
            ],
            outputs=[
                LongVideoShotPlanType.Output("shot_plan", display_name="选中范围镜头计划"),
                IO.Video.Output("selected_video", display_name="选中范围原视频"),
                IO.String.Output("report_json", display_name="范围选择报告 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        shot_plan: Any,
        start_shot: int,
        limit_mode: str,
        limit_minutes: float,
        limit_percent: float,
        shot_count: int,
    ):
        selected_plan, selected_video, report = select_long_video_length_range(
            shot_plan,
            start_shot=int(start_shot),
            limit_mode=str(limit_mode),
            limit_minutes=float(limit_minutes),
            limit_percent=float(limit_percent),
            shot_count=int(shot_count),
        )
        return IO.NodeOutput(selected_plan, selected_video, report, ui={"text": (report,)})


class CompanyLongVideoManualBatchRangeSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoManualBatchRangeSelector",
            display_name="手动批次范围与续接控制",
            category="company-remote/video/long video/manual batch",
            description="每次只选取一个可审阅批次；继续按原视频时间线推进，重试复用当前范围，并输出系列状态。",
            inputs=[
                LongVideoShotPlanType.Input("shot_plan", display_name="完整镜头计划"),
                IO.Combo.Input(
                    "action",
                    options=["新建系列", "继续下一批", "重试当前批"],
                    default="新建系列",
                    display_name="批次动作",
                ),
                IO.String.Input("series_id", display_name="系列 ID（新建可留空）", default=""),
                IO.Float.Input("batch_minutes", display_name="每批目标时长（分钟）", default=1.0, min=0.5, max=5.0, step=0.5),
                IO.Float.Input("boundary_tolerance", display_name="镜头边界容差（秒）", default=10.0, min=0.0, max=30.0, step=1.0),
                IO.Float.Input("start_minute", display_name="指定起始分钟（0=按顺序游标）", default=0.0, min=0.0, max=99999.0, step=0.1),
                IO.Float.Input("end_minute", display_name="指定结束分钟（0=按每批时长/到结尾）", default=0.0, min=0.0, max=99999.0, step=0.1),
            ],
            outputs=[
                LongVideoShotPlanType.Output("shot_plan", display_name="当前批次镜头计划"),
                IO.Video.Output("selected_video", display_name="当前批次原视频"),
                IO.String.Output("batch_report_json", display_name="批次范围报告 JSON"),
                IO.String.Output("series_state_json", display_name="系列状态 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        shot_plan: Any,
        action: str = "新建系列",
        series_id: str = "",
        batch_minutes: float = 1.0,
        boundary_tolerance: float = 10.0,
        start_minute: float = 0.0,
        end_minute: float = 0.0,
    ):
        selected_plan, selected_video, report, state = select_manual_batch_range(
            shot_plan,
            action=str(action),
            series_id=str(series_id),
            batch_minutes=float(batch_minutes),
            boundary_tolerance=float(boundary_tolerance),
            start_second=float(start_minute) * 60.0,
            end_second=float(end_minute) * 60.0,
        )
        return IO.NodeOutput(selected_plan, selected_video, report, state, ui={"text": (report, state)})


class CompanyLongVideoManualBatchFinalizerV1(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoManualBatchFinalizerV1",
            display_name="提交当前批次并等待人工审阅",
            category="company-remote/video/long video/manual batch",
            description="当前批次完整合并后提交系列状态；输出下一次继续所需的系列状态和人工暂停提示。",
            inputs=[
                LongVideoJobType.Input("job", display_name="已合并当前批次的任务"),
                IO.String.Input("series_state_json", display_name="系列状态 JSON", default=""),
            ],
            outputs=[
                IO.Video.Output("video", display_name="当前批次视频"),
                IO.String.Output("final_path", display_name="当前批次视频路径"),
                IO.String.Output("series_state_json", display_name="已提交系列状态 JSON"),
                IO.String.Output("status_json", display_name="人工审阅状态 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any, series_state_json: str = ""):
        if not getattr(job, "manifest", None):
            raise ValueError("没有收到有效的批次任务。")
        if int(job.manifest.get("processing_contract_version", 0)) != MANUAL_BATCH_PROCESSING_CONTRACT_VERSION:
            raise ValueError("当前任务不是 contract=4 手动批次任务。")
        if series_state_json:
            try:
                submitted_state = json.loads(series_state_json)
            except json.JSONDecodeError as exc:
                raise ValueError("传入的系列状态 JSON 无法读取。") from exc
            if not isinstance(submitted_state, dict) or submitted_state.get("contract") != MANUAL_BATCH_CONTRACT:
                raise ValueError("传入的系列状态不是当前手动批次版本。")
            submitted_batch = submitted_state.get("current_batch")
            manifest_batch = job.manifest.get("manual_batch")
            if (
                not isinstance(submitted_batch, dict)
                or not isinstance(manifest_batch, dict)
                or submitted_state.get("series_id") != manifest_batch.get("series_id")
                or submitted_batch.get("batch_id") != manifest_batch.get("batch_id")
                or int(submitted_batch.get("attempt", -1)) != int(manifest_batch.get("attempt", -2))
            ):
                raise ValueError("传入的系列状态与当前批次任务不一致。")
        if job.manifest.get("status") != "success" or not Path(str(job.manifest.get("final") or "")).is_file():
            merge_long_video_job(job)
        final_path = Path(str(job.manifest.get("final") or ""))
        if not final_path.is_file():
            raise ValueError("当前批次提交后找不到最终视频。")
        state_info = job.manifest.get("manual_batch") if isinstance(job.manifest.get("manual_batch"), dict) else {}
        series_id = str(state_info.get("series_id") or "")
        state, _ = _manual_batch_read_state(series_id)
        current_batch = state.get("current_batch") if isinstance(state.get("current_batch"), dict) else {}
        report = {
            "contract": MANUAL_BATCH_CONTRACT,
            "series_id": series_id,
            "batch_id": current_batch.get("batch_id"),
            "status": "completed",
            "manual_pause_after_batch": True,
            "series_complete": bool(state.get("series_complete")),
            "next_cursor": state.get("next_cursor"),
            "next_action": "检查当前批次满意后，将动作改为“继续下一批”再执行。",
            "retry_action": "不满意时选择“重试当前批”，源范围和上一批尾帧保持不变。",
        }
        return IO.NodeOutput(
            InputImpl.VideoFromFile(str(final_path)),
            str(final_path),
            json.dumps(state, ensure_ascii=False, indent=2),
            json.dumps(report, ensure_ascii=False, indent=2),
            ui={"text": (json.dumps(report, ensure_ascii=False, indent=2),)},
        )


class CompanyLongVideoDurationAdapter(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoDurationAdapter",
            display_name="镜头时长适配与任务规划",
            category="company-remote/video/long video/stages",
            description="按视频模型时长限制拆分超长镜头，并为极短镜头记录补帧和裁回参数。",
            inputs=[
                LongVideoShotPlanType.Input("shot_plan", display_name="镜头计划"),
                LongVideoAssetsType.Input("assets", display_name="已加载资产"),
                IO.String.Input("prompt", display_name="视频提示词", multiline=True, default=""),
                IO.Combo.Input("engine", options=["Seedance 2.0", "Wan2.7 R2V"], default="Seedance 2.0", display_name="视频引擎"),
                IO.Combo.Input(
                    "model",
                    options=["Seedance 2.0 Fast", "Seedance 2.0", "wan2.7-r2v-2026-06-12"],
                    default="Seedance 2.0 Fast",
                    display_name="模型",
                ),
                _gpt_text_model_input("analysis_model", display_name="分段分析模型"),
                IO.Int.Input("max_retries", display_name="每段最大重试次数", default=2, min=0, max=5, step=1),
                IO.Boolean.Input("resume", display_name="复用已完成分段", default=True),
                IO.Boolean.Input("force_rerun", display_name="强制重跑全部分段", default=False, advanced=True),
                IO.String.Input("negative_prompt", display_name="负面提示词", multiline=True, default="", optional=True, advanced=True),
            ],
            outputs=[
                LongVideoJobType.Output("job", display_name="长视频任务"),
                IO.String.Output("plan_json", display_name="时长适配计划 JSON"),
                IO.String.Output("manifest_path", display_name="任务 manifest 路径"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, analysis_model: str):
        return _validate_gpt_text_model(analysis_model)

    @classmethod
    def execute(
        cls,
        shot_plan: Any,
        assets: Any,
        prompt: str,
        engine: str,
        model: str,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        max_retries: int = 2,
        resume: bool = True,
        force_rerun: bool = False,
        negative_prompt: str = "",
    ):
        normalized_engine, normalized_model = _normalize_long_video_engine_model(engine, model)
        job = plan_long_video_job(
            video=shot_plan.video,
            assets=assets,
            prompt=prompt,
            engine=normalized_engine,
            model=normalized_model,
            segment_duration=int(shot_plan.fixed_duration),
            ai_model=analysis_model,
            max_retries=int(max_retries),
            resume=bool(resume),
            force_rerun=bool(force_rerun),
            negative_prompt=negative_prompt or "",
            shot_plan=shot_plan,
        )
        status = json.dumps(job.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(job, status, str(job.manifest_path), ui={"text": (status,)})


class CompanyLongVideoSegmentPlanner(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoSegmentPlanner",
            display_name="长视频切分与任务规划",
            category="company-remote/video/long video/stages",
            description="检查视频时长并建立连续的 10/15 秒逻辑片段及模型请求任务。",
            inputs=[
                IO.Video.Input("video", display_name="长视频"),
                LongVideoAssetsType.Input("assets", display_name="已加载资产"),
                IO.String.Input("prompt", display_name="视频提示词", multiline=True, default=""),
                IO.Combo.Input("engine", options=["Seedance 2.0", "Wan2.7 R2V"], default="Seedance 2.0", display_name="视频引擎"),
                IO.Combo.Input(
                    "model",
                    options=["Seedance 2.0 Fast", "Seedance 2.0", "wan2.7-r2v-2026-06-12"],
                    default="Seedance 2.0 Fast",
                    display_name="模型",
                ),
                IO.Combo.Input("segment_duration", options=[10, 15], default=10, display_name="目标分段时长（秒）"),
                _gpt_text_model_input("analysis_model", display_name="分段分析模型"),
                IO.Int.Input("max_retries", display_name="每段最大重试次数", default=2, min=0, max=5, step=1),
                IO.Boolean.Input("resume", display_name="复用已完成分段", default=True),
                IO.Boolean.Input("force_rerun", display_name="强制重跑全部分段", default=False, advanced=True),
                IO.String.Input("negative_prompt", display_name="负面提示词", multiline=True, default="", optional=True, advanced=True),
            ],
            outputs=[
                LongVideoJobType.Output("job", display_name="长视频任务"),
                IO.String.Output("plan_json", display_name="切分计划 JSON"),
                IO.String.Output("manifest_path", display_name="任务 manifest 路径"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, analysis_model: str):
        return _validate_gpt_text_model(analysis_model)

    @classmethod
    def execute(
        cls,
        video: Any,
        assets: Any,
        prompt: str,
        engine: str,
        model: str,
        segment_duration: int,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        max_retries: int = 2,
        resume: bool = True,
        force_rerun: bool = False,
        negative_prompt: str = "",
    ):
        normalized_engine, normalized_model = _normalize_long_video_engine_model(engine, model)
        job = plan_long_video_job(
            video=video,
            assets=assets,
            prompt=prompt,
            engine=normalized_engine,
            model=normalized_model,
            segment_duration=int(segment_duration),
            ai_model=analysis_model,
            max_retries=int(max_retries),
            resume=bool(resume),
            force_rerun=bool(force_rerun),
            negative_prompt=negative_prompt or "",
        )
        status = json.dumps(job.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(job, status, str(job.manifest_path), ui={"text": (status,)})


class CompanyLongVideoAutoAssetPlanner(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoAutoAssetPlanner",
            display_name="按镜头自动资产任务规划",
            category="company-remote/video/long video/auto assets",
            description="不要求预先列人物或背景；根据镜头计划建立首尾帧自动提取、欧美化资产和视频生成任务。",
            inputs=[
                LongVideoShotPlanType.Input("shot_plan", display_name="镜头计划"),
                IO.String.Input("prompt", display_name="视频提示词", multiline=True, default=""),
                IO.Combo.Input("engine", options=["Seedance 2.0", "Wan2.7 R2V"], default="Seedance 2.0", display_name="视频引擎"),
                IO.Combo.Input(
                    "model",
                    options=["Seedance 2.0 Fast", "Seedance 2.0", "wan2.7-r2v-2026-06-12"],
                    default="Seedance 2.0 Fast",
                    display_name="模型",
                ),
                _gpt_text_model_input("analysis_model", display_name="镜头分析模型"),
                IO.Combo.Input("image_model", options=["gpt-image-2"], default="gpt-image-2", display_name="自动资产图片模型"),
                IO.Combo.Input("image_quality", options=["auto", "low", "medium", "high"], default="medium", display_name="自动资产图片质量"),
                IO.Float.Input(
                    "reuse_threshold",
                    display_name="跨镜头复用置信度阈值",
                    default=0.92,
                    min=0.5,
                    max=1.0,
                    step=0.01,
                ),
                IO.Int.Input("max_retries", display_name="每段最大重试次数", default=2, min=0, max=5, step=1),
                IO.Boolean.Input("resume", display_name="复用已完成任务和资产", default=True),
                IO.Boolean.Input("force_rerun", display_name="强制重跑视频分段", default=False, advanced=True),
                IO.Boolean.Input("force_rerun_assets", display_name="强制重建镜头资产", default=False, advanced=True),
                IO.String.Input("negative_prompt", display_name="负面提示词", multiline=True, default="", optional=True, advanced=True),
                IO.Combo.Input(
                    "image_provider",
                    options=GPT_IMAGE_PROVIDER_OPTIONS,
                    default=GPT_IMAGE_PROVIDER_WISART,
                    display_name="自动资产图片服务",
                ),
            ],
            outputs=[
                LongVideoJobType.Output("job", display_name="自动资产长视频任务"),
                IO.String.Output("plan_json", display_name="任务计划 JSON"),
                IO.String.Output("manifest_path", display_name="任务 manifest 路径"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, analysis_model: str):
        return _validate_gpt_text_model(analysis_model)

    @classmethod
    def execute(
        cls,
        shot_plan: Any,
        prompt: str,
        engine: str,
        model: str,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        image_model: str = "gpt-image-2",
        image_quality: str = "medium",
        reuse_threshold: float = 0.92,
        max_retries: int = 2,
        resume: bool = True,
        force_rerun: bool = False,
        force_rerun_assets: bool = False,
        negative_prompt: str = "",
        image_provider: str = GPT_IMAGE_PROVIDER_WISART,
    ):
        normalized_engine, normalized_model = _normalize_long_video_engine_model(engine, model)
        job = plan_long_video_auto_asset_job(
            shot_plan=shot_plan,
            prompt=prompt,
            engine=normalized_engine,
            model=normalized_model,
            ai_model=analysis_model,
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
            reuse_threshold=float(reuse_threshold),
            max_retries=int(max_retries),
            resume=bool(resume),
            force_rerun=bool(force_rerun),
            force_rerun_assets=bool(force_rerun_assets),
            negative_prompt=negative_prompt or "",
        )
        status = json.dumps(job.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(job, status, str(job.manifest_path), ui={"text": (status,)})


ANIME_LONG_VIDEO_PROMPT = (
    "把本段重新演绎为统一的高质量二维动漫视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；"
    "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。"
    "从第一帧到最后一帧，人物与完整背景都必须保持清晰线稿、赛璐璐分层上色和统一动漫光影；"
    "不得出现真人脸、真实皮肤、照片纹理、真人摄影画面、半真人半动漫、写实 3D 人物、风格闪回、字幕、Logo 或水印。"
)

ANIME_LONG_VIDEO_NEGATIVE_PROMPT = (
    "真人，真实人脸，真实皮肤，照片，摄影，写实，半真人，真人背景，皮肤毛孔，镜头噪点，"
    "写实3D，风格漂移，真人闪回，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印"
)

WESTERN_LONG_VIDEO_PROMPT = (
    "用参考图片中的欧美人物和环境，替换本段原视频中的人物与环境。参考图片是最终人物身份、服装、道具和环境美术的唯一视觉依据；"
    "人物不能只换脸，环境不能只调色。人物的面部地域特征、发型发色、妆容、服装鞋履、配饰、版型剪裁、材质配色和气质，"
    "以及环境的建筑语言、道路与公共设施、家具陈设、材质、植被、照明和生活细节，都可以与原视频明显不同，"
    "但必须形成完整可信、地域统一的欧美版本，并清除中式/东方造型和中文视觉符号。"
    "保持输入参考图对应的视觉媒介：真人素材保持高质量欧美真人电影质感，动漫、漫画或 CG 素材保持同一媒介并替换为欧美版本。"
    "严格保持实际人物数量、人物关系、主要剧情顺序、镜头构图、动作、空间关系和场景功能；"
    "从第一帧到最后一帧保持人物身份、整体造型、欧美环境、媒介和光影稳定一致，不得恢复原人物或原背景，不得出现字幕、Logo 或水印。"
)

WESTERN_LONG_VIDEO_NEGATIVE_PROMPT = (
    "只换脸，亚洲面孔残留，本土东方造型，中式古装，中式建筑，中式家具，中文招牌，本土化道路设施，"
    "原人物残留，原服装残留，原背景残留，轻微调色，少量道具替换，地域混乱，媒介变化，风格漂移，"
    "人物复制，身份变化，多余人物，肢体畸形，额外手指，背景跳变，字幕，文字，Logo，水印"
)

PHOTOREAL_LONG_VIDEO_PROMPT = (
    "把本段重新演绎为统一的高质量真人影视视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；"
    "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。"
    "从第一帧到最后一帧保持自然真实的人脸、皮肤、头发、布料、建筑材质和电影光影，人物与完整背景必须稳定一致；"
    "不得出现动漫线稿、卡通脸、插画笔触、游戏 CG、塑料皮肤、半真人半卡通、风格闪回、字幕、Logo 或水印。"
)

PHOTOREAL_LONG_VIDEO_NEGATIVE_PROMPT = (
    "动漫，漫画，卡通，插画，二维线稿，赛璐璐，游戏CG，3D建模，塑料皮肤，假脸，过度磨皮，"
    "半真人半卡通，风格漂移，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印"
)

CG_3D_LONG_VIDEO_PROMPT = (
    "把本段重新演绎为统一的高质量 3D 游戏 CG 视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；"
    "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。"
    "从第一帧到最后一帧保持稳定三维造型、PBR 材质、体积光和影视级游戏过场渲染，人物与完整环境必须属于同一美术体系；"
    "不得出现真人摄影、二维线稿、平面插画、低模、塑料质感、材质跳变、字幕、Logo 或水印。"
)

CG_3D_LONG_VIDEO_NEGATIVE_PROMPT = (
    "真人摄影，真实照片，二维动漫，漫画线稿，平面插画，低模，塑料材质，材质穿帮，贴图错误，"
    "风格漂移，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印"
)

COMIC_LONG_VIDEO_PROMPT = (
    "把本段重新演绎为统一的高质量漫画插画视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；"
    "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。"
    "从第一帧到最后一帧保持稳定的手绘墨线、明确明暗块面、细腻插画上色和一致透视，人物与完整背景画风必须统一；"
    "不得出现真人摄影纹理、3D 建模感、廉价卡通贴纸感、拼贴、画风闪回、字幕、Logo 或水印。"
)

COMIC_LONG_VIDEO_NEGATIVE_PROMPT = (
    "真人摄影，真实皮肤，照片纹理，3D建模，游戏CG，低模，廉价卡通，贴纸感，拼贴，线条抖动，"
    "画风漂移，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印"
)

CUSTOM_LONG_VIDEO_PROMPT = (
    "根据用户指定的目标视觉方向重新演绎本段视频。参考图片定义人物身份、服装、道具和环境，"
    "镜头分析文字定义剧情、动作、表情、走位和镜头运动；严格保持实际人物数量、人物关系、剧情顺序和场景功能，"
    "并确保从第一帧到最后一帧的人物、完整背景、材质、色彩和画风稳定一致。"
)

CUSTOM_LONG_VIDEO_NEGATIVE_PROMPT = "风格漂移，多余人物，人物复制，身份变化，肢体畸形，字幕，文字，Logo，水印"

TARGET_RESOURCE_WESTERN = "欧美化资源"
TARGET_RESOURCE_PHOTOREAL = "真人写实资源"
TARGET_RESOURCE_ANIME = "二维动漫资源"
TARGET_RESOURCE_CG_3D = "3D / 游戏 CG 资源"
TARGET_RESOURCE_COMIC = "漫画插画资源"
TARGET_RESOURCE_CUSTOM = "自定义"
TARGET_RESOURCE_OPTIONS = [
    TARGET_RESOURCE_WESTERN,
    TARGET_RESOURCE_PHOTOREAL,
    TARGET_RESOURCE_ANIME,
    TARGET_RESOURCE_CG_3D,
    TARGET_RESOURCE_COMIC,
    TARGET_RESOURCE_CUSTOM,
]
TARGET_RESOURCE_PRESETS = {
    TARGET_RESOURCE_WESTERN: (
        AUTO_ASSET_STYLE_WESTERN,
        WESTERN_LONG_VIDEO_PROMPT,
        WESTERN_LONG_VIDEO_NEGATIVE_PROMPT,
    ),
    TARGET_RESOURCE_PHOTOREAL: (
        AUTO_ASSET_STYLE_PHOTOREAL,
        PHOTOREAL_LONG_VIDEO_PROMPT,
        PHOTOREAL_LONG_VIDEO_NEGATIVE_PROMPT,
    ),
    TARGET_RESOURCE_ANIME: (
        AUTO_ASSET_STYLE_ANIME,
        ANIME_LONG_VIDEO_PROMPT,
        ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
    ),
    TARGET_RESOURCE_CG_3D: (
        AUTO_ASSET_STYLE_CG_3D,
        CG_3D_LONG_VIDEO_PROMPT,
        CG_3D_LONG_VIDEO_NEGATIVE_PROMPT,
    ),
    TARGET_RESOURCE_COMIC: (
        AUTO_ASSET_STYLE_COMIC,
        COMIC_LONG_VIDEO_PROMPT,
        COMIC_LONG_VIDEO_NEGATIVE_PROMPT,
    ),
    TARGET_RESOURCE_CUSTOM: (
        AUTO_ASSET_STYLE_CUSTOM,
        CUSTOM_LONG_VIDEO_PROMPT,
        CUSTOM_LONG_VIDEO_NEGATIVE_PROMPT,
    ),
}


def _target_resource_settings(target_resource_type: str, prompt: str, negative_prompt: str) -> tuple[str, str, str, str]:
    requested_type = str(target_resource_type or "").strip()
    current_prompt = str(prompt or "").strip()
    known_prompts = {value[1] for value in TARGET_RESOURCE_PRESETS.values()}
    if not requested_type:
        requested_type = TARGET_RESOURCE_CUSTOM if current_prompt and current_prompt not in known_prompts else TARGET_RESOURCE_ANIME
    normalized_type = requested_type
    if normalized_type not in TARGET_RESOURCE_PRESETS:
        normalized_type = TARGET_RESOURCE_ANIME
    visual_style, preset_prompt, preset_negative = TARGET_RESOURCE_PRESETS[normalized_type]
    known_negatives = {value[2] for value in TARGET_RESOURCE_PRESETS.values()}
    resolved_prompt = current_prompt
    resolved_negative = str(negative_prompt or "").strip()
    if normalized_type != TARGET_RESOURCE_CUSTOM or not resolved_prompt or resolved_prompt in known_prompts:
        resolved_prompt = preset_prompt
    if normalized_type != TARGET_RESOURCE_CUSTOM or not resolved_negative or resolved_negative in known_negatives:
        resolved_negative = preset_negative
    return normalized_type, visual_style, resolved_prompt, resolved_negative


class CompanyLongVideoAnimeAssetPlanner(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoAnimeAssetPlanner",
            display_name="人物视频多风格资产任务规划",
            category="company-remote/video/long video/auto assets",
            description=(
                "根据目标资源类型逐镜头生成人物与场景参考图，并让 Seedance 仅接收生成后的参考图和剧情分析文字，"
                "不上传原分镜视频。"
            ),
            inputs=[
                LongVideoShotPlanType.Input("shot_plan", display_name="镜头计划"),
                IO.String.Input(
                    "prompt",
                    display_name="视频提示词",
                    multiline=True,
                    default=ANIME_LONG_VIDEO_PROMPT,
                ),
                IO.Combo.Input(
                    "model",
                    options=["Seedance 2.0 Fast", "Seedance 2.0"],
                    default="Seedance 2.0 Fast",
                    display_name="Seedance 模型",
                ),
                _gpt_text_model_input("analysis_model", display_name="镜头分析模型"),
                IO.Combo.Input("image_model", options=["gpt-image-2"], default="gpt-image-2", display_name="资产图片模型"),
                IO.Combo.Input("image_quality", options=["auto", "low", "medium", "high"], default="medium", display_name="资产图片质量"),
                IO.Float.Input(
                    "reuse_threshold",
                    display_name="跨镜头复用置信度阈值",
                    default=0.92,
                    min=0.5,
                    max=1.0,
                    step=0.01,
                ),
                IO.Int.Input("max_retries", display_name="每段最大重试次数", default=2, min=0, max=5, step=1),
                IO.Boolean.Input("resume", display_name="复用已完成任务和资产", default=True),
                IO.Boolean.Input("force_rerun", display_name="强制重跑视频分段", default=False, advanced=True),
                IO.Boolean.Input("force_rerun_assets", display_name="强制重建资产", default=False, advanced=True),
                IO.Combo.Input(
                    "image_provider",
                    options=GPT_IMAGE_PROVIDER_OPTIONS,
                    default=GPT_IMAGE_PROVIDER_WISART,
                    display_name="资产图片服务",
                ),
                IO.String.Input(
                    "negative_prompt",
                    display_name="负面提示词",
                    multiline=True,
                    default=ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "target_resource_type",
                    options=TARGET_RESOURCE_OPTIONS,
                    default=TARGET_RESOURCE_ANIME,
                    display_name="目标资源类型",
                    tooltip="切换后自动填写对应的视频提示词和负面提示词，并同步改变人物、背景资产的生成风格。",
                    optional=True,
                ),
            ],
            outputs=[
                LongVideoJobType.Output("job", display_name="多风格长视频任务"),
                IO.String.Output("plan_json", display_name="任务计划 JSON"),
                IO.String.Output("manifest_path", display_name="任务 manifest 路径"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, analysis_model: str):
        return _validate_gpt_text_model(analysis_model)

    @classmethod
    def execute(
        cls,
        shot_plan: Any,
        prompt: str,
        model: str,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        image_model: str = "gpt-image-2",
        image_quality: str = "medium",
        reuse_threshold: float = 0.92,
        max_retries: int = 2,
        resume: bool = True,
        force_rerun: bool = False,
        force_rerun_assets: bool = False,
        negative_prompt: str = ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
        image_provider: str = GPT_IMAGE_PROVIDER_WISART,
        target_resource_type: str = "",
    ):
        normalized_model = str(model).strip()
        if not normalized_model.lower().startswith("seedance"):
            normalized_model = "Seedance 2.0 Fast"
        normalized_type, visual_style, resolved_prompt, resolved_negative = _target_resource_settings(
            target_resource_type,
            prompt,
            negative_prompt,
        )
        job = plan_long_video_auto_asset_job(
            shot_plan=shot_plan,
            prompt=resolved_prompt,
            engine="seedance",
            model=normalized_model,
            ai_model=analysis_model,
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
            reuse_threshold=float(reuse_threshold),
            max_retries=int(max_retries),
            resume=bool(resume),
            force_rerun=bool(force_rerun),
            force_rerun_assets=bool(force_rerun_assets),
            negative_prompt=resolved_negative,
            visual_style=visual_style,
            send_source_video=False,
            target_resource_type=normalized_type,
        )
        status = json.dumps(job.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(job, status, str(job.manifest_path), ui={"text": (status,)})


class CompanyLongVideoAnimeAssetPlannerV3(CompanyLongVideoAnimeAssetPlanner):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "CompanyLongVideoAnimeAssetPlannerV3"
        schema.display_name = "人物视频多风格资产任务规划 v3"
        schema.description = (
            "按原逻辑镜头生成资产，再将不足 4 秒的相邻镜头合并为一次 Seedance 请求；"
            "可选择保留原视频音频，默认由 Seedance 同时生成音频。"
        )
        schema.inputs.append(
            IO.Boolean.Input(
                "use_original_audio",
                display_name="使用原视频音频",
                default=False,
                optional=True,
                tooltip="关闭时由 Seedance 同时生成音频；开启时关闭生成音频并按请求组恢复原视频音频。",
            )
        )
        schema.inputs.append(
            IO.Boolean.Input(
                "use_integrated_frame_references",
                display_name="生成整帧融合参考图",
                default=False,
                optional=True,
                advanced=True,
                tooltip="默认关闭：复用人物素材库 asset_id 与场景主素材，避免每个镜头额外生成整帧图片。开启后才会逐镜头生成并筛选整帧融合图。",
            )
        )
        schema.inputs.append(
            IO.String.Input(
                "identity_mapping_json",
                display_name="人物映射 JSON 或文件路径",
                multiline=True,
                default="",
                optional=True,
                advanced=True,
                tooltip=(
                    "人工确认的人物映射：global_people 定义固定人物（含火山素材 asset_id），"
                    "shot_people 用“镜头号:槽位”指派或忽略人物。留空时按自动识别与身份门控执行。"
                ),
            )
        )
        return schema

    @classmethod
    def execute(
        cls,
        shot_plan: Any,
        prompt: str,
        model: str,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        image_model: str = "gpt-image-2",
        image_quality: str = "medium",
        reuse_threshold: float = 0.92,
        max_retries: int = 2,
        resume: bool = True,
        force_rerun: bool = False,
        force_rerun_assets: bool = False,
        negative_prompt: str = ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
        image_provider: str = GPT_IMAGE_PROVIDER_WISART,
        target_resource_type: str = "",
        use_original_audio: bool = False,
        use_integrated_frame_references: bool = False,
        identity_mapping_json: str = "",
    ):
        normalized_model = str(model).strip()
        if not normalized_model.lower().startswith("seedance"):
            normalized_model = "Seedance 2.0 Fast"
        normalized_type, visual_style, resolved_prompt, resolved_negative = _target_resource_settings(
            target_resource_type,
            prompt,
            negative_prompt,
        )
        job = plan_long_video_auto_asset_job(
            shot_plan=shot_plan,
            prompt=resolved_prompt,
            engine="seedance",
            model=normalized_model,
            ai_model=analysis_model,
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
            reuse_threshold=float(reuse_threshold),
            max_retries=int(max_retries),
            resume=bool(resume),
            force_rerun=bool(force_rerun),
            force_rerun_assets=bool(force_rerun_assets),
            negative_prompt=resolved_negative,
            visual_style=visual_style,
            send_source_video=False,
            target_resource_type=normalized_type,
            processing_contract_version=3,
            use_original_audio=bool(use_original_audio),
            use_integrated_frame_references=bool(use_integrated_frame_references),
            identity_mapping=identity_mapping_json,
        )
        status = json.dumps(job.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(job, status, str(job.manifest_path), ui={"text": (status,)})


class CompanyLongVideoManualBatchPlannerV1(CompanyLongVideoAnimeAssetPlannerV3):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "CompanyLongVideoManualBatchPlannerV1"
        schema.display_name = "手动批次 Seedance 资产任务规划 v1"
        schema.description = "为当前可审阅批次建立 contract=4 任务；不会改动旧 v3 任务缓存。"
        return schema

    @classmethod
    def execute(
        cls,
        shot_plan: Any,
        prompt: str,
        model: str,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        image_model: str = "gpt-image-2",
        image_quality: str = "medium",
        reuse_threshold: float = 0.92,
        max_retries: int = 2,
        resume: bool = True,
        force_rerun: bool = False,
        force_rerun_assets: bool = False,
        negative_prompt: str = ANIME_LONG_VIDEO_NEGATIVE_PROMPT,
        image_provider: str = GPT_IMAGE_PROVIDER_WISART,
        target_resource_type: str = "",
        use_original_audio: bool = False,
        use_integrated_frame_references: bool = False,
        identity_mapping_json: str = "",
    ):
        normalized_type, visual_style, resolved_prompt, resolved_negative = _target_resource_settings(
            target_resource_type,
            prompt,
            negative_prompt,
        )
        batch_config = (shot_plan.config or {}).get("manual_batch", {})
        if not isinstance(batch_config, dict) or not batch_config.get("state_path"):
            raise ValueError("请先连接“手动批次范围与续接控制”节点的当前批次镜头计划。")
        job = plan_long_video_auto_asset_job(
            shot_plan=shot_plan,
            prompt=resolved_prompt,
            engine="seedance",
            model=str(model or "Seedance 2.0 Fast"),
            ai_model=analysis_model,
            image_model=image_model,
            image_quality=image_quality,
            image_provider=image_provider,
            reuse_threshold=float(reuse_threshold),
            max_retries=int(max_retries),
            resume=bool(resume),
            force_rerun=bool(force_rerun),
            force_rerun_assets=bool(force_rerun_assets),
            negative_prompt=resolved_negative,
            visual_style=visual_style,
            send_source_video=False,
            target_resource_type=normalized_type,
            processing_contract_version=MANUAL_BATCH_PROCESSING_CONTRACT_VERSION,
            use_original_audio=bool(use_original_audio),
            use_integrated_frame_references=bool(use_integrated_frame_references),
            manual_batch=dict(batch_config),
            identity_mapping=identity_mapping_json,
        )
        status = json.dumps(job.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(job, status, str(job.manifest_path), ui={"text": (status,)})


class CompanyLongVideoAutoAssetBuilder(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoAutoAssetBuilder",
            display_name="分镜资产识别与复用",
            category="company-remote/video/long video/auto assets",
            description=(
                "逐镜头读取首中尾帧，先识别并复用人物素材库和场景主素材；默认不再为每个镜头生成整帧融合图。"
                "只有规划节点显式开启“生成整帧融合参考图”时，才执行整帧生成与质量筛选。"
            ),
            inputs=[
                LongVideoJobType.Input("job", display_name="自动资产长视频任务"),
                IO.Int.Input(
                    "image_concurrency",
                    display_name="图片并发数（0=无上限）",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="每分析完一个镜头就立即提交人物和背景图片；0 不限制整个任务的图片并发数，大于 0 时按此数值排队。",
                ),
            ],
            outputs=[
                LongVideoJobType.Output("job", display_name="已生成镜头资产的任务"),
                IO.String.Output("asset_report_json", display_name="镜头资产报告 JSON"),
                IO.Image.Output("asset_previews", display_name="源帧、复用素材与可选整帧预览"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, job: Any, image_concurrency: int = 0):
        result, report, previews = await asyncio.to_thread(
            build_long_video_auto_assets,
            job,
            int(image_concurrency),
        )
        return IO.NodeOutput(result, report, previews, ui={"text": (report,)})


ASSET_LIBRARY_VIEW_ALL = "全部样式"


class CompanyLongVideoAssetLibraryViewer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoAssetLibraryViewer",
            display_name="查看入库素材库资源",
            category="company-remote/video/long video/auto assets",
            description=(
                "只读列出已入库的人物与场景资源：缩略图、火山素材 asset_id、入库状态和来源标识。"
                "默认读取共享素材库；连接当前任务时并入尚未落库的资源。不调用任何付费接口。"
            ),
            inputs=[
                IO.Combo.Input(
                    "visual_style",
                    options=[*TARGET_RESOURCE_OPTIONS, ASSET_LIBRARY_VIEW_ALL],
                    default=TARGET_RESOURCE_WESTERN,
                    display_name="资源样式",
                    tooltip="按目标资源类型筛选素材库；选择“全部样式”列出所有已入库资源。",
                ),
                IO.Combo.Input("columns", options=["2", "3", "4"], default="3", display_name="每行数量"),
                LongVideoJobType.Input(
                    "job",
                    display_name="可选：当前任务（并入未落库资源）",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output("library_previews", display_name="入库资源预览"),
                IO.String.Output("inventory_json", display_name="入库资源清单 JSON"),
                IO.String.Output("summary", display_name="汇总"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, visual_style: str = TARGET_RESOURCE_WESTERN, columns: str = "3", job: Any = None):
        if str(visual_style) == ASSET_LIBRARY_VIEW_ALL:
            style = ""
        else:
            preset = TARGET_RESOURCE_PRESETS.get(str(visual_style))
            style = preset[0] if preset else AUTO_ASSET_STYLE_WESTERN
        grid, report, summary = build_asset_library_view(style, int(columns), job=job)
        preview = UI.PreviewImage(grid, cls=cls)
        return IO.NodeOutput(grid, report, summary, ui={"images": preview.values, "text": (summary,)})


class CompanyLongVideoAutoReferencePacker(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoAutoReferencePacker",
            display_name="自动参考素材打包",
            category="company-remote/video/long video/auto assets",
            description=(
                "Seedance 默认优先打包人物素材库 asset_id 和去重后的场景主素材；"
                "仅在开启整帧融合参考图时，才附加通过质量筛选的整帧图。Wan 保留原有参考包兼容逻辑。"
            ),
            inputs=[LongVideoJobType.Input("job", display_name="已生成镜头资产的任务")],
            outputs=[
                LongVideoJobType.Output("job", display_name="已打包参考素材的任务"),
                IO.String.Output("reference_report_json", display_name="参考素材报告 JSON"),
                IO.Image.Output("reference_previews", display_name="自动参考包预览"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any):
        result, report, previews = pack_long_video_auto_references(job)
        return IO.NodeOutput(result, report, previews, ui={"text": (report,)})


class CompanyLongVideoPipelineAssetVideoGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoPipelineAssetVideoGenerator",
            display_name="实时预览素材并生成 Seedance",
            category="company-remote/video/long video/auto assets",
            description=(
                "实时查看源首尾帧、人物/场景替换母版和可选完整替换画面。默认先复用人物素材库与场景母版，"
                "再交给 Seedance；只有开启整帧参考时才做逐镜头整帧质量比较与自动重试。不同场景不再传上一段末帧；仅同一逻辑镜头被模型时长上限拆开时续接。"
                "人物图上传 TOS 失败会阻断该分镜，素材库登记失败只提示警告。"
            ),
            inputs=[
                LongVideoJobType.Input("job", display_name="当前批次自动资产任务"),
                IO.Int.Input(
                    "image_concurrency",
                    display_name="图片并发数（0=无上限）",
                    default=5,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="后续镜头可同时提交的图片请求数。建议 4-8，过大可能触发图片服务限流。",
                ),
            ],
            outputs=[
                LongVideoJobType.Output("job", display_name="已生成当前批次任务"),
                IO.String.Output("generation_json", display_name="流水线状态 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, job: Any, image_concurrency: int = 5):
        result, status = await asyncio.to_thread(
            generate_long_video_pipeline,
            job,
            int(image_concurrency),
        )
        return IO.NodeOutput(result, status, ui={"text": (status,)})


class CompanyLongVideoSegmentAnalyzer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoSegmentAnalyzer",
            display_name="GPT 分段人物与背景分析",
            category="company-remote/video/long video/stages",
            description="为每个分段抽取首中尾关键帧，只分析实际人物、背景和剧情动作。",
            inputs=[LongVideoJobType.Input("job", display_name="长视频任务")],
            outputs=[LongVideoJobType.Output("job", display_name="已分析任务"), IO.String.Output("analysis_json", display_name="分析结果 JSON")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any):
        result = analyze_long_video_job(job)
        status = json.dumps(result.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(result, status, ui={"text": (status,)})


class CompanyLongVideoReferenceMatcher(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoReferenceMatcher",
            display_name="分段参考素材匹配",
            category="company-remote/video/long video/stages",
            description="根据 GPT 分析结果，为每段匹配人物、背景和连续性参考。",
            inputs=[LongVideoJobType.Input("job", display_name="已分析任务")],
            outputs=[LongVideoJobType.Output("job", display_name="已匹配任务"), IO.String.Output("match_json", display_name="匹配结果 JSON")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any):
        result = match_long_video_references(job)
        status = json.dumps(result.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(result, status, ui={"text": (status,)})


class CompanyLongVideoSegmentGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoSegmentGenerator",
            display_name="顺序生成全部视频分段",
            category="company-remote/video/long video/stages",
            description="按顺序调用 Seedance/Wan，失败时保存断点并只重试未完成分段。",
            inputs=[LongVideoJobType.Input("job", display_name="已匹配任务")],
            outputs=[LongVideoJobType.Output("job", display_name="已生成任务"), IO.String.Output("generation_json", display_name="生成状态 JSON")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any):
        result = generate_long_video_segments(job)
        status = json.dumps(result.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(result, status, ui={"text": (status,)})


class CompanyLongVideoParallelSegmentGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoParallelSegmentGenerator",
            display_name="并行生成视频分段（实时进度）",
            category="company-remote/video/long video/stages",
            description=(
                "并发调用 Seedance/Wan 生成独立视频分段；每完成一段立即推送视频预览，"
                "失败时保存 manifest 和进度 JSON，重新执行可复用已完成分段。"
            ),
            inputs=[
                LongVideoJobType.Input("job", display_name="已打包参考素材的任务"),
                IO.Int.Input(
                    "concurrency",
                    display_name="并发分段数",
                    default=3,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="同时提交的远程视频分段数量；建议 2-3，过大可能触发服务端限流。",
                ),
            ],
            outputs=[
                LongVideoJobType.Output("job", display_name="已生成任务"),
                IO.String.Output("generation_json", display_name="生成状态 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any, concurrency: int = 3):
        result = generate_long_video_segments_parallel(job, int(concurrency))
        status = json.dumps(result.manifest, ensure_ascii=False, indent=2)
        return IO.NodeOutput(result, status, ui={"text": (status,)})


class CompanyLongVideoResultCollector(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoResultCollector",
            display_name="分段结果列表",
            category="company-remote/video/long video/stages",
            description="核对全部分段文件并输出每段末帧预览和结果路径列表。",
            inputs=[LongVideoJobType.Input("job", display_name="已生成任务")],
            outputs=[
                LongVideoJobType.Output("job", display_name="可合并任务"),
                IO.String.Output("results_json", display_name="分段结果 JSON"),
                IO.Image.Output("previews", display_name="各段末帧预览"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any):
        result, summary, previews = collect_long_video_results(job)
        return IO.NodeOutput(result, summary, previews, ui={"text": (summary,)})


class CompanyLongVideoContinuityPreview(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoContinuityPreview",
            display_name="连续分镜生成结果预览",
            category="company-remote/video/long video/testing",
            description="逐段播放连续分镜生成结果，并输出每段末帧用于检查人物、动作和场景衔接。",
            inputs=[LongVideoJobType.Input("job", display_name="已生成任务")],
            outputs=[
                LongVideoJobType.Output("job", display_name="可合并任务"),
                IO.String.Output("results_json", display_name="连续性检查 JSON"),
                IO.Image.Output("end_frames", display_name="各段末帧"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any):
        result, _summary, end_frames = collect_long_video_results(job)
        output_root = Path(folder_paths.get_output_directory()).resolve()
        saved_results = []
        segments = []
        for sequence, task in enumerate(result.manifest["tasks"], start=1):
            path = Path(task["result"]).resolve()
            try:
                relative = path.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(f"第 {sequence} 段结果不在 ComfyUI output 目录内：{path}") from exc
            saved_results.append(
                UI.SavedResult(relative.name, relative.parent.as_posix(), IO.FolderType.output)
            )
            roles = list(task.get("reference_roles") or [])
            segments.append(
                {
                    "sequence": sequence,
                    "logical_shot": task.get("logical_segment"),
                    "source_start": task.get("source_start", task.get("start")),
                    "duration": task.get("duration"),
                    "reference_roles": roles,
                    "uses_previous_end_frame": "previous_segment_end_frame" in roles,
                    "video": str(path),
                }
            )
        report = json.dumps(
            {
                "job_id": result.manifest["job_id"],
                "segment_count": len(segments),
                "continuity": result.manifest.get("continuity"),
                "segments": segments,
                "check": "从第 2 段开始，uses_previous_end_frame 应为 true；请逐段播放检查动作、人物和背景衔接。",
            },
            ensure_ascii=False,
            indent=2,
        )
        preview = UI.PreviewVideo(saved_results).as_dict()
        preview["text"] = (report,)
        return IO.NodeOutput(result, report, end_frames, ui=preview)


class CompanyLongVideoFinalMerger(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoFinalMerger",
            display_name="合并分段并恢复原音频",
            category="company-remote/video/long video/stages",
            description="按时间顺序合并全部分段，并重新挂回原长视频音频。",
            inputs=[LongVideoJobType.Input("job", display_name="可合并任务")],
            outputs=[
                IO.Video.Output("video", display_name="最终视频"),
                IO.String.Output("final_path", display_name="最终视频路径"),
                IO.String.Output("manifest_path", display_name="任务 manifest 路径"),
                IO.String.Output("status_json", display_name="任务状态 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, job: Any):
        return IO.NodeOutput(*merge_long_video_job(job))


class CompanyLongVideoRestyle(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyLongVideoRestyle",
            display_name="长视频欧美化分段转绘",
            category="company-remote/video/long video",
            description="将长视频按 10/15 秒分段，复用确认的人物与背景资产，逐段远程转绘并恢复原音频。",
            search_aliases=["Long Video Restyle", "Long Video Batch Restyle", "长视频批处理"],
            inputs=[
                IO.Video.Input("video", display_name="长视频"),
                IO.String.Input(
                    "assets_manifest",
                    display_name="资产清单 JSON 或路径",
                    multiline=True,
                    default="",
                    tooltip="连接“长视频欧美化资产清单”的 JSON 输出，或填写 output 下 manifest.json 路径。",
                ),
                IO.String.Input("prompt", display_name="视频提示词", multiline=True, default=""),
                IO.Combo.Input(
                    "engine",
                    options=["Seedance 2.0", "Wan2.7 R2V"],
                    default="Seedance 2.0",
                    display_name="视频引擎",
                ),
                IO.Combo.Input(
                    "model",
                    options=["Seedance 2.0 Fast", "Seedance 2.0", "wan2.7-r2v-2026-06-12"],
                    default="Seedance 2.0 Fast",
                    display_name="模型",
                ),
                IO.Combo.Input("segment_duration", options=[10, 15], default=10, display_name="目标分段时长（秒）"),
                _gpt_text_model_input("analysis_model", display_name="分段分析模型"),
                IO.Int.Input("max_retries", display_name="每段最大重试次数", default=2, min=0, max=5, step=1),
                IO.Boolean.Input("resume", display_name="复用已完成分段", default=True),
                IO.Boolean.Input("force_rerun", display_name="强制重跑全部分段", default=False, advanced=True),
                IO.String.Input("negative_prompt", display_name="负面提示词", multiline=True, default="", optional=True, advanced=True),
                *asset_image_inputs(),
            ],
            outputs=[
                IO.Video.Output(display_name="最终视频"),
                IO.String.Output(display_name="最终视频路径"),
                IO.String.Output(display_name="任务 manifest 路径"),
                IO.String.Output(display_name="任务状态 JSON"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, analysis_model: str):
        return _validate_gpt_text_model(analysis_model)

    @classmethod
    def execute(
        cls,
        video: Any,
        assets_manifest: str,
        prompt: str,
        engine: str,
        model: str,
        segment_duration: int,
        analysis_model: str = DEFAULT_OPENAI_TEXT_MODEL,
        max_retries: int = 2,
        resume: bool = True,
        force_rerun: bool = False,
        negative_prompt: str = "",
        **kwargs,
    ):
        people = connected_asset_dict(kwargs, "person_", PERSON_IDS)
        backgrounds = connected_asset_dict(kwargs, "", BACKGROUND_IDS)
        normalized_engine = "wan" if str(engine).lower().startswith("wan") else "seedance"
        normalized_model = str(model).strip()
        if normalized_engine == "wan":
            normalized_model = "wan2.7-r2v-2026-06-12"
        elif not normalized_model.lower().startswith("seedance"):
            normalized_model = "Seedance 2.0 Fast"
        return IO.NodeOutput(
            *process_long_video(
                video=video,
                assets_manifest=assets_manifest,
                prompt=prompt,
                engine=normalized_engine,
                model=normalized_model,
                segment_duration=int(segment_duration),
                ai_model=analysis_model,
                max_retries=int(max_retries),
                resume=bool(resume),
                force_rerun=bool(force_rerun),
                negative_prompt=negative_prompt or "",
                people=people,
                backgrounds=backgrounds,
            )
        )


class CompanyGPTImage2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyGPTImage2",
            display_name="公司 GPT Image 2 图片生成",
            category="company-remote/image/GPT Image 2",
            description="通过 WisArt 或 AI-Zero-Token OpenAI 兼容接口生成 GPT Image 2 图片。",
            search_aliases=["Company GPT Image 2", "WisArt GPT Image 2", "AI Zero Token GPT Image 2", "gpt-image-2"],
            inputs=[
                IO.String.Input(
                    "prompt",
                    display_name="提示词",
                    default="",
                    multiline=True,
                    tooltip="用于生成图片的纯文本提示词，可直接连接“公司提示词优化”或“图片提示词优化节点”的输出。",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "gpt-image-2",
                            [
                                IO.Combo.Input(
                                    "size",
                                    default="auto",
                                    options=[
                                        "auto",
                                        "1024x1024",
                                        "1024x1536",
                                        "1536x1024",
                                        "2048x2048",
                                        "2048x1152",
                                        "1152x2048",
                                        "3840x2160",
                                        "2160x3840",
                                        "Custom",
                                    ],
                                    tooltip="Image size. Select 'Custom' to use the custom width and height.",
                                ),
                                IO.Int.Input(
                                    "custom_width",
                                    default=1024,
                                    min=1024,
                                    max=3840,
                                    step=16,
                                    tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16.",
                                ),
                                IO.Int.Input(
                                    "custom_height",
                                    default=1024,
                                    min=1024,
                                    max=3840,
                                    step=16,
                                    tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16.",
                                ),
                                IO.Combo.Input(
                                    "background",
                                    default="auto",
                                    options=["auto", "opaque"],
                                    tooltip="Return image with or without background.",
                                ),
                                *_gpt_image_shared_inputs(),
                            ],
                        ),
                        IO.DynamicCombo.Option("gpt-image-1.5", _gpt_image_legacy_model_inputs()),
                        IO.DynamicCombo.Option("gpt-image-1", _gpt_image_legacy_model_inputs()),
                    ],
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=1,
                    step=1,
                    tooltip="当前 WisArt 与 AI-Zero-Token 每次请求均生成 1 张图片。",
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="not implemented yet in backend",
                ),
                IO.Combo.Input(
                    "provider",
                    options=GPT_IMAGE_PROVIDER_OPTIONS,
                    default=GPT_IMAGE_PROVIDER_WISART,
                    display_name="图片服务",
                    tooltip="WisArt 使用当前 gptimage2 配置；AI-Zero-Token 使用本地 gpttext 地址并自动切换到图片接口。",
                ),
            ],
            outputs=[IO.Image.Output()],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        n: int,
        seed: int = 0,
        provider: str = GPT_IMAGE_PROVIDER_WISART,
    ):
        validate_string(prompt, field_name="prompt", min_length=1)

        model_id = model["model"]
        size = model["size"]
        background = model["background"]
        quality = model["quality"]
        custom_width = model.get("custom_width", 1024)
        custom_height = model.get("custom_height", 1024)

        images_dict = model.get("images") or {}
        image_tensors = [t for t in images_dict.values() if t is not None]
        n_images = sum(get_number_of_images(t) for t in image_tensors)
        mask = model.get("mask")

        if mask is not None and n_images == 0:
            raise ValueError("Cannot use a mask without an input image")

        if size == "Custom":
            if custom_width % 16 != 0 or custom_height % 16 != 0:
                raise ValueError(
                    f"Custom width and height must be multiples of 16, got {custom_width}x{custom_height}"
                )
            if max(custom_width, custom_height) > 3840:
                raise ValueError(
                    f"Custom resolution max edge must be <= 3840, got {custom_width}x{custom_height}"
                )
            ratio = max(custom_width, custom_height) / min(custom_width, custom_height)
            if ratio > 3:
                raise ValueError(
                    f"Custom resolution aspect ratio must not exceed 3:1, got {custom_width}x{custom_height}"
                )
            total_pixels = custom_width * custom_height
            if not 655_360 <= total_pixels <= 8_294_400:
                raise ValueError(
                    f"Custom resolution total pixels must be between 655,360 and 8,294,400, got {total_pixels}"
                )
            size = f"{custom_width}x{custom_height}"

        files = None
        if image_tensors:
            flat: list[torch.Tensor] = []
            for tensor in image_tensors:
                if len(tensor.shape) == 4:
                    flat.extend(tensor[i : i + 1] for i in range(tensor.shape[0]))
                else:
                    flat.append(tensor.unsqueeze(0))

            files = []
            for i, single_image in enumerate(flat):
                scaled_image = downscale_image_tensor(single_image, total_pixels=2048 * 2048).squeeze()
                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                if len(flat) == 1:
                    files.append(("image", (f"image_{i}.png", img_byte_arr, "image/png")))
                else:
                    files.append(("image[]", (f"image_{i}.png", img_byte_arr, "image/png")))

            if mask is not None:
                if len(flat) != 1:
                    raise Exception("Cannot use a mask with multiple image")
                ref_image = flat[0]
                if mask.shape[1:] != ref_image.shape[1:-1]:
                    raise Exception("Mask and Image must be the same size")
                _, height, width = mask.shape
                rgba_mask = torch.zeros(height, width, 4, device="cpu")
                rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
                scaled_mask = downscale_image_tensor(
                    rgba_mask.unsqueeze(0), total_pixels=2048 * 2048
                ).squeeze()
                mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_img_byte_arr = BytesIO()
                mask_img.save(mask_img_byte_arr, format="PNG")
                mask_img_byte_arr.seek(0)
                files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

        image = await asyncio.to_thread(
            generate_openai_image,
            get_gpt_image_provider_config(provider),
            prompt=prompt,
            model=model_id,
            size=size,
            background=background,
            quality=quality,
            n=n,
            files=files,
        )
        return IO.NodeOutput(image)


class CompanySeedreamImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedreamImage",
            display_name="公司 Seedream 图片生成 / 编辑",
            category="company-remote/image/Seedream",
            description="公司远程订阅 Seedream 风格图片生成 / 编辑节点，请求发送到可配置的公司接口。",
            search_aliases=["Company Seedream 4.5 & 5.0", "Seedream Image"],
            inputs=[
                IO.Combo.Input("model", options=SEEDREAM_MODEL_OPTIONS, display_name="模型"),
                IO.String.Input(
                    "prompt",
                    display_name="提示词",
                    multiline=True,
                    default="",
                    tooltip="用于生成或编辑图片的提示词。",
                ),
                IO.Image.Input(
                    "image",
                    display_name="图片",
                    tooltip="用于图生图或多参考图生成的输入图片。",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size_preset",
                    options=[label for label, _, _ in SEEDREAM_PRESETS],
                    display_name="尺寸",
                    tooltip="图片尺寸。选择 Custom 后使用下面的自定义宽度和高度。",
                ),
                IO.Int.Input(
                    "width",
                    display_name="自定义宽度",
                    default=2048,
                    min=1024,
                    max=6240,
                    step=2,
                    tooltip="自定义图片宽度，仅在尺寸选择 Custom 时生效。",
                    optional=True,
                ),
                IO.Int.Input(
                    "height",
                    display_name="自定义高度",
                    default=2048,
                    min=1024,
                    max=4992,
                    step=2,
                    tooltip="自定义图片高度，仅在尺寸选择 Custom 时生效。",
                    optional=True,
                ),
                IO.Combo.Input(
                    "sequential_image_generation",
                    options=["disabled", "auto"],
                    display_name="连续出图",
                    tooltip="连续出图模式。disabled 为单张生成，auto 由模型决定是否生成相关图片。",
                    optional=True,
                ),
                IO.Int.Input(
                    "max_images",
                    display_name="数量",
                    default=1,
                    min=1,
                    max=15,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="连续出图为 auto 时最多生成的图片数量。",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    display_name="种子",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="生成使用的随机种子。",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    display_name="水印",
                    default=False,
                    tooltip="如果平台支持，是否为图片添加 AI 生成水印。",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "fail_on_partial",
                    display_name="部分失败时中断",
                    default=True,
                    tooltip="传递给支持批量部分返回的平台；开启后可在部分失败时中断。",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[IO.Image.Output(display_name="图片"), IO.String.Output(display_name="图片路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model: str,
        prompt: str,
        image: Any = None,
        size_preset: str = SEEDREAM_PRESETS[0][0],
        width: int = 2048,
        height: int = 2048,
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        seed: int = 0,
        watermark: bool = False,
        fail_on_partial: bool = True,
    ):
        final_width, final_height = _parse_seedream_size(size_preset, width, height)
        return generate_image(
            _load_provider_config("seedream"),
            operation="seedream_image",
            model=model,
            prompt=prompt,
            width=final_width,
            height=final_height,
            seed=seed,
            max_images=max_images,
            reference_images=[image] if image is not None else [],
            extra_values={
                "size_preset": size_preset,
                "sequential_image_generation": sequential_image_generation,
                "watermark": watermark,
                "fail_on_partial": fail_on_partial,
            },
        )


class CompanySeedance2TextToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedance2TextToVideo",
            display_name="公司 Seedance 2.0 文生视频",
            category="company-remote/video/Seedance",
            description="公司远程订阅 Seedance 2.0 文生视频节点。",
            search_aliases=["Company Seedance 2.0 Text to Video", "Seedance Text to Video"],
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=_dynamic_seedance_options(_seedance2_text_inputs),
                    display_name="模型",
                    tooltip="Seedance 2.0 偏质量；Seedance 2.0 Fast 偏速度。",
                ),
                IO.Int.Input(
                    "seed",
                    display_name="种子",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="种子会影响节点是否重新运行；实际结果是否稳定取决于平台。",
                ),
                IO.Boolean.Input(
                    "watermark",
                    display_name="水印",
                    default=False,
                    tooltip="如果平台支持，是否为视频添加水印。",
                    advanced=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model: dict, seed: int = 0, watermark: bool = False):
        return generate_video(
            _load_provider_config("seedance2"),
            operation="seedance2_text_to_video",
            model=model.get("model", "Seedance 2.0"),
            prompt=model.get("prompt", ""),
            resolution=model.get("resolution", "720p"),
            ratio=model.get("ratio", "16:9"),
            duration=model.get("duration", 7),
            seed=seed,
            extra_values={"generate_audio": model.get("generate_audio", True), "watermark": watermark},
        )


class CompanySeedance2FirstLastFrame(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedance2FirstLastFrame",
            display_name="公司 Seedance 2.0 首尾帧视频",
            category="company-remote/video/Seedance",
            description="公司远程订阅 Seedance 2.0 首尾帧视频节点。",
            search_aliases=["Company Seedance 2.0 First-Last-Frame to Video", "Seedance First Last Frame"],
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=_dynamic_seedance_options(_seedance2_text_inputs, default_ratio="adaptive"),
                    display_name="模型",
                    tooltip="Seedance 2.0 偏质量；Seedance 2.0 Fast 偏速度。",
                ),
                IO.Image.Input("first_frame", display_name="首帧图", tooltip="视频首帧图片。", optional=True),
                IO.Image.Input("last_frame", display_name="尾帧图", tooltip="视频尾帧图片。", optional=True),
                IO.String.Input(
                    "first_frame_asset_id",
                    display_name="首帧资源 ID",
                    default="",
                    tooltip="平台侧首帧资源 ID；如平台要求，不能和首帧图片同时使用。",
                    optional=True,
                ),
                IO.String.Input(
                    "last_frame_asset_id",
                    display_name="尾帧资源 ID",
                    default="",
                    tooltip="平台侧尾帧资源 ID；如平台要求，不能和尾帧图片同时使用。",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    display_name="种子",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="种子会影响节点是否重新运行；实际结果是否稳定取决于平台。",
                ),
                IO.Boolean.Input(
                    "watermark",
                    display_name="水印",
                    default=False,
                    tooltip="如果平台支持，是否为视频添加水印。",
                    advanced=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        seed: int = 0,
        watermark: bool = False,
        first_frame: Any = None,
        last_frame: Any = None,
        first_frame_asset_id: str = "",
        last_frame_asset_id: str = "",
    ):
        return generate_video(
            _load_provider_config("seedance2"),
            operation="seedance2_first_last_frame",
            model=model.get("model", "Seedance 2.0"),
            prompt=model.get("prompt", ""),
            resolution=model.get("resolution", "720p"),
            ratio=model.get("ratio", "adaptive"),
            duration=model.get("duration", 7),
            seed=seed,
            first_frame=first_frame,
            last_frame=last_frame,
            extra_values={
                "generate_audio": model.get("generate_audio", True),
                "watermark": watermark,
                "first_frame_asset_id": first_frame_asset_id,
                "last_frame_asset_id": last_frame_asset_id,
            },
        )


class CompanySeedance2ReferenceVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanySeedance2ReferenceVideo",
            display_name="公司 Seedance 2.0 参考生成视频",
            category="company-remote/video/Seedance",
            description="公司远程订阅 Seedance 2.0 参考图 / 参考视频生成节点。",
            search_aliases=["Company Seedance 2.0 Reference to Video", "Seedance Reference Video"],
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=_dynamic_seedance_options(_seedance2_reference_inputs, default_ratio="adaptive"),
                    display_name="模型",
                    tooltip="Seedance 2.0 偏质量；Seedance 2.0 Fast 偏速度。",
                ),
                IO.Int.Input(
                    "seed",
                    display_name="种子",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="种子会影响节点是否重新运行；实际结果是否稳定取决于平台。",
                ),
                IO.Boolean.Input(
                    "watermark",
                    display_name="水印",
                    default=False,
                    tooltip="如果平台支持，是否为视频添加水印。",
                    advanced=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model: dict, seed: int = 0, watermark: bool = False):
        reference_images = _dict_values(model.get("reference_images", {}))
        reference_videos = _dict_values(model.get("reference_videos", {}))
        reference_assets = _dict_values(model.get("reference_assets", {}))
        return generate_video(
            _load_provider_config("seedance2"),
            operation="seedance2_reference_video",
            model=model.get("model", "Seedance 2.0"),
            prompt=model.get("prompt", ""),
            resolution=model.get("resolution", "720p"),
            ratio=model.get("ratio", "adaptive"),
            duration=model.get("duration", 7),
            seed=seed,
            reference_images=reference_images,
            reference_video=reference_videos[0] if reference_videos else None,
            reference_videos=reference_videos,
            extra_values={
                "generate_audio": model.get("generate_audio", True),
                "watermark": watermark,
                "auto_downscale": model.get("auto_downscale", True),
                "auto_upscale": model.get("auto_upscale", False),
                "reference_assets": reference_assets,
            },
        )


class CompanyKlingImageToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CompanyKlingImageToVideo",
            display_name="公司 Kling 图生视频",
            category="company-remote/video/Kling",
            description="公司远程订阅 Kling 图生视频节点。",
            search_aliases=["Company Kling Image to Video", "Kling Image(First Frame) to Video"],
            inputs=[
                IO.Image.Input("start_frame", display_name="首帧图", tooltip="用于生成视频的参考图片。"),
                IO.String.Input("prompt", display_name="提示词", multiline=True, tooltip="正向提示词。"),
                IO.String.Input("negative_prompt", display_name="负面提示词", multiline=True, tooltip="负向提示词。"),
                IO.Combo.Input("model_name", options=KLING_MODEL_OPTIONS, default="kling-v2-master", display_name="模型"),
                IO.Float.Input("cfg_scale", display_name="提示词相关度", default=0.8, min=0.0, max=1.0),
                IO.Combo.Input("mode", options=KLING_MODE_OPTIONS, default="std", display_name="模式"),
                IO.Combo.Input("aspect_ratio", options=KLING_ASPECT_RATIO_OPTIONS, default="16:9", display_name="比例"),
                IO.Combo.Input("duration", options=KLING_DURATION_OPTIONS, default="5", display_name="时长"),
            ],
            outputs=[
                IO.Video.Output(display_name="视频"),
                IO.String.Output(display_name="视频路径"),
                IO.String.Output(display_name="视频 ID"),
                IO.String.Output(display_name="时长"),
            ],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        start_frame: Any,
        prompt: str,
        negative_prompt: str,
        model_name: str,
        cfg_scale: float,
        mode: str,
        aspect_ratio: str,
        duration: str,
    ):
        video, path = generate_video(
            _load_provider_config("kling"),
            operation="kling_image_to_video",
            model=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=float(duration),
            ratio=aspect_ratio,
            image=start_frame,
            extra_values={"model_name": model_name, "cfg_scale": cfg_scale, "mode": mode},
        )
        return video, path, "", str(duration)


class CompanyViduImageToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyViduImageToVideo",
            display_name="公司 Vidu Q3 图生视频",
            category="company-remote/video/Vidu",
            description="公司远程订阅 Vidu Q3 图生视频节点。",
            search_aliases=["Company Vidu Q3 Image-to-Video Generation", "Vidu Image to Video"],
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    display_name="模型",
                    options=[
                        IO.DynamicCombo.Option(
                            "viduq3-pro",
                            [
                                IO.Combo.Input("resolution", options=["720p", "1080p", "2K"], display_name="分辨率", tooltip="输出视频分辨率。"),
                                IO.Int.Input(
                                    "duration",
                                    display_name="时长",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="输出视频时长，单位秒。",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    display_name="生成音频",
                                    default=False,
                                    tooltip="开启后输出带声音的视频，包含对白和音效。",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "viduq3-turbo",
                            [
                                IO.Combo.Input("resolution", options=["720p", "1080p"], display_name="分辨率", tooltip="输出视频分辨率。"),
                                IO.Int.Input(
                                    "duration",
                                    display_name="时长",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="输出视频时长，单位秒。",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    display_name="生成音频",
                                    default=False,
                                    tooltip="开启后输出带声音的视频，包含对白和音效。",
                                ),
                            ],
                        ),
                    ],
                    tooltip="用于视频生成的模型。",
                ),
                IO.Image.Input("image", display_name="图片", tooltip="作为生成视频首帧的图片。"),
                IO.String.Input("prompt", display_name="提示词", multiline=True, default="", tooltip="可选的视频生成提示词。"),
                IO.Int.Input(
                    "seed",
                    display_name="种子",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model: dict, image: Any, prompt: str, seed: int = 1):
        return generate_video(
            _load_provider_config("vidu"),
            operation="vidu_image_to_video",
            model=model.get("model", "viduq3-pro"),
            prompt=prompt,
            resolution=model.get("resolution", "720p"),
            duration=model.get("duration", 5),
            seed=seed,
            image=image,
            extra_values={"audio": model.get("audio", False)},
        )


class CompanyMiniMaxHailuoVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="CompanyMiniMaxHailuoVideo",
            display_name="公司 MiniMax / Hailuo 视频",
            category="company-remote/video/MiniMax",
            description="公司远程订阅 MiniMax / Hailuo 视频生成节点。",
            search_aliases=["Company MiniMax Hailuo Video", "MiniMax Hailuo Video"],
            inputs=[
                IO.String.Input(
                    "prompt_text",
                    display_name="提示词",
                    multiline=True,
                    default="",
                    tooltip="用于引导视频生成的提示词。",
                ),
                IO.Int.Input(
                    "seed",
                    display_name="种子",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="用于生成的随机种子。",
                    optional=True,
                ),
                IO.Image.Input(
                    "first_frame_image",
                    display_name="首帧图",
                    tooltip="可选，用作视频首帧的图片。",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "prompt_optimizer",
                    display_name="优化提示词",
                    default=True,
                    tooltip="按需优化提示词以提升生成质量。",
                    optional=True,
                ),
                IO.Combo.Input(
                    "duration",
                    options=[6, 10],
                    display_name="时长",
                    default=6,
                    tooltip="输出视频时长，单位秒。",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["768P", "1080P"],
                    display_name="分辨率",
                    default="768P",
                    tooltip="输出视频分辨率。1080P 为 1920x1080，768P 为 1366x768。",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        prompt_text: str,
        seed: int = 0,
        first_frame_image: Any = None,
        prompt_optimizer: bool = True,
        duration: int = 6,
        resolution: str = "768P",
    ):
        return generate_video(
            _load_provider_config("minimax_hailuo"),
            operation="minimax_hailuo_video",
            model="MiniMax-Hailuo-02",
            prompt=prompt_text,
            duration=float(duration),
            resolution=resolution,
            seed=seed,
            image=first_frame_image,
            extra_values={"prompt_optimizer": prompt_optimizer},
        )


def _aliyun_video_common_inputs(models: list[str], *, with_ratio: bool = True) -> list[Any]:
    inputs: list[Any] = [
        IO.Combo.Input("model", options=models, display_name="模型"),
        IO.String.Input("prompt", display_name="提示词", multiline=True, default=""),
        IO.Combo.Input("resolution", options=["720P", "1080P"], default="720P", display_name="分辨率"),
    ]
    if with_ratio:
        inputs.append(IO.Combo.Input("ratio", options=ALIYUN_VIDEO_RATIOS, default="16:9", display_name="画幅比例"))
    inputs.extend([
        IO.Int.Input(
            "duration",
            display_name="时长（秒）",
            default=5,
            min=2,
            max=15,
            step=1,
            display_mode=IO.NumberDisplay.slider,
        ),
        IO.String.Input(
            "negative_prompt",
            display_name="负面提示词",
            multiline=True,
            default="",
            optional=True,
            advanced=True,
        ),
        IO.Boolean.Input("prompt_extend", display_name="智能改写", default=True, advanced=True),
        IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
        IO.Int.Input(
            "seed",
            display_name="种子",
            default=0,
            min=0,
            max=2147483647,
            step=1,
            display_mode=IO.NumberDisplay.number,
            control_after_generate=True,
        ),
    ])
    return inputs


class CompanyAliyunQwenImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyAliyunQwenImage",
            display_name="阿里云 Qwen Image 2.0 生成 / 编辑",
            category="company-remote/image/Alibaba Cloud",
            description="调用阿里云百炼 Qwen Image 2.0 Pro。未连接参考图时文生图，连接后图像编辑。",
            search_aliases=["Alibaba Cloud Qwen Image", "DashScope Qwen Image"],
            inputs=[
                IO.String.Input("prompt", display_name="提示词", multiline=True, default=""),
                IO.Combo.Input("model", options=ALIYUN_QWEN_IMAGE_MODELS, display_name="模型"),
                IO.Autogrow.Input(
                    "reference_images",
                    display_name="参考图片（可选）",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_image", display_name="参考图片"),
                        names=[f"image_{index}" for index in range(1, 4)],
                        min=0,
                    ),
                ),
                IO.Combo.Input(
                    "size",
                    options=["2048x2048", "2688x1536", "1536x2688", "2368x1728", "1728x2368"],
                    default="2048x2048",
                    display_name="尺寸",
                ),
                IO.String.Input(
                    "negative_prompt",
                    display_name="负面提示词",
                    multiline=True,
                    default="",
                    optional=True,
                ),
                IO.Int.Input("n", display_name="图片数量", default=1, min=1, max=6, step=1),
                IO.Boolean.Input("prompt_extend", display_name="智能改写", default=True),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input(
                    "seed",
                    display_name="种子",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
            ],
            outputs=[IO.Image.Output(display_name="图片")],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        size: str,
        negative_prompt: str = "",
        n: int = 1,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: int = 0,
        reference_images: Any = None,
    ):
        image = await asyncio.to_thread(
            generate_dashscope_image,
            _load_provider_config("aliyun_dashscope_image"),
            prompt=prompt,
            model=model,
            size=size,
            negative_prompt=negative_prompt,
            n=n,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            reference_images=_dict_values(reference_images),
        )
        return IO.NodeOutput(image)


class CompanyAliyunTextToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyAliyunTextToVideo",
            display_name="阿里云 Wan / HappyHorse 文生视频",
            category="company-remote/video/Alibaba Cloud",
            description="调用阿里云百炼 Wan 2.7 或 HappyHorse 文生视频接口。",
            inputs=_aliyun_video_common_inputs(ALIYUN_TEXT_TO_VIDEO_MODELS),
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model: str, prompt: str, resolution: str, ratio: str, duration: int,
                negative_prompt: str = "", prompt_extend: bool = True, watermark: bool = False, seed: int = 0):
        return generate_dashscope_video(
            _load_provider_config("aliyun_dashscope_video"),
            operation="dashscope_text_to_video",
            model=model,
            prompt=prompt,
            resolution=resolution,
            ratio=ratio,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
        )


class CompanyAliyunImageToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyAliyunImageToVideo",
            display_name="阿里云 Wan / HappyHorse 首帧图生视频",
            category="company-remote/video/Alibaba Cloud",
            description="调用阿里云百炼 Wan 2.7 或 HappyHorse 首帧图生视频接口。",
            inputs=[
                IO.Image.Input("first_frame", display_name="首帧图片"),
                *_aliyun_video_common_inputs(ALIYUN_IMAGE_TO_VIDEO_MODELS, with_ratio=False),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, first_frame: Any, model: str, prompt: str, resolution: str, duration: int,
                negative_prompt: str = "", prompt_extend: bool = True, watermark: bool = False, seed: int = 0):
        return generate_dashscope_video(
            _load_provider_config("aliyun_dashscope_video"),
            operation="dashscope_image_to_video",
            model=model,
            prompt=prompt,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            first_frame=first_frame,
        )


class CompanyAliyunReferenceToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyAliyunReferenceToVideo",
            display_name="阿里云 Wan / HappyHorse 参考生视频",
            category="company-remote/video/Alibaba Cloud",
            description="Wan 2.7 支持参考图片和视频；HappyHorse 仅支持参考图片。",
            inputs=[
                *_aliyun_video_common_inputs(ALIYUN_REFERENCE_TO_VIDEO_MODELS),
                IO.Autogrow.Input(
                    "reference_images",
                    display_name="参考图片",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_image", display_name="参考图片"),
                        names=[f"image_{index}" for index in range(1, 10)],
                        min=0,
                    ),
                ),
                IO.Autogrow.Input(
                    "reference_videos",
                    display_name="参考视频（仅 Wan）",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("reference_video", display_name="参考视频"),
                        names=[f"video_{index}" for index in range(1, 6)],
                        min=0,
                    ),
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model: str, prompt: str, resolution: str, ratio: str, duration: int,
                negative_prompt: str = "", prompt_extend: bool = True, watermark: bool = False,
                seed: int = 0, reference_images: Any = None, reference_videos: Any = None):
        return generate_dashscope_video(
            _load_provider_config("aliyun_dashscope_video"),
            operation="dashscope_reference_to_video",
            model=model,
            prompt=prompt,
            resolution=resolution,
            ratio=ratio,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            reference_images=_dict_values(reference_images),
            reference_videos=_dict_values(reference_videos),
        )


class CompanyWan27ThreeImageDirectVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyWan27ThreeImageDirectVideo",
            display_name="Wan 2.7 三人物参考图直传视频",
            category="company-remote/video/Alibaba Cloud",
            description="把三张参考图以 Base64 直接提交给阿里云百炼 Wan 2.7，不经过 TOS 或火山资产库。",
            inputs=[
                IO.Image.Input("image_a", display_name="参考图片 1（人物 A）"),
                IO.Image.Input("image_b", display_name="参考图片 2（人物 B）"),
                IO.Image.Input("image_c", display_name="参考图片 3（人物 C）"),
                IO.String.Input("prompt", display_name="视频提示词", multiline=True, default=""),
                IO.Combo.Input(
                    "model",
                    display_name="模型",
                    options=["wan2.7-r2v-2026-06-12"],
                    default="wan2.7-r2v-2026-06-12",
                ),
                IO.Combo.Input("resolution", display_name="分辨率", options=["720P", "1080P"], default="720P"),
                IO.Combo.Input("ratio", display_name="画幅比例", options=ALIYUN_VIDEO_RATIOS, default="16:9"),
                IO.Int.Input("duration", display_name="时长（秒）", default=10, min=2, max=15, step=1),
                IO.Boolean.Input("prompt_extend", display_name="智能改写", default=True, advanced=True),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input("seed", display_name="种子", default=0, min=0, max=2147483647, step=1),
                IO.String.Input("negative_prompt", display_name="负面提示词", multiline=True, default="", optional=True, advanced=True),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        image_a: Any,
        image_b: Any,
        image_c: Any,
        prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        duration: int,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: int = 0,
        negative_prompt: str = "",
    ):
        config = copy.copy(_load_provider_config("aliyun_dashscope_video_direct"))
        config.tos_enabled = False
        config.media_delivery = "base64"
        return generate_dashscope_video(
            config,
            operation="dashscope_reference_to_video",
            model=model,
            prompt=prompt,
            resolution=resolution,
            ratio=ratio,
            duration=duration,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            reference_images=[image_a, image_b, image_c],
            reference_videos=[],
        )


class CompanyWan27ThreePersonVideoEdit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyWan27ThreePersonVideoEdit",
            display_name="Wan 2.7 原视频三人物替换",
            category="company-remote/video/Alibaba Cloud",
            description=(
                "输入原视频和三张人物参考图。原视频自动上传到百炼 48 小时临时 OSS，"
                "三张图片以 Base64 直传；不使用 TOS 或火山资产库。"
            ),
            inputs=[
                IO.Video.Input("video", display_name="原视频（2-10 秒）"),
                IO.Image.Input("image_a", display_name="图1：替换原视频人物 A"),
                IO.Image.Input("image_b", display_name="图2：替换原视频人物 B"),
                IO.Image.Input("image_c", display_name="图3：替换原视频人物 C"),
                IO.String.Input("prompt", display_name="人物替换指令", multiline=True, default=""),
                IO.Combo.Input(
                    "model",
                    display_name="模型",
                    options=["wan2.7-videoedit"],
                    default="wan2.7-videoedit",
                ),
                IO.Combo.Input("resolution", display_name="分辨率", options=["720P", "1080P"], default="720P"),
                IO.Int.Input("duration", display_name="截断时长（0=原视频）", default=0, min=0, max=10, step=1),
                IO.Combo.Input("audio_setting", display_name="声音", options=["origin", "auto"], default="origin"),
                IO.Boolean.Input("prompt_extend", display_name="智能改写", default=True, advanced=True),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input("seed", display_name="种子", default=0, min=0, max=2147483647, step=1),
                IO.String.Input(
                    "negative_prompt",
                    display_name="负面提示词",
                    multiline=True,
                    default="",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        image_a: Any,
        image_b: Any,
        image_c: Any,
        prompt: str,
        model: str,
        resolution: str,
        duration: int = 0,
        audio_setting: str = "origin",
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: int = 0,
        negative_prompt: str = "",
    ):
        config = copy.copy(_load_provider_config("aliyun_dashscope_video_direct"))
        config.tos_enabled = False
        config.media_delivery = "base64"
        config.extra_headers = {
            **config.extra_headers,
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        }
        return generate_dashscope_video(
            config,
            operation="dashscope_video_edit",
            model=model,
            prompt=prompt,
            resolution=resolution,
            duration=int(duration),
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            edit_video=video,
            reference_images=[image_a, image_b, image_c],
            audio_setting=audio_setting,
        )


class CompanyWan30ThreePersonVideoEdit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyWan30ThreePersonVideoEdit",
            display_name="Wan 3.0 原视频三人物替换",
            category="company-remote/video/Alibaba Cloud",
            description=(
                "输入原视频和三张人物参考图，调用百炼 wan3.0-video。"
                "原视频自动上传到百炼临时 OSS，三张图片以 Base64 直传；不使用 TOS 或火山资产库。"
            ),
            inputs=[
                IO.Video.Input("video", display_name="原视频（2-30 秒）"),
                IO.Image.Input("image_a", display_name="图1：替换原视频人物 A"),
                IO.Image.Input("image_b", display_name="图2：替换原视频人物 B"),
                IO.Image.Input("image_c", display_name="图3：替换原视频人物 C"),
                IO.String.Input("prompt", display_name="人物替换指令", multiline=True, default=""),
                IO.Combo.Input("model", display_name="模型", options=["wan3.0-video"], default="wan3.0-video"),
                IO.Combo.Input("resolution", display_name="分辨率", options=["480P", "720P", "1080P"], default="720P"),
                IO.Int.Input("duration", display_name="目标时长（秒）", default=8, min=2, max=30, step=1),
                IO.Combo.Input("audio_setting", display_name="模型生成音频", options=["origin", "auto"], default="origin"),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input("seed", display_name="种子", default=0, min=0, max=2147483647, step=1),
                IO.String.Input(
                    "negative_prompt",
                    display_name="负面提示词",
                    multiline=True,
                    default="",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Any,
        image_a: Any,
        image_b: Any,
        image_c: Any,
        prompt: str,
        model: str,
        resolution: str,
        duration: int = 8,
        audio_setting: str = "origin",
        watermark: bool = False,
        seed: int = 0,
        negative_prompt: str = "",
    ):
        config = copy.copy(_load_provider_config("aliyun_dashscope_video_direct"))
        config.tos_enabled = False
        config.media_delivery = "base64"
        config.extra_headers = {
            key: value
            for key, value in config.extra_headers.items()
            if key.lower() != "x-dashscope-async"
        }
        config.extra_headers["X-DashScope-OssResourceResolve"] = "enable"
        return generate_dashscope_video(
            config,
            operation="dashscope_video_edit",
            model=model,
            prompt=prompt,
            resolution=resolution,
            duration=int(duration),
            negative_prompt=negative_prompt,
            prompt_extend=False,
            watermark=watermark,
            seed=seed,
            edit_video=video,
            reference_images=[image_a, image_b, image_c],
            audio_setting=audio_setting,
        )


class CompanyAliyunVideoEdit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyAliyunVideoEdit",
            display_name="阿里云 HappyHorse 视频编辑",
            category="company-remote/video/Alibaba Cloud",
            description="调用阿里云百炼 HappyHorse 视频编辑接口。",
            inputs=[
                IO.Video.Input("video", display_name="待编辑视频"),
                IO.String.Input("prompt", display_name="编辑指令", multiline=True, default=""),
                IO.Combo.Input("model", options=ALIYUN_VIDEO_EDIT_MODELS, display_name="模型"),
                IO.Combo.Input("resolution", options=["720P", "1080P"], default="720P", display_name="分辨率"),
                IO.Autogrow.Input(
                    "reference_images",
                    display_name="参考图片",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_image", display_name="参考图片"),
                        names=[f"image_{index}" for index in range(1, 6)],
                        min=0,
                    ),
                ),
                IO.Combo.Input("audio_setting", options=["auto", "origin"], default="auto", display_name="声音"),
                IO.Boolean.Input("watermark", display_name="水印", default=False, advanced=True),
                IO.Int.Input(
                    "seed", display_name="种子", default=0, min=0, max=2147483647, step=1,
                    display_mode=IO.NumberDisplay.number, control_after_generate=True,
                ),
            ],
            outputs=[IO.Video.Output(display_name="视频"), IO.String.Output(display_name="视频路径")],
            is_api_node=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Any, prompt: str, model: str, resolution: str,
                reference_images: Any = None, audio_setting: str = "auto",
                watermark: bool = False, seed: int = 0):
        return generate_dashscope_video(
            _load_provider_config("aliyun_dashscope_video"),
            operation="dashscope_video_edit",
            model=model,
            prompt=prompt,
            resolution=resolution,
            watermark=watermark,
            seed=seed,
            edit_video=video,
            reference_images=_dict_values(reference_images),
            audio_setting=audio_setting,
        )


NODE_CLASS_MAPPINGS = {
    "CompanyPromptEnhancer": CompanyPromptEnhancer,
    "CompanyImagePromptEnhancer": CompanyImagePromptEnhancer,
    "CompanyPersistentPromptDisplay": CompanyPersistentPromptDisplay,
    "CompanyFixedColumnImagePreview": CompanyFixedColumnImagePreview,
    "CompanyMultiPersonPromptAnalyzer": CompanyMultiPersonPromptAnalyzer,
    "CompanyLongVideoAssetManifest": CompanyLongVideoAssetManifest,
    "CompanyLongVideoMappingAnalyzer": CompanyLongVideoMappingAnalyzer,
    "CompanyLongVideoAssetLoader": CompanyLongVideoAssetLoader,
    "CompanyLongVideoShotDetector": CompanyLongVideoShotDetector,
    "CompanyLongVideoShotInspector": CompanyLongVideoShotInspector,
    "CompanyLongVideoContinuityRangeSelector": CompanyLongVideoContinuityRangeSelector,
    "CompanyLongVideoLengthRangeSelector": CompanyLongVideoLengthRangeSelector,
    "CompanyLongVideoManualBatchRangeSelector": CompanyLongVideoManualBatchRangeSelector,
    "CompanyLongVideoManualBatchPlannerV1": CompanyLongVideoManualBatchPlannerV1,
    "CompanyLongVideoManualBatchFinalizerV1": CompanyLongVideoManualBatchFinalizerV1,
    "CompanyLongVideoDurationAdapter": CompanyLongVideoDurationAdapter,
    "CompanyLongVideoSegmentPlanner": CompanyLongVideoSegmentPlanner,
    "CompanyLongVideoAutoAssetPlanner": CompanyLongVideoAutoAssetPlanner,
    "CompanyLongVideoAnimeAssetPlanner": CompanyLongVideoAnimeAssetPlanner,
    "CompanyLongVideoAnimeAssetPlannerV3": CompanyLongVideoAnimeAssetPlannerV3,
    "CompanyLongVideoAutoAssetBuilder": CompanyLongVideoAutoAssetBuilder,
    "CompanyLongVideoAssetLibraryViewer": CompanyLongVideoAssetLibraryViewer,
    "CompanyLongVideoAutoReferencePacker": CompanyLongVideoAutoReferencePacker,
    "CompanyLongVideoPipelineAssetVideoGenerator": CompanyLongVideoPipelineAssetVideoGenerator,
    "CompanyLongVideoSegmentAnalyzer": CompanyLongVideoSegmentAnalyzer,
    "CompanyLongVideoReferenceMatcher": CompanyLongVideoReferenceMatcher,
    "CompanyLongVideoSegmentGenerator": CompanyLongVideoSegmentGenerator,
    "CompanyLongVideoParallelSegmentGenerator": CompanyLongVideoParallelSegmentGenerator,
    "CompanyLongVideoResultCollector": CompanyLongVideoResultCollector,
    "CompanyLongVideoContinuityPreview": CompanyLongVideoContinuityPreview,
    "CompanyLongVideoFinalMerger": CompanyLongVideoFinalMerger,
    "CompanyLongVideoRestyle": CompanyLongVideoRestyle,
    "CompanyGPTImage2": CompanyGPTImage2,
    "CompanySeedreamImage": CompanySeedreamImage,
    "CompanySeedance2TextToVideo": CompanySeedance2TextToVideo,
    "CompanySeedance2FirstLastFrame": CompanySeedance2FirstLastFrame,
    "CompanySeedance2ReferenceVideo": CompanySeedance2ReferenceVideo,
    "CompanyKlingImageToVideo": CompanyKlingImageToVideo,
    "CompanyViduImageToVideo": CompanyViduImageToVideo,
    "CompanyMiniMaxHailuoVideo": CompanyMiniMaxHailuoVideo,
    "CompanyAliyunQwenImage": CompanyAliyunQwenImage,
    "CompanyAliyunTextToVideo": CompanyAliyunTextToVideo,
    "CompanyAliyunImageToVideo": CompanyAliyunImageToVideo,
    "CompanyAliyunReferenceToVideo": CompanyAliyunReferenceToVideo,
    "CompanyWan27ThreeImageDirectVideo": CompanyWan27ThreeImageDirectVideo,
    "CompanyWan27ThreePersonVideoEdit": CompanyWan27ThreePersonVideoEdit,
    "CompanyWan30ThreePersonVideoEdit": CompanyWan30ThreePersonVideoEdit,
    "CompanyAliyunVideoEdit": CompanyAliyunVideoEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompanyPromptEnhancer": "公司提示词优化",
    "CompanyImagePromptEnhancer": "图片提示词优化节点",
    "CompanyPersistentPromptDisplay": "持久化提示词显示",
    "CompanyFixedColumnImagePreview": "固定列数图片预览",
    "CompanyMultiPersonPromptAnalyzer": "多人角色识别与提示词拆分",
    "CompanyLongVideoAssetManifest": "长视频欧美化资产清单",
    "CompanyLongVideoMappingAnalyzer": "长视频人物与背景映射分析",
    "CompanyLongVideoAssetLoader": "读取长视频资产清单",
    "CompanyLongVideoShotDetector": "长视频镜头检测",
    "CompanyLongVideoShotInspector": "分镜检测结果检查",
    "CompanyLongVideoContinuityRangeSelector": "连续分镜测试范围选择",
    "CompanyLongVideoLengthRangeSelector": "按时长/百分比选择生成范围",
    "CompanyLongVideoManualBatchRangeSelector": "手动批次范围与续接控制",
    "CompanyLongVideoManualBatchPlannerV1": "手动批次 Seedance 资产任务规划 v1",
    "CompanyLongVideoManualBatchFinalizerV1": "提交当前批次并等待人工审阅",
    "CompanyLongVideoDurationAdapter": "镜头时长适配与任务规划",
    "CompanyLongVideoSegmentPlanner": "长视频切分与任务规划",
    "CompanyLongVideoAutoAssetPlanner": "按镜头自动资产任务规划",
    "CompanyLongVideoAnimeAssetPlanner": "人物视频多风格资产任务规划",
    "CompanyLongVideoAnimeAssetPlannerV3": "人物视频多风格资产任务规划 v3（短镜头合并 / 音频可选）",
    "CompanyLongVideoAutoAssetBuilder": "分镜首尾帧分析与自动欧美替换素材",
    "CompanyLongVideoAssetLibraryViewer": "查看入库素材库资源",
    "CompanyLongVideoAutoReferencePacker": "自动参考素材打包",
    "CompanyLongVideoPipelineAssetVideoGenerator": "实时预览素材并生成 Seedance",
    "CompanyLongVideoSegmentAnalyzer": "GPT 分段人物与背景分析",
    "CompanyLongVideoReferenceMatcher": "分段参考素材匹配",
    "CompanyLongVideoSegmentGenerator": "顺序生成全部视频分段",
    "CompanyLongVideoParallelSegmentGenerator": "并行生成视频分段（每段完成立即显示）",
    "CompanyLongVideoResultCollector": "分段结果列表",
    "CompanyLongVideoContinuityPreview": "连续分镜生成结果预览",
    "CompanyLongVideoFinalMerger": "合并分段并恢复原音频",
    "CompanyLongVideoRestyle": "长视频欧美化分段转绘",
    "CompanyGPTImage2": "公司 GPT Image 2 图片生成",
    "CompanySeedreamImage": "公司 Seedream 图片生成 / 编辑",
    "CompanySeedance2TextToVideo": "公司 Seedance 2.0 文生视频",
    "CompanySeedance2FirstLastFrame": "公司 Seedance 2.0 首尾帧视频",
    "CompanySeedance2ReferenceVideo": "公司 Seedance 2.0 参考生成视频",
    "CompanyKlingImageToVideo": "公司 Kling 图生视频",
    "CompanyViduImageToVideo": "公司 Vidu Q3 图生视频",
    "CompanyMiniMaxHailuoVideo": "公司 MiniMax / Hailuo 视频",
    "CompanyAliyunQwenImage": "阿里云 Qwen Image 2.0 生成 / 编辑",
    "CompanyAliyunTextToVideo": "阿里云 Wan / HappyHorse 文生视频",
    "CompanyAliyunImageToVideo": "阿里云 Wan / HappyHorse 首帧图生视频",
    "CompanyAliyunReferenceToVideo": "阿里云 Wan / HappyHorse 参考生视频",
    "CompanyWan27ThreeImageDirectVideo": "Wan 2.7 三人物参考图直传视频",
    "CompanyWan27ThreePersonVideoEdit": "Wan 2.7 原视频三人物替换",
    "CompanyWan30ThreePersonVideoEdit": "Wan 3.0 原视频三人物替换",
    "CompanyAliyunVideoEdit": "阿里云 HappyHorse 视频编辑",
}
