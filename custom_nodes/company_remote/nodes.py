from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import IO
from comfy_api_nodes.util import downscale_image_tensor, get_number_of_images, validate_string

from .client import (
    DEFAULT_OPENAI_TEXT_MODEL,
    generate_image,
    generate_openai_chat_text,
    generate_openai_image,
    generate_openai_image_prompt_text,
    generate_video,
    get_cached_openai_model_ids,
)
from .config_store import ConfigError, get_config, load_configs


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

PROVIDER_ALIASES = {
    "gptimage2": ["gptimage2", "gpt_image2", "gpt-image-2", "gpt image 2", "ai_zero_token"],
    "gpttext": ["gpttext", "gpt_text", "gpt-text", "openai_chat", "prompt_enhancer", "ai_zero_token_text"],
    "seedream": ["seedream", "seedream_image", "bytedance_seedream"],
    "seedance2": ["seedance2", "seedance", "seedance_2", "seedance 2.0", "bytedance_seedance"],
    "kling": ["kling", "kling_image_to_video"],
    "vidu": ["vidu", "viduq3", "vidu_q3"],
    "minimax_hailuo": ["minimax_hailuo", "minimax", "hailuo", "minimax-hailuo"],
}


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


def _gpt_text_model_input():
    models = get_cached_openai_model_ids()
    default = DEFAULT_OPENAI_TEXT_MODEL if DEFAULT_OPENAI_TEXT_MODEL in models else models[0]
    return IO.Combo.Input(
        "model",
        options=models,
        default=default,
        display_name="模型",
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


class CompanyGPTImage2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CompanyGPTImage2",
            display_name="公司 GPT Image 2 图片生成",
            category="company-remote/image/GPT Image 2",
            description="通过 AI-Zero-Token 本地 OpenAI 兼容接口生成 GPT Image 2 图片。",
            search_aliases=["Company GPT Image 2", "AI Zero Token GPT Image 2", "gpt-image-2"],
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
                    tooltip="AI-Zero-Token 当前每次只支持生成 1 张图片。",
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
            _load_provider_config("gptimage2"),
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


NODE_CLASS_MAPPINGS = {
    "CompanyPromptEnhancer": CompanyPromptEnhancer,
    "CompanyImagePromptEnhancer": CompanyImagePromptEnhancer,
    "CompanyGPTImage2": CompanyGPTImage2,
    "CompanySeedreamImage": CompanySeedreamImage,
    "CompanySeedance2TextToVideo": CompanySeedance2TextToVideo,
    "CompanySeedance2FirstLastFrame": CompanySeedance2FirstLastFrame,
    "CompanySeedance2ReferenceVideo": CompanySeedance2ReferenceVideo,
    "CompanyKlingImageToVideo": CompanyKlingImageToVideo,
    "CompanyViduImageToVideo": CompanyViduImageToVideo,
    "CompanyMiniMaxHailuoVideo": CompanyMiniMaxHailuoVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompanyPromptEnhancer": "公司提示词优化",
    "CompanyImagePromptEnhancer": "图片提示词优化节点",
    "CompanyGPTImage2": "公司 GPT Image 2 图片生成",
    "CompanySeedreamImage": "公司 Seedream 图片生成 / 编辑",
    "CompanySeedance2TextToVideo": "公司 Seedance 2.0 文生视频",
    "CompanySeedance2FirstLastFrame": "公司 Seedance 2.0 首尾帧视频",
    "CompanySeedance2ReferenceVideo": "公司 Seedance 2.0 参考生成视频",
    "CompanyKlingImageToVideo": "公司 Kling 图生视频",
    "CompanyViduImageToVideo": "公司 Vidu Q3 图生视频",
    "CompanyMiniMaxHailuoVideo": "公司 MiniMax / Hailuo 视频",
}
