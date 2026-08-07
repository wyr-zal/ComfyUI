import { app } from "../../scripts/app.js";

const NODE_TYPE = "CompanyLongVideoAnimeAssetPlanner";
const TYPE_WIDGET = "target_resource_type";
const PROMPT_WIDGET = "prompt";
const NEGATIVE_WIDGET = "negative_prompt";
const LAST_TYPE_PROPERTY = "company_target_resource_type";

const PRESETS = {
  "欧美化资源": {
    prompt:
      "把本段重新演绎为完整、鲜明且统一的欧美化视频。参考图片是最终人物身份、服装、道具和环境美术的唯一视觉依据；" +
      "人物必须整体重设计为符合当代欧美审美的外国人物，而不是只更换面孔：统一调整面部地域特征、发型发色、妆容、服装鞋履、配饰、版型剪裁、材质配色和人物气质，清除残留的本土东方造型语言。" +
      "环境必须彻底重构为真实可信、地域统一的欧美国家场景，而不是轻微调色或只替换少量道具；建筑语言、道路与公共设施、家具陈设、材质、植被、照明和生活细节都应符合选定地域的真实逻辑。" +
      "保持输入参考图对应的视觉媒介：真人素材保持高质量欧美真人电影质感，动漫、漫画或 CG 素材保持同一媒介并改成欧美版本。" +
      "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能，根据镜头分析文字自然设计动作、表情、走位和镜头运动；" +
      "从第一帧到最后一帧保持人物身份、整体造型、欧美环境、媒介和光影稳定一致，不得恢复原人物或原背景，不得出现字幕、Logo 或水印。",
    negative:
      "只换脸，亚洲面孔残留，本土东方造型，中式古装，中式建筑，中式家具，中文招牌，本土化道路设施，" +
      "原人物残留，原服装残留，原背景残留，轻微调色，少量道具替换，地域混乱，媒介变化，风格漂移，" +
      "人物复制，身份变化，多余人物，肢体畸形，额外手指，背景跳变，字幕，文字，Logo，水印",
  },
  "真人写实资源": {
    prompt:
      "把本段重新演绎为统一的高质量真人影视视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；" +
      "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。" +
      "从第一帧到最后一帧保持自然真实的人脸、皮肤、头发、布料、建筑材质和电影光影，人物与完整背景必须稳定一致；" +
      "不得出现动漫线稿、卡通脸、插画笔触、游戏 CG、塑料皮肤、半真人半卡通、风格闪回、字幕、Logo 或水印。",
    negative:
      "动漫，漫画，卡通，插画，二维线稿，赛璐璐，游戏CG，3D建模，塑料皮肤，假脸，过度磨皮，" +
      "半真人半卡通，风格漂移，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印",
  },
  "二维动漫资源": {
    prompt:
      "把本段重新演绎为统一的高质量二维动漫视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；" +
      "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。" +
      "从第一帧到最后一帧，人物与完整背景都必须保持清晰线稿、赛璐璐分层上色和统一动漫光影；" +
      "不得出现真人脸、真实皮肤、照片纹理、真人摄影画面、半真人半动漫、写实 3D 人物、风格闪回、字幕、Logo 或水印。",
    negative:
      "真人，真实人脸，真实皮肤，照片，摄影，写实，半真人，真人背景，皮肤毛孔，镜头噪点，" +
      "写实3D，风格漂移，真人闪回，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印",
  },
  "3D / 游戏 CG 资源": {
    prompt:
      "把本段重新演绎为统一的高质量 3D 游戏 CG 视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；" +
      "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。" +
      "从第一帧到最后一帧保持稳定三维造型、PBR 材质、体积光和影视级游戏过场渲染，人物与完整环境必须属于同一美术体系；" +
      "不得出现真人摄影、二维线稿、平面插画、低模、塑料质感、材质跳变、字幕、Logo 或水印。",
    negative:
      "真人摄影，真实照片，二维动漫，漫画线稿，平面插画，低模，塑料材质，材质穿帮，贴图错误，" +
      "风格漂移，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印",
  },
  "漫画插画资源": {
    prompt:
      "把本段重新演绎为统一的高质量漫画插画视频。参考图片是人物身份、服装、道具和环境美术的唯一视觉依据；" +
      "严格保持实际人物数量、人物关系、主要剧情顺序和场景功能。根据镜头分析文字自然设计动作、表情、走位和镜头运动。" +
      "从第一帧到最后一帧保持稳定的手绘墨线、明确明暗块面、细腻插画上色和一致透视，人物与完整背景画风必须统一；" +
      "不得出现真人摄影纹理、3D 建模感、廉价卡通贴纸感、拼贴、画风闪回、字幕、Logo 或水印。",
    negative:
      "真人摄影，真实皮肤，照片纹理，3D建模，游戏CG，低模，廉价卡通，贴纸感，拼贴，线条抖动，" +
      "画风漂移，多余人物，人物复制，肢体畸形，字幕，文字，Logo，水印",
  },
  "自定义": {
    prompt:
      "根据用户指定的目标视觉方向重新演绎本段视频。参考图片定义人物身份、服装、道具和环境，" +
      "镜头分析文字定义剧情、动作、表情、走位和镜头运动；严格保持实际人物数量、人物关系、剧情顺序和场景功能，" +
      "并确保从第一帧到最后一帧的人物、完整背景、材质、色彩和画风稳定一致。",
    negative: "风格漂移，多余人物，人物复制，身份变化，肢体畸形，字幕，文字，Logo，水印",
  },
};

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetValue(widget, value) {
  if (!widget || widget.value === value) return;
  widget.value = value;
  widget.callback?.(value);
  widget.inputEl?.dispatchEvent(new Event("input", { bubbles: true }));
}

function applyPreset(node, resourceType) {
  const preset = PRESETS[String(resourceType || "")];
  if (!preset) return;
  setWidgetValue(findWidget(node, PROMPT_WIDGET), preset.prompt);
  setWidgetValue(findWidget(node, NEGATIVE_WIDGET), preset.negative);
  node.properties ??= {};
  node.properties[LAST_TYPE_PROPERTY] = resourceType;
  node.graph?.setDirtyCanvas?.(true, true);
  node.setDirtyCanvas?.(true, true);
}

function bindResourceType(node) {
  const typeWidget = findWidget(node, TYPE_WIDGET);
  if (!typeWidget || typeWidget.companyPresetBound) return;
  typeWidget.companyPresetBound = true;
  const originalCallback = typeWidget.callback;
  typeWidget.callback = function (value) {
    const result = originalCallback?.apply(this, arguments);
    applyPreset(node, value);
    return result;
  };

  const previousType = node.properties?.[LAST_TYPE_PROPERTY];
  const configuredType = String(typeWidget.value || "二维动漫资源");
  const promptWidget = findWidget(node, PROMPT_WIDGET);
  const knownPrompts = new Set(Object.values(PRESETS).map((preset) => preset.prompt));
  const hasCustomLegacyPrompt =
    !previousType &&
    configuredType === "二维动漫资源" &&
    String(promptWidget?.value || "").trim() &&
    !knownPrompts.has(String(promptWidget.value).trim());
  if (hasCustomLegacyPrompt) {
    typeWidget.value = "自定义";
    node.properties ??= {};
    node.properties[LAST_TYPE_PROPERTY] = "自定义";
    node.setDirtyCanvas?.(true, true);
    return;
  }

  if (!previousType || previousType !== configuredType) {
    applyPreset(node, configuredType);
  }
}

app.registerExtension({
  name: "company_remote.target_resource_prompt_presets",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      setTimeout(() => bindResourceType(this), 0);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      setTimeout(() => bindResourceType(this), 0);
      return result;
    };
  },
});
