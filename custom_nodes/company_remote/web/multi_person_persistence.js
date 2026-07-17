import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_TYPE = "CompanyMultiPersonPromptAnalyzer";
const PROPERTY_NAME = "last_multi_person_outputs";
const DISPLAY_WIDGET_NAME = "last_multi_person_outputs_display";

const OUTPUT_FIELDS = [
  ["person_count", "人物数量"],
  ["identity_manifest", "统一身份表"],
  ["person_a_prompt", "人物 A 提示词"],
  ["person_b_prompt", "人物 B 提示词"],
  ["person_c_prompt", "人物 C 提示词"],
  ["background_prompt", "背景处理提示词"],
  ["final_prompt", "最终合成提示词"],
];

function firstValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function normalizeOutputs(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;

  const normalized = {};
  for (const [key] of OUTPUT_FIELDS) {
    const fieldValue = firstValue(value[key]);
    normalized[key] = key === "person_count"
      ? Number.parseInt(fieldValue, 10)
      : String(fieldValue ?? "");
  }
  if (!Number.isInteger(normalized.person_count)) return null;
  return normalized;
}

function formatOutputs(outputs) {
  if (!outputs) return "尚未生成多人分析结果。";

  return OUTPUT_FIELDS
    .map(([key, label]) => `【${label}】\n${outputs[key] || "（无）"}`)
    .join("\n\n");
}

function ensureDisplayWidget(node) {
  let widget = node.widgets?.find((item) => item.name === DISPLAY_WIDGET_NAME);
  if (widget) return widget;

  widget = ComfyWidgets.STRING(
    node,
    DISPLAY_WIDGET_NAME,
    ["STRING", { multiline: true }],
    app,
  ).widget;
  widget.label = "上次多人分析结果";
  widget.options ??= {};
  widget.options.serialize = false;
  widget.inputEl.readOnly = true;
  widget.inputEl.style.minHeight = "320px";
  widget.inputEl.style.fontFamily = "ui-monospace, SFMono-Regular, Consolas, monospace";
  widget.inputEl.style.fontSize = "12px";
  widget.inputEl.style.lineHeight = "1.5";
  widget.value = "尚未生成多人分析结果。";
  return widget;
}

function showOutputs(node, outputs) {
  const widget = ensureDisplayWidget(node);
  widget.value = formatOutputs(outputs);
  if (node.size?.[0] < 650) node.size[0] = 650;
  if (node.size?.[1] < 880) node.size[1] = 880;
  node.setDirtyCanvas?.(true, true);
}

function persistOutputs(node, outputs) {
  node.graph?.beforeChange?.();
  node.properties ??= {};
  node.properties[PROPERTY_NAME] = outputs;
  node.graph?.afterChange?.();
  node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "company_remote.multi_person_persistence",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      showOutputs(this, null);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = originalOnConfigure?.apply(this, arguments);
      const outputs = normalizeOutputs(
        info?.properties?.[PROPERTY_NAME] ?? this.properties?.[PROPERTY_NAME],
      );
      showOutputs(this, outputs);
      return result;
    };

    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = originalOnExecuted?.apply(this, arguments);
      const outputs = normalizeOutputs(message);
      if (!outputs) {
        console.warn("[company_remote] Multi-person output is incomplete; skipping persistence.", message);
        return result;
      }
      persistOutputs(this, outputs);
      showOutputs(this, outputs);
      return result;
    };
  },
});
