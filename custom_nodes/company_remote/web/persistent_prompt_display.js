import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_TYPE = "CompanyPersistentPromptDisplay";
const PROPERTY_NAME = "last_prompt_text";
const DISPLAY_WIDGET_NAME = "last_prompt_text_display";
const EMPTY_TEXT = "尚未生成提示词。";

function firstValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function normalizeText(value) {
  const normalized = firstValue(value);
  return normalized == null ? "" : String(normalized);
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
  widget.label = "已保存的提示词";
  widget.options ??= {};
  widget.options.serialize = false;
  widget.inputEl.readOnly = true;
  widget.inputEl.style.minHeight = "150px";
  widget.inputEl.style.fontFamily = "ui-monospace, SFMono-Regular, Consolas, monospace";
  widget.inputEl.style.fontSize = "12px";
  widget.inputEl.style.lineHeight = "1.5";
  widget.value = EMPTY_TEXT;
  return widget;
}

function showText(node, value) {
  const widget = ensureDisplayWidget(node);
  widget.value = value || EMPTY_TEXT;
  if (node.size?.[0] < 390) node.size[0] = 390;
  if (node.size?.[1] < 220) node.size[1] = 220;
  node.setDirtyCanvas?.(true, true);
}

function persistText(node, value) {
  if (node.properties?.[PROPERTY_NAME] === value) return;

  node.graph?.beforeChange?.();
  node.properties ??= {};
  node.properties[PROPERTY_NAME] = value;
  node.graph?.afterChange?.();
  node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "company_remote.persistent_prompt_display",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      showText(this, "");
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = originalOnConfigure?.apply(this, arguments);
      const value = normalizeText(
        info?.properties?.[PROPERTY_NAME] ?? this.properties?.[PROPERTY_NAME],
      );
      showText(this, value);
      return result;
    };

    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = originalOnExecuted?.apply(this, arguments);
      const value = normalizeText(message?.text);
      persistText(this, value);
      showText(this, value);
      return result;
    };
  },
});
