import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set([
  "CompanyPromptEnhancer",
  "CompanyImagePromptEnhancer",
  "CompanyMultiPersonPromptAnalyzer",
]);
const MODELS_URL = "/api/company_remote/models?config=gpttext";

async function fetchModels() {
  const response = await fetch(MODELS_URL, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const models = await response.json();
  if (!Array.isArray(models) || models.length === 0) {
    throw new Error("模型列表为空");
  }
  return [...new Set(models.map((model) => String(model).trim()).filter(Boolean))];
}

function updateModelWidget(node, models) {
  const widget = node.widgets?.find((item) => item.name === "model");
  if (!widget) return;

  const current = String(widget.value || "").trim();
  const options = current && !models.includes(current) ? [current, ...models] : models;
  widget.options ??= {};
  widget.options.values = options;
  if (!current) {
    widget.value = options[0];
    widget.callback?.(widget.value);
  }
  node.setDirtyCanvas?.(true, true);
}

async function refreshModels(node) {
  try {
    updateModelWidget(node, await fetchModels());
  } catch (error) {
    console.warn("[company_remote] Failed to refresh model list; keeping current options.", error);
  }
}

app.registerExtension({
  name: "company_remote.prompt_model_options",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODES.has(nodeData.name)) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      this.addWidget(
        "button",
        "refresh_models",
        "刷新模型列表",
        () => refreshModels(this),
        { serialize: false },
      );
      setTimeout(() => refreshModels(this), 0);
      return result;
    };
  },
});
