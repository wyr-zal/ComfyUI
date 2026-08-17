import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPES = new Set([
  "CompanyLongVideoAutoAssetBuilder",
  "CompanyLongVideoPipelineAssetVideoGenerator",
]);
const EVENT_NAME = "company_remote.auto_asset_progress";
const WIDGET_NAME = "auto_asset_progress";
const PROPERTY_NAME = "last_auto_asset_progress";
const EMPTY_TEXT = "等待自动资产进度。";
const MIN_HEIGHT = 360;
const MIN_WIDTH = 600;
const NODE_CHROME_HEIGHT = 158;

function firstValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function normalizePayload(value) {
  const payload = firstValue(value);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  return payload;
}

function nodeMatches(node) {
  return node && (NODE_TYPES.has(node.type) || NODE_TYPES.has(node.comfyClass));
}

function statusSummary(counts) {
  if (!counts || typeof counts !== "object") return "";
  const labels = {
    planned: "待处理",
    building: "生成中",
    masters_ready: "基础素材完成",
    ready: "完成",
    degraded: "部分完成",
    failed: "失败",
    source_frames_failed: "取帧失败",
    analysis_failed: "分析失败",
  };
  const order = [
    "planned",
    "building",
    "masters_ready",
    "ready",
    "degraded",
    "failed",
    "source_frames_failed",
    "analysis_failed",
  ];
  return order
    .filter((key) => Number(counts[key] || 0) > 0)
    .map((key) => `${labels[key] || key}:${counts[key]}`)
    .join("  ");
}

function payloadFromFinalReport(message) {
  const text = firstValue(message?.text);
  if (!text) return null;
  try {
    const report = JSON.parse(String(text));
    const tasks = Array.isArray(report.tasks) ? report.tasks : [];
    return {
      job_id: report.job_id || "",
      percent: 100,
      value: tasks.length,
      total: tasks.length || 1,
      phase: "completed",
      message: "自动资产阶段已完成，报告已返回。",
      auto_asset_status_counts: tasks.reduce((counts, task) => {
        const status = String(task?.status || "unknown");
        counts[status] = (counts[status] || 0) + 1;
        return counts;
      }, {}),
    };
  } catch (_error) {
    return null;
  }
}

function createLabel(text) {
  const element = document.createElement("div");
  element.textContent = text;
  element.style.margin = "9px 0 5px";
  element.style.color = "rgba(255, 255, 255, 0.82)";
  element.style.fontWeight = "600";
  return element;
}

function createGrid() {
  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(2, minmax(0, 1fr))";
  grid.style.gap = "7px";
  return grid;
}

function createProgressElement() {
  const box = document.createElement("div");
  box.className = "company-auto-asset-progress";
  box.style.width = "100%";
  box.style.minHeight = `${MIN_HEIGHT}px`;
  box.style.boxSizing = "border-box";
  box.style.padding = "9px";
  box.style.border = "1px solid rgba(255, 255, 255, 0.13)";
  box.style.borderRadius = "4px";
  box.style.background = "rgba(0, 0, 0, 0.24)";
  box.style.color = "rgba(255, 255, 255, 0.9)";
  box.style.font = "12px/1.42 ui-monospace, SFMono-Regular, Consolas, monospace";
  box.style.overflow = "hidden";

  const title = document.createElement("div");
  title.textContent = EMPTY_TEXT;
  title.style.marginBottom = "7px";
  title.style.fontWeight = "600";

  const track = document.createElement("div");
  track.style.width = "100%";
  track.style.height = "8px";
  track.style.borderRadius = "2px";
  track.style.background = "rgba(255, 255, 255, 0.12)";
  track.style.overflow = "hidden";

  const bar = document.createElement("div");
  bar.style.width = "0%";
  bar.style.height = "100%";
  bar.style.background = "linear-gradient(90deg, #48c7ff, #74e38d)";
  bar.style.transition = "width 160ms ease";
  track.appendChild(bar);

  const detail = document.createElement("div");
  detail.style.marginTop = "8px";
  detail.style.color = "rgba(255, 255, 255, 0.72)";
  detail.style.whiteSpace = "pre-wrap";

  const sourceLabel = createLabel("当前分镜源帧");
  const sourceGrid = createGrid();
  const convertedLabel = createLabel("已完成的转绘素材");
  const convertedGrid = createGrid();
  sourceLabel.style.display = "none";
  sourceGrid.style.display = "none";
  convertedLabel.style.display = "none";
  convertedGrid.style.display = "none";

  box.append(title, track, detail, sourceLabel, sourceGrid, convertedLabel, convertedGrid);

  for (const eventName of ["pointerdown", "pointermove", "pointerup", "click", "dblclick", "contextmenu"]) {
    box.addEventListener(eventName, (event) => event.stopPropagation());
  }
  box.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });

  return { box, title, bar, detail, sourceLabel, sourceGrid, convertedLabel, convertedGrid };
}

function ensureProgressWidget(node) {
  if (node.autoAssetProgressUi) return node.autoAssetProgressUi;
  const ui = createProgressElement();
  const progressNode = node;
  const widget = node.addDOMWidget(WIDGET_NAME, "auto-asset-progress", ui.box, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.computeSize = function (width) {
    const availableHeight = Math.max(MIN_HEIGHT, Number(progressNode.size?.[1] || 0) - NODE_CHROME_HEIGHT);
    return [Math.max(MIN_WIDTH, width), availableHeight];
  };
  node.autoAssetProgressUi = ui;
  return ui;
}

function viewUrl(file) {
  if (!file || typeof file !== "object" || file.type !== "output" || !file.filename) return "";
  const query = new URLSearchParams({
    filename: String(file.filename),
    subfolder: String(file.subfolder || ""),
    type: "output",
    timestamp: String(Date.now()),
  });
  return api.apiURL(`/view?${query.toString()}`);
}

function clearGrid(grid) {
  grid.replaceChildren();
}

function addPreviewCard(grid, label, file, state) {
  const url = viewUrl(file);
  if (!url) return false;

  const card = document.createElement("div");
  card.style.minWidth = "0";
  card.style.border = "1px solid rgba(255, 255, 255, 0.12)";
  card.style.background = "rgba(255, 255, 255, 0.045)";
  card.style.padding = "5px";

  const image = document.createElement("img");
  image.src = url;
  image.alt = label;
  image.loading = "lazy";
  image.style.display = "block";
  image.style.width = "100%";
  image.style.height = "118px";
  image.style.objectFit = "contain";
  image.style.background = "rgba(0, 0, 0, 0.28)";

  const name = document.createElement("div");
  name.textContent = label;
  name.style.marginTop = "4px";
  name.style.overflow = "hidden";
  name.style.textOverflow = "ellipsis";
  name.style.whiteSpace = "nowrap";

  const note = document.createElement("div");
  note.textContent = state;
  note.style.marginTop = "2px";
  note.style.color = state.includes("失败") || state.includes("未通过")
    ? "#ff9b9b"
    : state.includes("警告") || state.includes("偏弱") || state.includes("重试")
      ? "#ffd17c"
      : "#93e5b3";
  note.style.fontSize = "11px";
  note.style.lineHeight = "1.35";
  note.style.whiteSpace = "normal";
  note.style.wordBreak = "break-word";

  card.append(image, name, note);
  grid.appendChild(card);
  return true;
}

function personState(item) {
  const tos = String(item?.tos_status || "pending");
  const library = String(item?.asset_library_status || "pending");
  const warning = String(item?.warning || "").trim();
  if (tos === "failed") return `上传 TOS 失败，已停止该分镜${warning ? `：${warning}` : ""}`;
  if (tos === "uploaded") {
    if (library === "active") return "已上传 TOS，素材库已入库";
    if (library === "warning") return `已上传 TOS，入库警告${warning ? `：${warning}` : ""}`;
    return "已上传 TOS，正在入库";
  }
  if (tos === "reused") {
    if (library === "active") return "已复用 TOS，素材库已入库";
    if (library === "warning") return `已复用 TOS，入库警告${warning ? `：${warning}` : ""}`;
    return "已复用 TOS，正在确认入库";
  }
  return "等待上传 TOS";
}

function sceneState() {
  return "场景图只用来约束完整画面转换，不直接发送 Seedance";
}

function integratedFrameState(item) {
  const quality = item?.quality && typeof item.quality === "object" ? item.quality : {};
  const verdict = String(quality.verdict || "pending");
  const reasons = Array.isArray(quality.reasons) ? quality.reasons.filter(Boolean).join("；") : "";
  if (verdict === "approved") return "整帧转换已通过，Seedance 会使用这张图";
  if (verdict === "retry") return `转换效果偏弱，等待自动重试${reasons ? `：${reasons}` : ""}`;
  return `整帧转换未通过${reasons ? `：${reasons}` : ""}`;
}

function renderPreview(ui, preview) {
  clearGrid(ui.sourceGrid);
  clearGrid(ui.convertedGrid);
  const sourceStart = preview?.source_start;
  const sourceEnd = preview?.source_end;
  const hasSourceStart = addPreviewCard(ui.sourceGrid, "源首帧", sourceStart, "当前分镜起点");
  const hasSourceEnd = addPreviewCard(ui.sourceGrid, "源尾帧", sourceEnd, "当前分镜终点");
  const hasSource = hasSourceStart || hasSourceEnd;
  ui.sourceLabel.style.display = hasSource ? "block" : "none";
  ui.sourceGrid.style.display = hasSource ? "grid" : "none";

  let hasConverted = false;
  for (const item of Array.isArray(preview?.converted) ? preview.converted : []) {
    const state = item?.kind === "person"
      ? personState(item)
      : item?.kind === "integrated_frame"
        ? integratedFrameState(item)
        : sceneState();
    hasConverted = addPreviewCard(ui.convertedGrid, item?.label || "转绘素材", item?.preview, state) || hasConverted;
  }
  ui.convertedLabel.style.display = hasConverted ? "block" : "none";
  ui.convertedGrid.style.display = hasConverted ? "grid" : "none";
}

function renderProgress(node, payload) {
  const ui = ensureProgressWidget(node);
  const progress = normalizePayload(payload);
  if (!progress) {
    ui.title.textContent = EMPTY_TEXT;
    ui.bar.style.width = "0%";
    ui.detail.textContent = "";
    renderPreview(ui, null);
    return;
  }

  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  ui.title.textContent = progress.message || EMPTY_TEXT;
  ui.bar.style.width = `${percent}%`;

  const lines = [`进度：${percent.toFixed(1)}%  (${Number(progress.value || 0).toFixed(2)} / ${progress.total || 1})`];
  if (progress.task_index) {
    lines.push(`当前分镜：第 ${progress.task_index} 段  阶段：${progress.phase || "-"}`);
  } else if (progress.phase) {
    lines.push(`阶段：${progress.phase}`);
  }
  const summary = statusSummary(progress.auto_asset_status_counts);
  if (summary) lines.push(summary);
  if (progress.extra?.asset_total) {
    lines.push(`当前段素材：${progress.extra.asset_done || 0}/${progress.extra.asset_total}`);
  }
  const error = progress.extra?.error || progress.extra?.errors?.[0]?.message;
  if (error) lines.push(`提示：${typeof error === "string" ? error : JSON.stringify(error)}`);
  ui.detail.textContent = lines.join("\n");
  renderPreview(ui, progress.preview || progress.extra?.preview);

  if (node.size?.[0] < MIN_WIDTH) node.size[0] = MIN_WIDTH;
  if (node.size?.[1] < 620) node.size[1] = 620;
  node.setDirtyCanvas?.(true, true);
}

function persistProgress(node, payload) {
  const progress = normalizePayload(payload);
  if (!progress) return;
  node.properties ??= {};
  node.properties[PROPERTY_NAME] = progress;
  node.graph?.setDirtyCanvas?.(true, true);
}

function updateNode(node, payload) {
  persistProgress(node, payload);
  renderProgress(node, payload);
}

function updateMatchingNodes(payload) {
  const graph = app.graph;
  if (!graph) return;
  if (payload.node != null) {
    const node = graph._nodes_by_id?.[payload.node] || graph._nodes_by_id?.[String(payload.node)];
    if (nodeMatches(node)) {
      updateNode(node, payload);
      return;
    }
  }
  for (const node of graph._nodes || []) {
    if (nodeMatches(node)) updateNode(node, payload);
  }
}

api.addEventListener(EVENT_NAME, ({ detail }) => {
  const payload = normalizePayload(detail);
  if (payload) updateMatchingNodes(payload);
});

app.registerExtension({
  name: "company_remote.auto_asset_progress",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_TYPES.has(nodeData.name)) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      renderProgress(this, null);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = originalOnConfigure?.apply(this, arguments);
      renderProgress(this, info?.properties?.[PROPERTY_NAME] ?? this.properties?.[PROPERTY_NAME]);
      return result;
    };

    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = originalOnExecuted?.apply(this, arguments);
      const progress = payloadFromFinalReport(message) ?? this.properties?.[PROPERTY_NAME];
      updateNode(this, progress);
      return result;
    };
  },
});
