import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "CompanyLongVideoAutoAssetBuilder";
const EVENT_NAME = "company_remote.auto_asset_progress";
const WIDGET_NAME = "auto_asset_progress";
const PROPERTY_NAME = "last_auto_asset_progress";
const EMPTY_TEXT = "等待自动资产进度。";
const MIN_HEIGHT = 132;
const NODE_CHROME_HEIGHT = 150;

function firstValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function normalizePayload(value) {
  const payload = firstValue(value);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  return payload;
}

function nodeMatches(node) {
  return node && (node.type === NODE_TYPE || node.comfyClass === NODE_TYPE);
}

function statusSummary(counts) {
  if (!counts || typeof counts !== "object") return "";
  const labels = {
    planned: "待处理",
    building: "生成中",
    ready: "完成",
    degraded: "部分完成",
    failed: "失败",
    source_frames_failed: "取帧失败",
    analysis_failed: "分析失败",
  };
  const order = [
    "planned",
    "building",
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
      manifest: report.manifest || "",
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

function createProgressElement() {
  const box = document.createElement("div");
  box.className = "company-auto-asset-progress";
  box.style.width = "100%";
  box.style.minHeight = `${MIN_HEIGHT}px`;
  box.style.boxSizing = "border-box";
  box.style.padding = "8px";
  box.style.border = "1px solid rgba(255, 255, 255, 0.12)";
  box.style.borderRadius = "4px";
  box.style.background = "rgba(0, 0, 0, 0.24)";
  box.style.color = "rgba(255, 255, 255, 0.88)";
  box.style.font = "12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace";
  box.style.whiteSpace = "pre-wrap";
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
  track.style.marginBottom = "8px";

  const bar = document.createElement("div");
  bar.style.width = "0%";
  bar.style.height = "100%";
  bar.style.background = "linear-gradient(90deg, #48c7ff, #74e38d)";
  bar.style.transition = "width 160ms ease";
  track.appendChild(bar);

  const detail = document.createElement("div");
  detail.textContent = "";
  detail.style.color = "rgba(255, 255, 255, 0.72)";

  box.append(title, track, detail);

  for (const eventName of ["pointerdown", "pointermove", "pointerup", "click", "dblclick", "contextmenu"]) {
    box.addEventListener(eventName, (event) => event.stopPropagation());
  }
  box.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });

  return { box, title, bar, detail };
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
    return [width, availableHeight];
  };
  node.autoAssetProgressUi = ui;
  return ui;
}

function renderProgress(node, payload) {
  const ui = ensureProgressWidget(node);
  const progress = normalizePayload(payload);
  if (!progress) {
    ui.title.textContent = EMPTY_TEXT;
    ui.bar.style.width = "0%";
    ui.detail.textContent = "";
    return;
  }

  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  ui.title.textContent = progress.message || EMPTY_TEXT;
  ui.bar.style.width = `${percent}%`;

  const lines = [];
  lines.push(`进度：${percent.toFixed(1)}%  (${Number(progress.value || 0).toFixed(2)} / ${progress.total || 1})`);
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
  if (progress.manifest) lines.push(`manifest：${progress.manifest}`);
  if (progress.progress_path) lines.push(`progress：${progress.progress_path}`);
  ui.detail.textContent = lines.join("\n");

  if (node.size?.[0] < 430) node.size[0] = 430;
  if (node.size?.[1] < 260) node.size[1] = 260;
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
    if (nodeData.name !== NODE_TYPE) return;

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
