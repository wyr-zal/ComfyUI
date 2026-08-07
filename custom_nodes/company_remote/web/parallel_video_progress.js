import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "CompanyLongVideoParallelSegmentGenerator";
const EVENT_NAME = "company_remote.parallel_video_progress";
const WIDGET_NAME = "parallel_video_progress";
const PROPERTY_NAME = "last_parallel_video_progress";
const MIN_HEIGHT = 260;
const NODE_CHROME_HEIGHT = 150;

function normalizePayload(value) {
  const payload = Array.isArray(value) ? value[0] : value;
  return payload && typeof payload === "object" && !Array.isArray(payload) ? payload : null;
}

function nodeMatches(node) {
  return node && (node.type === NODE_TYPE || node.comfyClass === NODE_TYPE);
}

function mediaUrl(item) {
  const params = new URLSearchParams({
    filename: String(item?.filename || ""),
    subfolder: String(item?.subfolder || ""),
    type: String(item?.type || "output"),
    timestamp: String(Date.now()),
  });
  return api.apiURL(`/view?${params}`);
}

function createProgressElement() {
  const box = document.createElement("div");
  box.style.width = "100%";
  box.style.minHeight = `${MIN_HEIGHT}px`;
  box.style.boxSizing = "border-box";
  box.style.padding = "8px";
  box.style.border = "1px solid rgba(255, 255, 255, 0.12)";
  box.style.borderRadius = "4px";
  box.style.background = "rgba(0, 0, 0, 0.24)";
  box.style.color = "rgba(255, 255, 255, 0.88)";
  box.style.font = "12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace";
  box.style.overflow = "hidden";

  const title = document.createElement("div");
  title.textContent = "等待并行视频进度。";
  title.style.fontWeight = "600";
  title.style.marginBottom = "7px";

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
  detail.style.color = "rgba(255, 255, 255, 0.72)";
  detail.style.marginBottom = "8px";

  const list = document.createElement("div");
  list.style.display = "grid";
  list.style.gridTemplateColumns = "repeat(auto-fit, minmax(150px, 1fr))";
  list.style.gap = "6px";
  list.style.maxHeight = "260px";
  list.style.overflowY = "auto";
  list.style.overscrollBehavior = "contain";

  box.append(title, track, detail, list);
  for (const eventName of ["pointerdown", "pointermove", "pointerup", "click", "dblclick", "contextmenu"]) {
    box.addEventListener(eventName, (event) => event.stopPropagation());
  }
  box.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  return { box, title, bar, detail, list };
}

function ensureWidget(node) {
  if (node.parallelVideoProgressUi) return node.parallelVideoProgressUi;
  const ui = createProgressElement();
  const widget = node.addDOMWidget(WIDGET_NAME, "parallel-video-progress", ui.box, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.computeSize = function (width) {
    const height = Math.max(MIN_HEIGHT, Number(node.size?.[1] || 0) - NODE_CHROME_HEIGHT);
    return [width, height];
  };
  node.parallelVideoProgressUi = ui;
  return ui;
}

function render(node, rawPayload) {
  const ui = ensureWidget(node);
  const payload = normalizePayload(rawPayload);
  if (!payload) {
    ui.title.textContent = "等待并行视频进度。";
    ui.bar.style.width = "0%";
    ui.detail.textContent = "";
    ui.list.replaceChildren();
    return;
  }

  const total = Number(payload.total || 0);
  const completed = Number(payload.completed_count || payload.value || 0);
  const percent = Math.max(0, Math.min(100, Number(payload.percent || 0)));
  ui.title.textContent = payload.message || "并行视频生成中。";
  ui.bar.style.width = `${percent}%`;
  ui.detail.textContent = [
    `进度：${percent.toFixed(1)}%  (${completed}/${total || 1})`,
    `运行中：${(payload.running_segments || []).join(", ") || "无"}  失败：${(payload.failed_segments || []).map((item) => item.sequence).join(", ") || "无"}`,
    payload.phase ? `阶段：${payload.phase}` : "",
  ].filter(Boolean).join("\n");

  ui.list.replaceChildren();
  for (const item of payload.completed_segments || []) {
    if (!item?.preview?.filename) continue;
    const card = document.createElement("div");
    card.style.border = "1px solid rgba(255, 255, 255, 0.14)";
    card.style.borderRadius = "3px";
    card.style.padding = "4px";
    card.style.background = "rgba(0, 0, 0, 0.24)";
    const label = document.createElement("div");
    label.textContent = `第 ${item.sequence} 段${item.reused ? "（复用）" : ""}`;
    label.style.marginBottom = "3px";
    const video = document.createElement("video");
    video.src = mediaUrl(item.preview);
    video.controls = true;
    video.muted = true;
    video.preload = "metadata";
    video.style.display = "block";
    video.style.width = "100%";
    video.style.aspectRatio = "16 / 9";
    video.style.objectFit = "contain";
    video.style.background = "#111";
    video.addEventListener("error", () => {
      label.textContent = `第 ${item.sequence} 段（视频预览加载失败，可从输出目录打开）`;
    });
    card.append(label, video);
    ui.list.appendChild(card);
  }
  if (!ui.list.childElementCount && payload.phase !== "completed") {
    const empty = document.createElement("div");
    empty.textContent = "分段完成后会在这里立即显示视频。";
    empty.style.color = "rgba(255, 255, 255, 0.58)";
    ui.list.appendChild(empty);
  }
  if (node.size?.[0] < 520) node.size[0] = 520;
  if (node.size?.[1] < 480) node.size[1] = 480;
  node.setDirtyCanvas?.(true, true);
}

function updateNode(node, payload) {
  const normalized = normalizePayload(payload);
  if (!normalized) return;
  node.properties ??= {};
  node.properties[PROPERTY_NAME] = normalized;
  render(node, normalized);
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
  name: "company_remote.parallel_video_progress",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      render(this, null);
      return result;
    };
    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = originalOnConfigure?.apply(this, arguments);
      render(this, info?.properties?.[PROPERTY_NAME] ?? this.properties?.[PROPERTY_NAME]);
      return result;
    };
    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = originalOnExecuted?.apply(this, arguments);
      updateNode(this, this.properties?.[PROPERTY_NAME] ?? message?.parallel_video_progress);
      return result;
    };
  },
});
