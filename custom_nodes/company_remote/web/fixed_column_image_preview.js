import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "CompanyFixedColumnImagePreview";
const WIDGET_NAME = "fixed_column_grid";
const PROPERTY_NAME = "last_fixed_grid_images";
const MIN_VIEWPORT_HEIGHT = 220;
const NODE_CHROME_HEIGHT = 132;

function firstValue(value, fallback) {
  const normalized = Array.isArray(value) ? value[0] : value;
  return normalized == null ? fallback : normalized;
}

function imageUrl(item) {
  const params = new URLSearchParams({
    filename: String(item.filename || ""),
    subfolder: String(item.subfolder || ""),
    type: String(item.type || "temp"),
    timestamp: String(Date.now()),
  });
  return api.apiURL(`/view?${params}`);
}

function normalizeImages(value) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item) => item && typeof item === "object" && !Array.isArray(item))
    .map((item) => ({
      filename: String(item.filename || ""),
      subfolder: String(item.subfolder || ""),
      type: String(item.type || "temp"),
    }))
    .filter((item) => item.filename);
}

function sameImages(left, right) {
  if (left.length !== right.length) return false;
  return left.every((item, index) => (
    item.filename === right[index].filename
    && item.subfolder === right[index].subfolder
    && item.type === right[index].type
  ));
}

function persistImages(node, images) {
  const current = normalizeImages(node.properties?.[PROPERTY_NAME]);
  if (sameImages(current, images)) return;

  node.graph?.beforeChange?.();
  node.properties ??= {};
  node.properties[PROPERTY_NAME] = images;
  node.graph?.afterChange?.();
  node.graph?.setDirtyCanvas?.(true, true);
}

function refreshPersistedGrid(node) {
  renderGrid(node, normalizeImages(node.properties?.[PROPERTY_NAME]));
  node.setDirtyCanvas?.(true, true);
}

function watchLayoutWidgets(node) {
  for (const name of ["columns", "gap"]) {
    const widget = node.widgets?.find((item) => item.name === name);
    if (!widget || widget.fixedGridCallbackInstalled) continue;
    const originalCallback = widget.callback;
    widget.callback = function () {
      const result = originalCallback?.apply(this, arguments);
      refreshPersistedGrid(node);
      return result;
    };
    widget.fixedGridCallbackInstalled = true;
  }
}

function createGridElement() {
  const viewport = document.createElement("div");
  viewport.className = "company-fixed-grid-viewport";
  viewport.style.width = "100%";
  viewport.style.height = "100%";
  viewport.style.minHeight = `${MIN_VIEWPORT_HEIGHT}px`;
  viewport.style.overflowY = "auto";
  viewport.style.overflowX = "hidden";
  viewport.style.boxSizing = "border-box";
  viewport.style.padding = "6px";
  viewport.style.border = "1px solid rgba(255, 255, 255, 0.12)";
  viewport.style.borderRadius = "4px";
  viewport.style.background = "rgba(0, 0, 0, 0.24)";
  viewport.style.overscrollBehavior = "contain";

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.width = "100%";
  grid.style.alignItems = "start";
  viewport.appendChild(grid);

  for (const eventName of ["pointerdown", "pointermove", "pointerup", "click", "dblclick", "contextmenu"]) {
    viewport.addEventListener(eventName, (event) => event.stopPropagation());
  }
  viewport.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });

  return { viewport, grid };
}

function renderGrid(node, images) {
  const grid = node.fixedColumnGrid;
  if (!grid) return;

  const widgetColumns = node.widgets?.find((widget) => widget.name === "columns")?.value;
  const widgetGap = node.widgets?.find((widget) => widget.name === "gap")?.value;
  const columns = Math.max(1, Number(firstValue(widgetColumns, 2)) || 2);
  const gap = Math.max(0, Number(firstValue(widgetGap, 0)) || 0);

  grid.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
  grid.style.gap = `${gap}px`;
  grid.replaceChildren();

  if (images.length === 0) {
    const empty = document.createElement("div");
    empty.textContent = "运行节点后显示图片";
    empty.style.gridColumn = "1 / -1";
    empty.style.padding = "28px 12px";
    empty.style.color = "rgba(255, 255, 255, 0.58)";
    empty.style.textAlign = "center";
    grid.appendChild(empty);
    return;
  }

  images.forEach((item, index) => {
    const image = document.createElement("img");
    image.src = imageUrl(item);
    image.alt = `预览图片 ${index + 1}`;
    image.loading = "lazy";
    image.style.display = "block";
    image.style.width = "100%";
    image.style.height = "auto";
    image.style.minWidth = "0";
    image.style.borderRadius = "2px";
    image.style.background = "#111";
    image.style.cursor = "zoom-in";
    image.addEventListener("click", () => window.open(image.src, "_blank", "noopener,noreferrer"));
    image.addEventListener("error", () => {
      image.alt = `第 ${index + 1} 张图片加载失败`;
      image.style.aspectRatio = "16 / 9";
      image.style.objectFit = "contain";
    });
    grid.appendChild(image);
  });
}

app.registerExtension({
  name: "company_remote.fixed_column_image_preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      const { viewport, grid } = createGridElement();
      const previewNode = this;
      const widget = this.addDOMWidget(WIDGET_NAME, "fixed-grid", viewport, {
        serialize: false,
        hideOnZoom: false,
      });
      widget.computeSize = function (width) {
        const availableHeight = Math.max(MIN_VIEWPORT_HEIGHT, Number(previewNode.size?.[1] || 0) - NODE_CHROME_HEIGHT);
        return [width, availableHeight];
      };
      this.fixedColumnGrid = grid;
      watchLayoutWidgets(this);
      renderGrid(this, []);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = originalOnConfigure?.apply(this, arguments);
      const images = normalizeImages(
        info?.properties?.[PROPERTY_NAME] ?? this.properties?.[PROPERTY_NAME],
      );
      renderGrid(this, images);
      return result;
    };

    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = originalOnExecuted?.apply(this, arguments);
      const images = normalizeImages(message?.fixed_grid_images);
      persistImages(this, images);
      renderGrid(this, images);
      this.setDirtyCanvas?.(true, true);
      return result;
    };
  },
});
