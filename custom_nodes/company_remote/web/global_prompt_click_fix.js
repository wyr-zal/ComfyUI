import { app } from "../../scripts/app.js";

const PROMPT_SELECTOR = ".prompt-dialog-content";
const PATCHED_ATTRIBUTE = "data-company-remote-prompt-fix";
const NATIVE_PROMPT_MARKER = Symbol("companyRemoteNativeWorkflowPrompt");

function selectInput(input) {
  input.focus({ preventScroll: true });
  input.select();
}

function keepEventInsideDialog(event) {
  event.stopPropagation();
}

function patchPrompt(prompt) {
  if (!(prompt instanceof HTMLElement) || prompt.hasAttribute(PATCHED_ATTRIBUTE)) {
    return;
  }

  const input = prompt.querySelector('input[type="text"]');
  if (!(input instanceof HTMLInputElement)) return;

  prompt.setAttribute(PATCHED_ATTRIBUTE, "true");

  // Some custom-node canvas handlers treat dialog clicks as canvas clicks. Keep
  // those events inside the filename dialog while preserving Vue button handlers.
  for (const eventName of ["pointerdown", "mousedown", "click", "dblclick"]) {
    prompt.addEventListener(eventName, keepEventInsideDialog);
  }

  input.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
    queueMicrotask(() => selectInput(input));
  });

  input.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
      event.preventDefault();
      input.select();
    }
  });

  requestAnimationFrame(() => selectInput(input));
}

function patchVisiblePrompts(root = document) {
  if (root instanceof Element && root.matches(PROMPT_SELECTOR)) {
    patchPrompt(root);
  }
  root.querySelectorAll?.(PROMPT_SELECTOR).forEach(patchPrompt);
}

function installNativeWorkflowPrompt() {
  const workflow = app.extensionManager?.workflow?.activeWorkflow;
  const prototype = workflow ? Object.getPrototypeOf(workflow) : null;
  const originalPromptSave = prototype?.promptSave;
  if (typeof originalPromptSave !== "function") return false;
  if (originalPromptSave[NATIVE_PROMPT_MARKER]) return true;

  const nativePromptSave = async function () {
    const filename = window.prompt(
      "请输入工作流文件名",
      String(this.filename || "workflow"),
    );
    return filename?.trim() || null;
  };
  nativePromptSave[NATIVE_PROMPT_MARKER] = true;
  prototype.promptSave = nativePromptSave;
  return true;
}

app.registerExtension({
  name: "company_remote.workflow_filename_prompt_fix",
  setup() {
    patchVisiblePrompts();
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.addedNodes) {
          if (node instanceof Element) patchVisiblePrompts(node);
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    if (!installNativeWorkflowPrompt()) {
      const retry = window.setInterval(() => {
        if (installNativeWorkflowPrompt()) window.clearInterval(retry);
      }, 250);
      window.setTimeout(() => window.clearInterval(retry), 10_000);
    }
  },
});
