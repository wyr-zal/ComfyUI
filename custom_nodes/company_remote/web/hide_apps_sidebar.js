(function () {
  const STYLE_ID = "company-remote-hide-apps-sidebar-style";
  const HIDDEN_ATTR = "data-company-remote-hidden-apps-module";

  function showElement(element) {
    if (!(element instanceof HTMLElement)) return;
    element.hidden = false;
    element.removeAttribute(HIDDEN_ATTR);
    element.style.removeProperty("display");
    element.style.removeProperty("visibility");
    element.style.removeProperty("pointer-events");
  }

  function revealAppsModule(root = document) {
    document.getElementById(STYLE_ID)?.remove();
    const candidates = [];
    if (root instanceof HTMLElement) candidates.push(root);
    root.querySelectorAll?.(`[${HIDDEN_ATTR}="true"]`).forEach((element) => candidates.push(element));
    candidates.forEach(showElement);
  }

  function install() {
    revealAppsModule();
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node instanceof Element) revealAppsModule(node);
        }
      }
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
    window.addEventListener("hashchange", () => revealAppsModule());
    window.addEventListener("popstate", () => revealAppsModule());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install, { once: true });
  } else {
    install();
  }
})();
