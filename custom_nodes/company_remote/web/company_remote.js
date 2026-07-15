(function () {
  const API_BASE = "/api/company_remote";
  const STYLE_ID = "company-remote-style";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .crapi-backdrop { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center; padding: 16px; background: rgba(15,23,42,.38); color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
      .crapi-panel { width: min(1100px, calc(100vw - 32px)); max-height: calc(100vh - 32px); overflow: hidden; background: #fff; border: 1px solid #eef2f7; border-radius: 12px; box-shadow: 0 18px 45px rgba(15,23,42,.14); font-size: 14px; line-height: 1.5; }
      .crapi-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 20px 24px; border-bottom: 1px solid #eef2f7; background: linear-gradient(180deg, #fff, #f9fafb); }
      .crapi-title { display: grid; gap: 4px; }
      .crapi-kicker { width: max-content; padding: 4px 10px; border: 1px solid #dbeafe; border-radius: 999px; background: #eff6ff; color: #2563eb; font-size: 12px; font-weight: 800; }
      .crapi-head h2 { margin: 0; font-size: 20px; line-height: 1.2; font-weight: 760; }
      .crapi-head p { margin: 0; color: #6b7280; font-size: 13px; }
      .crapi-body { display: grid; grid-template-columns: 248px 1fr; min-height: 580px; max-height: calc(100vh - 132px); background: #f3f4f6; }
      .crapi-list { border-right: 1px solid #eef2f7; padding: 16px; background: #fff; overflow: auto; }
      .crapi-form { padding: 20px 24px 24px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 18px; overflow: auto; }
      .crapi-field { display: flex; flex-direction: column; gap: 7px; min-width: 0; }
      .crapi-field.full { grid-column: 1 / -1; }
      .crapi-field label { color: #1f2937; font-size: 13px; font-weight: 700; }
      .crapi-field input, .crapi-field select, .crapi-field textarea { width: 100%; min-height: 40px; background: #fff; color: #374151; border: 1px solid #e5e7eb; border-radius: 8px; padding: 9px 12px; font: inherit; min-width: 0; }
      .crapi-field textarea { min-height: 108px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
      .crapi-actions { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding-top: 6px; }
      .crapi-status { grid-column: 1 / -1; min-height: 22px; color: #6b7280; white-space: pre-wrap; }
      .crapi-list button, .crapi-actions button, .crapi-close, .crapi-menu-button { min-height: 40px; display: inline-flex; align-items: center; justify-content: center; gap: 8px; cursor: pointer; border: 1px solid #e5e7eb; background: #fff; color: #4b5563; border-radius: 8px; padding: 9px 14px; font: inherit; font-weight: 650; white-space: nowrap; }
      .crapi-close { width: 40px; height: 40px; padding: 0; border-radius: 10px; font-size: 20px; color: #6b7280; }
      .crapi-item { width: 100%; justify-content: flex-start; text-align: left; margin: 6px 0; overflow: hidden; text-overflow: ellipsis; }
      .crapi-item.active { border-color: #bfdbfe; background: #eff6ff; color: #2563eb; box-shadow: inset 3px 0 0 #2563eb; }
      .crapi-template-title { margin: 14px 2px 6px; color: #6b7280; font-size: 12px; font-weight: 800; }
      .crapi-template { justify-content: flex-start !important; margin: 4px 0; min-height: 34px !important; padding: 7px 10px !important; border-color: #e0e7ff !important; background: #f8fafc !important; color: #334155 !important; }
      .crapi-danger { border-color: #fecaca !important; background: #fef2f2 !important; color: #dc2626 !important; }
      .crapi-primary { border-color: #2563eb !important; background: #2563eb !important; color: #fff !important; }
      .crapi-menu-button { margin: 4px; min-height: 36px; border-color: #dbeafe; background: #eff6ff; color: #2563eb; }
      @media (max-width: 760px) { .crapi-backdrop { align-items: stretch; padding: 0; } .crapi-panel { width: 100vw; max-height: 100vh; border-radius: 0; } .crapi-body { grid-template-columns: 1fr; } .crapi-form { grid-template-columns: 1fr; padding: 16px; } }
    `;
    document.head.appendChild(style);
  }

  function emptyConfig() {
    return {
      name: "default",
      base_url: "https://your-subscription-api.example.com",
      submit_path: "/generate",
      method: "POST",
      auth_header: "Authorization",
      auth_prefix: "Bearer",
      api_key: "",
      api_key_env: "",
      timeout_seconds: 600,
      poll_enabled: true,
      poll_path_template: "/tasks/{task_id}",
      poll_interval_seconds: 5,
      max_poll_attempts: 120,
      test_path: "",
      request_template: {},
      image_field: "image",
      images_field: "images",
      first_frame_field: "first_frame",
      last_frame_field: "last_frame",
      reference_images_field: "reference_images",
      video_field: "video",
      reference_videos_field: "reference_videos",
      image_format: "png",
      media_delivery: "base64",
      tos_enabled: false,
      tos_bucket: "",
      tos_endpoint: "",
      tos_region: "",
      tos_key_prefix: "comfyui/seedance/",
      tos_access_key_env: "VOLC_ACCESS_KEY",
      tos_secret_key_env: "VOLC_SECRET_KEY",
      tos_url_expires_seconds: 7200,
      response_image_url_path: "image_url",
      response_video_url_path: "video_url",
      response_result_url_path: "url",
      response_task_id_path: "task_id",
      response_status_path: "status",
      success_statuses: ["succeeded", "success", "completed", "done"],
      failure_statuses: ["failed", "error", "cancelled", "canceled"],
      extra_headers: {},
    };
  }

  function templateConfig(name, overrides) {
    return { ...emptyConfig(), name, ...overrides };
  }

  const CONFIG_TEMPLATES = {
    gpttext: {
      label: "GPT Text",
      build: () => templateConfig("gpttext", {
        base_url: "http://127.0.0.1:8787/v1",
        submit_path: "/chat/completions",
        auth_header: "",
        auth_prefix: "",
        poll_enabled: false,
        test_path: "/models",
        request_template: {
          model: "{model}",
          messages: [
            {
              role: "system",
              content: "{skill}",
            },
            {
              role: "user",
              content: "{user_prompt}",
            },
          ],
          temperature: "{temperature}",
          max_tokens: "{max_tokens}",
        },
        image_field: "",
        images_field: "",
        response_image_url_path: "",
        response_result_url_path: "",
      }),
    },
    gptimage2: {
      label: "GPT Image 2",
      build: () => templateConfig("gptimage2", {
        base_url: "http://127.0.0.1:8787/v1",
        submit_path: "/images/generations",
        auth_header: "",
        auth_prefix: "",
        poll_enabled: false,
        test_path: "/models",
        request_template: {
          model: "{model}",
          prompt: "{prompt}",
          size: "{size}",
          quality: "{quality}",
          response_format: "{response_format}",
        },
        image_field: "",
        images_field: "",
        response_image_url_path: "data.0.b64_json",
        response_result_url_path: "",
      }),
    },
    seedream: {
      label: "Seedream",
      build: () => templateConfig("seedream", {
        submit_path: "/seedream/generations",
        poll_path_template: "/seedream/tasks/{task_id}",
        request_template: {
          provider: "seedream",
          operation: "{operation}",
          model: "{model}",
          prompt: "{prompt}",
          size: "{size_preset}",
          width: "{width}",
          height: "{height}",
          seed: "{seed}",
          max_images: "{max_images}",
          sequential_image_generation: "{sequential_image_generation}",
          watermark: "{watermark}",
        },
        response_image_url_path: "data.0.url",
        response_result_url_path: "url",
      }),
    },
    seedance2: {
      label: "Seedance 2.0",
      build: () => templateConfig("seedance2", {
        submit_path: "/seedance2/tasks",
        poll_path_template: "/seedance2/tasks/{task_id}",
        request_template: {
          provider: "seedance2",
          operation: "{operation}",
          model: "{model}",
          prompt: "{prompt}",
          resolution: "{resolution}",
          ratio: "{ratio}",
          duration: "{duration}",
          seed: "{seed}",
          generate_audio: "{generate_audio}",
          watermark: "{watermark}",
        },
        response_video_url_path: "content.video_url",
        response_result_url_path: "video_url",
        response_task_id_path: "id",
        media_delivery: "tos_presigned",
        tos_enabled: true,
        tos_bucket: "starshore",
        tos_endpoint: "tos-cn-beijing.volces.com",
        tos_region: "cn-beijing",
        tos_key_prefix: "comfyui/seedance/",
        tos_access_key_env: "VOLC_ACCESS_KEY",
        tos_secret_key_env: "VOLC_SECRET_KEY",
        tos_url_expires_seconds: 7200,
      }),
    },
    kling: {
      label: "Kling",
      build: () => templateConfig("kling", {
        submit_path: "/kling/videos/image2video",
        poll_path_template: "/kling/videos/image2video/{task_id}",
        image_field: "start_frame",
        request_template: {
          provider: "kling",
          operation: "{operation}",
          model_name: "{model_name}",
          model: "{model}",
          prompt: "{prompt}",
          negative_prompt: "{negative_prompt}",
          cfg_scale: "{cfg_scale}",
          mode: "{mode}",
          aspect_ratio: "{ratio}",
          duration: "{duration}",
        },
        response_video_url_path: "data.video_url",
        response_result_url_path: "video_url",
        response_task_id_path: "data.task_id",
      }),
    },
    vidu: {
      label: "Vidu",
      build: () => templateConfig("vidu", {
        submit_path: "/vidu/generations",
        poll_path_template: "/vidu/generations/{task_id}",
        request_template: {
          provider: "vidu",
          operation: "{operation}",
          model: "{model}",
          prompt: "{prompt}",
          resolution: "{resolution}",
          duration: "{duration}",
          seed: "{seed}",
          audio: "{audio}",
        },
        response_video_url_path: "creations.0.url",
        response_result_url_path: "url",
      }),
    },
    minimax_hailuo: {
      label: "MiniMax/Hailuo",
      build: () => templateConfig("minimax_hailuo", {
        submit_path: "/minimax/video_generation",
        poll_path_template: "/minimax/video_generation/{task_id}",
        image_field: "first_frame_image",
        request_template: {
          provider: "minimax_hailuo",
          operation: "{operation}",
          model: "{model}",
          prompt: "{prompt}",
          prompt_text: "{prompt}",
          duration: "{duration}",
          resolution: "{resolution}",
          seed: "{seed}",
          prompt_optimizer: "{prompt_optimizer}",
        },
        response_video_url_path: "file.video_url",
        response_result_url_path: "video_url",
      }),
    },
  };

  async function api(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);
    return data;
  }

  function field(form, key, label, type = "text", full = false) {
    const wrap = document.createElement("div");
    wrap.className = `crapi-field${full ? " full" : ""}`;
    const labelEl = document.createElement("label");
    labelEl.textContent = label;
    let input;
    if (type === "textarea") input = document.createElement("textarea");
    else if (type === "select") {
      input = document.createElement("select");
      ["GET", "POST"].forEach((item) => { const opt = document.createElement("option"); opt.value = item; opt.textContent = item; input.appendChild(opt); });
    } else { input = document.createElement("input"); input.type = type; }
    input.dataset.key = key;
    wrap.append(labelEl, input);
    form.appendChild(wrap);
    return input;
  }

  function setFormValues(form, config) {
    for (const input of form.querySelectorAll("[data-key]")) {
      const key = input.dataset.key;
      let value = config[key];
      if (key === "api_key" && config.has_api_key && !value) value = "";
      if (input.type === "checkbox") input.checked = Boolean(value);
      else if (typeof value === "object" && value !== null) input.value = JSON.stringify(value, null, 2);
      else input.value = value ?? "";
    }
  }

  function readFormValues(form) {
    const cfg = emptyConfig();
    for (const input of form.querySelectorAll("[data-key]")) {
      const key = input.dataset.key;
      if (input.type === "checkbox") cfg[key] = input.checked;
      else if (["timeout_seconds", "max_poll_attempts", "tos_url_expires_seconds"].includes(key)) cfg[key] = Number.parseInt(input.value || "0", 10);
      else if (["poll_interval_seconds"].includes(key)) cfg[key] = Number.parseFloat(input.value || "0");
      else if (["request_template", "extra_headers"].includes(key)) cfg[key] = input.value.trim() ? JSON.parse(input.value) : {};
      else if (["success_statuses", "failure_statuses"].includes(key)) cfg[key] = input.value.split(",").map((v) => v.trim()).filter(Boolean);
      else cfg[key] = input.value;
    }
    return cfg;
  }

  async function openPanel() {
    ensureStyle();
    const backdrop = document.createElement("div");
    backdrop.className = "crapi-backdrop";
    const panel = document.createElement("div");
    panel.className = "crapi-panel";
    panel.innerHTML = `<div class="crapi-head"><div class="crapi-title"><span class="crapi-kicker">公司远程订阅</span><h2>Company Remote 配置</h2><p>管理国内订阅/中转 API。保存后刷新节点定义即可更新下拉选项。</p></div><button class="crapi-close" title="关闭" aria-label="关闭">×</button></div><div class="crapi-body"><div class="crapi-list"></div><form class="crapi-form"></form></div>`;
    backdrop.appendChild(panel);
    document.body.appendChild(backdrop);

    const list = panel.querySelector(".crapi-list");
    const form = panel.querySelector(".crapi-form");
    const status = document.createElement("div");
    status.className = "crapi-status";
    panel.querySelector(".crapi-close").onclick = () => backdrop.remove();
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) backdrop.remove(); });

    field(form, "name", "配置名称");
    field(form, "base_url", "接口 Base URL");
    field(form, "submit_path", "提交路径");
    field(form, "method", "请求方法", "select");
    field(form, "auth_header", "鉴权 Header");
    field(form, "auth_prefix", "鉴权前缀");
    field(form, "api_key", "API Key");
    field(form, "api_key_env", "API Key 环境变量");
    field(form, "timeout_seconds", "超时时间（秒）", "number");
    const poll = field(form, "poll_enabled", "启用轮询", "checkbox"); poll.style.width = "18px";
    field(form, "poll_path_template", "轮询路径模板");
    field(form, "poll_interval_seconds", "轮询间隔（秒）", "number");
    field(form, "max_poll_attempts", "最大轮询次数", "number");
    field(form, "test_path", "测试路径");
    field(form, "image_field", "单图字段名");
    field(form, "images_field", "多图字段名");
    field(form, "first_frame_field", "首帧字段名");
    field(form, "last_frame_field", "尾帧字段名");
    field(form, "reference_images_field", "参考图字段名");
    field(form, "video_field", "视频字段名");
    field(form, "reference_videos_field", "参考视频字段名");
    field(form, "image_format", "图片格式");
    field(form, "media_delivery", "媒体传递方式");
    const tosEnabled = field(form, "tos_enabled", "启用 TOS URL", "checkbox"); tosEnabled.style.width = "18px";
    field(form, "tos_bucket", "TOS Bucket");
    field(form, "tos_endpoint", "TOS Endpoint");
    field(form, "tos_region", "TOS Region");
    field(form, "tos_key_prefix", "TOS 对象前缀");
    field(form, "tos_access_key_env", "TOS AK 环境变量");
    field(form, "tos_secret_key_env", "TOS SK 环境变量");
    field(form, "tos_url_expires_seconds", "TOS URL 有效期（秒）", "number");
    field(form, "response_image_url_path", "图片 URL JSON 路径");
    field(form, "response_video_url_path", "视频 URL JSON 路径");
    field(form, "response_result_url_path", "通用 URL JSON 路径");
    field(form, "response_task_id_path", "任务 ID JSON 路径");
    field(form, "response_status_path", "状态 JSON 路径");
    field(form, "success_statuses", "成功状态列表");
    field(form, "failure_statuses", "失败状态列表");
    field(form, "extra_headers", "额外 Header JSON", "textarea", true);
    field(form, "request_template", "请求模板 JSON", "textarea", true);

    const actions = document.createElement("div");
    actions.className = "crapi-actions";
    const saveBtn = button("保存配置", "crapi-primary");
    const newBtn = button("新建");
    const testBtn = button("测试连接");
    const deleteBtn = button("删除", "crapi-danger");
    actions.append(saveBtn, newBtn, testBtn, deleteBtn);
    form.append(actions, status);

    let configs = [];
    let selectedName = "";

    async function refresh(selectName) {
      const data = await api("/configs");
      configs = data.configs || [];
      selectedName = selectName || selectedName || configs[0]?.name || "";
      renderList();
      const selected = configs.find((item) => item.name === selectedName) || emptyConfig();
      setFormValues(form, selected);
    }

    function renderList() {
      list.innerHTML = "";
      const add = button("＋ 新建配置", "crapi-primary");
      add.style.width = "100%";
      add.onclick = () => { selectedName = ""; setFormValues(form, { ...emptyConfig(), name: nextName() }); renderList(); };
      list.appendChild(add);
      const templateTitle = document.createElement("div");
      templateTitle.className = "crapi-template-title";
      templateTitle.textContent = "从模板创建";
      list.appendChild(templateTitle);
      Object.entries(CONFIG_TEMPLATES).forEach(([key, tpl]) => {
        const tplBtn = button(tpl.label, "crapi-template");
        tplBtn.style.width = "100%";
        tplBtn.onclick = () => {
          selectedName = "";
          setFormValues(form, tpl.build());
          status.textContent = `已套用 ${tpl.label} 模板。请按公司订阅平台文档填写 Base URL / 路径 / Header 后保存。`;
          renderList();
        };
        list.appendChild(tplBtn);
      });
      configs.forEach((cfg) => {
        const item = button(cfg.name, "crapi-item" + (cfg.name === selectedName ? " active" : ""));
        item.onclick = () => { selectedName = cfg.name; setFormValues(form, cfg); renderList(); };
        list.appendChild(item);
      });
    }

    saveBtn.onclick = async (event) => {
      event.preventDefault();
      try {
        const cfg = readFormValues(form);
        const original = configs.some((item) => item.name === selectedName) ? `/${encodeURIComponent(selectedName)}` : "";
        const data = await api(`/configs${original}`, { method: original ? "PUT" : "POST", body: JSON.stringify(cfg) });
        status.textContent = "已保存。请刷新节点定义以更新配置下拉选项。";
        await refresh(data.config.name);
      } catch (err) { status.textContent = `保存失败：${err.message}`; }
    };
    newBtn.onclick = (event) => { event.preventDefault(); selectedName = ""; setFormValues(form, { ...emptyConfig(), name: nextName() }); status.textContent = ""; renderList(); };
    testBtn.onclick = async (event) => {
      event.preventDefault();
      try { const data = await api("/test", { method: "POST", body: JSON.stringify(readFormValues(form)) }); status.textContent = `测试${data.ok ? "成功" : "完成"}：HTTP ${data.status}\n${data.body || ""}`; }
      catch (err) { status.textContent = `测试失败：${err.message}`; }
    };
    deleteBtn.onclick = async (event) => {
      event.preventDefault();
      const name = form.querySelector('[data-key="name"]').value;
      if (!name || !confirm(`删除公司远程配置“${name}”？`)) return;
      try { await api(`/configs/${encodeURIComponent(name)}`, { method: "DELETE" }); status.textContent = "已删除。"; selectedName = ""; await refresh(); }
      catch (err) { status.textContent = `删除失败：${err.message}`; }
    };

    function nextName() { let i = configs.length + 1; while (configs.some((item) => item.name === `config_${i}`)) i += 1; return `config_${i}`; }

    try { await refresh(); } catch (err) { status.textContent = `加载失败：${err.message}`; setFormValues(form, emptyConfig()); }
  }

  function button(text, className = "") { const btn = document.createElement("button"); btn.type = "button"; btn.textContent = text; btn.className = className; return btn; }

  function installEntry() {
    const open = () => openPanel().catch((err) => alert(err.message));
    window.CompanyRemoteSettings = { open };
    const attach = () => {
      const menu = document.querySelector(".comfy-menu, .comfyui-menu, nav, body");
      if (!menu || document.getElementById("company-remote-settings-button")) return;
      const btn = button("Company Remote", "crapi-menu-button");
      btn.id = "company-remote-settings-button";
      btn.onclick = open;
      menu.appendChild(btn);
    };
    attach(); setTimeout(attach, 1000); setTimeout(attach, 3000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", installEntry, { once: true });
  else installEntry();
})();
