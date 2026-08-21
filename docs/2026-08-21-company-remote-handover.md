# Company Remote 定制版对接与交接说明

> **交接对象**：接手本项目的开发、运维或工作流使用人员
> **文档日期**：2026-08-21
> **已提交基线**：`f09cbbd9`（源码与测试：`dc07e983`；交接文档：`f09cbbd9`）
> **适用范围**：本项目新增的 Company Remote 节点、远端媒体调用、长视频资产处理、Seedance/Wan 工作流和运行数据
>
> 本文不是原生 ComfyUI 的通用安装手册。接手方仍需先准备一个可正常运行的 ComfyUI Windows Portable 环境，再按本文接入本项目的定制模块。

## 1. 先看结论

本项目的核心交付不是单个节点，而是一条“本地分镜与任务管理 + 远端图片/文本/视频服务 + TOS/资产网关 + 本地断点恢复与合并”的工作流链路。

接手时必须同时区分三类内容：

| 类别 | 说明 | 是否随普通 Git checkout 得到 |
| --- | --- | --- |
| 已提交源码 | `custom_nodes/company_remote/`、相关测试和 `server.py` 配套改动，当前源码/测试基线为 `dc07e983`。 | 已跟踪的文件可以得到；被忽略目录中的新增文件仍需单独确认。 |
| 已忽略运行扩展 | 输入视频、配置、资产库、manifest、断点和输出视频，以及未列入第 6.2 节的本机工作流。 | 大部分位于 `.gitignore` 忽略目录，必须作为独立交付物或按需迁移；第 6.2 节明确列出的 8 份工作流除外。 |
| 其他本机文件 | 当前工作树还存在与本交付无关的研究记录、缓存、日志和输出。 | 不属于 Company Remote 源码交付，不应混入交接包。 |

当前工作树还存在很多与本项目无关的上游文件行尾/格式差异，以及其他未提交文件。交付时不要把整个工作区压缩后直接交给接手方，也不要把整体 `git diff` 当作 Company Remote 的改动清单。

## 2. 版本边界与本次定制内容

### 2.1 业务提交链

| 提交 | 日期 | 主要内容 |
| --- | --- | --- |
| `45d659c7` | 2026-07-15 | 公司远端图片/文本/视频节点、配置界面、提示词增强和基础 API 适配。 |
| `81e95717` | 2026-07-17 | 多人转绘、提示词/任务持久化和前端恢复能力。 |
| `b65106ef` | 2026-07-17 | 多人工作流补充和远端视频工作流调整。 |
| `c8c8a373` | 2026-07-20 | 远端视频参数校验和工作流参数修正。 |
| `41ae589f` | 2026-08-07 | 长视频、资产网关、Seedance/Wan 多人物节点、前端组件和本地测试主体。 |
| `48555967` | 2026-08-17 | 长视频 v3 身份门控、人物映射、整帧参考、入库素材查看和手动批次分钟区间。 |
| `d509e130` | 2026-08-17 | 长视频 v3、资产网关和 Wan2.7 直传测试补充。 |
| `dc07e983` | 2026-08-21 | 仅人物转绘、人物母版质量门控、自动资产失败根因汇总、持久化固定列预览和对应回归测试。 |
| `f09cbbd9` | 2026-08-21 | Company Remote 对接、部署、运行数据迁移、安全边界和验收交接说明。 |

`41ae589f` 同时包含 `server.py` 的队列/历史清理配套改动。升级 ComfyUI 上游时，不要只复制 custom node 目录而遗漏该配套差异；升级后应重新评估并运行测试。

### 2.2 2026-08-21 已入库扩展

`dc07e983` 已将以下文件相对于 `d509e130` 的真实功能差异纳入版本库：

- `custom_nodes/company_remote/long_video.py`
- `custom_nodes/company_remote/nodes.py`
- `custom_nodes/company_remote/tests/test_long_video.py`

该提交包含：

- 仅人物转绘、背景保持原视频的 `person_only` 范围和对应提示词预设；
- 人物母版质量门控，检查目标风格强度、原风格残留、主体身份和人物数量；
- 自动资产失败根因汇总，区分认证、服务过载、分析失败、质量门控、TOS 上传和身份映射失败；
- 固定列预览的持久化输出；
- 对应的回归测试。

接手方应以 `dc07e983` 或更高版本作为 Company Remote 源码/测试基线；只 checkout `d509e130` 无法得到这些扩展。

## 3. 系统结构

```text
┌──────────────────────────────┐
│ ComfyUI Web UI / Workflow     │
└──────────────┬───────────────┘
               │ 节点注册、前端扩展、工作流执行
┌──────────────▼───────────────┐
│ custom_nodes/company_remote/  │
│ __init__.py                   │
│ nodes.py / web/*.js           │
└───────┬──────────┬────────────┘
        │          │
        │          ├── 配置与 API 路由
        │          │   config_store.py
        │          │   /api/company_remote/*
        │          │
        │          └── 远端媒体客户端
        │              client.py
        │              图片 / 文本 / Seedance / Wan
        │
        ├── long_video.py
        │   镜头检测 → 资产分析 → 身份门控 → 整帧质量筛选
        │   → 参考包 → 远端分段生成 → 断点恢复 → FFmpeg 合并
        │
        ├── asset_gateway.py / asset_gateway_video.py
        │   TOS 上传 → 资产注册 → asset_id / asset:// 引用 → 任务轮询
        │
        └── three_person_seedance_video.py
            three_person_wan27_video.py
            三人物独立分段转换与原音频恢复

持久化：
  user/default/company_remote/   配置、资产缓存、素材库、身份映射
  output/company_remote/         manifest、任务目录、请求视频、中间结果和最终视频
```

## 4. 功能概览

### 4.1 长视频 v3 主线

长视频 v3 以镜头和逻辑镜头组为单位运行：

1. `CompanyLongVideoShotDetector` 使用 PySceneDetect 检测镜头和转场。宽视频会先生成低分辨率代理，预览也会降采样，以降低内存峰值。
2. 规划节点读取镜头首、中、尾帧和文本分析，建立人物、地点、动作和请求组。
3. contract=4 先生成整帧转绘参考，并检查目标风格、原风格残留、构图、人物数量/身份和场景身份；未通过的参考不能进入付费视频阶段。
4. 通过 `identity_mapping_json` 固定全局人物与镜头槽位，避免同一人物在不同镜头重复建资产或错误续跑。
5. 参考打包节点只接受状态为 `ready` 的合格素材；`partial`、`degraded`、`identity_review_required` 和失败项不能静默进入 Seedance。
6. `CompanyLongVideoSegmentGenerator` 按请求组顺序调用 Seedance。不同逻辑镜头不传递上一组末帧；只有同一逻辑镜头因模型时长上限拆分时才保持连续性。
7. 最终合并节点用 FFmpeg 合并请求结果，并按 `use_original_audio` 选择恢复源音频或使用模型生成音频。

支持的目标资源类型包括：

- `二维动漫资源`
- `真人写实资源`
- `欧美化资源`
- `3D / 游戏 CG 资源`
- `漫画插画资源`
- `自定义`

手动批次的 `start_minute` / `end_minute` 单位是**分钟**。`0`/`0` 保持原顺序批次行为；只填起始分钟时按批次长度取范围；指定结束分钟时使用精确端点。修改节点 schema 后必须重启 ComfyUI 后端，旧进程可能仍显示“秒”版控件。

### 4.2 身份、资产和恢复

- 身份证据不足时记录为 `partial`，不生成或上传人物素材。
- 只有证据完整且存在高置信候选歧义时才进入 `identity_review_required`；这与远端临时 5xx 或缺少可比较字段不同。
- 人工映射中的 `global_people` 定义固定人物，`shot_people` 使用 `镜头号:槽位` 指向全局人物或 `ignore`。
- 修改人物映射会改变任务签名，防止使用旧人物资产错误续跑。
- 已有 active `asset_id` 的人物会跳过重复上传；资产库查看节点只读，不触发付费生成。
- 相同输入、相同配置和相同映射下，任务可从 manifest、帧缓存、分析结果、人物/场景素材和参考包继续执行。不要手动改 manifest 的 signature，也不要删除任务目录中的中间文件。

### 4.3 三人物独立流程

- Seedance 资产链：把源视频和人物 A/B/C 注册为 `Video`/`Image` 资产，再用 `asset://asset-id` 进行三人物整头转换。
- Wan2.7 直传链：本地拆成三个真实视频分支，分别调用三个独立付费节点，再按原帧数精确合并并恢复原音频。示例工作流采用 0–195、196–373、374–483 帧三段，实际使用时以输入视频和节点中的分段 JSON 为准。
- 三人物流程必须始终保持 A/B/C 映射一致；先用一段小视频验证，不要首次就运行完整付费视频。

### 4.4 非主线内容

旧 v2、纯换脸 CPU 节点、局部测试工作流和历史工作流仍可用于兼容或排查，但不是长视频 v3 的推荐生产入口。ReActor 的整头服饰/头部迁移能力不能由旧换脸节点替代；是否继续使用应由接手方按实际素材单独验收。

## 5. 代码地图

| 路径 | 职责 | 接手时优先关注 |
| --- | --- | --- |
| `custom_nodes/company_remote/__init__.py` | 注册节点、前端目录和 HTTP 路由。 | 路由、`NODE_CLASS_MAPPINGS`、导入失败。 |
| `custom_nodes/company_remote/nodes.py` | 节点 schema、输入输出和 ComfyUI 执行入口。 | 长视频规划、手动批次分钟控件、模型下拉项。 |
| `custom_nodes/company_remote/long_video.py` | 镜头检测、资产分析、质量门控、任务 manifest、恢复、参考包、分段生成和合并。 | 长视频问题通常先从这里追踪。 |
| `custom_nodes/company_remote/client.py` | OpenAI 兼容文本/图片调用、Seedance/Wan 请求、轮询、媒体上传、TOS 预签名、视频裁剪。 | 修改供应商协议时先核对 payload 和响应 JSON 路径。 |
| `custom_nodes/company_remote/config_store.py` | 配置校验、读写和环境变量取密钥。 | `RemoteMediaConfig` 字段和 `configs.json` 存储位置。 |
| `custom_nodes/company_remote/asset_gateway.py` | 图片/视频资产注册、Hash 缓存、TOS 上传和 active 状态轮询。 | 资产 ID 复用和授权边界。 |
| `custom_nodes/company_remote/asset_gateway_video.py` | Seedance 资产 ID 视频提交、轮询和结果下载。 | `asset://` 内容顺序与 role。 |
| `custom_nodes/company_remote/three_person_seedance_video.py` | 三人物 Seedance 分段处理、裁剪、结果保存和音频恢复。 | 付费开关和请求目录。 |
| `custom_nodes/company_remote/three_person_wan27_video.py` | Wan2.7 三段拆分、上传、生成、合并和音频恢复。 | 固定 24 FPS 上传副本与原帧数恢复。 |
| `custom_nodes/company_remote/web/` | 配置面板、进度事件、预览、提示词预设和持久化 UI。 | 节点 UI 改动必须与对应 JS 一起检查。 |
| `custom_nodes/company_remote/tests/` | 本地 mock/回归测试，不默认调用真实付费服务。 | `test_long_video.py` 是长视频主回归。 |
| `server.py` | ComfyUI 队列删除时补充历史记录清理。 | 上游升级时人工处理冲突。 |

建议的关键函数定位：

- 镜头与时长：`detect_long_video_shots`、`select_manual_batch_range`、`adapt_shot_plan_to_requests`；
- 身份映射：`load_identity_mapping_record`、`save_identity_mapping_record`；
- 素材库：`collect_registered_asset_inventory`、`build_asset_library_view`；
- 资产与参考：`build_long_video_auto_assets`、`pack_long_video_auto_references`；
- 执行与合并：`generate_long_video_segments`、`generate_long_video_segments_parallel`、`merge_long_video_job`；
- 配置 API：`__init__.py` 中 `/configs`、`/models`、`/test` 和 `/identity_mappings` 路由。

## 6. 交付物清单

### 6.1 必须交付

1. 与接手方约定好的 ComfyUI Portable 基线版本；至少确认其能启动并访问 `/system_stats`。
2. 已提交版本 `dc07e983` 对应的 Company Remote 源码与测试，以及既有 `server.py` 配套差异。
3. `custom_nodes/company_remote/requirements.txt`。
4. `custom_nodes/company_remote/tests/` 中的本地回归测试。
5. 本文档。
6. 经过 JSON 解析和节点版本核对的工作流包。工作流不应只依赖接手方本机已有文件。

### 6.2 推荐单独打包的工作流

这些工作流位于 `user/default/workflows/`。目录本身被 `.gitignore` 忽略，但以下**指定文件已使用强制暂存纳入本交付提交**；接手方 checkout 对应 commit 即可得到。不要因此把同目录的其他本机工作流、素材或运行数据一并打包。

所有交付工作流都新增了 ComfyUI 原生 `MarkdownNote` 画布笔记：说明整体思路、关键节点作用、远端费用边界、执行顺序和验收标准。笔记是前端虚拟节点，无输入、输出或连线，不进入后端执行队列，也不需要额外安装自定义节点。

| 文件 | 用途与画布说明 |
| --- | --- |
| `视频欧美转绘_长视频_分镜检测测试.json` | 只做本地镜头检测和预览，不调用视频模型；笔记指引先看全局切点，再逐镜头复核。 |
| `视频欧美转绘_长视频_连续分镜生成测试.json` | 选择连续 1–3 个镜头做低成本端到端测试；笔记区分本地、文本调用和 Seedance 付费阶段。 |
| `人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_流水线.json` | 推荐的 v3 手动批次生产入口，按批次审阅后继续；笔记说明 contract=4、系列游标、质量门控和人工放行点。 |
| `人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_含入库素材库查看.json` | 在批次旁查看当前和历史入库素材；笔记说明素材查看为只读，以及实际付费上游。 |
| `人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_仅人物转绘保留背景.json` | 仅人物转绘、原背景保持不变的手动批次版；笔记补充背景不可重绘的验收标准。 |
| `人物视频多风格转绘_长视频_按镜头自动资产转绘_Seedance版_素材库复用v3_短镜头合并_音频可选_限时生成.json` | 自动规划、短镜头合并和音频开关的 v3 长视频入口；笔记说明适合已完成小样验收后的自动处理。 |
| `Wan2.7_三人物参考图直传_无TOS无火山_20秒完整处理.json` | 三人物 Wan2.7 显式三段处理样例；笔记标明三条独立付费分支、帧范围和原音频恢复。 |
| `三人物ABC_火山资源ID_Seedance转换_10秒验证.json` | 资产 ID 注册与 Seedance 小样验证；笔记标明资产注册、视频生成和保存节点默认停用，需按顺序人工启用。 |

两份带素材库查看的手动审阅工作流已补齐已有 link 的源端反向引用，不新增、不删除或改写任何执行连线。

少量早期工作流已经被 Git 跟踪，但不能据此推断整个 `user/default/workflows/` 都会随 checkout 出现。交付前用 `git ls-files user/default/workflows` 和实际文件清单逐项核对。

### 6.3 可选迁移数据

| 路径 | 内容 | 迁移条件 |
| --- | --- | --- |
| `user/default/company_remote/long_video_asset_library.json` | 已入库人物/场景索引、状态、路径和远端 `asset_id`。 | 确认素材归属、远端 asset ID 和 TOS 对象均可转交。 |
| `user/default/company_remote/asset_gateway_cache.json` | 资产注册缓存。 | 与素材库和对应账号一起迁移，不能单独迁移。 |
| `output/company_remote/long_video_jobs/` | v3 manifest、源片段、帧、素材和请求/结果视频。 | 需要继续未完成作业时整体迁移对应 job 目录。 |
| `output/company_remote/manual_batch_series/` | 手动批次系列状态、批次游标和 attempt。 | 需要继续同一系列时必须迁移。 |
| `output/company_remote/identity_mappings/` | 保存的人物映射 JSON。 | 与系列、输入素材和资产库一并迁移。 |
| `input/` 中的样片 | 测试视频和参考图片。 | 仅在取得素材授权后迁移。 |

### 6.4 明确禁止直接交付

以下内容不能放入公共仓库、普通压缩包、截图或工作流 JSON：

- `user/default/company_remote/configs.json` 中的 API Key 或私有服务地址；
- `user/default/company_remote/debug/` 中的 payload、媒体地址和调试日志；
- TOS AK/SK、预签名 URL、Bearer 值和其他鉴权 Header；
- 未确认可转让的远端 `asset_id`、输入素材、人物图和视频；
- 当前机器专用的 CLIProxyAPI 启动脚本、快捷方式和绝对路径；
- 无关的工作区修改、缓存、`__pycache__`、个人账号数据和历史日志。

## 7. 安装与启动

### 7.1 准备 Portable Python

Windows 下必须使用 Portable 包内的 Python，不要使用 WSL 或系统 Python 代替：

```bat
cd /d F:\Path\To\ComfyUI_windows_portable
python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\company_remote\requirements.txt
```

Company Remote 直接声明的额外依赖为：

```text
scenedetect>=0.7,<0.8
```

其余运行时依赖由当前 ComfyUI 基线提供，例如 Torch、NumPy、Pillow、requests、OpenCV、PyAV、`imageio_ffmpeg` 和 `comfy_api`。启用 TOS 预签名媒体传递时，代码还需要 `tos` Python 包；请在接手环境用 Portable Python 安装并执行 `pip check`，不要从另一套 Python 复制 `site-packages`。

### 7.2 通用启动命令

本机访问：

```bat
cd /d F:\Path\To\ComfyUI_windows_portable
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen 127.0.0.1 --port 8188
```

可信局域网访问（仅在确有需要时）：

```bat
cd /d F:\Path\To\ComfyUI_windows_portable
python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen 0.0.0.0 --port 8188
```

`0.0.0.0` 只是监听地址，浏览器应使用服务器实际局域网 IP。不要把 8188 直接暴露到公网；跨网络访问应使用 VPN 或带身份认证的反向代理。

当前机器的 `run_amd_gpu.bat` 依赖本机专用的 CLIProxyAPI 启动器、绝对路径和本地健康检查，不能原样复制给接手方。接手方应替换为自己的网关地址和启动方式，或使用上面的通用命令。

### 7.3 启动后检查

浏览器打开 `http://127.0.0.1:8188`，然后检查：

```powershell
(Invoke-WebRequest http://127.0.0.1:8188/ -UseBasicParsing).StatusCode
(Invoke-RestMethod http://127.0.0.1:8188/system_stats).devices
(Invoke-RestMethod http://127.0.0.1:8188/object_info/CompanyLongVideoShotDetector).CompanyLongVideoShotDetector.name
```

启动日志不应出现 Company Remote 的 `IMPORT FAILED`、Traceback 或缺少模块错误。节点加载完成后，重新打开工作流；改过节点 schema 后不要只刷新浏览器，必须重启后端。

## 8. 远端服务配置

### 8.1 长视频 v3 的最低服务集合

| 配置名称 | 用途 | 是否必需 |
| --- | --- | --- |
| `gpttext` | 镜头、人物、场景和动作结构化分析；也可承担 AI-Zero-Token 图片接口的兼容基础。 | 长视频 v3 必需 |
| `gptimage2` 或 `gptimage2_wisart` | 生成目标人物/场景/整帧参考。 | 选择图片生成阶段时必需 |
| `seedance2` | Seedance 请求组提交、轮询和结果下载。 | Seedance 长视频必需 |
| `aliyun_dashscope_video` | Wan/HappHorse 等阿里云百炼视频请求。 | 使用对应 Wan 工作流时需要 |
| `aliyun_dashscope_video_direct` | 不使用 TOS 预签名、直接媒体传递的 Wan 变体。 | 只在对应工作流选择时需要 |
| `seedance_asset_gateway` | Seedance 资产注册和资产 ID 视频链路。 | 使用资产 ID 工作流时需要 |

服务实际可用的模型名、账户权限、请求路径和响应结构以提供商协议为准。配置界面模板只是起点，保存后必须点击“测试连接”。

### 8.2 脱敏配置字段模板

不要把以下模板中的占位符替换后提交到公共仓库；真实值应通过接手方自己的密钥系统或受控渠道注入。

```json
{
  "name": "seedance2",
  "base_url": "https://<provider-host>",
  "submit_path": "/<submit-path>",
  "method": "POST",
  "auth_header": "Authorization",
  "auth_prefix": "Bearer",
  "api_key_env": "COMPANY_REMOTE_API_KEY",
  "timeout_seconds": 600,
  "poll_enabled": true,
  "poll_path_template": "/<task-path>/{task_id}",
  "poll_interval_seconds": 5,
  "max_poll_attempts": 120,
  "test_path": "/<health-or-model-path>",
  "request_template": {},
  "media_delivery": "base64",
  "tos_enabled": false,
  "tos_bucket": "<bucket>",
  "tos_endpoint": "<tos-endpoint>",
  "tos_region": "<region>",
  "tos_key_prefix": "comfyui/<purpose>/",
  "tos_access_key_env": "VOLC_ACCESS_KEY",
  "tos_secret_key_env": "VOLC_SECRET_KEY",
  "tos_url_expires_seconds": 7200,
  "response_image_url_path": "<json.path>",
  "response_video_url_path": "<json.path>",
  "response_result_url_path": "<json.path>",
  "response_task_id_path": "<json.path>",
  "response_status_path": "<json.path>",
  "success_statuses": ["succeeded", "success", "completed", "done"],
  "failure_statuses": ["failed", "error", "cancelled", "canceled"],
  "extra_headers": {}
}
```

使用 TOS 预签名媒体传递时：

1. 将 `media_delivery` 设为 `tos_presigned`，或启用 `tos_enabled`；
2. 填写接手方有权限的 Bucket、Endpoint 和 Region；
3. 在**启动 ComfyUI 的同一进程环境**设置 `tos_access_key_env` 和 `tos_secret_key_env` 指定的环境变量；
4. 确认远端服务能访问生成的临时 URL；
5. 不要把签名 URL 写入日志、截图或问题单。

### 8.3 Company Remote 配置 API

以下是本插件提供的本地 ComfyUI 路由，不是远端供应商的公共 API：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/company_remote/configs` | 列出已保存配置，返回脱敏信息。 |
| `POST` | `/api/company_remote/configs` | 新建配置。 |
| `PUT` | `/api/company_remote/configs/{name}` | 更新配置。 |
| `DELETE` | `/api/company_remote/configs/{name}` | 删除配置。 |
| `GET` | `/api/company_remote/models?config=gpttext` | 获取或读取缓存的模型列表。 |
| `POST` | `/api/company_remote/test` | 测试一份未保存的配置。 |
| `GET` | `/api/company_remote/identity_mappings/{series_id}` | 读取系列人物映射。 |
| `PUT`/`POST` | `/api/company_remote/identity_mappings/{series_id}` | 保存系列人物映射。 |

插件也保留不带 `/api` 的兼容路径。当前这些自定义路由未替代 ComfyUI 的认证网关，因此不能因路径带 `/api` 就认为它们适合公网暴露。

## 9. 人物映射格式

规划节点的 `identity_mapping_json` 可以填写 JSON 文本，也可以填写已保存的映射文件路径。最小示例：

```json
{
  "global_people": [
    {
      "name": "人物A",
      "asset_id": "asset-<authorized-person-a>",
      "path": "<optional-local-image>",
      "status": "confirmed"
    },
    {
      "name": "人物B",
      "asset_id": "asset-<authorized-person-b>",
      "path": "<optional-local-image>",
      "status": "confirmed"
    }
  ],
  "shot_people": {
    "1:P1": "人物A",
    "1:P2": "人物B",
    "2:P1": "ignore"
  },
  "expected_distinct_people": 2
}
```

引用未定义人物、未确认人物或无权访问的资产 ID 会触发 `identity_mapping_failed`，不应通过手改 manifest 绕过。映射接口保存的文件位于 `output/company_remote/identity_mappings/`，应与对应系列状态和素材库一起迁移。

## 10. 推荐工作流与验收顺序

### 第 1 步：本地无付费验证

打开 `视频欧美转绘_长视频_分镜检测测试.json`，使用 3–10 秒样片运行：

- 视频能被读取；
- 镜头切点和首/中/尾帧预览正常；
- 没有 `PySceneDetect`、OpenCV、FFmpeg 或显存/内存错误；
- 不执行图片生成、资产注册或视频生成。

### 第 2 步：小范围远端链路验证

打开 `视频欧美转绘_长视频_连续分镜生成测试.json`，只选择连续 1–3 个镜头：

1. 先确认 `gpttext`、图片服务和视频服务的连接测试通过；
2. 先运行分析和素材阶段，检查人物/场景参考；
3. 确认人物数量、人物身份和场景符合预期后再执行视频节点；
4. 检查结果视频时长、帧率、音频和请求日志是否对应当前片段；
5. 失败时保留 manifest 和中间文件，用相同输入恢复，不要删除整个 job。

这一步可能产生远端调用费用，正式执行前需要接手方确认账号、预算和素材授权。

### 第 3 步：v3 手动批次生产

推荐使用：

```text
人物视频多风格转绘_长视频_Seedance版_v3手动批次_1分钟审阅_流水线.json
```

建议顺序：

```text
2. 检测镜头与转场
  → 3. 控制批次范围（分钟）
  → 4. 建立 contract=4 资产任务
  → 5. 整帧转绘并淘汰弱转换
  → 6. 打包合格参考图
  → 7. Seedance 顺序生成
  → 8. 播放检查
  → 9. 合并并按开关处理音频
```

第一次建议只填 `0.5`–`1` 分钟，完成一批人工检查后再扩大范围。需要查看素材库时使用带 `CompanyLongVideoAssetLibraryViewer` 的版本；该节点只读，不会因为查看而触发付费调用。

当前带素材库查看的分阶段工作流及旧分阶段审阅版存在两个保存元数据问题：部分节点输出仍引用缺失的 link 3/link 12。生产优先使用连线核验通过的“流水线”版本；若要使用分阶段版，先在 ComfyUI 中重新连线、保存并重新检查 JSON。

### 第 4 步：Wan2.7 三人物流程

打开：

```text
Wan2.7_三人物参考图直传_无TOS无火山_20秒完整处理.json
```

该工作流的画布明确显示：

```text
本地源视频拆分（不付费）
  → 第 1 段 Wan 生成
  → 第 2 段 Wan 生成
  → 第 3 段 Wan 生成
  → 按原帧数合并并恢复完整原音频
```

每段生成前检查：

- A/B/C 参考图和提示词映射一致；
- 上传副本的帧率、时长和大小符合供应商限制；
- `audio_setting` 与最终“恢复原音频”的意图一致；
- 三个付费节点是否按预期启用，避免误触发重复费用。

`Wan3.0_三人物参考图直传_无TOS无火山_20秒完整处理.json` 结构相近，但不要仅凭文件名断言模型参数完全相同；打开后以节点实际模型和控件为准。

## 11. 日常恢复与故障处理

### 11.1 断点恢复

| 现象 | 处理 |
| --- | --- |
| 某个图片/人物/场景失败 | 保留 job 目录，修复配置或远端服务后用相同输入恢复；系统会逐项补缺。 |
| `identity_review_required` | 检查观察图、人物映射和候选证据，完成确认后重新运行。不要绕过门控。 |
| 远端 HTTP 5xx 或超时 | 先确认服务健康和账号状态，再使用同一系列/同一任务恢复。不要删除已有中间文件。 |
| 页面显示“秒”而不是“分钟” | 完全重启 ComfyUI 后端，再重新载入工作流；不要在旧 schema 上保存。 |
| 结果音画不同步 | 检查源视频时长、请求组时长、FFmpeg 合并日志、原音频开关和最终输出帧率。 |
| 资产重复或找不到 | 检查 asset cache、素材库、映射和资产所属账号，不要只复制单个 `asset_id` 文本。 |

### 11.2 常见边界

- Seedance/Wan 的模型名称、时长上限、参考图数量和视频上传限制由供应商协议决定；节点虽有前置校验，仍需在接手账号上实际验证。
- TOS 预签名 URL 过期后，旧请求不能简单重用；应根据 manifest 状态重新生成或恢复上传。
- 远端服务返回 HTTP 200 但正文为空时，文本客户端会有限次重试；持续为空应检查网关认证和模型映射。
- 远程图片/视频和资产注册均可能产生费用；本地测试文件中的 mock 通过不代表真实服务一定可用。

## 12. 验证命令

以下命令应在 Windows Portable 根目录执行。命令只做本地检查，不会自动调用真实付费服务（除非接手方另行运行工作流或 smoke 脚本）：

```bat
cd /d F:\Path\To\ComfyUI_windows_portable

python_embeded\python.exe -s -m py_compile ^
  ComfyUI\custom_nodes\company_remote\__init__.py ^
  ComfyUI\custom_nodes\company_remote\asset_gateway.py ^
  ComfyUI\custom_nodes\company_remote\asset_gateway_video.py ^
  ComfyUI\custom_nodes\company_remote\client.py ^
  ComfyUI\custom_nodes\company_remote\config_store.py ^
  ComfyUI\custom_nodes\company_remote\long_video.py ^
  ComfyUI\custom_nodes\company_remote\nodes.py ^
  ComfyUI\custom_nodes\company_remote\three_person_seedance_video.py ^
  ComfyUI\custom_nodes\company_remote\three_person_wan27_video.py

python_embeded\python.exe -s ComfyUI\custom_nodes\company_remote\tests\test_long_video.py
python_embeded\python.exe -s ComfyUI\custom_nodes\company_remote\tests\test_asset_gateway.py
python_embeded\python.exe -s ComfyUI\custom_nodes\company_remote\tests\test_asset_gateway_video.py
python_embeded\python.exe -s ComfyUI\custom_nodes\company_remote\tests\test_wan27_three_image_direct.py
python_embeded\python.exe -s ComfyUI\custom_nodes\company_remote\tests\test_three_person_seedance_video.py
```

前端脚本检查：

```bat
node --check ComfyUI\custom_nodes\company_remote\web\company_remote.js
node --check ComfyUI\custom_nodes\company_remote\web\auto_asset_progress.js
node --check ComfyUI\custom_nodes\company_remote\web\fixed_column_image_preview.js
node --check ComfyUI\custom_nodes\company_remote\web\parallel_video_progress.js
```

接手验收分三层：

| 层级 | 通过标准 | 是否可能产生费用 |
| --- | --- | --- |
| 本地静态/单元测试 | 编译、节点注册、JSON 解析、mock 测试通过。 | 否 |
| 远端小样 | 配置测试通过，1–3 个镜头生成成功，音频/时长/人物映射正确。 | 是 |
| 生产验收 | 完成一个 0.5–1 分钟批次并能人工审阅、恢复和合并。 | 是 |

历史记录中曾有长视频本地回归全绿的结果，但这只是原开发环境的历史证据；接手方仍须在自己的 Portable Python 中重新执行，并保存实际输出。

## 13. 安全、授权和升级

1. 不要把 8188 直接暴露到公网；配置 API 没有替代外层认证。
2. API Key、TOS AK/SK、私有网关 URL 和预签名 URL 只通过受控密钥渠道传递。
3. 仅处理已取得授权的人物、视频、图片和远端资产；涉及人脸、人物身份和商业生成时由业务方确认合规范围。
4. 不要在工作正常时执行无约束的全量 `pip install -U`；先记录 Portable 版本和 Company Remote commit，再升级。
5. 升级 ComfyUI 上游时，优先保留 `custom_nodes/company_remote/`、其 requirements 和必要的 `server.py` 配套改动；升级后重新运行静态检查、本地测试和工作流节点注册检查。
6. 交付前先锁定 Company Remote 源码 commit（当前为 `dc07e983` 或更高版本），再同步匹配的工作流 JSON；不要交付“源码是旧版、工作流是新版”的混合包。

## 14. 最终交接验收清单

### 交付前

- [ ] 已确认接手方使用的 ComfyUI Portable 基线和 Python 路径。
- [ ] 已确认交付的 Company Remote 源码/测试基线为 `dc07e983` 或更高版本，并记录实际 commit。
- [ ] 已确认 8 份指定工作流已随交付 commit 获得，且画布中的 `MarkdownNote` 说明与目标工作流版本一致。
- [ ] 已从配置、日志、截图和压缩包中移除 API Key、TOS 凭据、预签名 URL 和未授权 asset ID。
- [ ] 已区分源码、工作流、输入素材、资产库和断点数据，不混装个人缓存。

### 接手方安装后

- [ ] ComfyUI 能启动，首页返回 HTTP 200。
- [ ] `/system_stats` 返回预期设备，Company Remote 节点能在 `/object_info` 找到。
- [ ] `scenedetect` 和 TOS（如使用）安装在同一个 Portable Python 中，`pip check` 无冲突。
- [ ] `gpttext`、图片服务、`seedance2`（以及 Wan/资产网关所需配置）测试通过。
- [ ] 分镜检测测试工作流能在本地样片上完成，不触发付费视频。
- [ ] 小范围真实工作流完成后，人物映射、视频时长、帧率和音频均已人工检查。
- [ ] 需要续跑时，job、系列状态、映射、资产缓存和素材库已成套迁移。

> **后端重启提醒**：交付节点源码、前端扩展、配置 schema 或分钟批次控件发生变化后，接手方必须完全关闭并重新启动 ComfyUI 后端，再重新载入工作流。仅刷新浏览器不能保证节点定义和旧工作流控件同步。
