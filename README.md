<div align="center">

# 🦙 MLX LoRA 训练工具 v2.0

**Apple Silicon 专属 · 零基础原生桌面 LoRA 微调工具 / Zero-barrier native desktop LoRA fine-tuning for Apple Silicon Macs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![macOS](https://img.shields.io/badge/macOS-13.0+-blue)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1/M2/M3/M4-black)
![Version](https://img.shields.io/badge/version-2.0.0-green)
![MLX](https://img.shields.io/badge/MLX-LoRA-orange)
[![Stars](https://img.shields.io/github/stars/www080327-creator/MLX-LoRA-Trainer)](https://github.com/www080327-creator/MLX-LoRA-Trainer)

<br>

### ⬇️ [📥 点此下载 v2.0 / Download v2.0](https://github.com/www080327-creator/MLX-LoRA-Trainer/releases/latest)

> 🔍 不要点绿色的「Code」按钮 — 点上面这个链接下载 / Don't click the green "Code" — click the link above

</div>

---

## ✨ 核心亮点 / Highlights

### 🖥️ 真正的原生桌面应用，开箱即用 / True Native Desktop App
- 彻底告别浏览器与 localhost 地址，**单 APP 一键启动/关闭**，无后台残留进程 / Single-click start/stop, no leftover processes
- **全量封装运行依赖**，无需安装 Python、配置 MLX 环境，下载 `.dmg` 拖入应用程序即可使用 / Download .dmg, drag to Applications, done
- **全程图形化操作**，无需敲任何命令行代码 / No command line, 100% GUI / No command line, 100% GUI

### 🧑‍💻 全链路小白友好，零门槛跑通微调 / Beginner Friendly
- 所有代码报错、运行日志全部转译为**大白话提示**，附带中英双语说明与解决方案 / Bilingual plain-language error messages with fixes
- 预设全场景最优训练参数，仅需 **3 步完成微调**：导入数据集 → 选择基础模型 → 点击开始 / Import data → select model → click start
- 训练进度、loss 曲线**实时可视化**，全程清晰掌握训练状态 / Real-time loss curve, always know what is happening

### ⚡ 原生 MLX 框架优化，性能拉满 / MLX Powered
- 基于 Apple MLX 框架深度适配，M 系列芯片满血运行，**本地微调无数据泄露风险 / Runs locally, no data ever leaves your machine**
- 保留完整高级参数面板，进阶用户可自由调整训练配置 / Full advanced settings for power users
- 支持 LoRA / QLoRA，自动检测模型量化级别（Q2~Q8） / Auto-detect quantization (Q2-Q8)

---

## 🆕 v2.0 重大更新 / What's New

| 更新 [Update] | 说明 [Description] |
|------|------|
| 🏗️ **架构全面重构** [Architecture Refactor] | 从 Web 前后端分离模式，升级为纯原生桌面 GUI 应用 [WebUI → native desktop app] |
| 💬 **全链路人话报错** [Plain-Language Errors] | 中英双语报错，附带问题说明与解决方案 [Bilingual errors with solutions] |
| 🌐 **中英双语界面** [Bilingual UI] | 全部 UI 文字中英双语呈现 [All UI in Chinese and English] |
| 🎨 **UI 大幅优化** [UI Overhaul] | 界面细节全面改进，操作逻辑更简单直观 [Major UI improvements] |
| 🔧 **稳定性提升** [Stability] | 修复进程残留、训练中断、图表数据交织等已知问题 [Fixed process leaks and chart bugs] |

---

## 📸 界面预览 / Screenshots

### 🎯 训练页面 / Training
<img src="assets/train.png" width="800" alt="训练页面">

### 💬 测试页面 / Test
<img src="assets/test.png" width="800" alt="测试页面">

### 📝 数据集管理 / Dataset
<img src="assets/dataset.png" width="800" alt="数据集管理">

---

## 🚀 快速上手 / Quick Start

### 安装（仅需 3 步）/ Installation

1. 前往 [Releases](https://github.com/www080327-creator/MLX-LoRA-Trainer/releases) 下载最新 `.dmg` / Download the latest `.dmg`
2. 打开安装包，将 `MLX训练.app` 拖入「应用程序」文件夹 / Drag to Applications folder
3. 首次打开：打开「**系统设置**」→「**隐私与安全性**」→ 往下滚动找到 `MLX训练` → 点击「**仍要打开**」 / First launch: System Settings → Privacy & Security → scroll down → "Open Anyway"

> ⚠️ 无需安装任何前置依赖，无需配置环境变量 / No dependencies, no environment setup
> ⚠️ 首次弹窗"无法验证"是正常现象（未签名应用），以上步骤只需操作一次 [Gatekeeper warning is normal, only needed once]
> 💡 如果系统设置里找不到"仍要打开"，可在终端运行： [If "Open Anyway" not found, run in Terminal:]
> `xattr -d com.apple.quarantine /Applications/MLX训练.app`

### 3 步完成微调 / 3 Steps to Fine-tune

1. 切换到 **📝 数据集** Tab，输入「问题」和「答案」 / Go to Dataset tab, enter Q&A
   > 💡 也可直接导入仓库的 [示例数据集](examples/sample_dataset.json) 快速测试 [Or import the sample dataset to test quickly]
2. 切换到 **🎯 训练** Tab，选择模型 → 选择数据集 → 创建适配器 / Select model → dataset → adapter
3. 点击 **▶️ 开始训练**，实时查看 loss 曲线 / Click Start Training, watch loss curve

### 测试效果 / Test

切换到 **💬 测试** Tab，选择模型和适配器，输入消息即可对话。 / Go to Test tab, select model + adapter, start chatting.

---

## 📂 项目结构 / Project Structure

```
MLX-LoRA-Trainer/
├── src/                    # 核心源码 / Core source
│   └── mlx_lora_tool.py    # 主程序 (~600 lines Python)
├── MLX训练.app/            # macOS app bundle
├── scripts/                # 打包脚本 / Build scripts
├── examples/               # 示例数据集 / Sample dataset
│   └── sample_dataset.json # 10条示例，可直接导入测试
├── assets/                 # 截图 / Screenshots
└── README.md
```

> 💡 源码在 `src/mlx_lora_tool.py`，全部本地运行，可自行审计安全性 / Source code is in `src/`, fully open for security audit

---

## 🤖 模型支持 / Model Support

### 支持的基础模型 / Supported Base Models
理论上支持所有能在 MLX 上运行的 HuggingFace 格式模型，实测包括： / Supports all HuggingFace-format models runnable on MLX, tested includes:

| 模型 [Model] | 参数量 [Params] | 推荐内存 [Recommended RAM] |
|------|------|------|
| Qwen2.5 / Qwen3 | 0.5B ~ 72B | 16GB+ |
| Llama 3 / 3.1 | 1B ~ 70B | 16GB+ |
| Mistral | 7B | 16GB+ |
| Gemma 2 | 2B ~ 27B | 16GB+ |
| DeepSeek | 7B ~ 67B | 32GB+ |
| 通义千问系列 / Qwen series | 全系列 | 16GB+ |

### 支持的模型格式 / Supported Formats
- ✅ HuggingFace 格式的 MLX 模型（`.safetensors` / `.npz`）
- ✅ 量化模型（Q2~Q8），自动检测 / Quantized models (Q2-Q8), auto-detected
- ✅ MTP/DSPARK 头的模型 / Models with MTP/DSPARK heads
- ❌ `.gguf` 格式（需用 oMLX 转换） / .gguf format (convert with oMLX)
- ❌ PyTorch 格式（需用 MLX 转换工具） / PyTorch format (convert via MLX tools)

### 模型获取方式 [How to Get Models]
- 推荐使用 [oMLX](https://github.com/Judn/omlx) 一键下载（Mac 上的 MLX 模型商店） [Recommended: oMLX, a Mac MLX model store]
- 或用命令行下载： `huggingface-cli download <模型名> --local-dir ~/.cache/huggingface/hub/models--<模型名>` [Or CLI: huggingface-cli download <model> --local-dir ...]
- 工具自动扫描 `~/.cache/huggingface/hub/` 目录，下载后重启工具即可识别 [Auto-scans HF cache, restart app to recognize]

### 🌟 新手推荐模型 [Beginner Recommendations]
入门建议从小模型开始，训练快、不挑配置： [Start with small models for fast training:]

| 模型 [Model] | 参数量 [Size] | 最低内存 [Min RAM] | 推荐理由 [Why] |
|------|------|------|------|
| Qwen2.5-0.5B | 0.5B | 8GB | 最快跑通流程，5 分钟看到效果 [Fastest, see results in 5 min] |
| Qwen3-0.8B | 0.8B | 8GB | 效果比 0.5B 好，速度依然很快 [Better quality, still fast] |
| Qwen2.5-1.5B | 1.5B | 16GB | 入门最佳平衡点 [Best beginner balance] |
| Llama-3.2-1B | 1B | 16GB | Meta 官方模型，社区资源多 [Official Meta model, rich community] |

---

## 🔒 隐私声明 / Privacy Statement

**✅ 100% 本地运行，零网络请求，零数据上传。** / **100% local, zero network, zero upload.**

- 训练全程在本地完成，不联网也可正常使用 [All training runs locally, works offline]
- 工具本身不发起任何网络请求，模型下载需用户自行操作 [The app makes zero network requests; model downloads are user-initiated]
- 无遥测、无埋点、无任何数据收集 [No telemetry, no tracking, no data collection]
- 源码公开可审计：[src/mlx_lora_tool.py](src/mlx_lora_tool.py) [Source code open for audit]
- 用户数据存储在 `~/.mlx_train/`，完全由你掌控 [Your data stays in ~/.mlx_train/, fully under your control]
- 后续版本升级不会覆盖或删除已有数据 / Future upgrades won't overwrite or delete existing data

---

## 📋 数据集格式 / Dataset Format

### 支持的格式 / Supported Formats

**JSON 格式**（推荐 / Recommended）
```json
[
  {"instruction": "问题", "input": "", "output": "答案"}
]
```

### 完整示例 / Complete Example

仓库已提供示例数据集 / Sample dataset included: [examples/sample_dataset.json](examples/sample_dataset.json)（10 条中英文问答 / 10 bilingual Q&A pairs）

### 目录结构 / Directory Structure

```
~/.mlx_train/
├── datasets/                    # 数据集存储
│   ├── my_data.json             # 用户编辑格式（GUI 可编辑）
│   └── my_data_mlx.jsonl       # 训练格式（自动生成）
└── adapters/                    # 训练输出 / Adapter output
    └── my_adapter/
        ├── meta.json            # 训练参数记录
        └── adapters.safetensors # LoRA 权重文件
```

---

## ⚠️ 断点续训 / Resume Training

**v2.0 暂不支持断点续训。 / Resume training is not supported in v2.0.** 训练中断（停止训练、app 闪退、电脑休眠）后需从头开始。

> 💡 训练每 5 步自动保存一次 adapter，中断后可用测试页面对话查看效果，但无法从上次进度继续训练。计划在 v2.1 支持此功能。

---

## ⚙️ 训练参数说明 / Training Parameters

| 参数 [Parameter] | 默认值 [Default] | 说明 [Description] |
|------|--------|------|
| 训练层数 [Train Layers] | 4 | LoRA 适配的目标层数。越大模型改动越多，不能超过模型总层数 [More = deeper fine-tuning. Must not exceed model layers] |
| 批处理大小 [Batch Size] | 1 | 每次训练的样本数。越大训练越快但越耗内存 [Larger = faster but more RAM] |
| 迭代步数 [Iterations] | 20 | 训练的总步数。200 步足以在 0.8B 模型上看到效果 [200 steps visibly improves 0.8B models] |
| 学习率 [Learning Rate] | 1e-4 | 控制参数更新幅度，典型范围 1e-5~1e-3。越小越稳定 [Smaller = more stable, typical range 1e-5~1e-3] |
| 最大文本长度 [Max Seq Length] | 256 | 单条文本的最大 token 数，长文本需调大 [Max tokens per sample. Increase for longer inputs] |

### LoRA 权重导出 / LoRA Weight Export

训练完成后，LoRA 权重保存在 / After training, LoRA weights are saved at `~/.mlx_train/adapters/<adapter_name>/adapters.safetensors`。

- 可在测试页面直接加载对话 / Load in Test tab for chatting
- 可用 `mlx-lm fuse` 与基础模型合并 [Merge with base model: `mlx-lm fuse --model <base> --adapter-path <adapter>`]
- Ollama 兼容路径：合并后通过 GGUF 转换工具导入 [Ollama path: fuse → convert to GGUF → import]
- 格式为标准 safetensors，兼容 MLX 生态 / Standard safetensors format, MLX ecosystem compatible

---

## 📋 系统要求 / System Requirements

- **macOS 13.0+**（Apple Silicon，M1/M2/M3/M4）
- **最低 8GB 内存**（仅支持 1B 以下小模型），**推荐 16GB+**，32GB+ 流畅运行 7B+ 模型 [**8GB+ RAM** (1B- models only), **16GB+ recommended**, 32GB+ for 7B+ models]

### 性能参考 / Performance Reference

| 芯片 [Chip] | 0.8B 模型 [Model] | 7B 模型 [Model] | 27B 模型 [Model] |
|------|------|------|------|
| M1 (16GB) | ✅ 流畅 [Smooth] | ⚠️ 仅量化版 [Q4 only] | ❌ 不足 [Insufficient] |
| M2 Max (32GB) | ✅ 流畅 [Smooth] | ✅ 流畅 [Smooth] | ⚠️ 可用 [OK] |
| M3 Max (36GB) | ✅ 流畅 [Smooth] | ✅ 流畅 [Smooth] | ✅ 可用 [OK] |
| M4 Max (48GB+) | ✅ 流畅 [Smooth] | ✅ 流畅 [Smooth] | ✅ 流畅 [Smooth] |

---

## ❓ 常见问题 / FAQ

| 问题 [Question] | 回答 [Answer] |
|------|------|
| **支持哪些设备？** [Supported devices?] | 仅 Apple Silicon Mac（M1/M2/M3/M4），Intel 不支持 [Apple Silicon Macs only, Intel not supported] |
| **需要安装 Python 吗？** [Need Python?] | 不需要，应用已封装全部依赖 [No, all dependencies bundled] |
| **训练数据会上传云端吗？** [Data privacy?] | 不会，所有训练全程本地运行 [All training runs locally] |
| **支持哪些模型？** [Supported models?] | HuggingFace-format MLX models，详见上方模型支持表 [See model support table above] |
| **支持断点续训吗？** / **Resume training?** | v2.0 不支持 / Not supported，每次训练从头开始 |
| **训练很慢怎么办？** / **Training too slow?** | 减少迭代步数 / Reduce iterations、减小文本长度、使用更小的模型 |
| **显存不足报错？** / **Out of memory?** | 减少 batch_size / Reduce batch size、减小文本长度、换更小的模型 |
| **模型加载失败？** / **Model load failed?** | 检查下载完整性 / Check download，磁盘空间是否充足 |
| **LoRA 权重怎么导出？** / **How to export weights?** | 在 `~/.mlx_train/adapters/` 目录，标准 safetensors 格式 |
| **能和 Ollama 一起用吗？** / **Compatible with Ollama?** | 需用 / Requires `mlx-lm fuse` 合并后转换，暂不直接兼容 |

---

## 🛠 技术栈 / Tech Stack

- **[pywebview](https://pywebview.flowrl.com/)** — 原生桌面窗口 / Native desktop window
- **[Gradio](https://www.gradio.app/)** — UI 框架 / UI framework
- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon ML 框架 / ML framework
- **[mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/lora)** — LLM 训练与推理 / Training & inference

---

## 🗺️ 路线图 / Roadmap

### ✅ 已实现 [Implemented]

| 功能 [Feature] | 说明 [Description] |
|------|------|
| 🖥️ 原生桌面应用 [Native Desktop] | pywebview 窗口，无需浏览器 [No browser needed] |
| 🎯 LoRA/QLoRA 训练 [Training] | 支持 Q2~Q8 量化级别 [Q2-Q8 quantization] |
| 💬 测试对话 [Chat Test] | 内置 mlx-lm generate [Built-in chat] |
| 📝 数据集管理 [Dataset] | GUI 编辑 + JSON 导入 [GUI edit + JSON import] |
| 📈 Loss 曲线 [Loss Curve] | 实时可视化 [Real-time visualization] |
| 🌐 中英双语 [Bilingual] | 全部 UI + 文档 [Full UI + docs] |
| 💬 大白话报错 [Plain Errors] | 中文解释 + 解决方案 [Chinese explanation + fix] |
| 🔒 隐私安全 [Privacy] | 全本地运行，零网络 [100% local, zero network] |
| 🔍 源码公开 [Open Source] | src/mlx_lora_tool.py 可审计 [Auditable] |
| 📋 新手引导 [Onboarding] | 示例数据集 + 入门模型推荐 [Sample dataset + model recs] |

### ⏳ 计划中 [Planned]

| 功能 [Feature] | 版本 [Version] | 说明 [Description] |
|------|------|------|
| 📉 学习率调度器 [LR Scheduler] | v2.1 | Cosine/Linear Decay，训练更稳定 [More stable training] |
| 🌀 启动动画 [Loading Screen] | v2.1 | 消除启动白屏 [Eliminates white screen at launch] |
| 🔄 断点续训 [Resume Training] | v2.2 | 中断后从 checkpoint 继续 [Continue from checkpoint] |
| 🔗 LoRA 合并 [Weight Fusion] | v2.3 | 与基础模型合并导出 [Merge with base model] |
| ✍️ Apple 签名 [Developer Signing] | 社区 | 消除 Gatekeeper 弹窗 [Eliminate Gatekeeper warning] |
| 📤 数据集导入/导出 [Dataset Export] | 待定 | 标准格式导出 [Export standard format] |

---

## 🤝 贡献 / Contributing

欢迎提交 Issue 和 PR！请先阅读 / Issues and PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证 / License

[MIT](LICENSE)

---

<div align="center">

**如果这个工具对你有帮助，请给一个 ⭐️！/ Give it a ⭐️ if it helps you!**

</div>
