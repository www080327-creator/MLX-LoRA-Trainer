<div align="center">

# 🦙 MLX LoRA 训练工具 v2.0

**Apple Silicon 专属 · 零基础原生桌面 LoRA 微调工具**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![macOS](https://img.shields.io/badge/macOS-13.0+-blue)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1/M2/M3/M4-black)
![Version](https://img.shields.io/badge/version-2.0.0-green)
![MLX](https://img.shields.io/badge/MLX-LoRA-orange)

</div>

---

## ✨ 核心亮点

### 🖥️ 真正的原生桌面应用，开箱即用

- 彻底告别浏览器与 localhost 地址，**单 APP 一键启动/关闭**，无后台残留进程
- **全量封装运行依赖**，无需安装 Python、配置 MLX 环境，下载 `.dmg` 拖入应用程序即可使用
- **全程图形化操作**，无需敲任何命令行代码

### 🧑‍💻 全链路小白友好，零门槛跑通微调

- 所有代码报错、运行日志全部转译为**大白话提示**，附带中英双语说明与解决方案，无需对着堆栈全网搜答案
- 预设全场景最优训练参数，仅需 **3 步完成微调**：导入数据集 → 选择基础模型 → 点击开始
- 训练进度、loss 曲线**实时可视化**，全程清晰掌握训练状态

### ⚡ 原生 MLX 框架优化，性能拉满

- 基于 Apple MLX 框架深度适配，M 系列芯片满血运行，**本地微调无数据泄露风险**
- 保留完整高级参数面板，进阶用户可自由调整训练配置
- 支持 LoRA / QLoRA，自动检测模型量化级别（Q2~Q8）

---

## 🆕 v2.0 重大更新

| 更新 | 说明 |
|------|------|
| 🏗️ **架构全面重构** | 从 Web 前后端分离模式，升级为纯原生桌面 GUI 应用，启停一体化 |
| 💬 **全链路人话报错** | 新增中英双语报错体系，所有异常提示通俗化，附带问题说明与解决方案 |
| 🌐 **中英双语界面** | 全部 UI 文字中英双语呈现 |
| 🎨 **UI 大幅优化** | 界面细节全面改进，操作逻辑更简单直观 |
| 🔧 **稳定性提升** | 修复进程残留、训练中断、图表数据交织等已知问题 |

---

## 📸 界面预览

> 截图待补充

---

## 🚀 快速上手

### 安装（仅需 2 步）

1. 前往 [Releases](https://github.com/PERRYGUO1215/MLX-LoRA-Trainer/releases) 下载最新 `.dmg`
2. 打开安装包，将 `MLX训练.app` 拖入「应用程序」文件夹即可启动

> ⚠️ 无需安装任何前置依赖，无需配置环境变量

### 3 步完成微调

1. 切换到 **📝 数据集** Tab，创建数据集并输入「问题」和「答案」
2. 切换到 **🎯 训练** Tab，选择模型 → 选择数据集 → 创建适配器
3. 点击 **▶️ 开始训练**，实时查看 loss 曲线与训练进度

### 测试效果

切换到 **💬 测试** Tab，选择模型和适配器，输入消息即可对话测试。

---

## 📋 系统要求

- **macOS 13.0+**（Apple Silicon，M1/M2/M3/M4）
- **至少 16GB 内存**（推荐 32GB+ 以运行 7B+ 模型）

---

## ❓ 常见问题

| 问题 | 回答 |
|------|------|
| **支持哪些设备？** | 仅支持 Apple Silicon 芯片的 Mac（M1/M2/M3/M4 全系列），Intel 芯片暂不支持 |
| **需要安装 Python 吗？** | 不需要，应用已封装全部依赖，下载即可使用 |
| **训练数据会上传云端吗？** | 不会，所有训练全程本地运行，无隐私风险 |
| **支持哪些模型？** | 支持 HuggingFace 格式的 MLX 模型，推荐使用 oMLX 下载 |
| **支持继续训练吗？** | 当前版本暂不支持，每次训练从头开始 |

---

## 🛠 技术栈

- **[pywebview](https://pywebview.flowrl.com/)** — 原生桌面窗口
- **[Gradio](https://www.gradio.app/)** — UI 框架
- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon 机器学习框架
- **[mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/lora)** — LLM 训练与推理

---

## 🤝 贡献

欢迎提交 Issue 和 PR！请先开 Issue 讨论再提交 PR。

## 📄 许可证

[MIT](LICENSE)

---

<div align="center">

**如果这个工具对你有帮助，请给一个 ⭐️！**

</div>
