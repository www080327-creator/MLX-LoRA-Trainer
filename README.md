<div align="center">

# 🦙 MLX LoRA 训练工具

**在 Mac（Apple Silicon）上对 MLX 量化模型进行 LoRA / QLoRA 微调的图形化工具**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![macOS](https://img.shields.io/badge/macOS-15+-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![MLX](https://img.shields.io/badge/MLX-LoRA-orange)

</div>

---

## 📸 界面预览

> 截图待补充

---

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 🧠 **模型发现** | 自动扫描 HuggingFace 缓存，列出所有已下载模型 |
| 📝 **数据集管理** | 创建、编辑、重命名、删除训练数据集 |
| 🎯 **LoRA / QLoRA 微调** | 自动检测模型量化级别（Q4/Q8），自动选择 LoRA/QLoRA |
| 🔒 **防呆测试** | 测试时按模型过滤适配器，防止加载错误 |
| 💬 **内置聊天** | 用 `mlx_lm generate` 直接测试微调效果 |
| 📈 **Loss 曲线** | 实时绘制训练损失曲线 |
| 🔄 **一键安装** | `.dmg` 安装包 + `install.sh` 自动配置环境 |

## 📋 系统要求

- **macOS 15+**（Apple Silicon，M1/M2/M3/M4）
- **至少 16GB 内存**（推荐 32GB+）
- [Miniforge3](https://github.com/conda-forge/miniforge)（conda 环境管理器）

## 🚀 快速开始

### 方案一：安装包（推荐）

1. 从 [Releases](https://github.com/PERRYGUO1215/MLX-LoRA-Trainer/releases) 下载最新 `.dmg`
2. 双击打开 `.dmg`，先运行 **「安装前请先运行我.command」**
3. 安装完成后，将 `MLX训练.app` 拖入 `/Applications`
4. 双击 `MLX训练.app` 启动

### 方案二：从源码运行

```bash
# 1. 创建 conda 环境
conda create -n llamafactory python=3.12 -y
conda activate llamafactory

# 2. 安装依赖
pip install gradio==5.50.0 mlx-lm matplotlib httpx

# 3. 下载模型（通过 oMLX 或直接用 HuggingFace）
# 例如：
# pip install huggingface-hub
# huggingface-cli download Jundot/Qwen3.6-27B-oQ8-mtp --local-dir ~/.cache/huggingface/hub/

# 4. 运行
python mlx_lora_tool.py
```

打开浏览器访问 **http://127.0.0.1:7878**

## 🎯 使用教程

### 1. 准备数据

切换到 **📝 数据集** Tab：
- 新建数据集 → 输入名称 → 创建
- 在表格中编辑「问题」和「答案」
- 点击「保存数据集」

### 2. 开始训练

切换到 **🎯 训练** Tab：
- **模型选择**：从下拉菜单选择已下载的模型
- **数据集**：选择刚才创建的数据集
- **输出文件**：新建或选择一个输出名称
- **超参数**：调整层数、批大小、迭代步数
- 点击「▶️ 开始训练」

### 3. 测试效果

切换到 **💬 测试** Tab：
- 先选择模型
- 再选择该模型对应的微调适配器（自动过滤）
- 输入消息开始对话

## ⚙️ 训练超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 训练层数 | 4 | LoRA 训练的层数，越大越精细 |
| 批处理大小 | 1 | 越大训练越快，越耗内存 |
| 迭代步数 | 20 | 总训练步数 |
| 学习率 | 1e-4 | 推荐 1e-4 ~ 5e-5 |
| 最大文本长度 | 256 | 单条文本最多 token 数 |

## 📁 数据存储

```
~/.mlx_train/
├── datasets/          # 训练数据集
└── adapters/          # 微调输出（每个适配器包含权重 + meta.json）
```

## 🛠 技术栈

- **[Gradio](https://www.gradio.app/)** — Web UI 框架
- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon 机器学习框架
- **[mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/lora)** — LLM 训练与推理
- **[Matplotlib](https://matplotlib.org/)** — Loss 曲线绘图

## 🤝 贡献

欢迎 Issue、PR！请先开 Issue 讨论再提交 PR。

## 📄 许可证

[MIT](LICENSE)

---

<div align="center">

**如果这个工具对你有帮助，请给一个 ⭐️！**

</div>
