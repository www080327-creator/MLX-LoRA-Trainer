# 更新日志 / Changelog

## v2.0.0 (2026-07-13)

### 🆕 新增 / Added
- 原生桌面 GUI 应用，基于 pywebview，无需浏览器 / Native desktop app with pywebview
- 中英双语全链路报错提示，附带问题说明与解决方案 / Bilingual plain-language error messages
- 中英双语 UI 界面 / Bilingual UI
- 训练中每5步自动保存 adapter 权重 / Auto-save adapter every 5 steps
- 高级模式折叠面板，可查看原始训练日志 / Advanced mode panel for raw logs
- 训练前校验：数据集大小、模型层数 / Pre-training validation: dataset size, model layers
- 学习率调度器支持（预留，暂未启用） / LR schedule support (reserved)

### 🔧 修复 / Fixed
- 进程残留导致端口占用 / Process leak causing port conflicts
- 暂停后重新训练图表数据交织 / Chart data corruption after pause-then-retrain
- 输出文件跨模型混显 / Output files mixed across different models
- 启动时量化级别显示为"无量化" / Quantization level showing "none" at startup

### 🗑️ 移除 / Removed
- WebUI 模式（浏览器访问）/ WebUI mode (browser access)
- 继续训练按钮（功能不稳定） / Resume training button (unstable)

---

## v1.0.0 (2026-07-08)

### 🆕 新增 / Added
- 基于 Gradio 的 WebUI 微调工具 / Gradio-based WebUI fine-tuning tool
- LoRA / QLoRA 微调支持 / LoRA/QLoRA support
- 数据集管理 / Dataset management
- 内置聊天测试 / Built-in chat testing
- Loss 曲线实时绘制 / Real-time loss curve
