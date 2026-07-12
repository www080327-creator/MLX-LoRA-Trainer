# 贡献指南 / Contributing

欢迎贡献！请先开 Issue 讨论，再提交 PR。 / Please open an Issue first before submitting a PR.

## 开发环境 / Dev Environment

- Python 3.12
- conda 环境：`conda create -n llamafactory python=3.12 -y`
- 依赖：`pip install gradio==5.50.0 mlx-lm matplotlib pywebview`
- 启动：`python src/mlx_lora_tool.py`

## 项目结构 / Project Structure

```
MLX-LoRA-Trainer/
├── src/                    # 核心源码 / Core source
│   └── mlx_lora_tool.py    # 主程序 / Main app
├── MLX训练.app/            # macOS app bundle
│   └── Contents/MacOS/
│       ├── launcher         # 启动脚本 / Launch script
│       └── mlx_lora_tool.py
├── scripts/                # 打包脚本 / Build scripts
│   └── create_dmg.sh
├── assets/                 # 截图 / Screenshots
└── README.md
```

## 提交规范 / Commit Convention

- `feat:` 新功能 / New feature
- `fix:` 修复 / Bug fix
- `docs:` 文档 / Documentation
- `refactor:` 重构 / Refactor
