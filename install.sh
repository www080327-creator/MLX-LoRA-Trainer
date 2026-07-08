#!/bin/bash
# MLX 训练工具 — 一键安装脚本
set -e

echo "============================================"
echo "  🦙 MLX LoRA 训练工具 — 安装脚本"
echo "============================================"
echo ""

# 检测 conda
CONDA_BASE=""
for p in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/homebrew/anaconda3"; do
    if [ -f "$p/bin/activate" ]; then
        CONDA_BASE="$p"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "❌ 未找到 Conda 环境！"
    echo "请先安装 Miniforge3:"
    echo "  https://github.com/conda-forge/miniforge"
    echo ""
    echo "安装命令（Apple Silicon）："
    echo "  curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
    echo "  bash Miniforge3.sh"
    exit 1
fi
echo "✅ 检测到 Conda: $CONDA_BASE"

# 创建 llamafactory 环境
CONDA_PY="$CONDA_BASE/envs/llamafactory/bin/python3"
if [ ! -f "$CONDA_PY" ]; then
    echo ""
    echo "📦 正在创建 llamafactory 环境..."
    "$CONDA_BASE/bin/conda" create -n llamafactory python=3.12 -y
    echo "✅ 环境创建完成"
else
    echo "✅ llamafactory 环境已存在"
fi

# 安装 Python 依赖
echo ""
echo "📦 正在安装 Python 依赖..."
source "$CONDA_BASE/bin/activate" llamafactory
pip install -q gradio==5.50.0 mlx-lm matplotlib httpx
echo "✅ 依赖安装完成"

# 复制应用
echo ""
echo "📂 正在安装到 /Applications..."
APP_SOURCE="$(cd "$(dirname "$0")" && pwd)/MLX训练.app"
if [ -d "$APP_SOURCE" ]; then
    rm -rf /Applications/MLX训练.app 2>/dev/null || true
    cp -R "$APP_SOURCE" /Applications/MLX训练.app
    echo "✅ 已安装到 /Applications/MLX训练.app"
else
    echo "⚠️ 未找到 MLX训练.app，请先下载完整包"
    exit 1
fi

echo ""
echo "============================================"
echo "  🎉 安装完成！"
echo "============================================"
echo ""
echo "双击 /Applications/MLX训练.app 启动"
echo "或点击 启动台 → MLX 训练工具"
echo ""
echo "首次启动会自动打开浏览器：http://127.0.0.1:7878"
