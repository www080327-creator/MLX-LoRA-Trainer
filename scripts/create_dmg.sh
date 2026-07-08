#!/bin/bash
# 创建 .dmg 安装镜像
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAGING_DIR="/tmp/mlx_trainer_dmg"
DMG_NAME="MLX-LoRA-Trainer-1.0.dmg"
DMG_PATH="$REPO_DIR/$DMG_NAME"

echo "🧹 清理旧文件..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

echo "📂 复制发布文件..."
cp -R "$REPO_DIR/MLX训练.app" "$STAGING_DIR/"
cp -R "$REPO_DIR/install.sh" "$STAGING_DIR/安装前请先运行我.command"
chmod +x "$STAGING_DIR/安装前请先运行我.command"

echo "📦 创建 .dmg..."
rm -f "$DMG_PATH"
hdiutil create -volname "MLX 训练工具" \
    -srcfolder "$STAGING_DIR" \
    -ov -format UDZO \
    -fs HFS+ "$DMG_PATH"

echo "🧹 清理..."
rm -rf "$STAGING_DIR"

echo ""
echo "✅ 打包完成: $DMG_PATH"
ls -lh "$DMG_PATH"
