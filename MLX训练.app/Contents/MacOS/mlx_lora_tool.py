# -*- coding: utf-8 -*-
"""
MLX LoRA 训练工具 v9 — 输出管理 + 防呆测试 + mlx_lm 聊天
"""

import gradio as gr
import subprocess, threading, os, time, re, json, httpx, shutil, socket
from pathlib import Path

CONDA_SH = "source /Users/perry_guo/miniforge3/bin/activate llamafactory"
CONDA_PY = "/Users/perry_guo/miniforge3/envs/llamafactory/bin/python3"
DATASET_DIR = str(Path.home() / ".mlx_train/datasets")
ADAPTER_DIR = str(Path.home() / ".mlx_train/adapters")
HF_HUB = str(Path.home() / ".cache/huggingface/hub")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

train_proc = None
train_thread = None
training_id = 0
train_log = []; train_losses = []; train_steps = []; train_done = False; train_ok = True

# ===== 模型发现 =====
def scan_hf_models():
    models = {}
    hub = Path(HF_HUB)
    if not hub.exists(): return models
    for d in sorted(hub.iterdir()):
        if not d.is_dir() or not d.name.startswith("models--"): continue
        snapshots = d / "snapshots"
        if not snapshots.exists(): continue
        snaps = sorted([s for s in snapshots.iterdir() if s.is_dir()])
        if not snaps: continue
        latest = snaps[-1]
        if (latest / "config.json").exists() or any(latest.glob("*.safetensors")) or any(latest.glob("*.npz")):
            models[d.name] = str(latest)
    return models

def get_model_choices():
    return list(scan_hf_models().keys())

def model_path(name):
    return scan_hf_models().get(name, name)

# 已知量化级别（用于 UI 判断，训练命令本身不依赖此值）
KNOWN_QUANTS = frozenset({'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8'})

def detect_quant(model_name):
    n = model_name.lower()
    # 按精度从高到低匹配，避免 q4 在 q8 之前误匹配
    if 'q8' in n or '8bit' in n or '8-bit' in n: return 'Q8'
    if 'q6' in n or '6bit' in n or '6-bit' in n: return 'Q6'
    if 'q5' in n or '5bit' in n or '5-bit' in n: return 'Q5'
    if 'q4' in n or '4bit' in n or '4-bit' in n: return 'Q4'
    if 'q3' in n or '3bit' in n or '3-bit' in n: return 'Q3'
    if 'q2' in n or '2bit' in n or '2-bit' in n: return 'Q2'
    return '无量化 / None'

def detect_mtp(model_name):
    n = model_name.lower()
    return 'mtp' in n or 'dspark' in n

def get_model_num_layers(model_path):
    """从 config.json 读取模型层数，失败返回 None"""
    cfg = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg):
        return None
    try:
        with open(cfg) as f:
            d = json.load(f)
        # 顶层常见字段
        for key in ("num_hidden_layers", "num_layers", "n_layer"):
            if key in d: return d[key]
        # Qwen3.5 等模型嵌套在 text_config 或 llm_config 中
        for sub in ("text_config", "llm_config", "language_config"):
            if sub in d and isinstance(d[sub], dict):
                for key in ("num_hidden_layers", "num_layers", "n_layer"):
                    if key in d[sub]: return d[sub][key]
    except: pass
    return None

# ===== 数据集管理 =====
def get_ds_list():
    names = []
    for f in Path(DATASET_DIR).glob("*.json"):
        if not f.name.endswith("_mlx.jsonl"):
            names.append(f.name.replace(".json", ""))
    return sorted(names)

def get_ds_choices(): return get_ds_list()

def load_ds(name):
    fp = os.path.join(DATASET_DIR, f"{name}.json")
    if not os.path.exists(fp): return [], f"❌ 不存在 / Not found: {name}"
    with open(fp, encoding="utf-8") as f: data = json.load(f)
    return data, f"✅ 已加载 / Loaded: {name}, {len(data)} examples"

def save_ds(name, data):
    fp = os.path.join(DATASET_DIR, f"{name}.json")
    with open(fp, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    mfp = os.path.join(DATASET_DIR, f"{name}_mlx.jsonl")
    with open(mfp, "w", encoding="utf-8") as f:
        for r in data:
            text = f"<|im_start|>user\n{r.get('instruction','')}<|im_end|>\n<|im_start|>assistant\n{r.get('output','')}<|im_end|>"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return f"✅ 已保存 / Saved: {name}, {len(data)} examples"

def rename_ds(old, new):
    if old == new: return f"⏭️ 名称未变化 / Name unchanged", old
    new = new.strip()
    for ext in [".json", "_mlx.jsonl"]:
        ofp = os.path.join(DATASET_DIR, f"{old}{ext}")
        nfp = os.path.join(DATASET_DIR, f"{new}{ext}")
        if os.path.exists(nfp): return f"❌「{new}」已存在 / Already exists", old
        if os.path.exists(ofp): os.rename(ofp, nfp)
    return f"✅ 已重命名 / Renamed to: {new}", new

def delete_ds(name):
    for ext in [".json", "_mlx.jsonl"]:
        fp = os.path.join(DATASET_DIR, f"{name}{ext}")
        if os.path.exists(fp): os.remove(fp)
    return f"🗑️ 已删除 / Deleted: {name}"

def prepare_train_data(name):
    d = f"/tmp/mlx_train_{name}"
    os.makedirs(d, exist_ok=True)
    src = os.path.join(DATASET_DIR, f"{name}_mlx.jsonl")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(d, "train.jsonl"))
        shutil.copy(src, os.path.join(d, "valid.jsonl"))
    return d

def data_to_table(data):
    if not data: return [["", ""]]
    return [[str(d.get("instruction","")), str(d.get("output",""))] for d in data]

def table_to_data(tbl):
    if tbl is None: return []
    if hasattr(tbl, 'values'): rows = tbl.values.tolist() if hasattr(tbl.values,'tolist') else list(tbl.values)
    elif isinstance(tbl, dict) and 'data' in tbl: rows = tbl['data']
    else: rows = list(tbl)
    result = []
    for row in rows:
        if row is None: continue
        if hasattr(row,'tolist'): row = row.tolist()
        if isinstance(row, str): row = [row, ""]
        else: row = list(row)
        inst = str(row[0] or "").strip() if len(row)>0 else ""
        out = str(row[1] or "").strip() if len(row)>1 else ""
        if inst or out: result.append({"instruction":inst,"input":"","output":out})
    return result

# ===== 适配器（输出）管理 =====
def get_adapter_list(model_filter=None):
    """返回适配器列表。如果提供 model_filter，只返回匹配该模型的。"""
    adapters = []
    for d in Path(ADAPTER_DIR).iterdir():
        if not d.is_dir(): continue
        meta_fp = d / "meta.json"
        if not meta_fp.exists(): continue
        try:
            with open(meta_fp) as f: meta = json.load(f)
        except: continue
        if model_filter and meta.get("model","") != model_filter: continue
        adapters.append(d.name)
    return sorted(adapters)

def get_adapter_choices(model_filter=None):
    return get_adapter_list(model_filter)

def save_adapter_meta(name, model_name, layers, batch, iters, lr, seq):
    """保存适配器元数据"""
    d = Path(ADAPTER_DIR) / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"model": model_name, "layers": layers, "batch": batch,
            "iters": iters, "lr": lr, "seq_len": seq,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def delete_adapter(name):
    d = Path(ADAPTER_DIR) / name
    if d.exists(): shutil.rmtree(d)
    return f"🗑️ 已删除 / Deleted: {name}"

def parse_training_error(log_lines):
    """将原始训练日志中的错误转换为简单的中英双语提示"""
    text = "".join(log_lines)
    # 层数超限
    m = re.search(r'Requested to train (\d+) layers but the model only has (\d+) layers', text)
    if m:
        return f"训练层数({m.group(1)})超过模型层数({m.group(2)}) / Train layers({m.group(1)}) exceed model layers({m.group(2)})"
    # 数据集太小
    m = re.search(r'Dataset must have at least batch_size=(\d+) examples but only has (\d+)', text)
    if m:
        return f"数据集太小(仅{m.group(2)}条)无法满足batch_size({m.group(1)}) / Dataset too small ({m.group(2)} examples) for batch_size({m.group(1)})"
    # 模型文件问题
    if 'FileNotFoundError' in text or 'No such file' in text:
        return "模型文件缺失或不完整 / Model file missing or incomplete"
    # 显存不足
    if 'out of memory' in text.lower() or 'OOM' in text:
        return "显存不足，请减小batch_size或文本长度 / Out of memory, reduce batch_size or seq length"
    # 其他错误 → 提取最后一行有意义的信息
    lines = [l.strip() for l in log_lines if l.strip() and 'Traceback' not in l]
    for l in reversed(lines):
        if 'Error' in l or 'Exception' in l:
            return f"训练出错: {l[:200]} / Training error: {l[:200]}"
    return "训练出错，详见高级模式 / Training error, see Advanced panel"

# ===== 训练 =====
def run_train(model, data_dir, adapter_dir, layers, batch, iters, lr, seq, my_id=0):
    global train_proc, train_log, train_losses, train_steps, train_done, train_ok, training_id
    train_log=[]; train_losses=[]; train_steps=[]; train_done=False; train_ok=True
    os.makedirs(adapter_dir, exist_ok=True)
    cmd = (f"{CONDA_SH} && {CONDA_PY} -m mlx_lm lora"
           f" --model \"{model}\" --train --data \"{data_dir}\""
           f" --num-layers {layers} --batch-size {batch}"
           f" --iters {iters} --learning-rate {lr}"
           f" --steps-per-report 1 --steps-per-eval {max(5,iters//3)}"
           f" --adapter-path \"{adapter_dir}\" --max-seq-length {seq}")
    train_proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash",
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1)
    try:
        for line in iter(train_proc.stdout.readline,""):
            if training_id != my_id:
                train_proc.terminate()
                break
            train_log.append(line)
            m = re.search(r'Iter\s+(\d+):.*Train loss\s+([\d.]+)', line)
            if m: train_steps.append(int(m.group(1))); train_losses.append(float(m.group(2)))
        rc = train_proc.wait()
        train_ok = (rc == 0)
    except Exception:
        train_ok = False
    finally:
        train_proc=None; train_done=True

def start_training(model_choice, custom_path, ds_name, out_name,
                   layers, batch, iters, lr, seq, quant_override):
    global train_done, train_losses, train_steps, train_log
    # 清理旧数据和图表
    train_losses=[]; train_steps=[]; train_log=[]
    if os.path.exists("/tmp/loss_chart.png"):
        os.remove("/tmp/loss_chart.png")
    # 生成空白图表清除旧图残留
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.close('all'); plt.figure(figsize=(1,1))
        plt.savefig("/tmp/loss_chart.png", dpi=10); plt.close('all')
    except: pass
    if custom_path and custom_path.strip():
        model = custom_path.strip()
        model_name = os.path.basename(model) or model
    elif model_choice:
        model = model_path(model_choice)
        model_name = model_choice
    else:
        yield "### ❌ 未选择模型 / No Model Selected\n\n**问题**: 请从下拉框选择模型或手动输入路径\n**Problem**: Please select a model or enter a path", "", None; return

    if not ds_name or ds_name not in get_ds_list():
        yield f"### ❌ 数据集不存在 / Dataset Not Found\n\n**问题**: 数据集「{ds_name}」不存在\n**Problem**: Dataset not found: {ds_name}\n\n**解决**: 请先在数据集管理中创建该数据集\n**Solution**: Create it in Dataset tab first", "", None; return
    if not out_name or not out_name.strip():
        yield "### ❌ 未选择适配器 / No Adapter Selected\n\n**问题**: 请选择已有适配器或输入新名称\n**Problem**: Please select an adapter or enter a new name\n\n**解决**: 点击创建适配器或从下拉框选择\n**Solution**: Click Create Adapter or select from dropdown", "", None; return

    # 量化判定（使用用户选择的值，默认已自动检测回填）
    quant = quant_override
    is_qlora = quant in KNOWN_QUANTS
    has_mtp = detect_mtp(model_name)
    mtp_warn = "\n⚠️ MTP/DSPARK detected" if has_mtp else ""
    mode_str = f"QLoRA {quant}" if is_qlora else "LoRA"
    header = f"📋 Model: {model_name}\n🎯 Mode: {mode_str}{mtp_warn}\n📦 Output: {out_name}\n"

    yield header + "⏳ 准备训练数据 / Preparing data...", "", None

    # ★ 训练前校验模型文件完整性 ★
    yield header + "🔍 校验模型文件 / Validating model files...", "", None
    check_cmd = (f"{CONDA_SH} && {CONDA_PY} -c \""
                 f"import mlx.core as mx, glob, os; "
                 f"fs=glob.glob('{model}/*.safetensors')+glob.glob('{model}/*.npz'); "
                 f"mx.load(fs[0]) if fs else (_ for _ in ()).throw(FileNotFoundError('无权重文件')); "
                 f"print('OK')\"")
    try:
        cp = subprocess.run(check_cmd, shell=True, executable="/bin/bash",
                           capture_output=True, text=True, timeout=60)
        if cp.returncode != 0 or "OK" not in cp.stdout:
            err = cp.stderr.strip() or cp.stdout.strip()
            yield header + f"### ❌ 模型文件校验失败 / Model Validation Failed\n\n**问题**: 模型文件可能损坏或不完整\n**Problem**: Model files may be corrupted or incomplete\n\n**解决**: 请在 oMLX 中重新下载该模型\n**Solution**: Re-download the model in oMLX\n\n<details><summary>详情 / Details</summary>\n\n```\n{err[:500]}\n```\n</details>", "", None
            return
    except subprocess.TimeoutExpired:
        yield header + "### ❌ 模型文件校验超时 / Model Validation Timed Out\n\n**问题**: 模型文件校验超时，可能是文件过大或磁盘慢\n**Problem**: Model validation timed out, file may be too large\n\n**解决**: 请检查磁盘空间，或重新下载模型\n**Solution**: Check disk space or re-download the model", "", None
        return

    data_dir = prepare_train_data(ds_name)
    adapter_dir = str(Path(ADAPTER_DIR) / out_name)

    # 校验数据集大小
    train_file = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(train_file):
        with open(train_file) as f:
            data_count = sum(1 for _ in f)
        if data_count < batch:
            yield header + f"### ❌ 数据集太小 / Dataset Too Small\n\n**问题**: 数据集仅有 {data_count} 条，但 batch_size={batch}，至少需要 {batch} 条\n**Problem**: Dataset has only {data_count} examples but batch_size={batch}, need at least {batch}\n\n**解决**: 减少 batch_size 到 {data_count} 以下，或在数据集中添加更多数据\n**Solution**: Reduce batch_size below {data_count} or add more data", "", None
            return

    # 校验训练层数
    num_layers = get_model_num_layers(model)
    if num_layers and layers > num_layers:
        yield header + f"### ❌ 训练层数超限 / Train Layers Exceed Model\n\n**问题**: 训练层数设为 {layers}，但模型只有 {num_layers} 层\n**Problem**: Requested {layers} layers but model only has {num_layers}\n\n**解决**: 将训练层数改为 {num_layers} 或更小\n**Solution**: Set train layers to {num_layers} or less", "", None
        return

    # 清理旧 adapter 目录并重新创建（避免 mlx_lm 自动续训导致 step 跳跃）
    if os.path.exists(adapter_dir):
        shutil.rmtree(adapter_dir)
    os.makedirs(adapter_dir, exist_ok=True)
    # 保存元数据
    save_adapter_meta(out_name, model_name, layers, batch, iters, lr, seq)

    yield header + "🚀 训练启动中 / Training started...", "", "/tmp/loss_chart.png"

    global train_thread, training_id
    # 递增训练 ID，使旧线程立即停止写入
    training_id += 1
    my_id = training_id
    # 确保旧训练线程已退出
    if train_thread and train_thread.is_alive():
        train_thread.join(timeout=5)
    train_thread = threading.Thread(target=run_train, args=(model, data_dir, adapter_dir, layers, batch, iters, lr, seq, my_id), daemon=True)
    train_thread.start()
    while train_thread.is_alive():
        time.sleep(0.5)
        cp = "/tmp/loss_chart.png"
        if len(train_losses) > 1:
            try:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.close('all'); plt.clf()
                plt.plot(train_steps, train_losses, 'b-o', markersize=3)
                plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Loss 曲线')
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(cp, dpi=90); plt.close('all')
            except: pass
        yield ("🚀 训练中... / Training...\n\n" + header), "".join(train_log[-60:]), cp if os.path.exists(cp) else None
    # 训练结束，检查退出码
    save_adapter_meta(out_name, model_name, layers, batch, iters, lr, seq)
    tail = "".join(train_log[-60:])
    if train_ok:
        last_loss = f"{train_losses[-1]:.4f}" if train_losses else "?"
        summary = f"""✅ 训练完成 / Training Complete!

📊 最终 Loss / Final Loss: {last_loss}
🔄 总步数 / Total Steps: {len(train_steps)}
📦 适配器 / Adapter: {out_name}
"""
        yield summary, tail, cp if os.path.exists(cp) else None
    else:
        err_msg = parse_training_error(train_log)
        summary = f"""❌ 训练失败 / Training Failed

{err_msg}

💡 点击下方「🔧 高级 / Advanced」查看原始日志 / Click Advanced below for raw log
"""
        yield summary, tail, cp if os.path.exists(cp) else None

def stop_training():
    global train_proc, train_thread, train_losses, train_steps, train_log, train_done
    if train_proc:
        train_proc.terminate(); train_proc=None
        if train_thread and train_thread.is_alive():
            train_thread.join(timeout=5)
        train_thread = None
        train_losses=[]; train_steps=[]; train_log=[]; train_done=True
        return "⏹ 训练已停止 / Training stopped"
    return "没有正在运行的训练 / No running training"

# ===== 测试聊天（mlx_lm + adapter）=====
def mlx_chat(msg, history, model_choice, custom_path, adapter_name):
    """用 mlx_lm generate 直接加载模型+adapter 聊天"""
    history = history or []
    if not msg or not msg.strip(): return history

    # 解析模型路径
    if custom_path and custom_path.strip():
        model = custom_path.strip()
    elif model_choice:
        model = model_path(model_choice)
    else:
        reply = "❌ 请选择模型 / Please select a model"
        history.append({"role":"user","content":msg})
        history.append({"role":"assistant","content":reply})
        return history

    if not adapter_name:
        reply = "❌ 请选择适配器文件 / Please select an adapter"
        history.append({"role":"user","content":msg})
        history.append({"role":"assistant","content":reply})
        return history

    adapter_path = str(Path(ADAPTER_DIR) / adapter_name)

    # 构建对话格式
    prompt = ""
    for h in history:
        prompt += f"<|im_start|>user\n{h.get('content','') if h.get('role')=='user' else ''}<|im_end|>\n"
        if h.get('role') == 'assistant':
            prompt += f"<|im_start|>assistant\n{h.get('content','')}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"

    try:
        cmd = (f"{CONDA_SH} && {CONDA_PY} -m mlx_lm generate"
               f" --model \"{model}\" --adapter-path \"{adapter_path}\""
               f" --max-tokens 200 --verbose False --prompt -")
        proc = subprocess.run(cmd, shell=True, executable="/bin/bash",
                              input=prompt, capture_output=True, text=True, timeout=120)
        reply = proc.stdout.strip() or "（无输出 / No output）"
        if proc.returncode != 0:
            err_text = proc.stderr.strip()
            if 'FileNotFoundError' in err_text or 'No such file' in err_text:
                reply = (f"### ❌ 适配器文件缺失 / Adapter file missing\n\n"
                         f"**可能原因**: 适配器「{adapter_name}」的权重文件不存在\n"
                         f"**Likely cause**: Weight file not found for adapter\n\n"
                         f"**解决**: 请重新训练或检查适配器目录\n"
                         f"**Solution**: Re-train or check adapter directory\n\n"
                         f"<details><summary>原始错误 / Raw error</summary>\n\n```\n{err_text[:500]}\n```\n</details>")
            elif 'out of memory' in err_text.lower():
                reply = (f"### ❌ 显存不足 / Out of Memory\n\n"
                         f"**解决**: 尝试使用更小的模型或减少 max-tokens\n"
                         f"**Solution**: Try a smaller model or reduce max-tokens\n\n"
                         f"<details><summary>原始错误 / Raw error</summary>\n\n```\n{err_text[:500]}\n```\n</details>")
            else:
                reply = (f"### ❌ 生成错误 / Generation error\n\n"
                         f"<details><summary>原始错误 / Raw error</summary>\n\n```\n{err_text[:500]}\n```\n</details>")
    except subprocess.TimeoutExpired:
        reply = "### ❌ 生成超时 / Generation timed out (120s)\n\n**解决**: 减小 max-tokens 或检查模型是否正常加载"
    except Exception as e:
        reply = f"### ❌ 错误 / Error: {str(e)[:200]}"

    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":reply})
    return history

# ===== 界面 =====
with gr.Blocks(title="MLX 训练工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦙 MLX LoRA 训练工具 / MLX LoRA Trainer")

    initial_models = get_model_choices()
    default_model = "models--Jundot--Qwen3.6-27B-oQ8-mtp"

    with gr.Tabs():

        # ═══ 训练 ═══
        with gr.TabItem("🎯 训练 / Train") as train_tab:
            gr.Markdown("### 训练参数设置 / Training Settings")

            # ---- 模型选择 ----
            with gr.Accordion("🧠 模型选择 / Model Selection", open=True):
                train_model_dd = gr.Dropdown(
                    label="已下载的模型 / Downloaded Models", choices=initial_models,
                    value=default_model if default_model in initial_models else None,
                    allow_custom_value=True, info="从 HF 缓存选择，或手动输入路径"
                )
                train_model_custom = gr.Textbox(label="或手动输入路径 / Or enter path", placeholder="粘贴完整路径 / Paste full path...")

                with gr.Row():
                    train_quant = gr.Dropdown(
                        label="量化级别 / Quantization", choices=["Q8","Q6","Q5","Q4","Q3","Q2","无量化 / None"],
                        value=detect_quant(default_model) if default_model in initial_models else "无量化 / None",
                        scale=2, info="自动检测后回填，可手动修改 / Auto-detected, can override")
                    train_mode_info = gr.Markdown("")

            def on_quant_change(mc, cp, qo):
                name = (cp or "").strip() or mc or ""
                if not name: return "⚙️ 请选择模型"
                has_m = detect_mtp(name)
                parts = [f"🧩 **{'QLoRA '+qo if qo in KNOWN_QUANTS else 'LoRA'}**"]
                if has_m: parts.append("⚠️ MTP/DSPARK")
                return "  |  ".join(parts)

            train_quant.change(fn=on_quant_change, inputs=[train_model_dd,train_model_custom,train_quant],
                               outputs=[train_mode_info])

            # ---- 数据集 + 输出 ----
            gr.Markdown("#### 📊 训练数据 & 适配器 / Training Data & Adapter")
            with gr.Row():
                train_ds = gr.Dropdown(label="训练数据集 / Dataset", choices=get_ds_choices(), value=None,
                                       allow_custom_value=True, info="选择数据集 / Select dataset", scale=1)
                train_out_dd = gr.Dropdown(label="适配器输出 / Adapter Output", choices=get_adapter_choices(), value=None,
                                           allow_custom_value=True, info="训练产出的 LoRA 适配器 / LoRA adapter from training")

            with gr.Row():
                out_new_name = gr.Textbox(label="新建输出 / New Adapter", placeholder="名称，如 my_lora_001 ... / Name, e.g. my_lora_001", scale=2)
                out_create_btn = gr.Button("✅ 创建适配器 / Create Adapter", scale=1)

            def create_out(name, model_choice, custom_path):
                name=(name or "").strip()
                if not name: return ("❌ 请输入名称 / Please enter a name", gr.update(), gr.update())
                model_name = (custom_path or "").strip() or model_choice or ""
                save_adapter_meta(name, model_name, 4, 1, 20, 1e-4, 256)
                adapters = get_adapter_choices(model_name)
                return (f"✅ 已创建适配器 / Adapter created: {name}", gr.update(choices=adapters, value=name), gr.update(value=""))
            out_create_btn.click(fn=create_out, inputs=[out_new_name, train_model_dd, train_model_custom],
                                 outputs=[train_mode_info, train_out_dd, out_new_name])

            # 模型切换时自动检测量化级别 + 过滤适配器列表
            def on_model_change(mc, cp):
                name = (cp or "").strip() or mc or ""
                if not name:
                    return ("⚙️ 请选择模型 / Please select a model",
                            gr.update(value="无量化 / None"),
                            gr.update(choices=[], value=None))
                quant = detect_quant(name)
                has_m = detect_mtp(name)
                parts = [f"🧩 **{'QLoRA '+quant if quant in KNOWN_QUANTS else 'LoRA'}**"]
                if has_m: parts.append("⚠️ MTP/DSPARK")
                adapters = get_adapter_choices(name)
                return ("  |  ".join(parts),
                        gr.update(value=quant),
                        gr.update(choices=adapters, value=adapters[0] if adapters else None))

            train_model_dd.change(fn=on_model_change, inputs=[train_model_dd,train_model_custom],
                                  outputs=[train_mode_info, train_quant, train_out_dd])
            train_model_custom.change(fn=on_model_change, inputs=[train_model_dd,train_model_custom],
                                  outputs=[train_mode_info, train_quant, train_out_dd])
            # ---- 全量微调提示 ----
            gr.Markdown("> 📦 Full fine-tuning is under development, only LoRA / QLoRA is supported | 全量微调开发中，仅支持 LoRA / QLoRA")

            # ---- 超参数 ----
            gr.Markdown("#### ⚙️ 训练超参数 / Hyperparameters")
            with gr.Row():
                nl = gr.Slider(1, 64, 4, step=1, label="训练层数 / Train Layers", info="LoRA 适配的目标层数，不能超过模型总层数 / Target LoRA layers, must ≤ model layers")
                bs = gr.Slider(1, 8, 1, step=1, label="批处理大小 / Batch Size", info="每次训练的样本数，不能超过数据集条数 / Samples per step, must ≤ dataset size")
                it = gr.Slider(5, 200, 20, step=5, label="迭代步数 / Iterations", info="训练的总步数 / Total training steps")
            with gr.Row():
                lr = gr.Number(1e-4, label="学习率 / Learning Rate", info="控制参数更新幅度，典型范围 1e-5~1e-3 / Parameter update step size")

                sl = gr.Slider(64, 1024, 256, step=64, label="最大文本长度 / Max Seq Length", info="单条文本的最大 token 数 / Max tokens per sample")

            with gr.Row():
                start = gr.Button("▶️ 开始训练 / Start Training", variant="primary", size="lg")
                stop = gr.Button("⏹ 停止训练 / Stop Training", variant="stop", size="lg")

            gr.Markdown("---")
            summary_box = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("🔧 高级 / Advanced", open=False):
                        raw_log = gr.Textbox(label="原始日志 / Raw Log", lines=14, max_lines=30, show_label=False)
                chart = gr.Image(label="📈 Loss 曲线 / Loss Curve", height=320, scale=1)

            ce = start.click(fn=start_training,
                inputs=[train_model_dd,train_model_custom,train_ds,train_out_dd,nl,bs,it,lr,sl,train_quant],
                outputs=[summary_box, raw_log, chart])
            stop.click(fn=stop_training, outputs=raw_log, cancels=[ce])

            def refresh_train(mc, cp):
                model_name = (cp or "").strip() or mc or ""
                adapters = get_adapter_choices(model_name) if model_name else get_adapter_choices()
                return (gr.update(choices=get_ds_choices()),
                        gr.update(choices=get_model_choices()),
                        gr.update(choices=adapters))
            train_tab.select(fn=refresh_train, inputs=[train_model_dd, train_model_custom],
                             outputs=[train_ds, train_model_dd, train_out_dd])

        # ═══ 测试（防呆 + mlx_lm 聊天）═══
        with gr.TabItem("💬 测试 / Test") as test_tab:
            gr.Markdown("### 测试微调模型 / Test Fine-tuned Model")

            # ---- 模型选择 ----
            with gr.Accordion("🧠 选择模型 / Select Model", open=True):
                test_model_dd = gr.Dropdown(
                    label="已下载的模型 / Downloaded Models", choices=initial_models,
                    value=default_model if default_model in initial_models else None,
                    allow_custom_value=True, info="必须先选模型，才能看到对应的适配器 / Select model first to see adapters"
                )
                test_model_custom = gr.Textbox(label="或手动输入路径 / Or enter path", placeholder="粘贴完整路径 / Paste full path...")

            # ---- 适配器选择（根据模型过滤）----
            gr.Markdown("#### 📦 适配器 / Adapter")
            test_adapter_dd = gr.Dropdown(
                label="适配器 / Adapter", choices=[], value=None,
                info="只显示当前模型训练出的适配器 / Only adapters for current model"
            )
            test_adapter_info = gr.Markdown("")

            # 选模型 → 过滤适配器
            def filter_adapters(model_choice, custom_path):
                model_name = (custom_path or "").strip() or model_choice
                if not model_name:
                    return gr.update(choices=[]), "⚙️ 请先选择模型 / Select a model first"
                adapters = get_adapter_choices(model_name)
                info = f"🔗 Model: {model_name} | Adapters: {len(adapters)}"
                if not adapters:
                    info += "\n> 暂无适配器 / No adapters yet, train one first"
                return gr.update(choices=adapters, value=adapters[0] if adapters else None), info

            test_model_dd.change(fn=filter_adapters, inputs=[test_model_dd, test_model_custom],
                                 outputs=[test_adapter_dd, test_adapter_info])
            test_model_custom.change(fn=filter_adapters, inputs=[test_model_dd, test_model_custom],
                                     outputs=[test_adapter_dd, test_adapter_info])

            # 选择适配器后显示信息
            def show_adapter_info(adapter_name):
                if not adapter_name: return ""
                meta_fp = Path(ADAPTER_DIR) / adapter_name / "meta.json"
                if not meta_fp.exists(): return f"⚠️ 无元数据 / No metadata: {adapter_name}"
                with open(meta_fp) as f: m = json.load(f)
                return (f"📋 **{adapter_name}**  |  Layers:{m.get('layers','?')}  "
                        f"Iters:{m.get('iters','?')}  |  {m.get('time','?')}")
            test_adapter_dd.change(fn=show_adapter_info, inputs=[test_adapter_dd], outputs=[test_adapter_info])

            # ---- 删除适配器 ----
            with gr.Row():
                test_delete_btn = gr.Button("🗑️ 删除此适配器 / Delete Adapter", variant="stop", size="sm")

            def del_adapter(adapter_name, model_choice, custom_path):
                if not adapter_name: return "❌ 请选择适配器 / Select an adapter", gr.update(), gr.update()
                msg = delete_adapter(adapter_name)
                model_name = (custom_path or "").strip() or model_choice
                adapters = get_adapter_choices(model_name)
                return msg, gr.update(choices=adapters, value=None)
            test_delete_btn.click(fn=del_adapter, inputs=[test_adapter_dd, test_model_dd, test_model_custom],
                                  outputs=[test_adapter_info, test_adapter_dd])

            # ---- 聊天 ----
            gr.Markdown("---")
            gr.Markdown("#### 💬 对话 / Chat")
            chatbot = gr.Chatbot(height=400, label="对话 / Chat", type="messages")
            msg = gr.Textbox(placeholder="输入消息后按回车 / Type message and press Enter...")

            # 用 mlx_lm generate 聊天
            msg.submit(fn=mlx_chat,
                       inputs=[msg, chatbot, test_model_dd, test_model_custom, test_adapter_dd],
                       outputs=[chatbot])

            def refresh_test(mc, cp, adapter_name):
                model_name = (cp or "").strip() or mc or ""
                adapters = get_adapter_choices(model_name) if model_name else get_adapter_choices()
                # 重新读取当前适配器的元数据（训练后时间会更新）
                info = ""
                if adapter_name:
                    meta_fp = Path(ADAPTER_DIR) / adapter_name / "meta.json"
                    if meta_fp.exists():
                        with open(meta_fp) as f: m = json.load(f)
                        info = (f"📋 **{adapter_name}**  |  Layers:{m.get('layers','?')}  "
                                f"Iters:{m.get('iters','?')}  |  {m.get('time','?')}")
                return (gr.update(choices=get_model_choices()),
                        gr.update(choices=adapters),
                        info)
            test_tab.select(fn=refresh_test, inputs=[test_model_dd, test_model_custom, test_adapter_dd],
                            outputs=[test_model_dd, test_adapter_dd, test_adapter_info])

        # ═══ 数据集 ═══
        with gr.TabItem("📝 数据集 / Dataset") as ds_tab:
            gr.Markdown("### 管理训练数据集 / Manage Datasets")

            ds_dropdown = gr.Dropdown(label="📋 数据集 / Dataset", choices=get_ds_choices(),
                allow_custom_value=True, info="选择或输入新名称 / Select or type new name")

            gr.Markdown("#### ➕ 新建 / Create Dataset")
            with gr.Row():
                ds_new_name = gr.Textbox(label="数据集名称 / Dataset Name", placeholder="输入后点创建 / Enter name then Create...", scale=3)
                ds_create_btn = gr.Button("✅ 创建 / Create", variant="primary", scale=1)

            gr.Markdown("#### ✏️ 管理 / Manage")
            with gr.Row():
                ds_rename_input = gr.Textbox(label="重命名为 / Rename to", placeholder="输入新名称 / Enter new name...", scale=2)
                ds_rename_btn = gr.Button("✅ 确认重命名 / Rename", scale=1)
                ds_delete_btn = gr.Button("🗑️ 删除 / Delete", variant="stop", scale=1)

            ds_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("#### 📝 数据编辑 / Edit Data")
            gr.Markdown("*双击单元格可编辑内容 / Double-click to edit*")

            ds_table = gr.Dataframe(headers=["问题 / Question","答案 / Answer"], datatype=["str","str"],
                col_count=(2,"fixed"), label="", interactive=True, row_count=(1,"dynamic"), wrap=True)

            with gr.Row():
                ds_add_btn = gr.Button("➕ 添加一行 / Add Row", size="sm")
                ds_del_btn = gr.Button("➖ 删除最后一行 / Delete Last Row", size="sm")
                ds_save_btn = gr.Button("💾 保存数据集 / Save Dataset", variant="primary")

            ds_current = gr.Textbox(visible=False, value="")
            N = gr.update()

            def refresh_ds(): return gr.update(choices=get_ds_choices())

            def on_select(choice):
                choice=(choice or "").strip()
                if not choice: return "", [["",""]], ""
                if choice not in get_ds_list(): return f"❌「{choice}」不存在 / Not found", [["",""]], ""
                data, msg = load_ds(choice)
                return msg, data_to_table(data), choice

            def on_create(name):
                name=(name or "").strip()
                if not name: return ("❌ 请输入名称 / Enter a name",N,N,N,refresh_ds())
                if name in get_ds_list(): return (f"❌「{name}」已存在 / Already exists",N,N,N,refresh_ds())
                save_ds(name,[])
                return (f"✅ 已创建 / Created: {name}",N,name,[["",""]],refresh_ds())

            def on_rename(old, new):
                if not old: return ("❌ 请先选择数据集 / Select a dataset first",N,N,N,refresh_ds())
                new=(new or "").strip()
                if not new: return ("❌ 请输入新名称 / Enter a new name",N,N,N,refresh_ds())
                msg,final=rename_ds(old,new)
                data,_=load_ds(final)
                return (msg,N,final,data_to_table(data),refresh_ds())

            def on_delete(name):
                if not name: return ("❌ 请先选择数据集 / Select a dataset first",N,N,N,refresh_ds())
                return (delete_ds(name),N,"",[["",""]],refresh_ds())

            def on_save(name,tbl):
                if not name: return "❌ 请先选择数据集 / Select a dataset first",N,refresh_ds()
                data=table_to_data(tbl)
                if not data: return "❌ 不能为空 / Cannot be empty",N,refresh_ds()
                return save_ds(name,data),N,refresh_ds()

            def on_add_row(tbl):
                if tbl is None: return [["",""]]
                if hasattr(tbl,'values'): rows=tbl.values.tolist()
                elif isinstance(tbl,dict) and 'data' in tbl: rows=list(tbl['data'])
                else: rows=list(tbl)
                rows.append(["",""]); return rows

            def on_del_row(tbl):
                if tbl is None: return [["",""]]
                if hasattr(tbl,'values'): rows=tbl.values.tolist()
                elif isinstance(tbl,dict) and 'data' in tbl: rows=list(tbl['data'])
                else: rows=list(tbl)
                if len(rows)>1: rows.pop()
                else: rows=[["",""]]
                return rows

            ds_dropdown.change(fn=on_select,inputs=[ds_dropdown],outputs=[ds_status,ds_table,ds_current])
            ds_create_btn.click(fn=on_create,inputs=[ds_new_name],outputs=[ds_status,ds_new_name,ds_current,ds_table,ds_dropdown])
            ds_new_name.submit(fn=on_create,inputs=[ds_new_name],outputs=[ds_status,ds_new_name,ds_current,ds_table,ds_dropdown])
            ds_rename_btn.click(fn=on_rename,inputs=[ds_current,ds_rename_input],outputs=[ds_status,ds_rename_input,ds_current,ds_table,ds_dropdown])
            ds_rename_input.submit(fn=on_rename,inputs=[ds_current,ds_rename_input],outputs=[ds_status,ds_rename_input,ds_current,ds_table,ds_dropdown])
            ds_delete_btn.click(fn=on_delete,inputs=[ds_current],outputs=[ds_status,ds_rename_input,ds_current,ds_table,ds_dropdown])
            ds_save_btn.click(fn=on_save,inputs=[ds_current,ds_table],outputs=[ds_status,ds_new_name,ds_dropdown])
            ds_add_btn.click(fn=on_add_row,inputs=[ds_table],outputs=[ds_table])
            ds_del_btn.click(fn=on_del_row,inputs=[ds_table],outputs=[ds_table])
            ds_tab.select(fn=refresh_ds,outputs=[ds_dropdown])

if __name__ == "__main__":
    import threading

    def start_gradio():
        demo.queue(default_concurrency_limit=1).launch(
            server_name="127.0.0.1", server_port=7878, show_error=True,
            inbrowser=False
        )

    try:
        import webview
        USE_WEBVIEW = True
    except ImportError:
        USE_WEBVIEW = False

    if USE_WEBVIEW:
        t = threading.Thread(target=start_gradio, daemon=True)
        t.start()
        # 等待 Gradio 就绪
        for _ in range(20):
            time.sleep(0.5)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if s.connect_ex(("127.0.0.1", 7878)) == 0:
                s.close()
                break
            s.close()
        webview.create_window(
            "MLX 训练工具 / MLX Trainer",
            "http://127.0.0.1:7878",
            width=1280, height=860, resizable=True, min_size=(900, 600)
        )
        webview.start()
    else:
        start_gradio()
