# -*- coding: utf-8 -*-
"""
MLX LoRA 训练工具 v9 — 输出管理 + 防呆测试 + mlx_lm 聊天
"""

import gradio as gr
import subprocess, threading, os, time, re, json, httpx, shutil
from pathlib import Path

# 自动检测 conda 路径
def _find_conda():
    for p in [os.path.expanduser("~/miniforge3"), os.path.expanduser("~/miniconda3"),
              os.path.expanduser("~/anaconda3"), "/opt/homebrew/anaconda3"]:
        if os.path.isfile(f"{p}/bin/python3"):
            return p
    return os.path.expanduser("~/miniforge3")
_CONDA_BASE = _find_conda()
CONDA_SH = f"source {_CONDA_BASE}/bin/activate llamafactory"
CONDA_PY = f"{_CONDA_BASE}/envs/llamafactory/bin/python3"
DATASET_DIR = str(Path.home() / ".mlx_train/datasets")
ADAPTER_DIR = str(Path.home() / ".mlx_train/adapters")
HF_HUB = str(Path.home() / ".cache/huggingface/hub")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

train_proc = None
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

def detect_quant(model_name):
    n = model_name.lower()
    if 'q8' in n or '8bit' in n or '8-bit' in n: return 'Q8'
    if 'q4' in n or '4bit' in n or '4-bit' in n: return 'Q4'
    return ''

def detect_mtp(model_name):
    n = model_name.lower()
    return 'mtp' in n or 'dspark' in n

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
    if not os.path.exists(fp): return [], f"❌ 不存在: {name}"
    with open(fp, encoding="utf-8") as f: data = json.load(f)
    return data, f"✅ 已加载「{name}」，共 {len(data)} 条"

def save_ds(name, data):
    fp = os.path.join(DATASET_DIR, f"{name}.json")
    with open(fp, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    mfp = os.path.join(DATASET_DIR, f"{name}_mlx.jsonl")
    with open(mfp, "w", encoding="utf-8") as f:
        for r in data:
            text = f"<|im_start|>user\n{r.get('instruction','')}<|im_end|>\n<|im_start|>assistant\n{r.get('output','')}<|im_end|>"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return f"✅ 已保存「{name}」，共 {len(data)} 条"

def rename_ds(old, new):
    if old == new: return f"⏭️ 名称未变化", old
    new = new.strip()
    for ext in [".json", "_mlx.jsonl"]:
        ofp = os.path.join(DATASET_DIR, f"{old}{ext}")
        nfp = os.path.join(DATASET_DIR, f"{new}{ext}")
        if os.path.exists(nfp): return f"❌「{new}」已存在", old
        if os.path.exists(ofp): os.rename(ofp, nfp)
    return f"✅ 已重命名为「{new}」", new

def delete_ds(name):
    for ext in [".json", "_mlx.jsonl"]:
        fp = os.path.join(DATASET_DIR, f"{name}{ext}")
        if os.path.exists(fp): os.remove(fp)
    return f"🗑️ 已删除「{name}」"

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
    return f"🗑️ 已删除「{name}」"

# ===== 训练 =====
def run_train(model, data_dir, adapter_dir, layers, batch, iters, lr, seq):
    global train_proc, train_log, train_losses, train_steps, train_done, train_ok
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
    for line in iter(train_proc.stdout.readline,""):
        train_log.append(line)
        m = re.search(r'Iter\s+(\d+):.*Train loss\s+([\d.]+)', line)
        if m: train_steps.append(int(m.group(1))); train_losses.append(float(m.group(2)))
    rc = train_proc.wait()
    train_ok = (rc == 0)
    train_proc=None; train_done=True

def start_training(model_choice, custom_path, ds_name, out_name,
                   layers, batch, iters, lr, seq, quant_override):
    global train_done
    if custom_path and custom_path.strip():
        model = custom_path.strip()
        model_name = os.path.basename(model) or model
    elif model_choice:
        model = model_path(model_choice)
        model_name = model_choice
    else:
        yield "❌ 请选择或输入模型路径", None; return

    if not ds_name or ds_name not in get_ds_list():
        yield f"❌ 数据集「{ds_name}」不存在", None; return
    if not out_name or not out_name.strip():
        yield "❌ 请选择或创建输出文件", None; return

    # 量化判定
    quant = quant_override if (quant_override and quant_override != "自动检测") else detect_quant(model_name)
    is_qlora = quant in ('Q4', 'Q8')
    has_mtp = detect_mtp(model_name)
    mtp_warn = "\n⚠️ 检测到 MTP/DSPARK 头" if has_mtp else ""
    mode_str = f"QLoRA {quant}" if is_qlora else "LoRA"
    header = f"📋 模型: {model_name}\n🎯 模式: {mode_str}{mtp_warn}\n📦 输出: {out_name}\n"

    yield header + "⏳ 准备训练数据...", None

    # ★ 训练前校验模型文件完整性 ★
    yield header + "🔍 校验模型文件...", None
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
            yield header + f"❌ 模型文件校验失败（可能损坏或不完整）：\n{err[:500]}\n\n请在 oMLX 中重新下载该模型", None
            return
    except subprocess.TimeoutExpired:
        yield header + "❌ 模型文件校验超时，请检查文件是否完整", None
        return

    data_dir = prepare_train_data(ds_name)
    adapter_dir = str(Path(ADAPTER_DIR) / out_name)

    # 保存元数据
    save_adapter_meta(out_name, model_name, layers, batch, iters, lr, seq)

    yield header + "🚀 训练启动中...", None

    t = threading.Thread(target=run_train, args=(model, data_dir, adapter_dir, layers, batch, iters, lr, seq), daemon=True)
    t.start()
    while t.is_alive():
        time.sleep(0.5)
        cp = "/tmp/loss_chart.png"
        if len(train_losses) > 1:
            try:
                import matplotlib; matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.clf(); plt.plot(train_steps, train_losses, 'b-o', markersize=3)
                plt.xlabel('Step'); plt.ylabel('Loss'); plt.title('Loss 曲线')
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(cp, dpi=90); plt.close()
            except: pass
        yield header + "".join(train_log[-60:]), cp if os.path.exists(cp) else None
    # 训练结束，检查退出码
    save_adapter_meta(out_name, model_name, layers, batch, iters, lr, seq)
    tail = "".join(train_log[-60:])
    if train_ok:
        yield header + tail + "\n\n✅ 训练完成！", cp if os.path.exists(cp) else None
    else:
        yield header + tail + "\n\n❌ 训练失败，详见上方日志", cp if os.path.exists(cp) else None

def stop_training():
    global train_proc
    if train_proc: train_proc.terminate(); train_proc=None; return "⏹ 训练已停止"
    return "没有正在运行的训练"

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
        reply = "❌ 请选择模型"
        history.append({"role":"user","content":msg})
        history.append({"role":"assistant","content":reply})
        return history

    if not adapter_name:
        reply = "❌ 请选择适配器文件"
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
        reply = proc.stdout.strip() or "（无输出）"
        if proc.returncode != 0:
            reply = f"❌ 生成错误:\n{proc.stderr[:200]}"
    except subprocess.TimeoutExpired:
        reply = "❌ 生成超时（120秒）"
    except Exception as e:
        reply = f"❌ 错误: {str(e)[:200]}"

    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":reply})
    return history

# ===== 界面 =====
with gr.Blocks(title="MLX 训练工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦙 MLX LoRA 训练工具")

    initial_models = get_model_choices()
    default_model = "models--Jundot--Qwen3.6-27B-oQ8-mtp"

    with gr.Tabs():

        # ═══ 训练 ═══
        with gr.TabItem("🎯 训练") as train_tab:
            gr.Markdown("### 训练参数设置")

            # ---- 模型选择 ----
            gr.Markdown("#### 🧠 模型选择")
            train_model_dd = gr.Dropdown(
                label="已下载的模型", choices=initial_models,
                value=default_model if default_model in initial_models else None,
                allow_custom_value=True, info="从 HF 缓存选择，或手动输入路径"
            )
            train_model_custom = gr.Textbox(label="或手动输入路径", placeholder="粘贴完整路径...")

            with gr.Row():
                train_quant = gr.Dropdown(
                    label="量化级别", choices=["自动检测","Q8","Q4","无量化"],
                    value="自动检测", scale=2)
                train_mode_info = gr.Markdown("")

            def update_mode_info(mc, cp, qo):
                name = (cp or "").strip() or mc or ""
                if not name: return "⚙️ 请选择模型"
                quant = qo if (qo and qo!="自动检测") else detect_quant(name)
                is_q = quant in ('Q4','Q8')
                has_m = detect_mtp(name)
                parts = [f"🧩 **{'QLoRA '+quant if is_q else 'LoRA'}**"]
                if has_m: parts.append("⚠️ MTP/DSPARK")
                return "  |  ".join(parts)

            train_model_dd.change(fn=update_mode_info, inputs=[train_model_dd,train_model_custom,train_quant], outputs=[train_mode_info])
            train_model_custom.change(fn=update_mode_info, inputs=[train_model_dd,train_model_custom,train_quant], outputs=[train_mode_info])
            train_quant.change(fn=update_mode_info, inputs=[train_model_dd,train_model_custom,train_quant], outputs=[train_mode_info])

            # ---- 数据集 + 输出 ----
            gr.Markdown("#### 📊 训练数据 & 输出")
            with gr.Row():
                train_ds = gr.Dropdown(label="训练数据集", choices=get_ds_choices(), value=None,
                                       allow_custom_value=True, info="选择数据集", scale=1)
                train_out_dd = gr.Dropdown(label="输出文件", choices=get_adapter_choices(), value=None,
                                           allow_custom_value=True, info="选择已有输出或输入新名称", scale=1)

            with gr.Row():
                out_new_name = gr.Textbox(label="新建输出", placeholder="名称，如 my_lora_001 ...", scale=2)
                out_create_btn = gr.Button("✅ 创建输出", scale=1)

            def create_out(name, model_choice, custom_path):
                name=(name or "").strip()
                if not name: return ("❌ 请输入名称", gr.update(), gr.update())
                model_name = (custom_path or "").strip() or model_choice or ""
                save_adapter_meta(name, model_name, 4, 1, 20, 1e-4, 256)
                return (f"✅ 已创建输出「{name}」", gr.update(choices=get_adapter_choices(), value=name), gr.update(value=""))
            out_create_btn.click(fn=create_out, inputs=[out_new_name, train_model_dd, train_model_custom],
                                 outputs=[train_mode_info, train_out_dd, out_new_name])

            # ---- 全量微调提示 ----
            gr.Markdown("> 📦 **全量微调** 开发中，仅支持 LoRA / QLoRA")

            # ---- 超参数 ----
            gr.Markdown("#### ⚙️ 训练超参数")
            with gr.Row():
                nl = gr.Slider(1, 64, 4, step=1, label="训练层数")
                bs = gr.Slider(1, 8, 1, step=1, label="批处理大小")
                it = gr.Slider(5, 200, 20, step=5, label="迭代步数")
            with gr.Row():
                lr = gr.Number(1e-4, label="学习率")
                sl = gr.Slider(64, 1024, 256, step=64, label="最大文本长度")

            with gr.Row():
                start = gr.Button("▶️ 开始训练", variant="primary", size="lg")
                stop = gr.Button("⏹ 停止训练", variant="stop", size="lg")

            gr.Markdown("---")
            with gr.Row():
                log = gr.Textbox(label="📋 训练日志", lines=12, max_lines=30, scale=2)
                chart = gr.Image(label="📈 Loss 曲线", height=320, scale=1)

            ce = start.click(fn=start_training,
                inputs=[train_model_dd,train_model_custom,train_ds,train_out_dd,nl,bs,it,lr,sl,train_quant],
                outputs=[log, chart])
            stop.click(fn=stop_training, outputs=log, cancels=[ce])

            def refresh_train():
                return (gr.update(choices=get_ds_choices()),
                        gr.update(choices=get_model_choices()),
                        gr.update(choices=get_adapter_choices()))
            train_tab.select(fn=refresh_train, outputs=[train_ds, train_model_dd, train_out_dd])

        # ═══ 测试（防呆 + mlx_lm 聊天）═══
        with gr.TabItem("💬 测试") as test_tab:
            gr.Markdown("### 测试微调模型")

            # ---- 模型选择 ----
            gr.Markdown("#### 🧠 选择模型")
            test_model_dd = gr.Dropdown(
                label="已下载的模型", choices=initial_models,
                value=default_model if default_model in initial_models else None,
                allow_custom_value=True, info="必须先选模型，才能看到对应的适配器"
            )
            test_model_custom = gr.Textbox(label="或手动输入路径", placeholder="粘贴完整路径...")

            # ---- 适配器选择（根据模型过滤）----
            gr.Markdown("#### 📦 选择微调文件")
            test_adapter_dd = gr.Dropdown(
                label="适配器", choices=[], value=None,
                info="只显示当前模型训练出的适配器"
            )
            test_adapter_info = gr.Markdown("")

            # 选模型 → 过滤适配器
            def filter_adapters(model_choice, custom_path):
                model_name = (custom_path or "").strip() or model_choice
                if not model_name:
                    return gr.update(choices=[]), "⚙️ 请先选择模型"
                adapters = get_adapter_choices(model_name)
                info = f"🔗 绑定模型「{model_name}」的适配器：{len(adapters)} 个"
                if not adapters:
                    info += "\n> 暂无适配器，请先在「🎯 训练」中训练一个"
                return gr.update(choices=adapters, value=adapters[0] if adapters else None), info

            test_model_dd.change(fn=filter_adapters, inputs=[test_model_dd, test_model_custom],
                                 outputs=[test_adapter_dd, test_adapter_info])
            test_model_custom.change(fn=filter_adapters, inputs=[test_model_dd, test_model_custom],
                                     outputs=[test_adapter_dd, test_adapter_info])

            # 选择适配器后显示信息
            def show_adapter_info(adapter_name):
                if not adapter_name: return ""
                meta_fp = Path(ADAPTER_DIR) / adapter_name / "meta.json"
                if not meta_fp.exists(): return f"⚠️ 无元数据: {adapter_name}"
                with open(meta_fp) as f: m = json.load(f)
                return (f"📋 **{adapter_name}**  |  层数:{m.get('layers','?')}  "
                        f"步数:{m.get('iters','?')}  |  {m.get('time','?')}")
            test_adapter_dd.change(fn=show_adapter_info, inputs=[test_adapter_dd], outputs=[test_adapter_info])

            # ---- 删除适配器 ----
            with gr.Row():
                test_delete_btn = gr.Button("🗑️ 删除此适配器", variant="stop", size="sm")

            def del_adapter(adapter_name, model_choice, custom_path):
                if not adapter_name: return "❌ 请选择适配器", gr.update(), gr.update()
                msg = delete_adapter(adapter_name)
                model_name = (custom_path or "").strip() or model_choice
                adapters = get_adapter_choices(model_name)
                return msg, gr.update(choices=adapters, value=None)
            test_delete_btn.click(fn=del_adapter, inputs=[test_adapter_dd, test_model_dd, test_model_custom],
                                  outputs=[test_adapter_info, test_adapter_dd])

            # ---- 聊天 ----
            gr.Markdown("---")
            gr.Markdown("#### 💬 对话")
            chatbot = gr.Chatbot(height=400, label="对话", type="messages")
            msg = gr.Textbox(placeholder="输入消息后按回车...")

            # 用 mlx_lm generate 聊天
            msg.submit(fn=mlx_chat,
                       inputs=[msg, chatbot, test_model_dd, test_model_custom, test_adapter_dd],
                       outputs=[chatbot])

            def refresh_test():
                return (gr.update(choices=get_model_choices()),
                        gr.update(choices=get_adapter_choices()))
            test_tab.select(fn=refresh_test, outputs=[test_model_dd, test_adapter_dd])

        # ═══ 数据集 ═══
        with gr.TabItem("📝 数据集") as ds_tab:
            gr.Markdown("### 管理训练数据集")

            ds_dropdown = gr.Dropdown(label="📋 选择数据集", choices=get_ds_choices(),
                allow_custom_value=True, info="选择已有数据集，或输入新名称后按回车加载")

            gr.Markdown("#### ➕ 新建数据集")
            with gr.Row():
                ds_new_name = gr.Textbox(label="数据集名称", placeholder="输入后点创建...", scale=3)
                ds_create_btn = gr.Button("✅ 创建", variant="primary", scale=1)

            gr.Markdown("#### ✏️ 管理")
            with gr.Row():
                ds_rename_input = gr.Textbox(label="重命名为", placeholder="输入新名称...", scale=2)
                ds_rename_btn = gr.Button("✅ 确认重命名", scale=1)
                ds_delete_btn = gr.Button("🗑️ 删除", variant="stop", scale=1)

            ds_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("#### 📝 数据编辑")
            gr.Markdown("*双击单元格可编辑内容*")

            ds_table = gr.Dataframe(headers=["问题","答案"], datatype=["str","str"],
                col_count=(2,"fixed"), label="", interactive=True, row_count=(1,"dynamic"), wrap=True)

            with gr.Row():
                ds_add_btn = gr.Button("➕ 添加一行", size="sm")
                ds_del_btn = gr.Button("➖ 删除最后一行", size="sm")
                ds_save_btn = gr.Button("💾 保存数据集", variant="primary")

            ds_current = gr.Textbox(visible=False, value="")
            N = gr.update()

            def refresh_ds(): return gr.update(choices=get_ds_choices())

            def on_select(choice):
                choice=(choice or "").strip()
                if not choice: return "", [["",""]], ""
                if choice not in get_ds_list(): return f"❌「{choice}」不存在", [["",""]], ""
                data, msg = load_ds(choice)
                return msg, data_to_table(data), choice

            def on_create(name):
                name=(name or "").strip()
                if not name: return ("❌ 请输入名称",N,N,N,refresh_ds())
                if name in get_ds_list(): return (f"❌「{name}」已存在",N,N,N,refresh_ds())
                save_ds(name,[])
                return (f"✅ 已创建「{name}」",N,name,[["",""]],refresh_ds())

            def on_rename(old, new):
                if not old: return ("❌ 请先选择数据集",N,N,N,refresh_ds())
                new=(new or "").strip()
                if not new: return ("❌ 请输入新名称",N,N,N,refresh_ds())
                msg,final=rename_ds(old,new)
                data,_=load_ds(final)
                return (msg,N,final,data_to_table(data),refresh_ds())

            def on_delete(name):
                if not name: return ("❌ 请先选择数据集",N,N,N,refresh_ds())
                return (delete_ds(name),N,"",[["",""]],refresh_ds())

            def on_save(name,tbl):
                if not name: return "❌ 请先选择数据集",N,refresh_ds()
                data=table_to_data(tbl)
                if not data: return "❌ 不能为空",N,refresh_ds()
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
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7878, show_error=True
    )
