"""PMNN 文本模型训练脚本 —— 双通道优化（v0.1 §7）。

    L = L_ce + w_fm·L_fm + w_sync·L_sync + w_drift·L_drift + w_free·L_F

    L_ce     语言建模交叉熵（末步反嵌入）
    L_fm     ELF 流匹配速度场（去噪分支，与解码分支共享权重）
    L_sync   序参量课程跟踪（K 值导航的同步度）
    L_F      自由能正则（热力学；含熵下限防全同步坍缩）

路径全部参数化（CLI > 环境变量 > 自动探测），云端零改代码复用。

本地：
    python train.py --preset small --steps 3000
    python train.py --dry_run --preset 100m        # 只测显存

云端（1B）：
    export PMNN_DATA_DIR=/data/03_dialogue_clean
    export PMNN_OUT_DIR=/workspace/pmnn_out
    python train.py --preset 1b --steps 300000 --batch 16 --grad_accum 8 \
                    --num_workers 8 --optimizer adafactor
"""
import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import add_path_args, resolve_paths
from config import add_config_args, config_from_args
from data_utils import (ensure_tokenizer, prepare_tokens, load_meta,
                        MemmapDataset, collate_fn)
from pmnn.model import PMNNModel
from pmnn.physics import order_parameter


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(cfg, model):
    if cfg.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        print("[train] 优化器: Adafactor（省显存，1B 推荐）")
        return Adafactor(model.parameters(), lr=cfg.learning_rate,
                         scale_parameter=False, relative_step=False,
                         warmup_init=False, weight_decay=cfg.weight_decay)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bias" in n or "norm" in n or "natural_freq" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95))


def lr_at(step, base_lr, warmup, total):
    """学习率调度（warmup + cosine）。

    注意：step / total 都是**绝对步数**（不是本次运行的相对步数）。
    这样 `--resume` 续训时，从 ckpt 的 global_step 接着算 lr，
    与原调度曲线连续 —— 否则续训会从 step 0 重新 warmup 再 cosine，
    把已经退到接近 0 的 lr 又推回高位，等于把训练成果打乱。
    """
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def masked_mse(pred, target, mask):
    """mask 加权的 MSE（排除左 padding）。"""
    m = mask.unsqueeze(-1).float()
    denom = m.sum() * pred.shape[-1] + 1e-6
    return (((pred - target) ** 2) * m).sum() / denom


def param_report(model):
    n = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.token_embed.parameters())
    head = sum(p.numel() for p in model.lm_head.parameters())
    return n, embed, head


def _last_elapsed(log_file: Path) -> float:
    """读 TSV 日志最后一行的累计时长，用于续训时把 elapsed 接上。"""
    try:
        tail = log_file.read_text(encoding="utf-8").strip().splitlines()
        if len(tail) < 2:
            return 0.0
        return float(tail[-1].split("\t")[-1])
    except (ValueError, IndexError, OSError):
        return 0.0


# ---------------------------------------------------------------------------
# 中断保护（终极防御）：Ctrl+C / kill 时先把当前进度存下来再退出
# ---------------------------------------------------------------------------
# 血泪教训：checkpoint 只在固定间隔（如每 5000 步）落盘，中途一被打断，
# 最后一个间隔之后的所有训练成果直接蒸发（实测跑到 9000 步只存到 5000）。
#
# 设计要点：
#   1) 信号处理函数里**只置标志位**，绝不做 torch.save。
#      torch.save 要在 CUDA 算子 / autograd 图正中间执行，极易二次崩溃或存出
#      半更新的权重；置标志位后由主循环在**一步完整结束后**走正常保存路径。
#   2) 手抖连按两次 Ctrl+C 不会丢存档（3 秒内视为同一次）；
#      真要放弃存档强制退出，就等 3 秒后再按一次。
#   3) 正常结束 / 中断 / 异常退出，三条路径都会落盘（异常见 main 的 try/finally）。
_INTERRUPT = {"signum": None, "at": 0.0}
_FORCE_WINDOW = 3.0      # 秒：首次中断后多久，再次按下才视为「真的不想等了」


def _on_interrupt(signum, frame):
    now = time.time()
    if _INTERRUPT["signum"] is not None:
        # 第二次中断：若离首次太近，多半是手抖连按，按首次处理（绝不能因此丢掉存档）
        if now - _INTERRUPT["at"] < _FORCE_WINDOW:
            print(f"\n[train] 已在收尾存档中（连按无效；{_FORCE_WINDOW:.0f} 秒后再按"
                  f"可放弃存档强制退出）", flush=True)
            return
        print("\n[train] 再次收到中断：立即强制退出，放弃本次存档", flush=True)
        _LIVE["active"] = False          # 关掉兜底，别再试着存了
        signal.signal(signum, signal.SIG_DFL)
        raise KeyboardInterrupt
    name = getattr(signal.Signals(signum), "name", str(signum))
    _INTERRUPT["signum"] = signum
    _INTERRUPT["at"] = now
    print(f"\n[train] 捕获到 {name}：本步结束后立即保存当前状态并退出"
          f"（连按两次 = 强制退出，不保存）", flush=True)


def install_interrupt_handlers():
    """注册 SIGINT(Ctrl+C) / SIGTERM(kill) 处理器。"""
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_interrupt)
        except (ValueError, OSError):
            # 非主线程注册会抛 ValueError；忽略即可（训练始终在主线程）
            pass


def find_ckpts(ckpt_dir, preset):
    """列出 `pmnn_<preset>_stepN.pt`，返回按 step 升序的 [(step, Path), ...]。

    只认文件名里的 step 数字（不信 mtime）：续训 / 拷贝 / 云端同步都会
    打乱时间戳，只有步数才是可靠的新旧依据。
    """
    pat = re.compile(rf"^pmnn_{re.escape(str(preset))}_step(\d+)\.pt$")
    items = []
    d = Path(ckpt_dir)
    if not d.exists():
        return items
    for p in d.glob(f"pmnn_{preset}_step*.pt"):
        m = pat.match(p.name)
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items


def rotate_ckpts(ckpt_dir, preset, keep, protect=None):
    """只保留最新 keep 个周期性 checkpoint，旧的删掉（keep<=0 = 不删）。

    按**文件名里的 step 数字**排序，而不是 mtime —— 续训 / 拷贝 / 云端同步
    都会打乱时间戳，只有步数才是可靠的新旧依据。
    `protect` 里的路径永不删除（用于保底存档）。
    """
    if not keep or keep <= 0:
        return []
    protect = {Path(p) for p in (protect or ())}
    items = find_ckpts(ckpt_dir, preset)
    removed = []
    for _, p in items[:-keep]:
        if p in protect:
            continue
        try:
            p.unlink()
            removed.append(p.name)
        except OSError:
            pass
    # 顺手清掉写了一半的临时档（.pt.tmp：存档被中断 / 磁盘写满时留下的残骸）
    for p in Path(ckpt_dir).glob(f"pmnn_{preset}_step*.pt.tmp"):
        try:
            p.unlink()
        except OSError:
            pass
    return removed


def save_ckpt(path, model, optimizer, cfg, step, tag=""):
    """统一保存入口：模型 + 优化器动量 + 步数。

    必须存 optimizer.state_dict()，否则续训时 AdamW 的一阶/二阶动量从零重建，
    前一段训练积累的动量全部丢失，续训初期的 loss 会明显反弹。
    （--no_save_optimizer 可关掉，用于纯推理/交付场景，省约 3/4 空间。）
    """
    payload = {"step": step, "model": model.state_dict(),
               "cfg": cfg, "preset": cfg.preset, "tag": tag,
               "has_optimizer": bool(cfg.save_optimizer)}
    if cfg.save_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["optimizer_name"] = cfg.optimizer
    # 先写临时档再原子改名：存档途中被打断 / 磁盘写满，只会留下 .pt.tmp 残骸，
    # 绝不会出现「文件名正常、内容写了一半」的坏 checkpoint（那会让续训载入坏档）。
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# 训练现场登记：只存引用（零开销），供任何退出路径做兜底存档
_LIVE = {"active": False, "model": None, "optimizer": None, "cfg": None,
         "step": 0, "saved_step": 0, "ckpt_dir": None}


def emergency_save(reason=""):
    """兜底存档：把"已经算完但还没落盘"的步数存下来。

    与信号处理配合，覆盖所有非正常退出（Ctrl+C 置标志后由主循环正常收尾；
    异常 / 第二次 Ctrl+C 由这里 finally 兜底）。若当前步已落过盘则什么都不做。
    """
    if not _LIVE["active"]:
        return None
    if _LIVE["step"] <= _LIVE["saved_step"]:
        _LIVE["active"] = False
        return None
    cfg, model, opt = _LIVE["cfg"], _LIVE["model"], _LIVE["optimizer"]
    step = _LIVE["step"]
    ckpt_dir = Path(_LIVE["ckpt_dir"])
    target = ckpt_dir / f"pmnn_{cfg.preset}_step{step}.pt"
    tag = f"（{reason}）" if reason else ""
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_ckpt(target, model, opt, cfg, step, tag="emergency")
        removed = rotate_ckpts(ckpt_dir, cfg.preset, cfg.keep_ckpt, protect={target})
        print(f"\n🚨 检测到非正常退出{tag}，已紧急存档 -> {target.name}", flush=True)
        if removed:
            print(f"[train] checkpoint 轮转：删除旧档 {', '.join(removed)}", flush=True)
        print("✅ 紧急存档完毕（用 --resume 指向该文件即可续训）", flush=True)
        _LIVE["saved_step"] = step
    except Exception as e:          # 兜底存档自己失败，不能掩盖原始异常
        print(f"[train] 紧急存档失败: {type(e).__name__}: {e}", flush=True)
    finally:
        _LIVE["active"] = False
    return target


# ---------------------------------------------------------------------------
# 单实例锁：同一份 checkpoints/ 目录只允许一个训练进程写入
# ---------------------------------------------------------------------------
# 踩过的坑：反复 `nohup ... &` 启动、或旧的裸跑进程没清干净，就会有多个
# 训练进程同时往同一个 checkpoints/ 里写同名 stepN.pt —— 互相覆盖、白烧机时，
# 而且日志里完全看不出来。所以默认硬拒绝，要并行必须显式 --force_start。
def _pid_alive(pid: int) -> bool:
    """进程是否还活着。

    ⚠️ 不能用 os.kill(pid, 0) 判断 Windows 上的进程：Windows 版 os.kill
    会**真的把目标进程 TerminateProcess 掉**（已实测：调用后进程立刻消失）。
    所以 Windows 走 tasklist 查询，POSIX 才用 os.kill(pid, 0)。
    """
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout
            return str(pid) in out
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_run_lock(ckpt_dir, force: bool = False):
    """在 ckpt_dir 下登记 pmnn.lock，返回 (ok, lock_path, holder_desc)。

    ok=False 表示已有活着的训练进程占着这份存档目录。
    判定存活时尽量核对 /proc/<pid>/cmdline 是不是真的 train.py，
    以免 PID 复用造成假阳性（无 /proc 的平台退化为「PID 存活即占用」）。
    """
    ck = Path(ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    lock = ck / "pmnn.lock"
    if lock.exists():
        holder_pid, holder_argv = 0, ""
        try:
            lines = lock.read_text(encoding="utf-8", errors="replace").splitlines()
            holder_pid = int(lines[0].strip()) if lines else 0
            holder_argv = lines[1].strip() if len(lines) > 1 else ""
        except Exception:
            holder_pid = 0
        if holder_pid and holder_pid != os.getpid() and _pid_alive(holder_pid):
            genuine = True
            try:
                raw = Path(f"/proc/{holder_pid}/cmdline").read_bytes()
                genuine = "train.py" in raw.replace(b"\x00", b" ").decode(
                    "utf-8", "replace")
            except Exception:
                pass                       # 无 /proc：保守认定被占用
            if genuine and not force:
                return False, lock, f"PID={holder_pid}  {holder_argv or '(未记录命令行)'}"
    lock.write_text(f"{os.getpid()}\n{' '.join(sys.argv)}\n", encoding="utf-8")
    return True, lock, ""


def release_run_lock(lock_path):
    """只删自己写的那把锁，绝不误删别人的。"""
    try:
        p = Path(lock_path)
        if not p.exists():
            return
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines and lines[0].strip() == str(os.getpid()):
            p.unlink()
    except Exception:
        pass


# 当前进程持有的锁（供任何退出路径统一释放，包括异常/二次 Ctrl+C）
_RUN_LOCK = {"path": None}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        logits, _ = model(input_ids, mask, return_stats=True)
        tot += F.cross_entropy(logits.transpose(1, 2), labels,
                               ignore_index=-100).item()
        n += 1
        if n >= max_batches:
            break
    model.train()
    return tot / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="PMNN 文本模型训练")
    add_path_args(parser)
    add_config_args(parser)
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="构建词表时的大小（词表已存在则忽略）")
    parser.add_argument("--rebuild_tokenizer", action="store_true")
    parser.add_argument("--rebuild_prepared", action="store_true",
                        help="强制重新预分词")
    parser.add_argument("--tokenizer_files", type=int, default=10)
    args = parser.parse_args()

    cfg = config_from_args(args)
    paths = resolve_paths(args).ensure()

    print("=" * 72)
    print("[train] PMNN 物理矩阵神经网络 —— 文本版")
    # 把完整命令行写进日志：以后任何一次「到底续训没有」的争论，
    # 都只需要看这一行，不需要靠猜。
    print("[train] 命令行: " + " ".join(sys.argv))
    print(f"[train] 预设: {cfg.preset}  "
          f"(d_field={cfg.d_field}, blocks={cfg.n_blocks}, ffn={cfg.ffn_hidden})")
    print("[train] 路径：")
    print(paths.summary())
    # 把主进程 PID 写进日志：DataLoader 的子进程同名，`ps | grep train.py`
    # 数出来的个数天生多几个，认准这个 PID 才不会被吓一跳。
    print(f"[train] PID={os.getpid()}（DataLoader 的 {cfg.num_workers} 个 worker "
          "也会显示同名进程，属正常）")
    print("=" * 72)

    # ---- 单实例锁（dry_run 只测显存，不占锁）----
    lock_path = None
    if not cfg.dry_run:
        _ok, lock_path, _holder = acquire_run_lock(paths.ckpt_dir,
                                                   force=cfg.force_start)
        if not _ok:
            print("!" * 72)
            print("[train] 错误：已有训练进程正在使用同一份存档目录")
            print(f"[train]   存档目录: {paths.ckpt_dir}")
            print(f"[train]   占用者  : {_holder}")
            print(f"[train]   锁文件  : {lock_path}")
            print("[train] 多个进程同时写同一份存档会互相覆盖、白烧机时，已中止。")
            print("[train]   停掉旧的 -> pkill -f train.py   （或 kill 上面的 PID）")
            print("[train]   确要并行 -> 加 --force_start，或换一个 --out_dir")
            print("!" * 72)
            return 1
        print(f"[train] 单实例锁已获取: {lock_path}")
        _RUN_LOCK["path"] = lock_path

    torch.manual_seed(cfg.seed)
    device = pick_device()
    print(f"[train] 设备: {device}")

    # ---- 数据准备 ----
    files = paths.jsonl_files()
    meta_path = paths.prepared_dir / "meta.json"
    prepared_ready = meta_path.exists() and paths.tokenizer.exists()
    if not files and not prepared_ready:
        print(f"[train] 错误: {paths.data_dir} 下没有 *.jsonl")
        print("  用 --data_dir 或 PMNN_DATA_DIR 指定数据目录")
        print("  或提供已预分词产物 (prepared/) + 词表 (data/tokenizer.json) 后直接复用")
        return 1
    if not files:
        print("[train] 无原始 jsonl，但 prepared 产物与词表已就绪，直接复用预分词结果")
    tok = ensure_tokenizer(paths.tokenizer, files, args.vocab_size,
                           rebuild=args.rebuild_tokenizer,
                           max_files=args.tokenizer_files)

    need_prepare = args.rebuild_prepared or not meta_path.exists()
    if not need_prepare:
        meta = load_meta(paths.prepared_dir)
        if meta.get("max_len", 0) < cfg.max_seq_len:
            print(f"[train] 已预处理 max_len={meta['max_len']} < 需求 {cfg.max_seq_len}，重新预处理")
            need_prepare = True
    if need_prepare:
        print("[train] 开始预分词（流式落盘，内存恒定）...")
        prepare_tokens(tok, files, paths.prepared_dir, max_len=cfg.max_seq_len)
    meta = load_meta(paths.prepared_dir)

    cfg.vocab_size = int(meta["vocab_size"])
    cfg.pad_token_id = tok.token_to_id("<pad>")
    if cfg.pad_token_id is None:
        cfg.pad_token_id = 0
    print(f"[train] 样本 {meta['n_samples']} 条 / token {meta['total_tokens']} / "
          f"vocab {cfg.vocab_size} / pad {cfg.pad_token_id}")

    # ---- DataLoader（memmap，内存占用≈0）----
    ds = MemmapDataset(paths.prepared_dir, max_len=cfg.max_seq_len,
                       pad_id=cfg.pad_token_id, max_samples=cfg.max_samples)
    collate = partial(collate_fn, pad_id=cfg.pad_token_id,
                      max_len=cfg.max_seq_len, dynamic=cfg.dynamic_pad)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=cfg.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=True,
                        persistent_workers=cfg.num_workers > 0)
    print(f"[train] DataLoader: {len(ds)} 样本, {len(loader)} batch/epoch, "
          f"workers={cfg.num_workers}")

    # ---- 防呆：存档明明在，却没传 --resume（会静默从 0 重训，白烧机时）----
    # 这是目前踩过最贵的坑：脚本/命令行漏传 --resume 时日志毫无异常，
    # 直到几小时后才发现「一直在从头训」。所以宁可硬中止，也不许静默继续。
    if not cfg.resume and not cfg.dry_run:
        _have = find_ckpts(paths.ckpt_dir, cfg.preset)
        if _have and not cfg.fresh_ok:
            _last_step, _last_path = _have[-1]
            print("!" * 72)
            print(f"[train] 错误：{paths.ckpt_dir} 下已有 {len(_have)} 个存档，"
                  "但本次**没有**传 --resume")
            print(f"[train]   最新存档: {_last_path.name} (step={_last_step})")
            print("[train] 继续下去会从 step 0 重新训练：白烧几小时机时，"
                  "并且会在同名 stepN.pt 上覆盖旧存档")
            print("[train]   要续训 -> 加 --resume <存档路径>，"
                  "或用 RESUME=auto bash scripts/run_cloud_300m.sh")
            print("[train]   要重训 -> 加 --fresh_ok 显式确认")
            print("!" * 72)
            return 1

    # ---- 模型 ----
    model = PMNNModel(cfg).to(device)
    n_params, n_embed, n_head = param_report(model)
    print(f"[train] 参数量: {n_params/1e6:.2f}M "
          f"(embed {n_embed/1e6:.2f}M / lm_head {n_head/1e6:.2f}M)")

    resume_step = 0
    resume_opt = None
    if cfg.resume:
        # 存档不存在 -> 直接中止。绝不能「以为在续训，其实从零重训」：
        # 那会白烧几小时机时，而且日志里看不出任何异常。
        _rp = Path(cfg.resume).expanduser()
        if not _rp.exists():
            print("!" * 72)
            print(f"[train] 错误：--resume 指定的存档不存在 -> {_rp}")
            print("[train] 已中止，未开始训练。请检查路径（云端可用 RESUME=auto "
                  "让 scripts/run_cloud_300m.sh 自动挑最新档）")
            print("!" * 72)
            return 1
        ck = torch.load(str(_rp), map_location="cpu", weights_only=False)
        # strict=False：架构演进后旧 checkpoint 可能少/多几个 buffer，允许部分加载
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        resume_step = int(ck.get("step", 0) or 0)
        resume_opt = ck.get("optimizer")
        print(f"[train] 已加载续训权重: {_rp} (存档 step={resume_step})")
        n_state = len(model.state_dict())
        if missing:
            print(f"[train] !!! 警告：{len(missing)}/{n_state} 个权重**未**恢复，"
                  f"保持随机初始化（前 3 个: {list(missing)[:3]}）")
        else:
            print(f"[train] 权重键完全匹配（{n_state} 项全部命中）")
        if unexpected:
            print(f"[train] 存档多出 {len(unexpected)} 项（已跳过，前 3 个: "
                  f"{list(unexpected)[:3]}）")
        if resume_opt is None:
            print("[train] 该 checkpoint 未保存优化器状态 -> 动量从零开始"
                  "（lr 调度仍按绝对步数延续）")
        elif cfg.optimizer != ck.get("optimizer_name", cfg.optimizer):
            print(f"[train] 优化器类型变了({ck.get('optimizer_name')} -> "
                  f"{cfg.optimizer})，忽略已存优化器状态")
            resume_opt = None

    optimizer = build_optimizer(cfg, model)
    if resume_opt is not None:
        try:
            optimizer.load_state_dict(resume_opt)
            print("[train] 已恢复优化器动量状态")
        except (ValueError, KeyError) as e:
            print(f"[train] 优化器状态不兼容，已忽略: {e}")
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- 启动横幅 + 权重自检（「到底续训没有」必须一眼可辨）----
    if cfg.resume:
        print(f"[train] >>> 续训模式：global_step 从 {resume_step} 起"
              f"（本轮再跑 {max(0, cfg.max_steps - resume_step)} 步）")
    else:
        print("[train] >>> 全新训练：global_step 从 0 起（本次未使用 --resume）")

    if not cfg.dry_run:
        # 用固定 batch 做一次「仅前向」的自检（~1 秒）：
        #   全新训练 -> loss ≈ ln(vocab)（随机水平）
        #   续训成功 -> loss 应明显低于随机水平
        # 这是唯一能证明「权重真的进了模型」的硬证据。
        probe_loss = None
        try:
            probe = next(iter(loader))
            model.eval()
            with torch.no_grad():
                _pi = probe["input_ids"].to(device)
                _pl = probe["labels"].to(device)
                _pm = probe["attention_mask"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=amp_enabled):
                    _lg, _ = model(_pi, _pm)
                probe_loss = float(F.cross_entropy(
                    _lg.transpose(1, 2), _pl, ignore_index=-100).item())
            model.train()
            del probe, _pi, _pl, _pm, _lg
        except Exception as e:                  # 自检失败绝不能影响训练
            print(f"[train] 权重自检跳过（{type(e).__name__}: {e}）")
        if probe_loss is not None:
            rnd = math.log(max(2, cfg.vocab_size))
            print(f"[train] 起步自检：固定 batch loss={probe_loss:.4f} "
                  f"（随机初始化参考 ≈ ln(vocab) = {rnd:.2f}）")
            if resume_step > 0 and probe_loss > rnd:
                print("[train] " + "!" * 58)
                print("[train] !!! 严重警告：续训后 loss 仍高于随机水平，"
                      "权重很可能没有真正生效！")
                print("[train] !!! 请核对存档 preset/结构，并检查上方是否出现"
                      "「未恢复的权重」警告。")
                print("[train] " + "!" * 58)
            elif resume_step > 0 and probe_loss < rnd - 0.5:
                print(f"[train] >>> 自检通过：续训权重已生效"
                      f"（loss {probe_loss:.3f} 远低于随机 {rnd:.2f}）")
            elif resume_step > 0:
                print(f"[train] 注：loss {probe_loss:.3f} 仍在随机水平 {rnd:.2f} 附近。"
                      f"权重确实加载了，但「存档本身步数很少 / 尚未收敛」时就是这个样子；"
                      f"若你预期的是一个已收敛的存档，请重点核查上面的加载信息。")

    # ---- dry run：完整跑一步（含优化器 step，才能测到 AdamW 惰性分配的动量状态）----
    if cfg.dry_run:
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            logits, diag = model(input_ids, mask, return_stats=True)
            loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
        # 前向 → 反向 → 优化器一步（优化器状态在此刻才真正分配）
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[dry_run] batch={cfg.batch_size} seq={cfg.max_seq_len} "
                  f"params={n_params/1e6:.2f}M optimizer={cfg.optimizer}")
            print(f"[dry_run] 峰值显存 = {peak:.3f} GB / 总 {total:.2f} GB "
                  f"({peak/total*100:.0f}%)")
            if peak > total * 0.9:
                print("[dry_run] 警告：显存占用 >90%，建议减小 --batch 或改用 "
                      "--optimizer adafactor")
        print(f"[dry_run] loss={loss.item():.4f} 前向+反向+优化器一步成功")
        return 0

    # ---- 训练 ----
    # 按 preset 分文件，避免不同规模的训练互相覆盖日志。
    # 续训时必须 **追加** 而不是覆盖，否则前一段（可能几小时）的曲线全丢。
    log_file = paths.log_dir / f"train_log_{cfg.preset}.tsv"
    header = ("step\tloss\tloss_ce\tloss_fm\tloss_sync\tloss_drift\tloss_free\t"
              "r_mean\tentropy\tenergy\tlr\telapsed\n")
    elapsed_offset = 0.0
    if resume_step > 0 and log_file.exists():
        elapsed_offset = _last_elapsed(log_file)
        print(f"[train] 续训：日志追加到 {log_file} (已有时长 {elapsed_offset:.0f}s)")
    else:
        log_file.write_text(header, encoding="utf-8")

    global_step = resume_step
    accum = 0
    t0 = time.time()
    # ---- wall-clock 软退出 ----
    # 云端平台按「任务启动即计时」限时（常见 10 小时），到点直接 SIGKILL。
    # 与其被强杀丢掉最后一个存档间隔，不如自己到点前主动存盘、正常退出：
    # 下一次任务 --resume 指向新档即可无缝接续，机时零浪费。
    # 基准是**本进程的 t0**（不是日志累计 elapsed）—— 平台限的是「本次任务」，
    # 不是「这个模型累计训练」。
    _wall_hours = float(getattr(cfg, "max_hours", 0.0) or 0.0)
    _wall_margin = float(getattr(cfg, "wallclock_margin_min", 20.0) or 0.0) * 60.0
    wall_deadline = (t0 + _wall_hours * 3600.0 - _wall_margin) if _wall_hours > 0 else 0.0
    if wall_deadline:
        print(f"[train] wall-clock 软退出已开启：本次进程最长 {_wall_hours:.2f}h，"
              f"将于 {max(_wall_hours * 3600.0 - _wall_margin, 0) / 3600.0:.2f}h "
              f"处主动存盘退出（留 {_wall_margin/60:.0f} 分钟收尾）", flush=True)
    else:
        print("[train] wall-clock 软退出未开启（--max_hours 0）；"
              "若平台按任务限时，建议设为 9.0", flush=True)
    r_mean = ent = en = torch.tensor(0.0)
    loss = loss_ce = loss_fm = loss_sync = loss_free = loss_drift = torch.tensor(0.0)
    optimizer.zero_grad(set_to_none=True)
    print(f"[train] 开始训练：max_steps={cfg.max_steps}, "
          f"有效 batch={cfg.batch_size}×{cfg.grad_accum}, amp={amp_enabled}")

    # ---- 中断保护：注册信号处理器 + 登记训练现场（零开销，只存引用）----
    install_interrupt_handlers()
    _LIVE.update(active=True, model=model, optimizer=optimizer, cfg=cfg,
                 step=global_step, saved_step=resume_step,
                 ckpt_dir=paths.ckpt_dir)

    stop = False
    wall_clock_stopped = False
    # ---- 收敛早停：CE 长时间不再刷新最优就收工，不必训满 max_steps ----
    # 用「滑窗均值」而非单点判断，避免 CE 在 1~3 之间跳动造成误判。
    es_patience = max(0, int(getattr(cfg, "early_stop_patience", 0) or 0))
    es_window = max(1, int(getattr(cfg, "early_stop_window", 5) or 5))
    es_delta = float(getattr(cfg, "early_stop_min_delta", 0.01) or 0.0)
    ce_win = deque(maxlen=es_window)
    best_ce = float("inf")
    best_ce_step = 0
    last_improve_step = resume_step
    early_stopped = False
    if es_patience > 0:
        print(f"[train] 收敛早停已开启：CE 滑窗 {es_window} 个记录点 "
              f"(≈{es_window * cfg.log_every} 步) 连续 {es_patience} 步未改善 "
              f"(改善阈值 {es_delta:g}) 即自动停止", flush=True)

    while not stop and global_step < cfg.max_steps:
        for batch in loader:
            if global_step >= cfg.max_steps:
                stop = True
                break
            # wall-clock 到点：同样放在「上一步完整结束之后」触发，
            # 绝不在梯度/优化器中途打断，保证落盘状态自洽。
            if wall_deadline and time.time() >= wall_deadline:
                _ran = (time.time() - t0) / 3600.0
                print(f"[train] 触发 wall-clock 软退出：本次已运行 {_ran:.2f}h"
                      f"（上限 {_wall_hours:.2f}h），主动存盘退出；"
                      f"下轮用 --resume 指向新档即可无缝接续", flush=True)
                wall_clock_stopped = True
                stop = True
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            B = input_ids.shape[0]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                logits, diag = model(input_ids, mask, return_stats=True)
                a, theta = diag["a"], diag["theta"]
                final_stat = diag["final_stat"]

                # L_ce：语言建模
                loss_ce = F.cross_entropy(
                    logits.transpose(1, 2), labels, ignore_index=-100,
                    label_smoothing=cfg.label_smoothing)

                # L_sync：序参量课程跟踪（mask 加权）
                r, _, _ = order_parameter(a, theta, dim=1, mask=mask)
                if cfg.sync_curriculum and diag.get("stats_all") is not None:
                    # 逐层跟踪（设计稿 §7 的 r(t) 课程）：
                    #   早期层目标 r 低（混沌探索）→ 晚期层目标 r 高（同步整合）。
                    #   只约束末层会把整条课程曲线压成一个常数，等于丢掉 K 值导航。
                    nb = max(cfg.n_blocks, 1)
                    ls = torch.zeros((), device=device)
                    for i, st in enumerate(diag["stats_all"]):
                        ti = i / max(nb - 1, 1)
                        rt = torch.sigmoid(
                            torch.tensor((ti - cfg.t_c) / cfg.tau_c, device=device))
                        ls = ls + F.mse_loss(st["order_raw"],
                                             rt.expand_as(st["order_raw"]))
                    loss_sync = ls / nb
                else:
                    # 旧行为：只约束末层，目标是常数 σ((1−t_c)/τ_c) = 0.9656
                    r_target = torch.sigmoid(
                        torch.tensor((1.0 - cfg.t_c) / cfg.tau_c, device=device))
                    loss_sync = F.mse_loss(r, r_target.expand_as(r))

                # L_drift：漂移场一致性  MSE(φ(x), stopgrad(φ(x) + V))
                #   让漂移头预测物理核心算出的漂移方向 V = V⁺ − λV⁻
                if diag.get("drift_pred") is not None and diag.get("drift") is not None:
                    loss_drift = masked_mse(diag["drift_pred"],
                                            diag["drift"].detach(), mask)
                else:
                    loss_drift = torch.zeros((), device=device)

                # L_F：自由能正则（能量下降 / 熵平衡）
                #   F = U − T·S 本身已含「降能量 + 升熵」的平衡；
                #   额外只加**单边 hinge**：S < 目标时惩罚（防全同步坍缩）。
                #   注意：不能写成 −w·(S−target)，那会奖励熵无限增长，
                #   最终把振幅分布推成均匀分布 = 结构性坍缩。
                energy = final_stat["energy"].mean()
                entropy = final_stat["entropy"].mean()
                free = energy - cfg.T_min * entropy
                loss_free = free + cfg.w_entropy_floor * F.relu(
                    cfg.entropy_target - entropy)

                # L_fm：ELF 流匹配（去噪分支，mask 加权）
                with torch.no_grad():
                    z = diag["z"]
                    d = z.shape[-1] // 2
                    x1 = z[..., :d]
                x0 = torch.randn_like(x1)
                tr = torch.rand(B, device=device)
                x_t = tr.view(-1, 1, 1) * x1 + (1 - tr).view(-1, 1, 1) * x0
                v_pred = model.flow_matching(x_t, tr, mask)
                v_target = model.velocity_target(x0, x1)
                loss_fm = masked_mse(v_pred, v_target, mask)

                # mode 门控（设计稿 §6）：按热力学状态动态加权两条分支。
                #   g 大 → 解码分支（L_ce）主导；g 小 → 去噪分支（L_fm）主导。
                #   clamp 防止门控跑到极端、把另一条分支彻底掐死（梯度消失）。
                if diag.get("mode_gate") is not None:
                    g = diag["mode_gate"].mean().clamp(0.05, 0.95)
                    w_ce, w_fm = g, (1.0 - g)
                else:
                    w_ce, w_fm = 1.0, cfg.w_fm

                loss = (w_ce * loss_ce
                        + w_fm * loss_fm
                        + cfg.w_sync * loss_sync
                        + cfg.w_drift * loss_drift
                        + cfg.w_free * loss_free)

            scaler.scale(loss / cfg.grad_accum).backward()
            accum += 1
            if accum < cfg.grad_accum:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum = 0

            lr = lr_at(global_step, cfg.learning_rate, cfg.warmup_steps, cfg.max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            global_step += 1
            _LIVE["step"] = global_step      # 供兜底存档使用

            r_mean = r.mean().detach()
            ent = entropy.detach()
            en = energy.detach()

            if global_step % cfg.log_every == 0 or global_step == 1:
                elapsed = time.time() - t0 + elapsed_offset
                sps = (global_step - resume_step) / max(time.time() - t0, 1e-6)
                gate_txt = ""
                if diag.get("mode_gate") is not None:
                    gate_txt = f" g={w_ce.item():.3f}"
                line = (f"[step {global_step}/{cfg.max_steps}] "
                        f"loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                        f"fm={loss_fm.item():.4f} sync={loss_sync.item():.4f} "
                        f"drift={loss_drift.item():.4f} "
                        f"free={loss_free.item():.4f} | "
                        f"r={r_mean.item():.3f} Ent={ent.item():.3f} "
                        f"E={en.item():.3f}{gate_txt} lr={lr:.2e} ({sps:.2f} step/s)")
                print(line, flush=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{global_step}\t{loss.item():.5f}\t{loss_ce.item():.5f}"
                            f"\t{loss_fm.item():.5f}\t{loss_sync.item():.5f}"
                            f"\t{loss_drift.item():.5f}"
                            f"\t{loss_free.item():.5f}\t{r_mean.item():.5f}"
                            f"\t{ent.item():.5f}\t{en.item():.5f}\t{lr:.3e}"
                            f"\t{elapsed:.1f}\n")

                # ---- 收敛检测：CE 滑窗均值长时间不刷新最优 => 早停 ----
                if es_patience > 0:
                    ce_win.append(loss_ce.item())
                    if len(ce_win) >= es_window:
                        smooth = sum(ce_win) / len(ce_win)
                        if smooth < best_ce - es_delta:
                            best_ce, best_ce_step = smooth, global_step
                            last_improve_step = global_step
                        elif global_step - last_improve_step >= es_patience:
                            print(f"[train] 触发收敛早停：CE 滑窗均值 {smooth:.4f}，"
                                  f"历史最优 {best_ce:.4f} @ step {best_ce_step}，"
                                  f"已连续 {global_step - last_improve_step} 步未改善 "
                                  f"(阈值 {es_delta:g})", flush=True)
                            early_stopped = True
                            stop = True
                            break

            if cfg.eval_every and global_step % cfg.eval_every == 0:
                ev = evaluate(model, loader, device)
                print(f"[eval @ {global_step}] val_ce={ev:.4f}", flush=True)

            if cfg.save_every and global_step % cfg.save_every == 0:
                ckpt = paths.ckpt_dir / f"pmnn_{cfg.preset}_step{global_step}.pt"
                paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_ckpt(ckpt, model, optimizer, cfg, global_step)
                _LIVE["saved_step"] = global_step
                removed = rotate_ckpts(paths.ckpt_dir, cfg.preset, cfg.keep_ckpt)
                print(f"[train] 保存 checkpoint: {ckpt.name}"
                      + (f"（轮转删除 {len(removed)} 个旧档，"
                         f"只留最新 {cfg.keep_ckpt} 个）" if removed else ""))

            # 中断检查放在**一步完整结束之后**：绝不在梯度/优化器中途打断，
            # 保证存下来的状态是自洽的（不会出现半更新的权重）。
            if _INTERRUPT["signum"] is not None:
                stop = True
                break

    # ---- 收尾：正常完成 / 收敛早停 / 中断 / 异常，四条路径都必须落盘 ----
    interrupted = _INTERRUPT["signum"] is not None
    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = paths.ckpt_dir / f"pmnn_{cfg.preset}_step{global_step}.pt"
    if interrupted:
        _tag = "interrupted"
    elif wall_clock_stopped:
        _tag = "wallclock"
    elif early_stopped:
        _tag = "earlystop"
    else:
        _tag = "final"
    save_ckpt(ckpt, model, optimizer, cfg, global_step, tag=_tag)
    _LIVE["saved_step"] = global_step
    _LIVE["active"] = False                 # 已落盘，finally 里无需再兜底
    removed = rotate_ckpts(paths.ckpt_dir, cfg.preset, cfg.keep_ckpt, protect={ckpt})
    if interrupted:
        print(f"✅ 紧急存档完毕 -> {ckpt.name} (step {global_step})，"
              f"用 --resume 指向它即可续训")
    elif wall_clock_stopped:
        print(f"⏱️ wall-clock 软退出存档完毕 -> {ckpt.name} (step {global_step})，"
              f"本次机时已用满；下轮用 --resume 指向它即可无缝接续")
    elif early_stopped:
        print(f"[train] 收敛早停 -> {ckpt.name} (step {global_step})；"
              f"CE 最优 {best_ce:.4f} @ step {best_ce_step}"
              f"（未训满 {cfg.max_steps} 步，省下 {cfg.max_steps - global_step} 步）")
    else:
        print(f"[train] 训练完成，最终 checkpoint: {ckpt.name}")
    if removed:
        print(f"[train] checkpoint 轮转：删除旧档 {', '.join(removed)}"
              f"（保留最新 {cfg.keep_ckpt} 个）")
    return 0


if __name__ == "__main__":
    _code = 0
    try:
        _code = main()
    finally:
        # 终极兜底：任何未捕获的退出路径（异常、二次 Ctrl+C、外部 kill）都在这里
        # 补一次存档。若主流程已正常落盘，_LIVE["active"]=False，本函数直接返回。
        emergency_save("非正常退出")
        # 存档写完之后再放锁（顺序不能反，否则别人可能在我们写盘时启动）
        release_run_lock(_RUN_LOCK["path"])
    raise SystemExit(_code)
