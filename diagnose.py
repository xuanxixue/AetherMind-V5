"""PMNN 诊断脚本：按功能模块做梯度审计 + 物理量体检。

用来回答两个问题：
  1. 哪条通路是「死的」？—— 反向传播后逐模块看梯度范数。
     某模块梯度 ~1e-7 或为 None，说明它没被训练到（上游 detach / 饱和 / 被忽略）。
  2. 物理量是否健康？—— 序参量 r 是否卡在 0 或 1（梯度死区）、
     振幅是否非负、熵是否顶到 ln(D) 上限。

用法：
    python diagnose.py --preset small
    python diagnose.py --preset 100m --batch 4 --seq 128
    python diagnose.py --preset small --no_causal        # 对比因果通路
"""
import argparse
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import add_path_args, resolve_paths
from config import add_config_args, config_from_args
from data_utils import MemmapDataset, collate_fn, load_tokenizer
from pmnn.model import PMNNModel
from pmnn.physics import order_parameter
from torch.utils.data import DataLoader

DEAD_THRESHOLD = 1e-6


GROUP_PATTERNS = OrderedDict([
    ("① 词嵌入",            r"^token_embed\."),
    ("① 振子化投影",        r"^(amp_proj|phase_proj|embed_norm)\."),
    ("② 时间场",            r"^time_field\."),
    # 内容寻址耦合必须排在「相位同步」之前：按顺序归组 + 去重，
    # 否则 (cq|ck|cv) 会被笼统的 sync 组先吃掉，看不出它到底学没学。
    # 注意结尾的 ($|\.) ：cq/ck/cv 是子模块（名字带 .weight），
    # 而 c_scale 是裸 Parameter（名字就到此为止）。只写 $ 会漏掉前者。
    ("② 物理块-内容耦合",    r"^blocks\.\d+\.sync\.(cq|ck|cv|c_scale)($|\.)"),
    ("② 物理块-相位同步",    r"^blocks\.\d+\.sync\."),
    ("② 物理块-共振驻波",    r"^blocks\.\d+\.ffn\."),
    ("② 物理块-漂移场",      r"^blocks\.\d+\.drift\."),
    ("② 物理块-振幅归一",    r"^blocks\.\d+\.norm_a\."),
    ("③ 语言建模头",        r"^lm_head\."),
    ("③ 流匹配头",          r"^(vel_head|bottleneck)\."),
    ("③ 漂移一致性头",      r"^drift_head\."),
    ("③ mode 门控",         r"^mode_gate\."),
])


def grad_groups(model):
    """按功能模块归组参数（正则匹配，先匹配者优先），返回 {组名: [参数名...]}。

    去重是必要的：`blocks.N.sync.cq` 同时匹配「内容耦合」与「相位同步」两个模式。
    不去重会导致同一参数被两组重复统计，看梯度表时误以为某个模块参数很多。
    """
    import re
    out = OrderedDict()
    named = list(model.named_parameters())
    seen = set()
    for g, pat in GROUP_PATTERNS.items():
        rx = re.compile(pat)
        sel = [n for n, _ in named if rx.search(n) and n not in seen]
        seen.update(sel)
        out[g] = sel
    return out


def main():
    p = argparse.ArgumentParser(description="PMNN 诊断")
    add_path_args(p)
    add_config_args(p)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # --batch / --max_seq_len 由 add_config_args 提供（dest 为 batch_size / max_seq_len）
    cfg = config_from_args(args)
    if cfg.batch_size > 16:
        cfg.batch_size = 8
    paths = resolve_paths(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(cfg.seed)

    # ---- 数据 ----
    if not (paths.prepared_dir / "meta.json").exists():
        print(f"[diag] 未找到预处理产物 {paths.prepared_dir}，先跑 prepare_data.py")
        return 1
    tok = load_tokenizer(paths.tokenizer)
    pad_id = tok.token_to_id("<pad>") or 0
    ds = MemmapDataset(paths.prepared_dir, max_len=cfg.max_seq_len,
                       pad_id=pad_id, max_samples=256)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=partial(collate_fn, pad_id=pad_id,
                                           max_len=cfg.max_seq_len,
                                           dynamic=cfg.dynamic_pad),
                        drop_last=True)
    batch = next(iter(loader))

    # ---- 模型 ----
    cfg.vocab_size = int(ds.meta["vocab_size"])
    cfg.pad_token_id = pad_id
    model = PMNNModel(cfg).to(device)
    n = sum(q.numel() for q in model.parameters())
    print("=" * 74)
    print(f"[diag] 预设={cfg.preset} 参数量={n/1e6:.2f}M 设备={device}")
    print(f"[diag] 跨位置耦合={cfg.coupling}  "
          f"因果耦合={'ON' if cfg.causal_coupling else 'OFF'} "
          f"(halflife={cfg.causal_halflife}, mix={cfg.causal_mix})")
    if cfg.coupling == "content":
        print(f"[diag] content 耦合: halflife={cfg.coupling_halflife}, "
              f"temp={cfg.coupling_temp}, rank={cfg.r_rank}")
    print(f"[diag] 朗之万={'ON' if cfg.langevin else 'OFF'}  "
          f"mode门控={'ON' if cfg.mode_gate else 'OFF'}  "
          f"轨迹瓶颈={cfg.bottleneck_dim}维")
    print("=" * 74)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["attention_mask"].to(device)

    # ---- 前向 ----
    logits, diag = model(input_ids, mask, return_stats=True)
    loss_ce = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)

    # ---- 物理辅助项（与 train.py 完全同口径，否则梯度审计会误判为"死"）----
    B = input_ids.shape[0]
    r, _, _ = order_parameter(diag["a"], diag["theta"], dim=1, mask=mask)
    st = diag["final_stat"]
    energy = st["energy"].mean()
    entropy = st["entropy"].mean()
    mm = mask.unsqueeze(-1).float()

    # L_sync 口径必须与 train.py 完全一致，否则梯度审计对 sync 通路的判断会失真
    if cfg.sync_curriculum and diag.get("stats_all") is not None:
        nb = max(cfg.n_blocks, 1)
        ls = torch.zeros((), device=device)
        for i, st in enumerate(diag["stats_all"]):
            ti = i / max(nb - 1, 1)
            rt = torch.sigmoid(torch.tensor((ti - cfg.t_c) / cfg.tau_c, device=device))
            ls = ls + F.mse_loss(st["order_raw"], rt.expand_as(st["order_raw"]))
        loss_sync = ls / nb
    else:
        r_target = torch.sigmoid(torch.tensor((1.0 - cfg.t_c) / cfg.tau_c, device=device))
        loss_sync = F.mse_loss(r, r_target.expand_as(r))

    z = diag["z"]
    d = z.shape[-1] // 2
    with torch.no_grad():
        x1 = z[..., :d]
    x0 = torch.randn_like(x1)
    tr = torch.rand(B, device=device)
    x_t = tr.view(-1, 1, 1) * x1 + (1 - tr).view(-1, 1, 1) * x0
    v_pred = model.flow_matching(x_t, tr, mask)
    v_tgt = model.velocity_target(x0, x1)
    loss_fm = (((v_pred - v_tgt) ** 2) * mm).sum() / (mm.sum() * v_pred.shape[-1])

    loss_drift = torch.zeros((), device=device)
    if diag.get("drift") is not None and diag.get("drift_pred") is not None:
        loss_drift = (((diag["drift_pred"] - diag["drift"].detach()) ** 2)
                      * mm).sum() / (mm.sum() * z.shape[-1])

    loss_free = (energy - cfg.T_min * entropy) + cfg.w_entropy_floor * F.relu(
        cfg.entropy_target - entropy)

    # mode 门控分支权重（与 train.py 完全同口径，否则梯度审计会误判为"死"）
    if diag.get("mode_gate") is not None:
        g = diag["mode_gate"].mean().clamp(0.05, 0.95)
        w_ce, w_fm = g, (1.0 - g)
    else:
        w_ce, w_fm = 1.0, cfg.w_fm

    loss = (w_ce * loss_ce + w_fm * loss_fm + cfg.w_sync * loss_sync
            + cfg.w_drift * loss_drift + cfg.w_free * loss_free)
    loss.backward()

    # ---- 梯度审计 ----
    print("\n[梯度审计]（阈值 1e-6 以下视为该通路「死」）")
    print(f"  {'模块':<22} {'参数量':>10} {'梯度范数':>12}  状态")
    print("  " + "-" * 60)
    for g, names in grad_groups(model).items():
        ps = [dict(model.named_parameters())[nm] for nm in names]
        if not ps:
            continue
        cnt = sum(q.numel() for q in ps)
        sq = [float(q.grad.norm()) ** 2 for q in ps if q.grad is not None]
        gn = sum(sq) ** 0.5 if sq else 0.0
        state = "OK"
        if gn < DEAD_THRESHOLD:
            state = "<== 无梯度/近乎为 0"
        print(f"  {g:<22} {cnt:>10,} {gn:>12.3e}  {state}")

    # ---- 物理体检 ----
    d = cfg.d_field
    print("\n[物理体检]")
    print(f"  序参量 r        = {r.mean().item():.4f}   (∈[0,1]；贴 0 或 1 都进梯度死区)")
    print(f"  振幅 a          = min {diag['a'].min().item():.4f} / "
          f"max {diag['a'].max().item():.4f}   (min 必须 ≥ 0)")
    S = entropy.item()
    print(f"  振幅熵 S        = {S:.4f}   (上限 ln(D)={torch.log(torch.tensor(float(d))).item():.4f}；"
          f"占比 {S / float(torch.log(torch.tensor(float(d)))) * 100:.1f}%)")
    print(f"  XY 能量 U       = {energy.item():.4f}")
    print(f"  自由能 F        = {st['free_energy'].mean().item():.4f}")
    print(f"  K 值 κ(t)       = {st['kappa'].mean().item():.4f} / 上限 {cfg.kappa_max}")
    print(f"  温度 T          = {st['temp'].mean().item():.4f}")

    # ---- 维度完整性（有无静默截断）----
    z = diag["z"]
    nz = (z.detach().abs().mean(-1) > 1e-6).float().mean().item()
    print(f"  z 非零维占比    = {nz:.3f}   (应接近 1.0，远小于 1 说明有维度被静默置零)")

    print(f"\n  CE = {loss_ce.item():.4f}   总 loss = {loss.item():.4f}")
    if device.type == "cuda":
        print(f"  峰值显存 = {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
