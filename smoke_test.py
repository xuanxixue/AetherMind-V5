"""PMNN 冒烟测试：不读数据、不写文件，只验证模型前向/反向/生成/流匹配能跑通。

用法：
    python smoke_test.py
    python smoke_test.py --preset small --seq 128 --batch 4
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config, PRESETS
from pmnn.model import PMNNModel
from pmnn.physics import order_parameter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="tiny", choices=list(PRESETS))
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(0)

    cfg = get_config(preset=args.preset, vocab_size=512, max_seq_len=args.seq,
                     pad_token_id=0)
    model = PMNNModel(cfg).to(device)
    n = sum(p_.numel() for p_ in model.parameters())
    print(f"[smoke] 预设={args.preset} 参数量={n/1e6:.2f}M 设备={device}")

    ids = torch.randint(1, 500, (args.batch, args.seq), device=device)
    ids[:, :5] = 0  # 模拟左 padding
    mask = (ids != 0).float()

    # 1) 前向
    t0 = time.time()
    logits, diag = model(ids, mask, return_stats=True)
    print(f"[smoke] 前向 OK: logits={tuple(logits.shape)} "
          f"z={tuple(diag['z'].shape)} ({time.time()-t0:.2f}s)")

    # 2) 物理诊断（含不变量断言）
    st = diag["final_stat"]
    r, _, _ = order_parameter(diag["a"], diag["theta"], dim=1, mask=mask)
    print(f"[smoke] 物理: r={r.mean().item():.3f} "
          f"E={st['energy'].mean().item():.3f} "
          f"S={st['entropy'].mean().item():.3f} "
          f"F={st['free_energy'].mean().item():.3f} "
          f"kappa={st['kappa'].mean().item():.2f}")
    assert torch.isfinite(r).all(), "序参量出现 NaN/Inf"
    assert r.mean().item() <= 1.0 + 1e-4, "序参量越界 >1"
    assert diag["a"].min().item() >= 0, "振幅出现负值（物理不合法）"

    # 3) 反向
    t0 = time.time()
    loss = F.cross_entropy(logits.transpose(1, 2), ids, ignore_index=0)
    loss.backward()
    dt = time.time() - t0
    gnorm = sum((p_.grad ** 2).sum().item() for p_ in model.parameters()
                if p_.grad is not None) ** 0.5
    print(f"[smoke] 反向 OK: loss={loss.item():.3f} |grad|={gnorm:.2f} ({dt:.2f}s)")

    # 4) 流匹配分支
    with torch.no_grad():
        d = diag["z"].shape[-1] // 2
        x1 = diag["z"][..., :d]
    x0 = torch.randn_like(x1)
    tr = torch.rand(args.batch, device=device)
    x_t = tr.view(-1, 1, 1) * x1 + (1 - tr).view(-1, 1, 1) * x0
    v_pred = model.flow_matching(x_t, tr, mask)
    v_tgt = model.velocity_target(x0, x1)
    print(f"[smoke] 流匹配 OK: v_pred={tuple(v_pred.shape)} "
          f"mse={F.mse_loss(v_pred, v_tgt).item():.4f}")

    # 5) 生成
    t0 = time.time()
    gen = model.generate(torch.randint(1, 500, (1, 8), device=device),
                         max_new_tokens=12, temperature=0.8, eos_id=3)
    print(f"[smoke] 生成 OK: {tuple(gen.shape)} ({time.time()-t0:.2f}s)")

    if device.type == "cuda":
        print(f"[smoke] 峰值显存 = {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")
    print("[smoke] 全部通过 OK")


if __name__ == "__main__":
    main()
