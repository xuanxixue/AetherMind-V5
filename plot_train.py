"""绘制训练曲线与物理诊断（需 matplotlib）。

用法（路径参数化）：
    python plot_train.py
    python plot_train.py --log logs/train_log.tsv --out logs/train_curve.png

读取 train.py 输出的 logs/train_log.tsv（按表头列名解析，列顺序变化不会错位）。
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows 优先雅黑/黑体，Linux 回退思源/文泉驿）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_path_args, resolve_paths


def load_log(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    data = {}
    for k in rows[0].keys():
        vals = []
        for r in rows:
            v = r.get(k)
            if v in (None, ""):
                vals.append(float("nan"))
                continue
            vals.append(int(v) if k == "step" else float(v))
        data[k] = vals
    return data


def main():
    p = argparse.ArgumentParser(description="绘制 PMNN 训练曲线")
    add_path_args(p)
    p.add_argument("--log", default=None, help="日志 TSV（默认 <log_dir>/train_log.tsv）")
    p.add_argument("--out", default=None, help="输出 PNG（默认 <log_dir>/train_curve.png）")
    args = p.parse_args()

    paths = resolve_paths(args)
    if args.log:
        log_path = Path(args.log)
    else:
        # 默认取日志目录里最新的 train_log*.tsv（每次训练按 preset 分文件，不互相覆盖）
        cands = sorted(paths.log_dir.glob("train_log*.tsv"),
                       key=lambda p: p.stat().st_mtime)
        log_path = cands[-1] if cands else paths.log_dir / "train_log.tsv"
    out_path = Path(args.out) if args.out else paths.log_dir / "train_curve.png"

    if not log_path.exists():
        print(f"[plot] 日志不存在: {log_path}")
        return 1
    d = load_log(log_path)
    n = len(d.get("step", []))
    if n == 0:
        print("[plot] 日志为空")
        return 1
    s = d["step"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def panel(ax, keys, labels, colors, title, ylabel):
        for k, lb, c in zip(keys, labels, colors):
            if k in d:
                ax.plot(s, d[k], label=lb, color=c)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    panel(axes[0][0], ["loss", "loss_ce"],
          ["总损失", "语言建模 CE"], ["#2b6a8f", "#b33a3a"],
          "损失", "loss")

    panel(axes[0][1], ["loss_fm", "loss_sync"],
          ["ELF 流匹配", "序参量课程"], ["#c26b00", "#1e7a3a"],
          "物理辅助损失", "loss")

    panel(axes[1][0], ["r_mean"],
          ["序参量 r"], ["#1e7a3a"],
          "序参量（同步度 0→1）", "r")
    if "r_mean" in d:
        axes[1][0].set_ylim(0, 1)

    ax = axes[1][1]
    if "entropy" in d:
        ax.plot(s, d["entropy"], label="振幅熵 S", color="#9a6b00")
    if "energy" in d:
        ax2 = ax.twinx()
        ax2.plot(s, d["energy"], label="XY 能量 U", color="#6a4c93")
        ax2.set_ylabel("U", color="#6a4c93")
    ax.set_xlabel("step")
    ax.set_ylabel("S", color="#9a6b00")
    ax.set_title("热力学判据（熵 / 能量）")
    ax.grid(alpha=0.3)

    fig.suptitle("PMNN 物理矩阵神经网络 训练曲线", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"[plot] 已保存: {out_path} (共 {n} 步)")

    # 文字摘要
    print("[plot] 摘要：")
    for k in ("loss", "loss_ce", "r_mean", "entropy", "energy"):
        if k in d and len(d[k]) > 0:
            print(f"    {k:>9}: {d[k][0]:.4f} -> {d[k][-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
