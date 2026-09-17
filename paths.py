"""PMNN 统一路径解析层。

优先级（从高到低）：
    1. 命令行参数（--data_dir / --out_dir / ...）
    2. 环境变量（PMNN_DATA_DIR / PMNN_OUT_DIR / ...）
    3. 自动探测默认值（从工程根向上查找 datasets / data 目录）

设计目标：**云端零改代码复用**。本地怎么跑，云端设几个环境变量就能跑。

本地（Windows）:
    python train.py --preset small --steps 3000

云端（Linux）:
    export PMNN_DATA_DIR=/data/03_dialogue_clean
    export PMNN_OUT_DIR=/workspace/pmnn_out
    export PMNN_TOKENIZER=/workspace/pmnn_out/data/tokenizer.json
    python train.py --preset 300m --steps 300000

生成产物目录结构（全部可覆盖）:
    <out_dir>/
    ├── data/tokenizer.json     词表
    ├── prepared/               预分词产物 tokens.bin / index.npy / meta.json
    ├── checkpoints/            *.pt
    └── logs/                   train_log.tsv
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from pathlib import Path

# ---- 环境变量名（统一 PMNN_ 前缀）----
ENV_ROOT = "PMNN_ROOT"
ENV_DATA_DIR = "PMNN_DATA_DIR"
ENV_OUT_DIR = "PMNN_OUT_DIR"
ENV_TOKENIZER = "PMNN_TOKENIZER"
ENV_PREPARED_DIR = "PMNN_PREPARED_DIR"
ENV_CKPT_DIR = "PMNN_CKPT_DIR"
ENV_LOG_DIR = "PMNN_LOG_DIR"

# 自动探测时优先尝试的子目录名（按优先级）
_DATA_CANDIDATES = ("datasets/03_dialogue_clean", "datasets", "data", "corpus")
_MAX_ASCEND = 8  # 向上最多查找层数


def default_root() -> Path:
    """工程根目录 = paths.py 所在目录（pmnn_text/）。"""
    return Path(__file__).resolve().parent


def _pick(cli_value, env_name: str, default):
    """三级取值：命令行 > 环境变量 > 默认。空字符串视为未提供。"""
    if cli_value not in (None, ""):
        return cli_value
    env_val = os.environ.get(env_name)
    if env_val:
        return env_val
    return default


def _has_jsonl(p: Path) -> bool:
    try:
        if not p.is_dir():
            return False
        return next(p.glob("*.jsonl"), None) is not None
    except OSError:
        return False


def autodetect_data_dir(root: Path) -> Path:
    """从工程根向上逐层查找含 *.jsonl 的数据目录。

    默认工程结构下会命中 <root>/../../datasets/03_dialogue_clean。
    """
    cur = root
    for _ in range(_MAX_ASCEND):
        for name in _DATA_CANDIDATES:
            p = cur / name
            if _has_jsonl(p):
                return p
        if cur.parent == cur:  # 到达盘符根
            break
        cur = cur.parent
    return root / "data"


@dataclass
class Paths:
    root: Path          # 工程根（代码所在）
    data_dir: Path      # 原始 jsonl 数据目录
    out_dir: Path       # 输出根
    tokenizer: Path     # 词表文件
    prepared_dir: Path  # 预分词产物目录
    ckpt_dir: Path      # checkpoint 目录
    log_dir: Path       # 日志目录

    def ensure(self) -> "Paths":
        for p in (self.out_dir, self.tokenizer.parent, self.prepared_dir,
                  self.ckpt_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def summary(self) -> str:
        items = asdict(self)
        w = max(len(k) for k in items)
        return "\n".join(f"  {k:<{w}} : {v}" for k, v in items.items())

    def jsonl_files(self):
        return sorted(self.data_dir.glob("*.jsonl"))


def add_path_args(parser: argparse.ArgumentParser) -> None:
    """给 argparse 挂上全部路径参数（不传则走环境变量/自动探测）。"""
    g = parser.add_argument_group("路径（优先级：命令行 > 环境变量 > 自动探测）")
    g.add_argument("--root", default=None, help=f"工程根目录 (env {ENV_ROOT})")
    g.add_argument("--data_dir", default=None, help=f"原始 jsonl 数据目录 (env {ENV_DATA_DIR})")
    g.add_argument("--out_dir", default=None, help=f"输出根目录 (env {ENV_OUT_DIR})")
    g.add_argument("--tokenizer", default=None, help=f"词表文件 (env {ENV_TOKENIZER})")
    g.add_argument("--prepared_dir", default=None, help=f"预分词产物目录 (env {ENV_PREPARED_DIR})")
    g.add_argument("--ckpt_dir", default=None, help=f"checkpoint 目录 (env {ENV_CKPT_DIR})")
    g.add_argument("--log_dir", default=None, help=f"日志目录 (env {ENV_LOG_DIR})")


def resolve_paths(args=None) -> Paths:
    """把 args（可为 None）解析成完整路径集合。"""
    a = args if args is not None else argparse.Namespace()

    root = Path(_pick(getattr(a, "root", None), ENV_ROOT, default_root())).resolve()

    raw_data = _pick(getattr(a, "data_dir", None), ENV_DATA_DIR, None)
    data_dir = Path(raw_data).resolve() if raw_data else autodetect_data_dir(root)

    out_dir = Path(_pick(getattr(a, "out_dir", None), ENV_OUT_DIR, root)).resolve()

    tokenizer = Path(_pick(getattr(a, "tokenizer", None), ENV_TOKENIZER,
                           out_dir / "data" / "tokenizer.json")).resolve()
    prepared_dir = Path(_pick(getattr(a, "prepared_dir", None), ENV_PREPARED_DIR,
                              out_dir / "prepared")).resolve()
    ckpt_dir = Path(_pick(getattr(a, "ckpt_dir", None), ENV_CKPT_DIR,
                          out_dir / "checkpoints")).resolve()
    log_dir = Path(_pick(getattr(a, "log_dir", None), ENV_LOG_DIR,
                         out_dir / "logs")).resolve()

    return Paths(root=root, data_dir=data_dir, out_dir=out_dir,
                 tokenizer=tokenizer, prepared_dir=prepared_dir,
                 ckpt_dir=ckpt_dir, log_dir=log_dir)
