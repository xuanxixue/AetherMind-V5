"""预处理脚本：原始 jsonl -> 磁盘 memmap 产物（tokens.bin / index.npy / meta.json）。

这一步把「tokenize」从训练循环里彻底剥离，训练时内存占用≈0，
从根本上解决 20 万样本把宿主内存撑爆的问题。

用法（所有路径参数化，云端可直接复用）：
    # 本地默认（自动探测 datasets/03_dialogue_clean）
    python prepare_data.py

    # 限制规模快速验证
    python prepare_data.py --max_samples 20000

    # 云端：全部数据 + 更大序列长度
    export PMNN_DATA_DIR=/data/03_dialogue_clean
    export PMNN_OUT_DIR=/workspace/pmnn_out
    python prepare_data.py --max_seq_len 512 --vocab_size 32768

    # 只重建词表
    python prepare_data.py --rebuild_tokenizer
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import add_path_args, resolve_paths
from data_utils import ensure_tokenizer, prepare_tokens, load_meta


def main():
    p = argparse.ArgumentParser(description="PMNN 数据预处理")
    add_path_args(p)
    p.add_argument("--vocab_size", type=int, default=8192,
                   help="BPE 词表大小（云端可加到 32768/65536）")
    p.add_argument("--max_seq_len", type=int, default=128,
                   help="单条样本最大 token 数（本语料真实 p95≈485，建议云端用 512）")
    p.add_argument("--no_keep_eom", action="store_true",
                   help="截断时**不**保留句尾 <eom>（默认保留；关掉会让模型学不会收尾）")
    p.add_argument("--max_samples", type=int, default=0, help="0=全量")
    p.add_argument("--rebuild_tokenizer", action="store_true", help="强制重建词表")
    p.add_argument("--tokenizer_files", type=int, default=10,
                   help="构建词表时最多读取多少个 jsonl 文件")
    p.add_argument("--log_every", type=int, default=100000)
    args = p.parse_args()

    paths = resolve_paths(args).ensure()
    print("[prepare] 路径解析：")
    print(paths.summary())

    files = paths.jsonl_files()
    if not files:
        print(f"[prepare] 错误：{paths.data_dir} 下没有 *.jsonl 文件")
        print("  提示：用 --data_dir 或环境变量 PMNN_DATA_DIR 指定数据目录")
        return 1
    print(f"[prepare] 数据文件 {len(files)} 个")

    tok = ensure_tokenizer(paths.tokenizer, files,
                           vocab_size=args.vocab_size,
                           rebuild=args.rebuild_tokenizer,
                           max_files=args.tokenizer_files)

    meta = prepare_tokens(tok, files, paths.prepared_dir,
                          max_len=args.max_seq_len,
                          max_samples=args.max_samples,
                          log_every=args.log_every,
                          keep_eom=not args.no_keep_eom)

    print("[prepare] 产物 meta：")
    for k, v in meta.items():
        print(f"    {k} = {v}")
    # 回读校验
    meta2 = load_meta(paths.prepared_dir)
    assert meta2["n_samples"] == meta["n_samples"], "meta 回读不一致"
    print(f"[prepare] 校验通过，共 {meta['n_samples']} 条样本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
