"""数据管道：BPE 词表 + 磁盘 memmap 预分词数据集。

数据格式：{"text": "<|Human|>: ...<eoh>\\n<|MOSS|>: ...<eom>"}

为什么改成 memmap（修复宿主内存耗尽崩溃）：
  旧实现把每条样本 tokenize 成 Python list[int] 常驻内存，
  20 万条 × 128 token ≈ 数 GB 的 Python 对象，加上 DataLoader 多进程
  在 Windows spawn 下再复制一份，直接 OOM。

  新实现分两阶段：
    ① 预处理（prepare_data.py）：流式 tokenize，结果落盘为
       tokens.bin（连续 token 流）+ index.npy（每条样本的 offset/length）
       + meta.json。全程内存恒定（只保留当前 batch）。
    ② 训练：Dataset 用 np.memmap 只读映射，进程内存占用 ≈ 0，
       数据靠操作系统页缓存，多 worker 共享同一份映射。

产物目录（默认 <out_dir>/prepared/）：
    tokens.bin   连续 token 流（uint16/uint32）
    index.npy    [N, 2] int64：每行 = (offset, length)
    meta.json    元信息（词表大小 / 样本数 / dtype / max_len）
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


# ============================================================================
# 词表
# ============================================================================
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>",
                  "<|Human|>", "<|MOSS|>", "<eoh>", "<eom>"]


def iter_texts(data_files: Iterable, max_samples: int = 0) -> Iterable[str]:
    """流式产出 jsonl 里的 text 字段（内存恒定）。"""
    n = 0
    for fp in data_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    txt = json.loads(line).get("text") or ""
                except Exception:
                    continue
                if not txt:
                    continue
                yield txt
                n += 1
                if max_samples and n >= max_samples:
                    return


def build_tokenizer(data_files, save_path, vocab_size: int = 8192,
                    max_files: int = 0, max_texts: int = 0) -> Tokenizer:
    """从原始 jsonl 流式训练 BPE 词表。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    files = list(data_files)[:max_files] if max_files else list(data_files)

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tk.train_from_iterator(iter_texts(files, max_samples=max_texts), trainer=trainer)
    tk.save(str(save_path))
    print(f"[tokenizer] 已保存 {save_path} (vocab={tk.get_vocab_size()})")
    return tk


def load_tokenizer(path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def ensure_tokenizer(tokenizer_path, data_files, vocab_size=8192,
                     rebuild: bool = False, max_files: int = 10) -> Tokenizer:
    """词表存在则加载，否则构建。"""
    tokenizer_path = Path(tokenizer_path)
    if rebuild or not tokenizer_path.exists():
        build_tokenizer(data_files, tokenizer_path, vocab_size, max_files=max_files)
    tk = load_tokenizer(tokenizer_path)
    print(f"[tokenizer] 加载 {tokenizer_path} (vocab={tk.get_vocab_size()})")
    return tk


# ============================================================================
# 预处理：jsonl -> tokens.bin / index.npy / meta.json
# ============================================================================
_EMIT_CHUNK = 512              # 每批 encode_batch 的文本数
_FLUSH_TOKENS = 2_000_000      # 每累积这么多 token 落盘一次


def prepare_tokens(tokenizer: Tokenizer, data_files, prepared_dir,
                   max_len: int = 128, max_samples: int = 0,
                   log_every: int = 100000, verbose: bool = True,
                   keep_eom: bool = True) -> dict:
    """流式 tokenize 并落盘。返回 meta 字典。

    keep_eom=True（默认）：超长样本截断时**保住句尾的 <eom>**。
    ⚠️ 实测教训（2026-09-13）：本语料真实平均长度 255 token、p95 达 485，
       而 max_len=128 时只有 18.6% 的样本能完整保留 —— 也就是说 81% 的
       训练样本「没有结尾」。模型因此学不会收尾，生成时不停复读、迟迟不
       输出 <eom>。改成「前 max_len-1 个 token + <eom>」后，每条样本都
       带终止符，这是让模型学会「说完就停」的最低成本修法。
    """
    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    tok_path = prepared_dir / "tokens.bin"
    idx_path = prepared_dir / "index.npy"
    meta_path = prepared_dir / "meta.json"

    vocab = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab <= 65535 else np.uint32
    eom_id = tokenizer.token_to_id("<eom>")

    offsets: List[int] = []
    lengths: List[int] = []
    cursor = 0
    n_samples = 0
    n_skipped = 0
    n_truncated = 0

    # 落盘缓冲
    buf_ids: List[np.ndarray] = []
    buf_off: List[int] = []
    buf_len: List[int] = []
    buf_tokens = 0

    def flush(fh):
        nonlocal cursor, buf_tokens
        if not buf_ids:
            return
        arr = np.concatenate(buf_ids)
        fh.write(arr.astype(dtype, copy=False).tobytes())
        offsets.extend(buf_off)
        lengths.extend(buf_len)
        cursor += int(arr.size)
        buf_ids.clear()
        buf_off.clear()
        buf_len.clear()
        buf_tokens = 0

    def emit(texts: List[str]):
        """批量编码；offset = 已落盘 token 数 + 当前缓冲 token 数。"""
        nonlocal n_samples, n_skipped, buf_tokens, n_truncated
        for enc in tokenizer.encode_batch(texts):
            ids = enc.ids
            if len(ids) > max_len:
                n_truncated += 1
                # 截断时必须保住句尾的 <eom>，否则模型永远见不到「说完就停」
                if keep_eom and eom_id is not None and ids and ids[-1] == eom_id:
                    ids = ids[:max_len - 1] + [eom_id]
                else:
                    ids = ids[:max_len]
            if len(ids) < 2:
                n_skipped += 1
                continue
            arr = np.asarray(ids, dtype=np.int64)
            buf_ids.append(arr)
            buf_off.append(cursor + buf_tokens)
            buf_len.append(int(arr.size))
            buf_tokens += int(arr.size)
            n_samples += 1

    chunk: List[str] = []
    last_log = 0
    with open(tok_path, "wb") as fh:
        for txt in iter_texts(data_files, max_samples=max_samples):
            chunk.append(txt)
            if len(chunk) >= _EMIT_CHUNK:
                emit(chunk)
                chunk.clear()
                if buf_tokens >= _FLUSH_TOKENS:
                    flush(fh)
                if verbose and log_every and n_samples - last_log >= log_every:
                    last_log = n_samples
                    print(f"[prepare] 已处理 {n_samples} 条样本 / "
                          f"{cursor + buf_tokens} token")
        if chunk:
            emit(chunk)
        flush(fh)

    index = np.zeros((n_samples, 2), dtype=np.int64)
    index[:, 0] = offsets[:n_samples]
    index[:, 1] = lengths[:n_samples]
    np.save(idx_path, index)

    meta = {
        "vocab_size": vocab,
        "n_samples": int(n_samples),
        "n_skipped": int(n_skipped),
        "total_tokens": int(cursor),
        "dtype": np.dtype(dtype).name,
        "max_len": int(max_len),
        "keep_eom": bool(keep_eom),
        "n_truncated": int(n_truncated),
        "truncated_ratio": round(n_truncated / max(n_samples, 1), 4),
        "tokens_file": tok_path.name,
        "index_file": idx_path.name,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"[prepare] 完成：{n_samples} 条样本 / {cursor} token / "
              f"跳过 {n_skipped} 条 -> {prepared_dir}")
        ratio = n_truncated / max(n_samples, 1)
        print(f"[prepare] 截断统计：{n_truncated} 条被截到 {max_len} token "
              f"（{ratio*100:.1f}%），keep_eom={keep_eom}")
        if ratio >= 0.30:
            print("[prepare] " + "!" * 60)
            print(f"[prepare] !!! 警告：{ratio*100:.0f}% 的样本长度超过 max_len={max_len}。")
            print("[prepare] !!! 样本被截断会丢掉对话细节；若 keep_eom=False 还会丢掉")
            print("[prepare] !!! 句尾 <eom>，模型将学不会收尾（生成变得啰嗦、复读）。")
            print(f"[prepare] !!! 建议：--max_seq_len 提到 512（实测可完整保留 97% 样本），")
            print("[prepare] !!!       并把训练的 --batch 同比例调小以控显存。")
            print("[prepare] " + "!" * 60)
    return meta


def load_meta(prepared_dir) -> dict:
    return json.loads((Path(prepared_dir) / "meta.json").read_text(encoding="utf-8"))


# ============================================================================
# 训练用数据集（memmap，内存占用 ≈ 0）
# ============================================================================
class MemmapDataset(Dataset):
    """从磁盘映射 token 流，按需切片。"""

    def __init__(self, prepared_dir, max_len: Optional[int] = None,
                 pad_id: int = 0, max_samples: int = 0):
        prepared_dir = Path(prepared_dir)
        self.meta = load_meta(prepared_dir)
        self.tokens = np.memmap(prepared_dir / self.meta["tokens_file"],
                                dtype=np.dtype(self.meta["dtype"]), mode="r")
        self.index = np.load(prepared_dir / self.meta["index_file"], mmap_mode="r")
        self.pad_id = pad_id
        self.max_len = int(max_len or self.meta["max_len"])
        self.n = int(len(self.index))
        if max_samples:
            self.n = min(self.n, int(max_samples))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        off, ln = int(self.index[i, 0]), int(self.index[i, 1])
        ln = min(ln, self.max_len)
        ids = np.asarray(self.tokens[off:off + ln], dtype=np.int64)
        return {"input_ids": torch.from_numpy(ids)}


def collate_fn(batch, pad_id: int = 0, max_len: Optional[int] = None,
               dynamic: bool = False):
    """左 padding（与 generate 的推理布局一致）。

    dynamic=False 时固定补齐到 max_len；True 时补齐到 batch 内最长。
    labels 在 pad 位置为 -100（不参与交叉熵）。
    """
    seqs = [b["input_ids"] for b in batch]
    lengths = [int(s.numel()) for s in seqs]
    L = max(lengths) if lengths else 2
    if not dynamic and max_len:
        L = max(L, int(max_len))
    L = max(L, 2)
    B = len(seqs)

    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:L]
        n = int(s.numel())
        input_ids[i, L - n:] = s
    mask = (input_ids != pad_id).float()

    # ⚠️ 因果语言建模：labels 必须是 input_ids **右移一位**。
    #    若直接令 labels = input_ids，模型只需学会"抄自己输入"就能把 CE 压到 ~0，
    #    那是纯作弊，学不到任何语言结构（曾把 CE 从 9 骗到 0.01）。
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]      # 位置 t 的目标 = 位置 t+1 的 token
    labels[:, -1] = -100                   # 末位没有下一个 token
    labels[input_ids == pad_id] = -100     # pad 位置不参与 loss
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}
