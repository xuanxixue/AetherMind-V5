"""PMNN 实时多轮对话脚本（SFT 格式 + 流式逐字输出）。

基于 PMNNModel.stream_generate 边生成边打印，并按 SFT 训练时的真实格式
（[convert_sft.py] 轮间纯换行、无 <s> 分隔符）拼接多轮历史作为上下文，
让模型能跨轮承接对话主题（max_seq_len=512 下可容纳多轮短历史）。

用法：
    python chat.py --ckpt checkpoints/pmnn_300m_step13000.pt
    python chat.py --ckpt checkpoints/pmnn_300m_step13000.pt --temp 0.7 --max_new 100

对话内可用命令：
    /reset        清空历史，重开一段对话
    /quit /exit   退出
    /help         帮助
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import load_tokenizer
from pmnn.model import PMNNModel

HUMAN, MOSS, EOH, EOM = "<|Human|>", "<|MOSS|>", "<eoh>", "<eom>"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = PMNNModel(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[chat] 权重差异: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    return model, cfg, int(ckpt.get("step", 0) or 0)


MAX_ROUNDS = 3


def _trim_history(history: str, keep_rounds: int = MAX_ROUNDS) -> str:
    """只保留最近 keep_rounds 轮，丢掉更早的轮次。

    欠拟合模型会抓住历史里最早的低质回复死缠烂打；截短后模型只能看到
    最近几轮，避免最早那坨垃圾无限污染后续回答。按 "<|Human|>: " 切分轮次，
    无论如何都保留最后一个提问位（当前正待模型回答的那句）。
    """
    marker = f"{HUMAN}: "
    idx = 0
    for _ in range(keep_rounds):
        nxt = history.find(marker, idx if idx else 0)
        if nxt == -1:
            break
        idx = nxt + len(marker)
    return history if idx == 0 else history[idx:]


def main():
    p = argparse.ArgumentParser(description="PMNN 实时对话")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint 路径（默认取 checkpoints/ 里最新）")
    p.add_argument("--tokenizer", default=None, help="词表路径（自动探测则无需）")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--rep_penalty", type=float, default=1.15,
                   help="重复惩罚（CTRL 口径，1.0=关闭，对抗未充分训练的复读）")
    p.add_argument("--no_repeat_ngram", type=int, default=4,
                   help="禁止重复 n-gram 长度（0/1=关闭）")
    args = p.parse_args()

    root = Path(__file__).resolve().parent

    tok_path = Path(args.tokenizer) if args.tokenizer else root / "data" / "tokenizer.json"
    if not tok_path.exists():
        print(f"[chat] 词表不存在: {tok_path}；用 --tokenizer 指定")
        return 1

    ckpt_dir = root / "checkpoints"
    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)
        if not ckpts:
            print(f"[chat] {ckpt_dir} 下没有 checkpoint")
            return 1
        ckpt = ckpts[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(tok_path)
    model, cfg, step = load_model(ckpt, device)
    eos_id = tokenizer.token_to_id(EOM)
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<eos>")

    print(f"[chat] 模型: {ckpt.name} (step={step}, preset={getattr(cfg, 'preset', '?')}, "
          f"vocab={cfg.vocab_size}, device={device})")
    print("[chat] 多轮对话。/reset 清空历史、/quit 退出。/help 帮助。\n")

    history = ""   # 累积的多轮上下文，SFT 格式：轮间 \n、无 <s>

    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] 再见")
            break
        if not q:
            continue
        low = q.lower()
        if low in ("/quit", "/exit", "退出"):
            print("[chat] 再见")
            break
        if low == "/reset":
            history = ""
            print("[chat] 历史已清空，重开一段对话\n")
            continue
        if low == "/help":
            print("[chat] 命令：/reset 清空历史、/quit 退出、/help 帮助\n")
            continue

        # 追加本轮提问到历史，然后拼上待补的 MOSS 位
        history += f"{HUMAN}: {q}{EOH}\n"
        history = _trim_history(history)   # 只保留最近几轮，别让最早垃圾污染后续

        def run_gen(rep, ngram, temp):
            """跑一次流式生成并打印，返回是否生成了非空回复。"""
            context = history + f"{MOSS}: "
            prompt_ids = tokenizer.encode(context).ids
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            sys.stdout.write("PMNN > ")
            sys.stdout.flush()
            new_ids = []
            prev_len = 0
            for step_gen in model.stream_generate(
                    input_ids, max_new_tokens=args.max_new, temperature=temp,
                    top_k=args.top_k, top_p=args.top_p, eos_id=eos_id,
                    repetition_penalty=rep, no_repeat_ngram_size=ngram):
                last = step_gen[0][-1].item()
                new_ids.append(last)
                full = tokenizer.decode(new_ids)
                frag = full[prev_len:]      # 只打本次新增，跨 token 合并也不错位
                prev_len = len(full)
                if last == eos_id:
                    break
                sys.stdout.write(frag)
                sys.stdout.flush()
            # new_ids 去掉末尾 <eom>；若没有任何内容 token，视为空回复
            reply = tokenizer.decode(
                new_ids[:-1] if new_ids and new_ids[-1] == eos_id else new_ids).strip()
            return reply, new_ids

        # 第一次：用默认参数。若空回复，放宽惩罚重试一次，避免只剩"直接结束"
        reply = None
        new_ids = None
        for attempt, (rep, ngram, temp) in enumerate([
                (args.rep_penalty, args.no_repeat_ngram, args.temp),
                (1.0, 0, min(args.temp + 0.2, 1.0))]):
            try:
                reply_cand, new_ids_cand = run_gen(rep, ngram, temp)
            except KeyboardInterrupt:
                print("\n[chat] 中断本次生成")
                history = history[: history.rfind(f"{HUMAN}: ")]
                break
            reply, new_ids = reply_cand, new_ids_cand
            if reply:
                break
            print("\n[chat] 空回复，放宽参数重试...\n")
        else:
            reply, new_ids = "", []
            print("[chat] 连续两次空回复，已跳过本轮")

        print("\n")

        # 今生成了内容，把本轮 MOSS 回复接进历史，供下一轮承接
        if reply:
            history += f"{MOSS}: {reply}{EOM}\n"


if __name__ == "__main__":
    raise SystemExit(main())