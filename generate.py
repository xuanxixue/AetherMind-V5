"""PMNN 文本生成：加载 checkpoint 生成回复。

路径全部参数化（CLI > 环境变量 > 自动探测）。

用法：
    python generate.py --prompt "你好呀"
    python generate.py --prompt "今天天气怎么样" --max_new 80 --temp 0.7
    python generate.py --ckpt D:/out/checkpoints/pmnn_step6000.pt --interactive

云端：
    export PMNN_OUT_DIR=/workspace/pmnn_out
    python generate.py --prompt "介绍一下你自己"
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import add_path_args, resolve_paths
from data_utils import load_tokenizer
from pmnn.model import PMNNModel


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = PMNNModel(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[gen] 权重差异: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    return model, cfg, int(ckpt.get("step", 0) or 0)


def find_latest_ckpt(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


@torch.no_grad()
def reply_to(model, tokenizer, prompt, device, max_new=64, temp=0.8,
             top_k=50, top_p=0.9, rep_penalty=1.15, no_repeat_ngram=4):
    # 注意不能写成 `token_to_id("<eom>") or token_to_id("<eos>")`：
    # 若 <eom> 恰好是 0 号 token，`0 or x` 会错误地退回 <eos>。
    eos_id = tokenizer.token_to_id("<eom>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<eos>")
    prompt_ids = tokenizer.encode(f"<|Human|>: {prompt}<eoh>\n<|MOSS|>: ").ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = model.generate(input_ids, max_new_tokens=max_new, temperature=temp,
                         top_k=top_k, top_p=top_p, eos_id=eos_id,
                         repetition_penalty=rep_penalty,
                         no_repeat_ngram_size=no_repeat_ngram)
    out_ids = gen[0].cpu().tolist()
    return tokenizer.decode(out_ids[len(prompt_ids):])


def main():
    p = argparse.ArgumentParser(description="PMNN 文本生成")
    add_path_args(p)
    p.add_argument("--ckpt", default=None, help="checkpoint 路径（默认取 ckpt_dir 内最新）")
    p.add_argument("--prompt", action="append", default=None, help="可重复传入多条")
    p.add_argument("--interactive", action="store_true", help="交互式输入")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    # 未充分训练的 checkpoint 会疯狂复读，这两个默认值能立刻把输出从
    # 「复读机」拉回可读句子；想看原汁原味的分布就传 --rep_penalty 1.0
    # --no_repeat_ngram 0。
    p.add_argument("--rep_penalty", type=float, default=1.15,
                   help="重复惩罚（CTRL 口径，1.0=关闭）")
    p.add_argument("--no_repeat_ngram", type=int, default=4,
                   help="禁止重复 n-gram 长度（0/1=关闭）")
    args = p.parse_args()

    paths = resolve_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt is None:
        ckpt = find_latest_ckpt(paths.ckpt_dir)
        if ckpt is None:
            print(f"[gen] {paths.ckpt_dir} 下没有 checkpoint，先跑 train.py")
            return 1
    else:
        ckpt = Path(args.ckpt)

    if not paths.tokenizer.exists():
        print(f"[gen] 词表不存在: {paths.tokenizer}")
        return 1
    tokenizer = load_tokenizer(paths.tokenizer)
    model, cfg, step = load_model(ckpt, device)
    print(f"[gen] 模型: {ckpt} (step={step}, preset={getattr(cfg, 'preset', '?')}, "
          f"vocab={cfg.vocab_size})")

    prompts = list(args.prompt or [])
    if args.interactive or not prompts:
        print("[gen] 交互模式，输入空行退出")
        while True:
            try:
                q = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            ans = reply_to(model, tokenizer, q, device, args.max_new,
                           args.temp, args.top_k, args.top_p,
                           args.rep_penalty, args.no_repeat_ngram)
            print(f"PMNN > {ans}\n")
        return 0

    for q in prompts:
        ans = reply_to(model, tokenizer, q, device, args.max_new,
                       args.temp, args.top_k, args.top_p,
                       args.rep_penalty, args.no_repeat_ngram)
        print("=" * 64)
        print(f"用户: {q}")
        print("-" * 64)
        print(f"PMNN: {ans}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
