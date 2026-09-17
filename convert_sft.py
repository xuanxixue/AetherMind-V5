"""SFT 后训练数据转换脚本：把外部指令/对话数据集转成 PMNN 模板。

目标模板（与现有预训练语料一致，见 data_utils.py）：
    {"text": "<|Human|>: {prompt}<eoh>\\n<|MOSS|>: {response}<eom>"}

支持两种来源格式，自动检测：
  1) instruction/input/output   （如 COIG-CQIA、firefly、alpaca 系）
     prompt   = instruction + ("\\n" + input 若 input 非空)
     response = output
  2) messages                   （如 UltraData-SFT、OpenAI 系）
     [{role: "system", content}, {role:"user", content}, {role:"assistant", content}, ...]
     system 内容并入第一条 user；user-><|Human|>，assistant-><|MOSS|>；
     最后一条必须是 assistant。

用法：
    # 单文件
    python convert_sft.py --input data.jsonl --output converted.jsonl

    # 目录（递归 *.jsonl）
    python convert_sft.py --input datasets/ultradata/COIG-CQIA/... --output out.jsonl

    # 抽样/限长
    python convert_sft.py --input d.jsonl --output o.jsonl --max_samples 50000 --max_chars 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HUMAN = "<|Human|>"
MOSS = "<|MOSS|>"
EOH = "<eoh>"
EOM = "<eom>"


def _norm(s) -> str:
    """去掉内部换行两侧多余空白，避免破坏模板结构。"""
    if s is None:
        return ""
    s = str(s).strip()
    # 内部连续换行压成单个换行
    lines = [ln.strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln != "")


def conv_instruction(obj: dict) -> str | None:
    """instruction/input/output -> text"""
    instr = _norm(obj.get("instruction"))
    inp = _norm(obj.get("input"))
    out = _norm(obj.get("output"))
    if not instr:
        return None
    prompt = instr if not inp else f"{instr}\n{inp}"
    if not out:
        return None
    return f"{HUMAN}: {prompt}{EOH}\n{MOSS}: {out}{EOM}"


def conv_messages(obj: dict) -> str | None:
    """messages -> text（支持多轮拼接）"""
    msgs = obj.get("messages") or obj.get("conversations") or []
    if not isinstance(msgs, list) or not msgs:
        return None

    parts: list[str] = []
    sys_buf: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        content = _norm(m.get("content") or m.get("value"))
        if not content:
            continue
        if role == "system":
            sys_buf.append(content)
            continue
        if role in ("user", "human") or (role == "" and not parts):
            head = HUMAN
        elif role in ("assistant", "moss", "bot") or (role == "" and parts):
            head = MOSS
        else:
            # 未知 role：跳过，避免污染
            continue
        if sys_buf and head == HUMAN and not parts:
            content = "\n".join(sys_buf) + "\n" + content
            sys_buf = []
        parts.append(f"{head}: {content}")

    if not parts:
        return None
    if len(parts) == 1:
        # 只有一条 user 没有 assistant，无法构成问答对
        return None

    # 拼成连续对话：每条 Human 以 <eoh> 结尾，每条 MOSS 以 <eom> 结尾
    text = ""
    for i, p in enumerate(parts):
        head, body = p.split(": ", 1)
        if head == HUMAN:
            text += f"{HUMAN}: {body}{EOH}\n"
        else:
            text += f"{MOSS}: {body}{EOM}"
            if i != len(parts) - 1:
                text += "\n"
    # 最后一条必须是 MOSS 收尾
    if not text.rstrip().endswith(EOM):
        return None
    return text


def detect_and_convert(obj: dict) -> str | None:
    if "messages" in obj or "conversations" in obj:
        return conv_messages(obj)
    if "instruction" in obj or ("output" in obj and "input" in obj):
        return conv_instruction(obj)
    return None


def iter_jsonl(inp: Path):
    targets = [inp] if inp.is_file() else sorted(inp.rglob("*.jsonl"))
    for fp in targets:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield obj
        except OSError:
            continue


def main() -> int:
    p = argparse.ArgumentParser(description="SFT 数据 -> PMNN 模板转换")
    p.add_argument("--input", required=True, help="输入 jsonl 文件或目录")
    p.add_argument("--output", required=True, help="输出 jsonl 文件")
    p.add_argument("--max_samples", type=int, default=0, help="0=全量")
    p.add_argument("--max_chars", type=int, default=0,
                   help="单条 text 最大字符数（超长丢弃，0=不限，建议 4000）")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        print(f"[convert] 错误：输入不存在 {inp}")
        return 1

    n_read = 0
    n_ok = 0
    n_skip = 0
    fmt_seen = {"instruction": 0, "messages": 0}

    with open(out, "w", encoding="utf-8") as w:
        for obj in iter_jsonl(inp):
            n_read += 1
            if args.max_samples and n_ok >= args.max_samples:
                break
            if "messages" in obj or "conversations" in obj:
                text = conv_messages(obj)
                bucket = "messages"
            else:
                text = conv_instruction(obj)
                bucket = "instruction"
            if text is None:
                n_skip += 1
                continue
            if args.max_chars and len(text) > args.max_chars:
                n_skip += 1
                continue
            fmt_seen[bucket] += 1
            w.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 50000 == 0:
                print(f"[convert] 已写入 {n_ok} 条 ...")

    print(f"[convert] 完成：读取 {n_read} 条 -> 有效 {n_ok} 条、跳过 {n_skip} 条")
    print(f"[convert] 格式分布：instruction={fmt_seen['instruction']}, "
          f"messages={fmt_seen['messages']}")
    print(f"[convert] 输出：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())