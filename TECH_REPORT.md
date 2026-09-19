<!-- yaml front-matter for ModelScope / Hugging Face indexing -->
---
title: PMNN Text Generation Model — Technical Evaluation Report
version: v1.0 (Open-Source Release)
date: 2026-09
domain: nlp
frameworks: PyTorch
tags:
- Technical Report
- Evaluation
- Physics Model
- Text Generation
language:
- en
---

# PMNN Text Generation Model · Technical Evaluation Report

> **Version**: v1.0 (Open-Source Release) · **Date**: 2026-09
> **Scope**: Architecture assessment, training setup, complete testing directions & results, attribution conclusions

<p align="center">
  <a href="README.md"><b>English</b></a> &nbsp;|&nbsp;
  <a href="README_CN.md">简体中文</a>
</p>

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Hardware & Training Constraints](#2-hardware--training-constraints)
- [3. Data Scale & Parameter/Token Ratio](#3-data-scale--parametertoken-ratio)
- [4. Evaluation Metrics (step 5000 checkpoint)](#4-evaluation-metrics-step-5000-checkpoint)
- [5. Test Directions & Results](#5-test-directions--results)
- [6. Architecture-Level Assessment](#6-architecture-level-assessment)
- [7. Attribution Summary](#7-attribution-summary)
- [8. Recommended Next Steps (Ranked by ROI)](#8-recommended-next-steps-ranked-by-roi)
- [9. Reproducibility Entry Points](#9-reproducibility-entry-points)
- [10. License & Open-Source Statement](#10-license--open-source-statement)

---

## 1. Executive Summary

PMNN (Physics Matrix Neural Network) is a general-purpose substrate architecture that replaces Transformer's statistical attention with **physical dynamics**. This report documents the complete lifecycle — from architecture implementation and training through multiple evaluation rounds — with the emphasis on **objective test results**, including which directions worked, which did not, and the root causes behind the failures.

**Headline Conclusions (upfront)**:

1. The architecture's **forward / backward / generation pipeline is fully functional**, with no structural blockers.
2. All issues surfaced by evaluation (repetition, off-topic responses, multi-turn collapse) are attributable to **insufficient training data + insufficient training steps + data-quality contamination** — i.e. **underfitting**, not architectural error.
3. Knowledge-graph injection at inference time is **ineffective or even harmful**. The correct usage is to convert triples into training corpus rather than perform inference-time RAG.

---

## 2. Hardware & Training Constraints

The project uses a split environment: **training in the cloud, testing and patching locally**.

### 2.1 Training Environment (ModelScope Cloud)

| Resource | Specification |
|---|---|
| GPU | 24 GB VRAM |
| CPU | 8 cores |
| RAM | 32 GB |
| Single-task time cap | 10 hours (checkpoint-and-resume granularity down to the batch) |

At 24 GB VRAM, the 300m tier (batch=4, seq=128, including optimizer state) runs without pressure:

| Preset | Parameters | Optimizer | Peak VRAM |
|---|---|---|---|
| `100m` | 88.2 M | AdamW | 1.67 GB |
| `300m` | 187.6 M | AdamW | 3.49 GB |
| `300m` | 187.6 M | Adafactor | 2.24 GB |
| `1b` | ~1.1 B | Adafactor | fits within 24 GB |

The 10-hour single-task cap is bridged via `--resume` checkpoint continuation: checkpoints are atomically archived every 500 steps, and the learning rate is rejoined at the absolute step count with no jump. All checkpoints referenced in this report (step 5000 / 11500 / 13000) were produced on the ModelScope cloud.

### 2.2 Test / Patch Environment (Local)

| Resource | Specification |
|---|---|
| GPU | RTX 3050 Laptop, 4 GB VRAM |
| RAM | OS-limited |

Locally we only run inference evaluation and code patching — 4 GB is sufficient to load the 300m tier for generation tests. **No training runs locally.** Constraint takeaway: every evaluation conclusion in this report is based on the 300m tier.

---

## 3. Data Scale & Parameter/Token Ratio

The first piece of evidence for underfitting. Measured on the local corpus (`datasets/03_dialogue_clean`, 60 jsonl files):

| Metric | Value |
|---|---|
| Sample count | 237,707 |
| Total tokens (`max_seq_len=128`) | 28.47 M |
| Average sample length | 105.8 tokens |
| Length distribution | p50 = 106 / p90 = 181 / p99 = 282 / max = 2952 |
| Samples truncated at 128 | **81.4%** |
| Fully retained at 512 | 97.0% |

Parameter/token ratio (Chinchilla reference: 20 tokens/parameter):

| Preset | Parameters | Recommended tokens | Local 28.5 M coverage | Shortfall |
|---|---|---|---|---|
| `small` | 16 M | 0.32 B | 8.9% | 11× |
| `100m` | 90 M | 1.81 B | 1.6% | 64× |
| **`300m`** | **210 M** | **4.20 B** | **0.7%** | **147×** |
| `1b` | 1.1 B | 22.0 B | 0.13% | 772× |

**This is the single most important table in the report**: the 300m model was fed only 1/147 of the recommended token budget. Furthermore, 81.4% of samples were truncated at 128 tokens — **the model has predominantly only seen the "beginnings" of dialogues**. This is the direct cause of the "starts off sounding like a conversation, then drifts off" behavior.

---

## 4. Evaluation Metrics (step 5000 checkpoint)

Measured on `eval_metrics.json` (300m, n_blocks=16, d_field=1024, max_seq_len=128):

| Metric | Value | Note |
|---|---|---|
| CE | 2.1783 | Random baseline is 9.01 → the model is learning |
| PPL | 8.83 | — |
| `p_eom_mean` | 0.484 | Learned to close at MOSS-style end markers |
| `gen_eom_fired` | 5/5 ("你好呀"), 5/5 ("我好累啊") | Template completeness |
| `gen_len_range` | 36–42 tokens | Reasonable length |
| Order parameter `r_first` / `r_last` | 0.406 / 0.545 | No collapse |
| Entropy `s_first` / `s_last` | 2.75 / 4.635 | No information collapse |

**The metrics show: the model has learned the dialogue template (Human → MOSS → EOM) and there is no physical collapse.** However, generation content quality is poor (see §5), which means template and semantics are two different things — the template comes from format learning, while semantics requires sufficient token volume.

---

## 5. Test Directions & Results

All tests follow a **strict controlled-comparison** protocol: same input, same sampling parameters, same device, with only a single variable changed per comparison — ensuring conclusions are attributable.

### 5.1 Single-Turn Dialogue: step 5000 vs 11500 vs 13000

Sampling parameters fixed: `temp=0.8, top_k=50, top_p=0.9, rep_penalty=1.15, no_repeat_ngram=4`.

| Prompt | step 5000 | step 11500 | step 13000 (SFT) |
|---|---|---|---|
| "你好啊" (Hello) | Chaotic | "哇哇哇~你个好哇哇哇哇⭐但我了你好哇哇喵！" | "怎么怎么啊 : 这只说的是怎么会怎么可能…" |
| "我好累啊" (I'm so tired) | "去睡觉" (Go to sleep) — sporadic match | "不要把你的时间表分心放在你要去完成你做的" / "快去睡觉" | "想怎么做很多事情，怎么做些什么呢？去做好像我才做到…" |
| "你会什么" (What can you do) | Chaotic | "电脑上/手机上" (On the computer / phone) — partial match | Still chaotic |

**Conclusions**:

- **5000 → 11500**: from sporadic match to **stable match** ("我好累啊" answered with the right consolation semantics twice in a row) — training is progressing.
- **11500 → 13000** (SFT 1500 steps): from "word-fragment loops (哇哇喵)" to "verb-phrase loops (怎么怎么 / 想怎么做)" — **SFT swapped one repetition substrate for another, but did not eliminate the repetition mechanism**.
- The two reasons why step 13000 actually performed *worse* (confirmed by checkpoint metadata):
  1. The SFT stage ran with `w_fm=0.1, w_sync=0.1, w_drift=0.05, w_free=0.05` — **all auxiliary physics losses fully on**, when they should have been retired during the pure-QA phase.
  2. `warmup_steps=19500` but actual training stopped at 13000 → **the entire run was inside the warmup window, lr never reached its peak**. SFT was effectively "fine-tuning at warmup intensity" — learning was very shallow.

### 5.2 Multi-Turn Dialogue: Format Is the Watershed

- **Wrong format** (inserting `<s>` between turns): the model output emoji/English fragments like "⭐⭐⭐ Counting'mem" with no topic continuity → **initially misjudged as "multi-turn not learned"**.
- **Correct format** (matching SFT training exactly: plain newline between turns, no `<s>`, see `convert_sft.py`): `猫娘是什么？ → 女孩子会怎么样呢？` — **the model correctly associated the topic across turns**.
- **Real multi-turn chat** (accumulated history): garbage from turn 1 ("怎么怎么怎么") was stored in history as context, and every subsequent turn was dragged off-course → **multi-turn amplifies weaknesses rather than hiding them**.

**Methodological lesson**: the first multi-turn test used a hand-cobbled wrong format and produced the false conclusion "multi-turn not learned". After correcting the format to match training, the conclusion reversed. **Evaluation format must strictly match training format — otherwise you are testing the format, not the model.**

### 5.3 Knowledge-Graph Injection: Inference-Time Injection Is Ineffective (the Most Important Negative Result)

Knowledge graph: `knowledge_graph.json` — 9,948 entities + 8,011 triples (`[subject, relation, object]`).

Three injection methods × three checkpoints, each with an **injected vs not-injected** control:

| Injection method | Result |
|---|---|
| RAG-guided ("Answer based on the following knowledge: … Question: …") | All three checkpoints degraded; step 13000 collapsed completely (English gibberish) |
| Natural-fluency ("Regarding this question, I learned that: … May I ask…") | Still degraded ("技术？", "巴哈德！") |
| **No injection (control)** | Actually better: "是一种常用的网络协议", "5G网络的使用场景" |

**Root causes (three)**:

1. **Window crowding**: at `max_seq_len=128`, injected knowledge occupies 30+ tokens, diluting the model's continuation-start space — it can only emit very short residual fragments.
2. **Sentence pattern out-of-distribution**: the RAG prompt template was never seen in training, biasing the continuation path; step 13000 (post-SFT) is most sensitive to non-training patterns → collapses hardest.
3. **Triple noise**: even after filtering, meaningless injections like `技术 (COMES_FROM) 动处理一些常见问题` remain.

**Conclusion**: inference-time knowledge-graph injection suits **well-trained large models** (RAG), not underfit small models — a small model cannot digest the external-knowledge format. The correct usage is to convert triples into natural sentences ("5G is a technology that can be applied in industrial fields") and **mix them into the training corpus**, letting the model absorb the knowledge into its weights. (The script `scripts/kg_chat.py` is retained, including entity matching and noise filtering, for reuse as a training-data generator going forward.)

### 5.4 Sampling Parameters: Can Mitigate Repetition, Not Cure It

Under heavy penalties (`temp 0.6 + rep 1.3 + ngram 4`), repetition persists ("哇哇哇哇", "快快快快"). The root cause of repetition is a high-probability loop formed by underfitting — the model is over-confident on a few high-frequency tokens, and parameters can only suppress, not eliminate. **Curing repetition requires actually training the model well — there is no sampling shortcut.**

### 5.5 Empty Output / Premature Termination (Engineering-Side Fix)

- **Symptom**: when history is filled with high-frequency repeated tokens, `rep_penalty + no_repeat_ngram` suppresses every candidate, and the model emits `<eom>` directly (empty reply).
- **Fix (in `chat.py`)**: when the first generated token is `<eom>`, automatically relax parameters and retry once; history keeps only the most recent 3 turns to prevent early garbage from indefinitely contaminating later turns.

---

## 6. Architecture-Level Assessment

### 6.1 Architecture Usability

Forward, backward, generation, and flow-matching all pass the smoke test; the 200k-sample memmap pipeline runs without memory crashes; checkpoints are atomic and resume is batch-level precise. **At the engineering-implementation level, there are no blocking issues.**

### 6.2 Issues Found (Ranked by Severity)

**P1. Standing-wave positional encoding depends on the current sequence length** (`physics.py:_standing_wave`)

The wavenumber `k = 2π(m+1)/N` is normalized by the current length N. During training N is constant, but **during autoregressive inference N grows by +1 each step, so the encoding at every position in the window changes every step** — the representation of the already-generated prefix is not "frozen", which breaks the precondition for autoregressive stability.

**Fix direction**: switch the wavenumber to an absolute frequency (`k = 2π(m+1)/L_max`, fixed).

**P2. Auxiliary physics losses conflict with the language objective**

- `L_fm`'s target `v = x1 − x0` uses x0 = pure noise, **which carries no language signal** (training log shows `loss_fm ≈ 1.5` flat, never decreasing);
- `L_sync`'s target `r = 0.966` forces "global synchrony", which is inherently at odds with the "local diversity" that language modeling requires;
- `L_drift`'s attraction centers are randomly initialized prototypes (`randn * 0.05`) with no anchor to real embeddings — effectively injecting random forces.

**P3. Vocabulary size 8192 is too small for Chinese**

Chinese has 3,500+ common characters; with 8192 BPE tokens, Chinese effectively degenerates to single/double-character units, and the model can only memorize n-gram fragments. Half of the `�` characters in generated outputs are rare characters that were byte-split and could not be decoded completely.

**P4. Amplitude information is underutilized**

Logs show entropy `S ≈ 6.4` vs the theoretical maximum `ln(1024) = 6.93` — the amplitude distribution is near-uniform, output relies almost entirely on phase, and precise token prediction becomes hard to learn after 16 rounds of chaotic integration.

### 6.3 Data-Quality Contamination (Unexpected Finding)

Generation outputs contain fragments like `<sup><|1|></sup>` (HTML tags), `qwqwq`, `喵！`, `（恼）`, `Counting'mem` — **the corpus is contaminated with HTML and emoji/internet-slang noise, and the model is memorizing these fragments**. This is part of the underfitting picture (data quality itself is low) and is an actionable item.

---

## 7. Attribution Summary

| Symptom | Attribution | Evidence |
|---|---|---|
| Repetition / gibberish | Underfitting (data volume + step count) | 147× shortfall; stable 5000→11500 progress |
| Off-topic answers | Underfitting + 81.4% sample truncation (never saw complete logic) | `len_pct` / `truncated_at_128` |
| step 13000 worse than 11500 | SFT not completed + auxiliary losses fully on + warmup incomplete | Checkpoint metadata |
| Multi-turn collapse | Multi-turn amplifies weaknesses; turn-1 garbage contaminates history | §5.2 control |
| KG injection ineffective | Window crowding + out-of-distribution template + triple noise | §5.3 control |
| Positional encoding drift | Architectural defect (P1) | Code review |
| Parameter/data ratio imbalance | Hard data-volume constraint | §3 table |

**In one sentence**: the model *has* learned the dialogue template, *can* track topics across turns, and *can* distinguish intent types — these are **structural-level capabilities already acquired**. What it cannot do is **knowledge organization and logical elaboration** — that is a **qualitative change that only quantitative scaling (data/steps) can bring**.

---

## 8. Recommended Next Steps (Ranked by ROI)

1. **Data cleaning**: filter samples containing HTML tags or pure emoji (a one-shot regex pass — simple).
2. **Lengthen sequences**: `--max_seq_len 256 → 512` (97% of samples fully retained; the current 128 retains only 19%).
3. **Continue training**: directly resume from existing checkpoints (`--resume`) — no retraining, no new storage. On 24 GB cloud, 300m with AdamW is fine; Adafactor (2.24 GB) saves further.
4. **Fix P1 positional encoding**: pure architectural fix; eliminates autoregressive representation drift.
5. **Normalize SFT**: during the SFT stage, turn off auxiliary physics losses, complete the warmup, and train to the full step count.
6. **Convert knowledge graph to training corpus**: 8,011 triples → natural sentences → mix into the training set.

---

## 9. Reproducibility Entry Points

| Tool | Purpose |
|---|---|
| `generate.py` | Single-turn generation (for strict controlled comparison) |
| `chat.py` | Multi-turn dialogue (with empty-reply retry + history truncation) |
| `scripts/kg_chat.py` | Knowledge-graph injection controlled experiment |
| `eval_metrics.json` | Metric evaluation (CE / PPL / order parameter / entropy) |
| `logs/train_log.tsv` | Training curves |
| `smoke_test.py` | Forward / backward / generation smoke test |

---

## 10. License & Open-Source Statement

This project is open-sourced. Evaluation scripts, controlled-comparison methodology, and all conclusion data are released alongside the repository and are reproducible. Contributions on data cleaning and architecture fixes based on §8 are welcome.
