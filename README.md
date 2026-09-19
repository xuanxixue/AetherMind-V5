<!-- yaml front-matter for ModelScope / Hugging Face indexing -->
---
datasets:
- DreamStarPaint/MOSS
- Moemuu/Muice-Dataset
frameworks: PyTorch
license: CC_BY_4.0
domain: nlp
tags:
- Text Generation Model
- Physics Model
- Kuramoto
- Flow Matching
- Oscillator Neural Network
tasks:
- text-generation
language:
- zh
- en
studios:
- xuanxixue/AetherMind-V5-300M-Instruct
---

# PMNN: Physics Matrix Neural Network · General-Purpose Substrate

> **Version** v0.1 · 2026-09-11 · Experimental validation just passed · Positionally analogous to Transformer but **non-statistical attention**
>
> Kuramoto phase synchronization replaces attention, a resonance–standing-wave FNN replaces the feed-forward network, a drift field + thermodynamics + time field act as explicit physical constraints, and the output stage fuses ELF continuous flow matching.

<p align="center">
  <a href="https://modelscope.cn/studios/xuanxixue/AetherMind-V5-300M-Instruct" target="_blank">
    <img alt="Online Demo" src="https://img.shields.io/badge/🚀_Online_Demo-ModelScope-4e2958?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJMNCA3djEwbDggNSA4LTVWNyIvPjwvc3ZnPg=="/>
  </a>
  &nbsp;
  <a href="https://modelscope.cn/models/xuanxixue/AetherMind-V5-300M-Instruct" target="_blank">
    <img alt="Model Weights" src="https://img.shields.io/badge/📦_Model-300M_Instruct-6c5ce7?style=flat-square"/>
  </a>
  &nbsp;
  <img alt="License" src="https://img.shields.io/badge/License-CC_BY_4.0-2ecc71?style=flat-square"/>
  &nbsp;
  <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch-ee4a2d?style=flat-square&logo=pytorch&logoColor=white"/>
  &nbsp;
  <img alt="Languages" src="https://img.shields.io/badge/Languages-zh%20%7C%20en-3498db?style=flat-square"/>
</p>

<p align="center">
  <a href="README.md"><b>English</b></a> &nbsp;|&nbsp;
  <a href="README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="README.md"><img alt="EN" src="https://img.shields.io/badge/lang-English-blue.svg"/></a>
  &nbsp;
  <a href="README_CN.md"><img alt="CN" src="https://img.shields.io/badge/lang-中文-red.svg"/></a>
</p>

---

## Table of Contents

- [1. One-Line Positioning](#1-one-line-positioning)
- [2. Core Innovation: PMNN vs Transformer](#2-core-innovation-pmnn-vs-transformer)
- [3. Architecture Overview](#3-architecture-overview)
- [4. Empirical Validation](#4-empirical-validation)
  - [4.1 Controlled Architecture Ablation](#41-controlled-architecture-ablation)
  - [4.2 Cross-Architecture BPB Benchmark](#42-cross-architecture-bpb-benchmark)
  - [4.3 Scaling Law](#43-scaling-law)
- [5. Internal Physics Diagnostics](#5-internal-physics-diagnostics)
  - [5.1 Layer-wise Physics Trajectory (step 5000)](#51-layer-wise-physics-trajectory-step-5000)
  - [5.2 No-Collapse Verification (step 11500)](#52-no-collapse-verification-step-11500)
- [6. Quick Start](#6-quick-start)
- [7. Directory Structure & Path Conventions](#7-directory-structure--path-conventions)
- [8. Scale Presets & Measured VRAM](#8-scale-presets--measured-vram)
- [9. Data Pipeline & Sequence-Length Analysis](#9-data-pipeline--sequence-length-analysis)
- [10. Cloud Deployment (1B, Zero Code Changes)](#10-cloud-deployment-1b-zero-code-changes)
- [11. Key Engineering Decisions](#11-key-engineering-decisions)
- [12. Loss Composition](#12-loss-composition)
- [13. Design-Doc Implementation Status](#13-design-doc-implementation-status)
- [14. Knowledge-Distillation Cloud Package](#14-knowledge-distillation-cloud-package)
- [15. FAQ](#15-faq)
- [16. Citation & License](#16-citation--license)
- [17. References](#17-references)
- [Appendix: Figure Attribution](#appendix-figure-attribution)

---

## 1. One-Line Positioning

> Replace Transformer's "token + attention (statistical affinity)" with "**complex-valued oscillator field + phase synchronization (physical coupling)**", make depth a physical-time ODE, and replace the output stage with an ELF-style continuous denoising trajectory — yielding a modality-agnostic substrate of the same isomorphic standing as Transformer.

Any modality is first "oscillatorized" into a continuous embedding field, evolved through R rounds of physical blocks, and discretized back to symbols at the final step. Current stage: **experimental validation just passed** — architectural feasibility has been verified at medium scale (≤300M params / 28.5M tokens); the next step is large-scale data expansion and knowledge distillation.

---

## 2. Core Innovation: PMNN vs Transformer

PMNN is not a Transformer variant — it is a substrate that differs at the **mechanism level**. The table below maps the two component-by-component:

| Dimension | Transformer | PMNN |
|---|---|---|
| **Basic unit** | Token / embedding vector | Complex oscillator `z = a·e^{iθ}` (amplitude + phase) |
| **Interaction mechanism** | Attention `softmax(QKᵀ)`, statistical affinity | Phase synchronization via Kuramoto coupling `Σ W·sin(Δθ)`, dynamical phase-locking |
| **Positional encoding** | RoPE / ALiBi (hand-designed functions) | Standing-wave modes `cos(k·p)` (wave interference; position = spatial coordinate) |
| **Feed-forward** | FFN (ReLU/GELU) | Resonance–standing-wave FNN (Lorentzian frequency selection + standing-wave modulation) |
| **Depth** | Layer stacking (discrete) | Physical-time ODE integration (continuous, reversible) |
| **Distribution modeling** | Autoregressive + softmax cross-entropy | Flow matching + drift field (continuous trajectory + distribution evolution) |
| **Output** | Vocabulary distribution | Continuous denoising trajectory → final-step de-embedding |
| **Training signal** | Token-level cross-entropy | Velocity-field matching + synchrony/entropy constraints |
| **Control parameter** | Sampling temperature / decoding strategy | **K-value** (time field × thermodynamics) × entropy budget |
| **Source of generality** | Tokenizing any modality | Oscillatorizing any modality (complex-valued field) |

> **Honest note**: The precise boundary of "non-statistical" is — what is removed is the attention mechanism and token-level step-by-step cross-entropy; flow matching and drift fields are still distribution modeling (statistical targets). Precise statement: **the mechanism is physical, the distribution target is statistical** — the two do not contradict.

---

## 3. Architecture Overview

The PMNN physics core is a stack of R physical blocks, each containing four layers that map one-to-one onto the Transformer block:

```
Input modality ──> ① Oscillatorize ──> ② Physics core · R rounds ──> ③ Continuous trajectory ──> ④ Final decode
(any discrete seq)  (z_i = a_i ⊙ e^(iθ_i))  (Kuramoto+standing wave+drift+thermo)  (ELF flow matching)  (de-embedding)
```

**Four-layer structure of each physical block (mapped to Transformer block):**

| Layer | PMNN implementation | Transformer analog |
|---|---|---|
| **Phase-synchronization layer** | `dθ/dt = ω + Σ W·sin(θⱼ−θᵢ) + κ(t)·g` | Attention |
| **Resonance–standing-wave FNN** | Lorentzian gain × standing-wave modulation | FFN |
| **Drift-field layer** | `V = V⁺ − λV⁻` (multi-temperature softmax) | Distribution evolution |
| **Thermodynamics–entropy layer** | XY energy / amplitude entropy / free energy `F = U − T·S` | Energy/temperature regularization |

**Navigation bus**: time field × thermodynamics → K-value (coupling strength κ(t) · temperature annealing T(t) · driving frequency ω_d(t) · entropy budget), bidirectionally fed — the bus controls the physical blocks, and the blocks feed back the order parameter.

> 📐 **Full design doc**: see [`PMNN_模型架构详尽网络层通用底层架构 (1).md`](./PMNN_模型架构详尽网络层通用底层架构%20%281%29.md) (v0.1 formal spec, in Chinese), which contains the unified state equation conflating the six extensions (ELF / drift field / UN-0 phase / thermodynamic entropy / resonance standing wave / time physics), source-verification records, and honest risk annotations. An English version is also available as `PMNN_Architecture_Design_v0.1_Professional_…` in the repo root.

**Unified state equation** (the core matrix formula conflating the six extensions):

```
dZ/dt = M_κ(t)·Z + ∇_Z[ −U(Z) + T(t)·S(Z) ] + v_φ(Z, t) + η(t)
```

- `M_κ(t)·Z`: linear matrix operator navigated by the K-value;
- `∇_Z[−U + T·S]`: physical potential descent + entropy pressure (auto-expands into Kuramoto sin coupling + free-energy dynamics + drift repulsion);
- `v_φ(Z,t)`: drift-field learning residual;
- `η(t)`: Langevin noise (fluctuation-dissipation).

---

## 4. Empirical Validation

> This section presents the core empirical results at the current experimental-validation stage. All comparisons are run under **the same tokenizer, the same corpus, and the same number of training steps** — only the architecture varies. This is the minimum credibility bar for evaluating "mechanism differences".

### 4.1 Controlled Architecture Ablation

**Figure 1: Controlled architecture ablation training curve** (same tokenizer / same corpus / same steps — only architecture varies)

<!-- Figure slot 1 -->
<p align="center">
  <img src="docs/images/fig_arch_ablation.png" alt="Controlled architecture ablation training curve" width="85%"/>
</p>
<p align="center"><sub>
  <b>Figure 1</b> · Controlled architecture ablation: training cross-entropy curves for Transformer / MoE / CNN / PMNN under the same tokenizer, same corpus, and same number of steps. PMNN (red line, ~9M) achieves the lowest training loss, converging to ~3.8–4.0 and outperforming the parameter-matched Transformer (~9M) and CNN (~9M).
  <br/><b>Source file</b>: <code>Image_20260917150553_985_2.png</code>
</sub></p>

**Key observations**:

- All six architectures (4M–9M params) start at CE ≈ 9 and drop rapidly in the first ~250 steps;
- **PMNN (~9M) achieves the lowest training loss throughout**, converging to the 3.8–4.0 band;
- Under parameter-matched comparison, PMNN beats the same-scale Transformer (~9M) and CNN (~9M);
- After step 500 the curves show fluctuation, related to small batches / learning-rate scheduling — this is normal.

> ⚠️ **Interpretation boundary**: training CE reflects **architectural optimization ease**, not generalization. For the generalization comparison see [§4.2](#42-cross-architecture-bpb-benchmark).

### 4.2 Cross-Architecture BPB Benchmark

**Figure 2: Cross-architecture bits-per-byte (BPB) benchmark** (left: dialogue domain, in-distribution; right: instruction domain, out-of-distribution)

<!-- Figure slot 2 -->
<p align="center">
  <img src="docs/images/fig_bpb_benchmark.png" alt="Cross-architecture BPB benchmark" width="95%"/>
</p>
<p align="center"><sub>
  <b>Figure 2</b> · Cross-architecture BPB comparison (tokenizer-agnostic fair metric; lower is better). PMNN 300M achieves <b>0.812 BPB</b> on the dialogue domain (best in-distribution) but degrades to 1.611 BPB on the instruction domain (worst OOD); Qwen2.5-0.5B shows the opposite pattern — 1.535 on dialogue / 0.691 on instruction.
  <br/><b>Source file</b>: <code>Image_20260917150555_986_2.png</code>
</sub></p>

| Model | Parameters | Dialogue BPB ↓ | Instruction BPB ↓ | Notes |
|---|---|---|---|---|
| **PMNN 300M (this work)** | 208 M | **0.812** ✅ | 1.611 ⚠️ | Best in-dist / worst OOD |
| Qwen2.5-0.5B dense | 494 M | 1.535 | **0.691** ✅ | Best OOD |
| Qwen3.5-hybrid (linear attn + conv) | 752 M | 1.602 | 0.760 | — |
| Mamba-370M SSM | 372 M | 2.002 | 1.313 | — |
| GPT2-Chinese dense | 102 M | 4.559 ⚠️ | 4.266 ⚠️ | Worst overall |

**Key insight**: Figure 2 reveals a clear **generalization–specialization trade-off**. PMNN leads significantly in-distribution (dialogue) but degrades most severely out-of-distribution (instruction); Qwen2.5-0.5B shows the opposite pattern. This indicates PMNN's current architecture exhibits a "strong memorization, weak transfer" character on small data — exactly the issue that [§9](#9-data-pipeline--sequence-length-analysis) and the knowledge-distillation track aim to address.

### 4.3 Scaling Law

**Figure 3: Parameter scaling law** (left: real checkpoints; right: compute-matched controlled sweep)

<!-- Figure slot 3 -->
<p align="center">
  <img src="docs/images/fig_scaling_law.png" alt="Scaling laws: parameters vs held-out NLL" width="95%"/>
</p>
<p align="center"><sub>
  <b>Figure 3</b> · Scaling-law verification. <b>Left</b>: real checkpoints (97M @ 14,856 steps → NLL ≈ 4.4; 208M @ 11,500 steps → NLL ≈ 2.5), with untrained models as random baseline anchors. <b>Right</b>: compute-matched controlled sweep shows a clean power-law decay <code>log(NLL) ~ -0.068 · log(N)</code>, confirming that "larger parameters → lower held-out loss under the same compute" holds for PMNN.
  <br/><b>Source file</b>: <code>Image_20260917150600_990_2.png</code>
</sub></p>

**Left panel** (real checkpoints) reflects the two actual training nodes: 97M (14,856 steps) and 208M (11,500 steps), NLL drops from 4.4 to 2.5; gray anchors are untrained models of the same parameter count (step 30), NLL ≈ 8.4–8.8, serving as random baselines.

**Right panel** (compute-matched controlled sweep) is the more rigorous verification: fixed corpus, tokenizer, and step count, only varying parameter scale (3.8M → 16.2M); a smooth power-law decay is observed, with fitted slope **−0.068**. This proves PMNN exhibits standard scaling-law behavior — a prerequisite any serious architecture must satisfy.

---

## 5. Internal Physics Diagnostics

> This section showcases PMNN's **physical interpretability**, which distinguishes it from traditional neural networks — layer-by-layer tracking of the order parameter `r(t)`, amplitude entropy `S`, XY energy `E`, and free energy `F = E − T·S`. All quantities are expanded on physical time `t = layer_index / (n_blocks−1)`.

### 5.1 Layer-wise Physics Trajectory (step 5000)

**Figure 4: PMNN 300M step 5000 layer-wise physics trajectory** (top: order parameter; bottom: thermodynamic quantities)

<!-- Figure slot 4 -->
<p align="center">
  <img src="docs/images/fig_physics_step5000.png" alt="Layer-wise physics trajectory at step 5000" width="75%"/>
</p>
<p align="center"><sub>
  <b>Figure 4</b> · step 5000 physics diagnostics. <b>Top</b>: order parameter r evolves across layers, from 0.406 at the first layer → 0.545 at the last (Δr = +0.139), with the lowest point r ≈ 0.03 near layer 6 (chaotic-exploration regime), then a sharp rise toward synchrony at the last layer. <b>Bottom</b>: amplitude entropy S (blue) peaks at ~6.8 in layers 11–12 then drops back to 4.6; XY energy E (green) is approximately conserved throughout (−0.2 to −0.4); free energy F = E − T·S (purple) rises monotonically from −3.2 to −0.7, matching the physical intuition of "cool exploration in early layers → warm integration in late layers".
  <br/><b>Source file</b>: <code>Image_20260917150557_988_2.png</code>
</sub></p>

**Physical interpretation**: Figure 4 validates the "order-parameter curriculum" from design doc §7 — early layers occupy the chaotic-exploration regime (r ≈ 0), late layers enter the synchrony-integration regime (r → 0.5+), and middle layers carry the heaviest information-mixing load (where entropy peaks). The "first dip then rise" shape of the free-energy curve is the hallmark of an open system relaxing from a non-equilibrium state toward quasi-equilibrium.

### 5.2 No-Collapse Verification (step 11500)

**Figure 5: PMNN 300M step 11500 full-synchronization collapse check** (left: order parameter vs collapse threshold; right: thermodynamic trajectories)

<!-- Figure slot 5 -->
<p align="center">
  <img src="docs/images/fig_physics_step11500.png" alt="Collapse threshold verification at step 11500" width="95%"/>
</p>
<p align="center"><sub>
  <b>Figure 5</b> · step 11500 physics diagnostics. <b>Left</b>: order parameter r stays well below the r = 0.99 full-synchrony collapse threshold (red dashed line) across all layers, peaking at only ~0.47 — proving that the amplitude channel + heterogeneous intrinsic frequencies + sparse coupling effectively prevent information homogenization. <b>Right</b>: amplitude entropy S (blue) peaks at ~7 in layers 11–12 then drops sharply to ~3 at the last layer; free energy F (green) rises from −2.2 to −0.7; energy E (orange) is approximately conserved near 0.
  <br/><b>Source file</b>: <code>Image_20260917150559_989_2.png</code>
</sub></p>

**Key conclusion**: Figure 5 is one of the core pieces of evidence for the **correctness** of PMNN's physical design. The foremost risk listed in design doc §11 is "full-synchronization collapse" — when r → 1, all oscillator phases converge, individual information is wiped out, and the model degenerates to a constant output. This figure proves that at step 11500 (significantly trained), r stays below 0.5 throughout, with ample margin to the 0.99 collapse threshold. This is thanks to three anti-collapse mechanisms:

1. **Amplitude channel preserves individual information** (`a ≥ 0` guaranteed by softplus);
2. **Heterogeneous intrinsic frequencies** `ω_i` prevent frequency locking;
3. **Sparse / local coupling** (drawing on the UN-0 conclusion).

---

## 6. Quick Start

### Local environment (RTX 3050 4 GB is enough)

```bash
cd AetherMind-V5/pmnn_text
PY=C:/Python312/python.exe

# ① Preprocessing: streaming tokenize to disk (constant memory, a few minutes)
$PY prepare_data.py --max_seq_len 128

# ② Smoke test: no data reads, validates forward/backward/generation/flow matching
$PY smoke_test.py --preset tiny

# ③ Training (preset and step count)
$PY train.py --preset small --steps 3000 --log_every 20

# ④ Generation
$PY generate.py --prompt "你好呀，今天天气怎么样"

# ⑤ Plot curves
$PY plot_train.py
```

**One-liners**:

| Platform | Command |
|---|---|
| Windows | `scripts\run_local.bat 100m 3000` |
| Linux cloud | `bash scripts/run_cloud.sh` |

> 💡 **First run?** Validate tier-by-tier in the order `tiny` → `small` → `100m`; at each tier, probe VRAM first with `--dry_run` (AdamW's momentum state is lazily allocated, so `--dry_run` must include one optimizer step to be accurate).

---

## 7. Directory Structure & Path Conventions

> 📌 This repo uses a **flat root structure**: all Python sources, scripts, and docs sit directly at the repo root; only image assets live under `docs/images/`. This way the GitHub landing page lists every file at a glance, without drilling down.

```
AetherMind-V5/                            ← GitHub repo root
├── README.md                             ← This file (English, primary entry)
├── README_CN.md                          ← Chinese README (mirror of this file)
├── PMNN README.md                        ← Legacy PMNN overview (kept for reference)
├── TECH_REPORT.md                        ← Technical report
├── CLOUD_GUIDE.md                        ← Cloud deployment guide (KD package steps 1–9)
├── PMNN_模型架构详尽网络层通用底层架构 (1).md   ← ⭐ v0.1 design doc, Chinese (click to view)
├── PMNN_Architecture_Design_v0.1_…       ← Design doc, English version
├── LICENSE                               ← CC_BY_4.0
├── requirements.txt                      ← Python dependencies
│
├── paths.py                              ← Path-resolution layer (CLI > env var > auto-detect)
├── config.py                             ← Hyperparameters + scale presets
├── data_utils.py                         ← Vocabulary + memmap dataset
├── prepare_data.py                       ← Preprocessing: jsonl → tokens.bin
├── train.py                              ← Training entry point
├── generate.py                           ← Inference / generation
├── plot_train.py                         ← Training-curve visualization
├── smoke_test.py                         ← Smoke test (does not read data)
├── chat.py                               ← Interactive chat demo
├── convert_sft.py                        ← SFT data-format conversion
├── diagnose.py                           ← Physics / training diagnostics
├── _env_check.py                         ← Environment self-check
├── _memtest.jsonl                        ← Memory-test sample
│
└── docs/
    └── images/                           ← All figures referenced by README
        ├── fig_arch_ablation.png         ← Fig 1 · Controlled architecture ablation
        ├── fig_bpb_benchmark.png         ← Fig 2 · Cross-architecture BPB benchmark
        ├── fig_scaling_law.png           ← Fig 3 · Parameter scaling law
        ├── fig_physics_step5000.png      ← Fig 4 · step 5000 layer-wise physics trajectory
        ├── fig_physics_step11500.png     ← Fig 5 · step 11500 no-collapse verification
        └── fig_seqlen_intact.png         ← Fig 6 · Sequence-length sensitivity
```

> 💡 **Design-doc entry point**: [`PMNN_模型架构详尽网络层通用底层架构 (1).md`](./PMNN_模型架构详尽网络层通用底层架构%20%281%29.md) at the repo root is the v0.1 formal spec. Clicking the filename on the GitHub web UI renders it in-place (formulas, tables, ASCII art and all). The English counterpart is `PMNN_Architecture_Design_v0.1_Professional_…`.

**Parameterized paths**: every script resolves paths through three levels (CLI > env var > auto-detect), so zero code changes are needed in the cloud.

| Environment variable | Meaning | Default |
|---|---|---|
| `PMNN_ROOT` | Project root | Directory containing the script |
| `PMNN_DATA_DIR` | Raw jsonl directory | Auto-search upward for `datasets/03_dialogue_clean` |
| `PMNN_OUT_DIR` | Output root | Project root |
| `PMNN_TOKENIZER` | Vocabulary path | `<out>/data/tokenizer.json` |
| `PMNN_PREPARED_DIR` | Preprocessed artifacts | `<out>/prepared` |
| `PMNN_CKPT_DIR` | Checkpoint path | `<out>/checkpoints` |
| `PMNN_LOG_DIR` | Log path | `<out>/logs` |

Command-line flags of the same name (`--data_dir`, etc.) take the highest priority.

---

## 8. Scale Presets & Measured VRAM

Measured on 4 GB VRAM (batch=4, seq=128, including optimizer state):

| Preset | Parameters | Optimizer | Peak VRAM | Use case |
|---|---|---|---|---|
| `tiny` | 2.7 M | AdamW | ~0.05 GB | Smoke test |
| `small` | 16.2 M | AdamW | 0.33 GB | Local default; can be pushed to batch 32+ |
| `100m` | 88.2 M | AdamW | 1.67 GB | **Workhorse for local architecture validation** |
| `300m` | 187.6 M | AdamW | 3.49 GB (87%) | Tight, not recommended |
| `300m` | 187.6 M | **Adafactor** | **2.24 GB (56%)** | ✅ Recommended config |
| `1b` | ~1.1 B | Adafactor | — | Cloud only (~100 GPU-hours) |

**VRAM pre-flight** — probe before a real run to avoid OOM:

```bash
$PY train.py --preset 100m --dry_run --batch 4
```

> ⚠️ `--dry_run` runs a full "forward + backward + one optimizer step" — because AdamW's momentum state is **lazily allocated**, running only forward and backward severely underestimates VRAM.

---

## 9. Data Pipeline & Sequence-Length Analysis

### 9.1 Dataset statistics (`datasets/03_dialogue_clean`, 60 jsonl files)

| Metric | Value |
|---|---|
| File size | 250 MB |
| Sample count | 237,707 |
| Total tokens (`max_seq_len=128`) | **28.47 M** |
| Average sample length | 105.8 tokens |
| Length distribution | p50 = 106 / p90 = 181 / p99 = 282 / max = 2952 |
| Retention at `max_len=128` | 85.7% |
| Retention at `max_len=256` | **99.0%** |

### 9.2 Sequence length vs sample-intact rate

**Figure 6: Sample-intact rate as a function of `max_seq_len`** (based on the corpus's true length distribution)

<!-- Figure slot 6 -->
<p align="center">
  <img src="docs/images/fig_seqlen_intact.png" alt="Sample intact rate vs max_seq_len" width="75%"/>
</p>
<p align="center"><sub>
  <b>Figure 6</b> · Sequence-length sensitivity analysis. The current <code>max_seq_len = 128</code> (red dashed line) keeps only 19% of samples intact; raising it to <code>max_seq_len = 512</code> (green dashed line) retains 97%; 768 tokens reaches 100%. The intermediate 256 tokens retains 54%.
  <br/><b>Source file</b>: <code>Image_20260917150556_987_2.png</code>
</sub></p>

**Free win**: raising `--max_seq_len` from 128 to 256 yields **≈15%** more tokens (28.47M → ≈32.9M) with almost no additional truncation loss. Raising it to 512 essentially eliminates truncation — the most cost-effective data-expansion lever.

### 9.3 Parameter/token ratio (Chinchilla reference: 20 tokens/parameter)

| Preset | Parameters | Recommended tokens | Local 28.5M coverage | Shortfall |
|---|---|---|---|---|
| `small` | 16 M | 0.32 B | 8.9% | 11× |
| `100m` | 90 M | 1.81 B | 1.6% | **64×** |
| `300m` | 210 M | 4.20 B | 0.7% | 147× |
| `1b` | 1.1 B | 22.0 B | 0.13% | **772×** |

**Conclusions**:

1. The local 28.5M tokens are only enough to **validate architectures up to 100M** — which is exactly the entire goal of the current stage.
2. To train 1B without severe overfitting, you need on the order of **~20B tokens (~200 GB of raw text)**. ModelScope's UltraData series (or any multi-source mixed corpus) is not an "optional optimization" but a **prerequisite** for 1B.
3. The only outcome of forcing 1B onto small data is: training CE keeps dropping while validation CE rebounds early — the model simply memorizes these 230k dialogues.
4. For reference: even under compute constraints, training 1B on 5–10B tokens (about 1/4 of Chinchilla) is far more realistic than training 1B on 0.03B tokens.

---

## 10. Cloud Deployment (1B, Zero Code Changes)

```bash
export PMNN_DATA_DIR=/data/03_dialogue_clean
export PMNN_OUT_DIR=/workspace/pmnn_out

# Larger datasets call for a larger vocabulary and longer sequences
python prepare_data.py --max_seq_len 512 --vocab_size 32768
python train.py --preset 1b --steps 300000 \
    --batch 16 --grad_accum 8 --num_workers 8 --optimizer adafactor
```

Or simply:

```bash
PMNN_DATA_DIR=/data/dialogue PMNN_OUT_DIR=/workspace/out bash scripts/run_cloud.sh
```

---

## 11. Key Engineering Decisions

This project went through multiple rounds of architecture-level fixes while landing the design doc. Below are the decisions with the largest impact.

### 11.1 The data pipeline was moved to on-disk memmap (fixing a memory crash)

The old implementation tokenized every sample into a `list[int]` held permanently in memory: 200k samples × 128 tokens ≈ several GB of Python objects; with `num_workers=2` under Windows spawn, another copy was made → host-memory exhaustion and a crash.

**New implementation has two stages**:

- **Preprocessing** (`prepare_data.py`): streaming `encode_batch`, written to disk as `tokens.bin` (contiguous uint16/uint32 token stream) + `index.npy` (offset/length) + `meta.json`. Memory stays constant throughout; only the current batch is held.
- **Training**: `np.memmap` read-only mapping, so process memory usage is ≈ 0 and relies on the page cache; multiple workers share the same mapping. `num_workers` defaults to 0 (most stable on Windows).

Measured after preprocessing 3,000 samples: `tokens.bin` 241 KB + `index.npy` 48 KB, and the tokens read back are **bit-for-bit identical** to the original tokenization.

### 11.2 Drift-field einsum rewrite

The original implementation computed distances with a 4D `[B,N,M,D]` tensor, costing O(N·M·D) memory. Rewriting as `dist² = ‖x‖² + ‖y‖² − 2xyᵀ` (einsum) reduces this to O(N·M + N·D):

| Metric | Before | After | Speedup |
|---|---|---|---|
| Backward time | 10.5 s | **0.46 s** | **23×** |
| Peak VRAM | 0.39 GB | **0.11 GB** | 3.5× |

### 11.3 Mask-leakage fix (correctness)

With left padding, pad tokens were polluting the Kuramoto mean field, the drift-field repulsion term, and the thermodynamic energy/entropy statistics. All of these are now mask-weighted:

- `KuramotoSync`: the mean field and low-rank aggregation are normalized by the number of valid tokens; pad phases are not updated;
- `DriftField._softmax_pull`: pad positions are masked out as attraction centers;
- `Thermodynamics.energy/entropy`: mask-weighted averaging;
- `order_parameter`: gained a new `mask` argument.

### 11.4 Cross-position information pathway: two rounds of architecture-level fixes

Language modeling requires position `t` to be able to "retrieve by content" the tokens before it. The original architecture could not do this — the single most central architectural fix in the project, which took two rounds.

#### Round one (did not work): cumsum distance-decay coupling

**Problem**: the original cross-position interaction consisted only of a "global long-range mean field" and a "global low-rank aggregation", both **permutation-invariant** — position `t` could not learn "what specific word precedes it", so the model could only learn to copy its own input.

> Measured: with `labels = input_ids` and no right shift, CE was pushed from 9.4 down to **0.01** — looked like excellent convergence but was pure cheating; once labels were corrected to a one-position right shift, CE got stuck at **5.16** and would not go lower.

**Approach**: add exponentially decaying causal memory to the low-rank aggregation: `A_i = Σ_{j≤i} e^{−λ(i−j)} · v_j = e^{−λi} ⊙ cumsum(e^{λj} ⊙ v_j)`, complexity O(N·r), naturally causal.

**Result**: A/B comparison (same seed, same data, 400 steps) gave CE 5.1596 (ON) vs 5.1622 (OFF) — a 0.05% difference, i.e. no difference at all. **This round failed.**

#### Root cause: `k ⊙ agg` is not a query-key interaction at all

Looking back at the original implementation:

```python
k = self.cpl_key(phi_sin)            # [B,N,r]
agg = (v.sum(dim=1, keepdim=True))   # [B,1,r] or causal version [B,N,r]
lr_coupling = self.cpl_out(k * agg)  # ← Hadamard product, not a dot product!
```

`agg` is merely an **indiscriminate weighted average** of past `v`, with weights fixed at `e^{−λ(i−j)}` that are **completely independent of `k` (its own key)**. So `k * agg` only acts as an element-wise scaling gate; the model has **no content-addressing capability whatsoever**. This explains why the A/B test showed no difference and why CE stayed stuck at 3.59.

#### Round two (current approach): content-addressed causal coupling

Let the coupling strength be determined **by the oscillator states themselves** — exactly Kuramoto's original intent (only phase-locked oscillators couple strongly), and also the original form written in `docx/理论.txt`: `θ̇ᵢ = ωᵢ + (K/N)·ΣWᵢⱼsin(θⱼ−θᵢ)` (note: this is a **pairwise sum**, not a mean-field approximation):

```
W_ij = softmax_j( q_i·k_j / √r − λ·(i−j) ),  i ≥ j
output = W @ sin(θ)
```

- `q, k` are low-rank projections of the complex field `z = a·e^{iθ}` → **coupling strength is content-determined**;
- the `−λ(i−j)` term simultaneously provides light-cone causality and long-range decay;
- softmax is computed in fp32 (the `[B,N,N]` dot product in bf16 easily loses precision), and masked entries use `-1e4` rather than `-inf` (so softmax does not produce NaN when an entire row is masked).

**Configuration**: `--coupling {mean_field, content}` (default `content`), `--coupling_halflife` (default 64), `--coupling_temp` (default 1.0).

### 11.5 Labels for causal language modeling must be shifted right by one

```python
# Anti-pattern: labels copy the input directly → the model only has to copy itself to push CE to 0
labels = input_ids.clone()

# Correct
labels = input_ids.clone()
labels[:, :-1] = input_ids[:, 1:]   # target at position t = token at position t+1
labels[:, -1]   = -100
labels[input_ids == pad_id] = -100
```

This is the most insidious class of bug: **the loss curve looks beautiful, but the model has learned nothing**.

### 11.6 Physical-invariant fixes

- After LayerNorm, amplitude could become **negative** (physically, amplitude must be ≥ 0) → changed to `softplus(norm(a + ffn))`;
- Phase was also being normalized by LayerNorm, which **destroyed the [−π, π] periodic structure** → removed; phase is now maintained solely by Kuramoto's internal wrap.

`smoke_test.py` asserts both of these (amplitude non-negative, order parameter ∈ [0,1]).

### 11.7 Langevin noise wired in (the η(t) term from design doc §3.4)

`Thermodynamics.langevin_noise()` was originally **dead code** — defined but never called. The injection point is now in the integration step of `KuramotoSync.forward`:

```
θ_next = θ + dt·f(θ) + √(2T·dt)·N(0,1)
```

> ⚠️ The key point is that the standard deviation must be `√(2T·dt)` and not `√(2T)`: that is the result of discretizing the continuous-time Langevin equation `dθ = f·dt + √(2T)·dW` (with `dW ~ N(0, dt)`). Dropping `dt` makes the noise magnitude scale incorrectly with the step size. The temperature `T(t)` anneals toward `T_min` over physical time, so the noise naturally tends to 0 later on (cooling).

**Injected only during training** (`self.training`); at inference time, randomness is the job of the sampling temperature. Switches: `--langevin` (off by default) / `--langevin_gain`.

Four checks (measured):

| Config | Difference between two forward passes |
|---|---|
| Off | 0 |
| On + train | 1.6e+0 |
| On + eval | 0 |
| gain = 4 | 2.2e+0 |

### 11.8 The 128-dim trajectory bottleneck wired in (the core ELF trick from design doc §6)

`self.bottleneck` was also originally **dead code**: it was created and written into `diag`, but no loss ever used it → its parameter gradient was always `None`. After the fix, the velocity field is predicted from the bottleneck:

```
x_t → shared physics core (1 round) → field [2d] → 128-dim bottleneck + LayerNorm → vel_head → [d]
```

**Verification**: `bottleneck.weight` gradient 0.212, `bottleneck_norm.weight` 0.026, `vel_head.weight` 0.844 (all were None before the fix).

**This also cleared out a class of "fake-dead pathways"**: in `content` mode, the mean-field-only pathway `cpl_key/cpl_val/cpl_out` does not participate in the forward pass, producing `3×n_blocks` gradient-free parameters (11 measured) that masquerade as "dead pathways" in a gradient audit and drown out the real problems. These two parameter groups are now created **conditionally** on `coupling` — in both modes, the live parameters have zero dead gradient entries.

### 11.9 Order-parameter curriculum: restored from a "single-point constant" to a "layer-by-layer curve"

Design doc §7 requires `L_sync = ‖r(t) − r_target(t)‖²`, where `r_target(t) = σ((t−t_c)/τ_c)`. The original implementation took the `r` of the **last layer** only, and `t_final` was always equal to 1.0 → the target was a constant `σ((1−0.5)/0.15) = 0.9656`. **The entire "chaotic exploration in early layers → synchronized integration in late layers" K-value navigation curriculum was collapsed into a single point** — the model was only told "the last layer must synchronize", with no constraint on the middle layers at all.

The shape the curriculum should have (100m, `n_blocks=12`):

| Layer i | 0 | 3 | 6 | 9 | 11 |
|---|---|---|---|---|---|
| Physical time t | 0.000 | 0.273 | 0.545 | 0.818 | 1.000 |
| `r_target(t)` | **0.034** | 0.180 | 0.575 | 0.893 | **0.966** |

**Fix**: `--sync_curriculum` now tracks layer by layer. The accompanying change is that `stats_list` gained an `order_raw` field that is **not detached** — previously every statistic was detached, so layer-wise `L_sync` could not obtain gradients. The remaining diagnostics (energy/entropy) are still detached, to prevent diagnostics from accidentally participating in backpropagation. Off by default (to keep comparability with the existing baseline).

---

## 12. Loss Composition

```
L = L_ce + w_fm·L_fm + w_sync·L_sync + w_drift·L_drift + w_free·L_F
```

| Term | Role | Default weight |
|---|---|---|
| `L_ce` | Language-modeling cross-entropy (final-step de-embedding) | 1.0 |
| `L_fm` | ELF flow-matching velocity field (denoising branch, shares the physics-core weights) | 0.1 |
| `L_sync` | Order-parameter curriculum tracking `r → σ((t−t_c)/τ_c)` | 0.1 |
| `L_drift` | Drift-field consistency | 0.05 (currently always 0, pending target embeddings) |
| `L_F` | Free-energy regularization `U − T·S`, with an entropy floor to prevent full-synchronization collapse | 0.05 |

The training log (`logs/train_log.tsv`) also records physical diagnostics: order parameter `r_mean`, amplitude entropy `entropy`, and XY energy `energy`.

---

## 13. Design-Doc Implementation Status

The authoritative reference is [`PMNN_模型架构详尽网络层通用底层架构 (1).md`](./PMNN_模型架构详尽网络层通用底层架构%20%281%29.md) (i.e. the original `docx/PMNN物理矩阵神经网络通用底层架构.md`, now relocated to the repo root). Each item below has been checked against the implementation. Every ❌ item has been grep-verified as genuinely not wired in (not merely "possibly unimplemented").

| Design-doc item | Status | Notes |
|---|---|---|
| §2 Complex-valued oscillator `z = a·e^{iθ}` | ✅ | Real-valued representation with a/θ separated; `a≥0` guaranteed by softplus |
| §3.1 Kuramoto **pairwise** coupling `Σ_j W_ij·sin(θ_j−θ_i)` | ⚠️ fixed this round | The original implementation degenerated into the mean-field approximation `r·sin(ψ−θ_i)`; see [§11.4](#114-cross-position-information-pathway-two-rounds-of-architecture-level-fixes) |
| §3.1 Low-rank coupling `W = UVᵀ` | ✅ | `r_rank` configurable (32/64/128) |
| §3.1 κ(t) synchronization strength (K-value navigation) | ✅ | — |
| §3.2 Resonance–standing-wave FNN (Lorentzian gain + standing-wave modulation) | ✅ | — |
| §3.3 Drift field `V = V⁺ − λV⁻` (multi-temperature softmax) | ✅ | einsum rewrite, 23× faster backward |
| §3.3 Train-time evolution → per-round active at inference | ⚠️ partial | Per-round active during training is implemented |
| §3.4 XY energy / amplitude entropy / free energy `F = U − T·S` | ✅ | All mask-weighted |
| §3.4 **Langevin noise** `η ~ N(0, 2T·Δt)` | ✅ wired in | Implemented in the integration step of `KuramotoSync.forward`, **injected during training only**; enabled via `--langevin` (off by default, pending A/B) |
| §3.5 Heun/RK4 reversible integration | ⚠️ explicit Euler | `dt=0.1`; accurate enough but does not save VRAM |
| §5 `∇_Z[−U + T·S]` free-energy gradient | ✅ amplitude component added | The phase component is handled by KuramotoSync (`∇_θU` = sin coupling); the amplitude component `∇_a[−U+T·S]` is newly added as `free_energy_amplitude_gradient`, enabled by `--free_energy_amp` (off by default, pending A/B) |
| §6 **128-dim trajectory bottleneck** | ✅ wired in | The velocity field is now predicted from the bottleneck; `bottleneck` parameter gradient 0.212 (was None before the fix) |
| §6 **mode gate** `g(t)=σ(W_g·[t, r(t), κ(t)])`, `t_dec≈0.98` | ✅ implemented | Enabled via `--mode_gate`; weights `L_ce` and `L_fm` by thermodynamic state (off by default, pending A/B) |
| §6 ELF flow matching `x_t = t·x₁ + (1−t)x₀` | ✅ | Rectified flow + velocity-field target |
| §6 Shared weights (physics core doubles as the velocity-field network) | ✅ | `flow_matching` reuses `blocks[0].sync` |
| §7 `L = L_fm + λ₁L_sync + λ₂L_drift + λ₃L_F` | ✅ + `L_ce` | Text tasks add language-modeling cross-entropy on top |
| §7 `L_sync` order-parameter curriculum `r(t)` tracked layer by layer | ⚠️ → ✅ | The original implementation constrained only the last layer with a constant target of 0.9656; `--sync_curriculum` restores the layer-wise curve |
| §7 K curriculum `κ(t) = κ_max·σ((t−t_c)/τ)` | ✅ | — |
| §7 Temperature annealing `T(t) = T_max(1−t)^p + T_min` | ✅ | — |
| §7 Two-stage training (A: pre-train the core / B: end-to-end flow matching) | ❌ not staged | Currently single-stage joint training |
| §8 Inference: forward ODE integration + per-round active drift field | ⚠️ deviates | Currently uses "physics-core encoding + autoregressive decoding" to preserve text controllability |

**Remaining gaps** (by priority):

1. **Two-stage training (§7)** — Stage A pre-trains the physics core / Stage B does end-to-end flow matching; currently a single joint stage.
2. **Heun/RK4 integration (§3.5)** — currently explicit Euler (`dt=0.1`); the reversible VRAM-saving capability is not yet realized.

---

## 14. Knowledge-Distillation Cloud Package

> **📦 What is this directory for? (read this first)**
>
> This directory (which unpacks to `pmnn_text_kd/`) is the **standalone cloud package for PMNN ← MiniCPM5-2B knowledge distillation**. It is a separate track from the `pmnn_text/` of the 300M pre-training round — do not confuse the two.

### Entry point & proof of life

- **Read `CLOUD_GUIDE.md` inside the package first** (follow steps 1–9 in order; it includes a symptom-to-cause table and a "never do this" section).
- **Single entry point**:

  ```bash
  bash scripts/run_cloud_kd.sh
  # STAGE=where|probe|prepare|losses|smoke|l1|l2|all
  ```

- **Proof that the patch is live**: the log must begin with `[cloud-kd] 补丁版本: kd-guard vN (日期)`. If it is missing, you are running the old script.

### Why distillation works despite completely different architectures

See the first section of `PMNN_蒸馏方案_MiniCPM5教师.md` (if present; otherwise see §1 of `CLOUD_GUIDE.md`).

> **Core thesis**: the `w` computed inside PMNN's `KuramotoSync` is isomorphic to Transformer attention, so attention alignment needs no projection layer.

> 📌 Sections 1–13 above are the engineering notes from the **300M pre-training** round, kept as background material on the model itself.

---

## 15. FAQ

**Q1: Not enough VRAM?**
Reduce `--batch`, or use `--optimizer adafactor`, or lower `--max_seq_len`.

**Q2: Data directory not found?**
Specify `--data_dir`, or set `PMNN_DATA_DIR`. The scripts automatically search upward for a `datasets` directory.

**Q3: I changed `--max_seq_len` but it had no effect?**
When the `max_len` of the preprocessed artifacts is smaller than required, preprocessing is re-run automatically; you can also force it with `--rebuild_prepared`.

**Q4: How do I change the vocabulary size?**
`prepare_data.py --vocab_size 32768 --rebuild_tokenizer`, then re-run preprocessing. The model's `vocab_size` automatically aligns with the actual vocabulary size.

**Q5: Training CE drops fast but generation quality is poor?**
Most likely the [§11.5](#115-labels-for-causal-language-modeling-must-be-shifted-right-by-one) labels-not-right-shifted bug — the model is copying its input. Check how `labels` is constructed in `train.py`.

**Q6: NaN in physical quantities?**
Check [§11.6](#116-physical-invariant-fixes) — is amplitude guaranteed non-negative via softplus? Is phase being wrongly normalized by LayerNorm? Is the Kuramoto softmax mask using `-1e4` instead of `-inf`?

---

## 16. Citation & License

**License**: CC_BY_4.0

**Datasets**:

- `DreamStarPaint/MOSS`
- `Moemuu/Muice-Dataset`

**Framework**: PyTorch

**Languages**: Chinese / English

**Suggested citation** (BibTeX, placeholder — replace upon formal publication):

```bibtex
@misc{pmnn2026,
  title  = {PMNN: Physics Matrix Neural Network — General-Purpose Substrate Architecture},
  author = {xuanxixue},
  year   = {2026},
  note   = {v0.1, experimental validation stage},
  url    = {https://modelscope.cn/models/xuanxixue/AetherMind-V5-300M-Instruct}
}
```

---

## 17. References

1. He, K. *ELF: Embedded Language Flows*. arXiv:2605.10938, 2026. <https://arxiv.org/pdf/2605.10938>
2. He, K. et al. *Generative Modeling via Drifting*. arXiv:2602.04770, 2026. <https://arxiv.org/pdf/2602.04770>
3. Unconventional AI. *Un-0: Image Generation via Kuramoto Dynamics*. GitHub, 2026. <https://github.com/unconv-ai/un-0>

---

## Appendix: Figure Attribution

This README references 6 figures, all under the repo's `docs/images/` directory (when uploading, rename the original files per the table below and place them in that directory).

| Fig | Suggested in-repo path | Original upload filename | Description | Section |
|---|---|---|---|---|
| Fig 1 | `docs/images/fig_arch_ablation.png` | `Image_20260917150553_985_2.png` | Controlled architecture ablation training curve (Transformer/MoE/CNN/PMNN) | §4.1 |
| Fig 2 | `docs/images/fig_bpb_benchmark.png` | `Image_20260917150555_986_2.png` | Cross-architecture BPB benchmark (dialogue vs instruction) | §4.2 |
| Fig 3 | `docs/images/fig_scaling_law.png` | `Image_20260917150600_990_2.png` | Parameter scaling law (real checkpoints + compute-matched sweep) | §4.3 |
| Fig 4 | `docs/images/fig_physics_step5000.png` | `Image_20260917150557_988_2.png` | PMNN 300M step 5000 layer-wise physics trajectory | §5.1 |
| Fig 5 | `docs/images/fig_physics_step11500.png` | `Image_20260917150559_989_2.png` | PMNN 300M step 11500 no-collapse verification | §5.2 |
| Fig 6 | `docs/images/fig_seqlen_intact.png` | `Image_20260917150556_987_2.png` | Sequence length vs sample-intact-rate curve | §9.2 |

> **Usage steps**:
> 1. Create the `docs/images/` directory at the repo root;
> 2. Copy the 6 original PNGs in by the "original upload filename" column and rename them per the "in-repo path" column;
> 3. The `<img>` tags in the README will then resolve correctly.
