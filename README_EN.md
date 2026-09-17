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
tasks:
- text-generation
language:
  - zh
  - en
studios:
- xuanxixue/AetherMind-V5-300M-Instruct
---
# PMNN Text Generation Model (Physics Matrix Neural Network · General-Purpose Substrate Architecture)

<p align="center">
<a href="https://modelscope.cn/studios/xuanxixue/AetherMind-V5-300M-Instruct" target="_blank">🚀 Online Demo</a> |
<a href="https://modelscope.cn/models/xuanxixue/AetherMind-V5-300M-Instruct" target="_blank">📦 Model</a>
</p>

A trainable text model built according to `docx/PMNN物理矩阵神经网络通用底层架构.md`
(the v0.1 formal specification). It is positionally analogous to a Transformer but
**non-statistical**: Kuramoto phase synchronization replaces attention,
a resonance–standing-wave FNN replaces the feed-forward network, and a drift field +
thermodynamics + time field act as explicit physical constraints. The output stage
fuses ELF continuous flow matching.

Note: we have only just passed the experimental validation stage.

---

> **📦 What is this directory for? (read this first)**
>
> This directory (which unpacks to `pmnn_text_kd/`) is the **standalone cloud package for
> PMNN ← MiniCPM5-2B knowledge distillation**. It is a separate track from the `pmnn_text/`
> of the 300M pre-training round — do not confuse the two.
>
> - **Read `CLOUD_GUIDE.md` inside the package first** (follow steps 1–9 in order; it includes
>   a symptom-to-cause table and a "never do this" section).
> - Single entry point: `bash scripts/run_cloud_kd.sh`, with
>   `STAGE=where|probe|prepare|losses|smoke|l1|l2|all`.
> - **Proof that the patch is live**: the log must begin with
>   `[cloud-kd] 补丁版本: kd-guard vN (日期)`. If it is missing, you are running the old script.
> - Why distillation works despite completely different architectures → see the first section of
>   `docx/PMNN_蒸馏方案_MiniCPM5教师.md`
>   (**the `w` computed inside PMNN's `KuramotoSync` is isomorphic to Transformer attention,
>   so attention alignment needs no projection layer**).
>
> Sections 1–9 below are the engineering notes from the **300M pre-training** round, kept as
> background material on the model itself.

---

## 1. Directory and Path Conventions

```
D:\AetherMind-V5\                 ← workspace root
├── AetherMind-V5\
│   └── pmnn_text\                ← this project (code lives only here; never mixed with docx/datasets)
│       ├── paths.py              path-resolution layer (CLI > env var > auto-detect)
│       ├── config.py             hyperparameters + scale presets
│       ├── data_utils.py         vocabulary + memmap dataset
│       ├── prepare_data.py       preprocessing: jsonl → tokens.bin
│       ├── train.py              training
│       ├── generate.py           generation
│       ├── plot_train.py         curve visualization
│       ├── smoke_test.py         smoke test (does not read data)
│       └── scripts\              launch scripts (.bat pure ASCII / .sh)
├── datasets\                     data area
└── docx\                         documentation area
```

**Parameterized paths**: every script resolves paths through three levels, so zero code
changes are needed in the cloud.

| Environment variable | Meaning | Default |
|---|---|---|
| `PMNN_ROOT` | Project root | Directory containing the script |
| `PMNN_DATA_DIR` | Raw jsonl directory | Auto-search upward for `datasets/03_dialogue_clean` |
| `PMNN_OUT_DIR` | Output root | Project root |
| `PMNN_TOKENIZER` | Vocabulary | `<out>/data/tokenizer.json` |
| `PMNN_PREPARED_DIR` | Preprocessed artifacts | `<out>/prepared` |
| `PMNN_CKPT_DIR` | Checkpoints | `<out>/checkpoints` |
| `PMNN_LOG_DIR` | Logs | `<out>/logs` |

Command-line flags of the same name (`--data_dir`, etc.) take the highest priority.

---

## 2. Quick Start (local RTX 3050, 4 GB)

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

One-liner on Windows: `scripts\run_local.bat 100m 3000`
One-liner on Linux cloud: `bash scripts/run_cloud.sh`

---

## 3. Scale Presets and Measured VRAM

Measured on 4 GB VRAM (batch=4, seq=128, including optimizer state):

| Preset | Parameters | Optimizer | Peak VRAM | Recommendation |
|---|---|---|---|---|
| `tiny` | 2.7M | AdamW | ~0.05 GB | Smoke test |
| `small` | 16.2M | AdamW | 0.33 GB | Local default; can be pushed to batch 32+ |
| `100m` | 88.2M | AdamW | 1.67 GB | **Workhorse for local architecture validation** |
| `300m` | 187.6M | AdamW | 3.49 GB (87%) | Too tight |
| `300m` | 187.6M | Adafactor | 2.24 GB (56%) | ✅ Recommended |
| `1b` | ~1.1B | Adafactor | — | Cloud only (roughly 100 GPU-hours) |

Always probe VRAM before a real run:

```bash
$PY train.py --preset 100m --dry_run --batch 4
```

`--dry_run` runs a full "forward + backward + one optimizer step" — because AdamW's momentum
state is **lazily allocated**, running only forward and backward severely underestimates VRAM.

---

## 4. Running 1B in the Cloud (without changing a single line of code)

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

## 5. Key Engineering Decisions

### 5.1 The data pipeline was moved to on-disk memmap (fixing a memory crash)

The old implementation tokenized every sample into a `list[int]` held permanently in memory:
200k samples × 128 tokens ≈ several GB of Python objects; with `num_workers=2` under Windows
spawn, another copy was made → host memory exhaustion and a crash.

The new implementation has two stages:

- **Preprocessing** (`prepare_data.py`): streaming `encode_batch`, written to disk as
  `tokens.bin` (a contiguous uint16/uint32 token stream) + `index.npy` (offset/length) +
  `meta.json`. Memory stays constant throughout; only the current batch is held.
- **Training**: `np.memmap` read-only mapping, so process memory usage is ≈ 0 and relies on
  the page cache; multiple workers share the same mapping. `num_workers` defaults to 0
  (the most stable option on Windows).

Measured after preprocessing 3,000 samples: `tokens.bin` 241 KB + `index.npy` 48 KB, and the
tokens read back are **bit-for-bit identical** to the original tokenization.

### 5.2 Drift-field einsum rewrite

The original implementation computed distances with a 4D `[B,N,M,D]` tensor, costing
O(N·M·D) memory. Rewriting as `dist² = ‖x‖² + ‖y‖² − 2xyᵀ` (einsum) reduces this to
O(N·M + N·D): backward time 10.5 s → **0.46 s** (23×), peak VRAM 0.39 GB → **0.11 GB**.

### 5.3 Mask-leakage fix (correctness)

With left padding, pad tokens were polluting the Kuramoto mean field, the drift-field
repulsion term, and the thermodynamic energy/entropy statistics. All of these are now
mask-weighted:

- `KuramotoSync`: the mean field and low-rank aggregation are normalized by the number of
  valid tokens; pad phases are not updated;
- `DriftField._softmax_pull`: pad positions are masked out as attraction centers;
- `Thermodynamics.energy/entropy`: mask-weighted averaging;
- `order_parameter`: gained a new `mask` argument.

### 5.4 Cross-position information pathway: two attempts (an architecture-level fix)

Language modeling requires that position t be able to "retrieve by content" the tokens before
it. The original architecture could not do this. This is the single most central architectural
fix in the project, and it took two rounds.

#### 5.4.1 Round one (did not work): cumsum distance-decay coupling

**Problem**: the original design's cross-position interaction consisted only of a "global
long-range mean field" and a "global low-rank aggregation", both of which are
**permutation-invariant** — position t cannot learn "what specific word precedes it", so the
model could only learn to copy its own input.

(Measured: with `labels = input_ids` and no right shift, CE was pushed from 9.4 down to
**0.01**. It looked like excellent convergence but was pure cheating; once labels were
corrected to a one-position right shift, CE got stuck at **5.16** and would not go lower.)

**Approach**: add exponentially decaying causal memory to the low-rank aggregation.

```
A_i = Σ_{j≤i} e^{−λ(i−j)} · v_j = e^{−λi} ⊙ cumsum(e^{λj} ⊙ v_j)
```

Complexity O(N·r), naturally causal, λ = ln2 / `causal_halflife`.

**Result: A/B comparison (same seed, same data, 400 steps) gave CE 5.1596 (ON) vs 5.1622 (OFF)
— a 0.05% difference, i.e. no difference at all. This round failed.**

#### 5.4.2 Root-cause analysis: `k ⊙ agg` is not a query-key interaction at all

Looking back at the original implementation:

```python
k = self.cpl_key(phi_sin)              # [B,N,r]
agg = (v.sum(dim=1, keepdim=True))     # [B,1,r] or, in the causal version, [B,N,r]
lr_coupling = self.cpl_out(k * agg)    # ← Hadamard product, not a dot product!
```

`agg` is merely an **indiscriminate weighted average** of past `v`, with weights fixed at
`e^{−λ(i−j)}` that are **completely independent of `k` (its own key)**. So `k * agg` only acts
as an element-wise scaling gate; the model has **no content-addressing capability whatsoever**.
Whether causality is ON or OFF, all it can obtain is a rank-1 summary of the context.

This explains why the A/B test showed no difference, and why CE stayed stuck at 3.59.

#### 5.4.3 Round two (the current approach): content-addressed causal coupling

Let the coupling strength be determined **by the oscillator states themselves** — which is
exactly Kuramoto's original intent (only phase-locked oscillators couple strongly), and also
the original form written in `docx/理论.txt`:
`θ̇ᵢ = ωᵢ + (K/N)·ΣWᵢⱼsin(θⱼ−θᵢ)`
(note: this is a **pairwise sum**, not a mean-field approximation):

```
W_ij = softmax_j( q_i·k_j / √r  −  λ·(i−j) ),   i ≥ j
output = W @ sin(θ)
```

- `q, k` are low-rank projections of the complex field `z = a·e^{iθ}` → **coupling strength is
  content-determined**;
- the `−λ(i−j)` term simultaneously provides light-cone causality and long-range decay;
- softmax is computed in fp32 (the [B,N,N] dot product in bf16 easily loses precision), and
  masked entries use `-1e4` rather than `-inf` (so softmax does not produce NaN when an entire
  row is masked).

Configuration: `--coupling {mean_field, content}` (default `content`),
`--coupling_halflife` (default 64), `--coupling_temp` (default 1.0).

### 5.5 Labels for causal language modeling must be shifted right by one

```python
# Anti-pattern: labels copy the input directly → the model only has to copy itself to push CE to 0
labels = input_ids.clone()

# Correct
labels = input_ids.clone()
labels[:, :-1] = input_ids[:, 1:]    # target at position t = token at position t+1
labels[:, -1] = -100
labels[input_ids == pad_id] = -100
```

This is the most insidious class of bug: **the loss curve looks beautiful, but the model has
learned nothing.**

### 5.6 Physical-invariant fixes

- After LayerNorm, amplitude could become **negative** (physically, amplitude must be ≥ 0) →
  changed to `softplus(norm(a + ffn))`;
- Phase was also being normalized by LayerNorm, which **destroyed the [-π, π] periodic
  structure** → removed; phase is now maintained solely by Kuramoto's internal wrap.

`smoke_test.py` asserts both of these (amplitude non-negative, order parameter ∈ [0,1]).

### 5.7 Langevin noise wired in (the η(t) term from design doc §3.4)

`Thermodynamics.langevin_noise()` was originally **dead code** — defined but never called.
The injection point is now in the integration step of `KuramotoSync.forward`:

```
θ_next = θ + dt·f(θ) + √(2T·dt)·N(0,1)
```

⚠️ The key point is that the standard deviation must be `√(2T·dt)` and not `√(2T)`: that is the
result of discretizing the continuous-time Langevin equation
`dθ = f·dt + √(2T)·dW` (with `dW ~ N(0, dt)`). Dropping `dt` makes the noise magnitude scale
incorrectly with the step size. The temperature `T(t)` anneals toward `T_min` over physical
time, so the noise naturally tends to 0 later on (cooling).

**Injected only during training** (`self.training`); at inference time, randomness is the job of
the sampling temperature. Switches: `--langevin` (off by default) / `--langevin_gain`.

Four checks (measured): off → two forward passes differ by 0; on + train → differ by 1.6e+0;
on + eval → differ by 0; gain=4 → differ by 2.2e+0 (magnitude scales with the coefficient).

### 5.8 The 128-dim trajectory bottleneck wired in (the core ELF trick from design doc §6)

`self.bottleneck` was also originally **dead code**: it was created and written into `diag`, but
no loss ever used it → its parameter gradient was always `None`. After the fix, the velocity
field is predicted from the bottleneck:

```
x_t → shared physics core (1 round) → field [2d] → 128-dim bottleneck + LayerNorm → vel_head → [d]
```

Verification: `bottleneck.weight` gradient 0.212, `bottleneck_norm.weight` 0.026,
`vel_head.weight` 0.844 (all were None before the fix).

**This also cleared out a class of "fake-dead pathways"**: in `content` mode, the mean-field-only
pathway `cpl_key/cpl_val/cpl_out` does not participate in the forward pass, producing `3×n_blocks`
gradient-free parameters (11 measured) that masquerade as "dead pathways" in a gradient audit and
drown out the real problems. These two parameter groups are now created **conditionally** on
`coupling` — in both modes, the live parameters have zero dead gradient entries.

### 5.9 Order-parameter curriculum: restored from a "single-point constant" to a "layer-by-layer curve"

Design doc §7 requires `L_sync = ‖r(t) − r_target(t)‖²`, where `r_target(t) = σ((t−t_c)/τ_c)`.

The original implementation took the `r` of the **last layer** only, and `t_final` was always
equal to `(n_blocks−1)/(n_blocks−1) = 1.0` → the target was a constant
`σ((1−0.5)/0.15) = 0.9656`. **The entire "chaotic exploration in early layers → synchronized
integration in late layers" K-value navigation curriculum was collapsed into a single point**:
the model was only told "the last layer must synchronize", with no constraint on the middle
layers at all.

The shape the curriculum should have (100m, `n_blocks=12`):

| Layer i | 0 | 3 | 6 | 9 | 11 |
|---|---|---|---|---|---|
| Physical time t | 0.000 | 0.273 | 0.545 | 0.818 | 1.000 |
| `r_target(t)` | **0.034** | 0.180 | 0.575 | 0.893 | **0.966** |

Fix: `--sync_curriculum` now tracks layer by layer. The accompanying change is that `stats_list`
gained an `order_raw` field that is **not detached** — previously every statistic was detached, so
layer-wise `L_sync` could not obtain gradients. The remaining diagnostics (energy/entropy) are
still detached, to prevent diagnostics from accidentally participating in backpropagation.

Off by default (keeping comparability with the existing baseline).

---

## 6. Training Scale vs. Data Volume (essential reading before 1B)

Current local data measurement (`datasets/03_dialogue_clean`, 60 jsonl files):

| Metric | Value |
|---|---|
| File size | 250 MB |
| Number of samples | 237,707 |
| Total tokens (`max_seq_len=128`) | **28.47 M** |
| Average sample length | 105.8 tokens |
| Length distribution | p50=106 / p90=181 / p99=282 / max=2952 |
| Retained at `max_len=128` | 85.7% |
| Retained at `max_len=256` | **99.0%** |

**Free win**: raising `--max_seq_len` from 128 to 256 yields **≈15%** more tokens
(28.47M → ≈32.9M) with almost no additional truncation loss.

### Parameter/token ratio (Chinchilla reference: 20 tokens/parameter)

| Preset | Parameters | Recommended tokens | Local 28.5M coverage | Shortfall |
|---|---|---|---|---|
| `small` | 16 M | 0.32 B | 8.9% | 11× |
| `100m` | 90 M | 1.81 B | 1.6% | **64×** |
| `300m` | 210 M | 4.20 B | 0.7% | 147× |
| `1b` | 1.1 B | 22.0 B | 0.13% | **772×** |

**Conclusions**:

1. The local 28.5M tokens are only enough to **validate architectures up to 100M** — which is
   exactly the entire goal of the current stage.
2. To train 1B without severe overfitting, you need on the order of **~20B tokens (about 200 GB
   of raw text)**. ModelScope's UltraData series (or any multi-source mixed corpus) is not an
   "optional optimization" but a **prerequisite** for 1B.
3. The only outcome of forcing 1B onto small data is this: training CE keeps dropping while
   validation CE rebounds early — the model simply memorizes these 230k dialogues.
4. For reference: even under compute constraints, training 1B on 5–10B tokens (about 1/4 of
   Chinchilla) is far more realistic than training 1B on 0.03B tokens.

---

## 7. Loss Composition (v0.1 §7)

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

The training log (`logs/train_log.tsv`) also records physical diagnostics: order parameter
`r_mean`, amplitude entropy `entropy`, and XY energy `energy`.

---

## 8. Design Doc (v0.1) Implementation Status

The authoritative reference is `docx/PMNN物理矩阵神经网络通用底层架构.md`; each item below has
been checked against the implementation. Every ❌ item has been grep-verified as genuinely not
wired in (not merely "possibly unimplemented").

| Design-doc item | Status | Notes |
|---|---|---|
| §2 Complex-valued oscillator `z = a·e^{iθ}` | ✅ | Real-valued representation with a/θ separated; `a≥0` guaranteed by softplus |
| §3.1 Kuramoto **pairwise** coupling `Σ_j W_ij·sin(θ_j−θ_i)` | ⚠️ fixed this round | The original implementation degenerated into the mean-field approximation `r·sin(ψ−θ_i)`; see §5.4 |
| §3.1 Low-rank coupling `W = UVᵀ` | ✅ | `r_rank` configurable (32/64/128) |
| §3.1 κ(t) synchronization strength (K-value navigation) | ✅ | |
| §3.2 Resonance–standing-wave FNN (Lorentzian gain + standing-wave modulation) | ✅ | |
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
| §7 K curriculum `κ(t) = κ_max·σ((t−t_c)/τ)` | ✅ | |
| §7 Temperature annealing `T(t) = T_max(1−t)^p + T_min` | ✅ | |
| §7 Two-stage training (A: pre-train the core / B: end-to-end flow matching) | ❌ not staged | Currently single-stage joint training |
| §8 Inference: forward ODE integration + per-round active drift field | ⚠️ deviates | Currently uses "physics-core encoding + autoregressive decoding" to preserve text controllability |

**Remaining gaps** (by priority):

1. Two-stage training (§7) — Stage A pre-trains the physics core / Stage B does end-to-end flow
   matching; currently a single joint stage.
2. Heun/RK4 integration (§3.5) — currently explicit Euler (`dt=0.1`); the reversible
   VRAM-saving capability is not yet realized.

---

## 9. FAQ

**Q: Not enough VRAM?**
Reduce `--batch`, or use `--optimizer adafactor`, or lower `--max_seq_len`.

**Q: Data directory not found?**
Specify `--data_dir`, or set `PMNN_DATA_DIR`. The scripts automatically search upward for a
`datasets` directory.

**Q: I changed `--max_seq_len` but it had no effect?**
When the `max_len` of the preprocessed artifacts is smaller than required, preprocessing is
re-run automatically; you can also force it with `--rebuild_prepared`.

**Q: How do I change the vocabulary size?**
`prepare_data.py --vocab_size 32768 --rebuild_tokenizer`, then re-run preprocessing. The model's
`vocab_size` automatically aligns with the actual vocabulary size.
