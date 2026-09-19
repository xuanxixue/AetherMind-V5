<div align="center">

# PMNN

### Physical Matrix Neural Network — Universal Foundational Architecture

*A modality-agnostic substrate that replaces token-attention with complex-valued oscillator fields governed by phase synchronization.*

**Design Specification · v0.1**

[![Status](https://img.shields.io/badge/status-design%20spec-8A2BE2)]()
[![Version](https://img.shields.io/badge/version-v0.1-1F6FEB)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()
[![Target](https://img.shields.io/badge/project-V6%20Video%20Gen-FF6B35)]()

</div>

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Core Idea](#core-idea)
- [Architecture](#architecture)
  - [Physical Block](#physical-block)
  - [Data Flow](#data-flow)
  - [Unified State Equation](#unified-state-equation)
- [Transformer ↔ PMNN Correspondence](#transformer--pmnn-correspondence)
- [Training & Inference](#training--inference)
- [Recommended Configuration](#recommended-configuration)
- [Source Verification](#source-verification)
- [Risk Assessment](#risk-assessment)
- [Repository Structure](#repository-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)
- [License](#license)

---

## Overview

**PMNN** (Physical Matrix Neural Network) is a proposed universal foundational architecture that occupies an isomorphic structural position relative to the Transformer. Instead of the now-standard *token + attention* paradigm — which is built on statistical affinity — PMNN adopts a *complex-valued oscillator field + phase synchronization* paradigm, grounded in physical coupling.

The architecture unifies six independently validated research threads into a single coherent substrate:

| # | Extension | Origin |
|---|-----------|--------|
| 1 | **ELF** — continuous embedding trajectories with terminal discretization | He et al., 2026 |
| 2 | **Drift Field** — attractive/repulsive distribution evolution | He et al., 2026 |
| 3 | **UN-0 Phase Synchronization** — Kuramoto coupling for generation | Unconventional AI, 2026 |
| 4 | **Thermodynamic Entropy** — XY-model free energy dynamics | Classical statistical mechanics |
| 5 | **Resonant Standing Waves** — Lorentzian frequency selection | Resonance theory |
| 6 | **Time Physics** — depth as reversible ODE integration | Neural ODE lineage |

Any input modality is first subjected to **oscillatorization** to enter a continuous embedding field, evolves through *R* rounds of physical blocks along a physical-time ODE, and is discretized back into symbolic representation at the terminal step.

---

## Motivation

The Transformer has been the dominant substrate for sequence modeling since 2017. However, three structural limitations motivate the search for an alternative:

1. **Discrete depth** — Layer stacking is a heuristic discretization of what could naturally be a continuous evolution.
2. **Statistical-only interaction** — Attention captures statistical co-occurrence but not dynamical coupling (phase locking, resonance, energy minimization).
3. **Token-level supervision** — Stepwise cross-entropy imposes discrete supervision where continuous trajectory matching may be more expressive.

PMNN addresses each of these by replacing the corresponding Transformer component with a physically-motivated analogue, while preserving computational tractability (same Big-O as attention).

---

## Core Idea

> Replace **token + attention (statistical affinity)** with **complex-valued oscillator field + phase synchronization (physical coupling)**. Replace the depth dimension with **physical-time ODE**. Replace the output layer with **ELF-style continuous denoising trajectories**.

The oscillator factor — the atomic unit of PMNN — is a complex-valued field element:

```
z_i = a_i ⊙ e^(iθ_i) ∈ ℂ^d
```

- **a_i ∈ ℝ^d₊** — amplitude vector (energy / confidence; preserves individual information, preventing full-synchronization collapse)
- **θ_i ∈ ℝ^d** — phase vector (carrier of information integration, evolved by coupling dynamics)

Complexification from any modality embedding `e_i`:

```python
a_i = softplus(W_a · e_i + b_a)
θ_i = (W_θ · e_i) mod 2π
```

This mapping is modality-agnostic: text tokens, image patches, spatiotemporal video patches, and audio frames all enter through the same gateway.

---

## Architecture

### Physical Block

A stack of *R* physical blocks is structurally homologous to Transformer layer stacking. Each block comprises four sublayers:

#### 3.1 · Phase Synchronization (replaces Attention)

Kuramoto coupling dynamics:

```
dθ_i^ν / dt = ω_i^ν + Σ_j W_ij · sin(θ_j^ν − θ_i^ν) + κ(t)·g_i^ν
```

| Symbol | Meaning |
|--------|---------|
| `ω_i^ν` | Intrinsic frequency (input drive) |
| `W ∈ ℝ^(N×N)` | Coupling matrix (low-rank `W = UVᵀ`, `r = 32`, or Toeplitz-structured) |
| `κ(t)` | Synchronization strength — the **K-value** |
| `g_i` | Drift field guidance term |
| `r(t) = |(1/N) Σ_j e^(iθ_j)|` | Order parameter (`r → 1` = full synchronization) |

**Complexity:** `O(N²·d)` full coupling → `O(N·d·r)` structured.

#### 3.2 · Resonance–Standing-Wave FNN (replaces FFN)

```
h = σ( W₂·( W₁·z + b₁ ) ⊙ sw(z, ω_d) + b₂ )
```

- **Resonance gain** (Lorentzian): `A(ω_d, ω_n) = 1 / √((ω_d² − ω_n²)² + (2γω_d)²)`
- **Standing-wave modulation**: `SW_m(p) = cos(k_m · p)`, `k_m = 2πm/N` — a physical analogue of RoPE.

#### 3.3 · Drift Field (replaces distribution evolution)

```
V(Z) = V⁺(Z) − λ·V⁻(Z)

V⁺ = Σ_k softmax_τ(−‖z_i − z_k^tar‖²)·(z_k^tar − z_i)    # real-sample attraction
V⁻ = Σ_j softmax_τ(−‖z_i − z_j‖²)·(z_j − z_i)             # generated-sample repulsion
```

> **Architectural departure from the original Drifting paper:** the drift direction `v_φ(Z, t)` is recomputed at every physical block during inference (not only during training). This is an intentional design choice — see [Risk Assessment](#risk-assessment).

#### 3.4 · Thermodynamics–Entropy

- **Energy** (XY Hamiltonian): `U(Z) = −Σ_ij W_ij · cos(θ_j − θ_i)`
- **Identity**: `∂U/∂θ_i = Σ_j W_ij · sin(θ_j − θ_i)` — Kuramoto's sin coupling *is* the XY energy gradient
- **Entropy**: `S(Z) = −Σ_i p_i log p_i`, `p_i = softmax(a_i)`
- **Free energy**: `F = U − T·S`, with annealing schedule `T(t) = T_max·(1−t)^p + T_min`
- **Langevin noise**: `η ~ N(0, 2T/Δt)`

#### 3.5 · Time Physics

- **Depth = physical time**: `dZ/dt = f_φ(Z, t, K(t))`, `t ∈ [0,1]`, integrated via Heun/RK4 (reversible → memory-independent of depth)
- **Time-reversal symmetry**: encoding = forward integration; generation = time-reversed integration (high-entropy → low-entropy)
- **Time field**: provides global temporal coordinate and driving protocols `κ(t)`, `T(t)`, `ω_d(t)`

### Data Flow

```
Input Modality ──> ① Oscillatorization ──> ② Physical Core (R Blocks) ──> ③ Continuous Trajectory ──> ④ Terminal Decoding
                     z_i = a_i ⊙ e^(iθ_i)     [see above]                   ELF Flow Matching            De-embedding
                     Z ∈ ℂ^(N×d)                                             x_t = t·x₁ + (1−t)x₀         Dual-Branch + Mode Gating
                                                                              t ∈ [0,1]
```

### Unified State Equation

The six extensions converge into a single hybrid dynamics:

```
dZ/dt = M_κ(t)·Z  +  ∇_Z[ −U(Z) + T(t)·S(Z) ]  +  v_φ(Z, t)  +  η(t)
        ─────────    ─────────────────────────    ───────────    ───────
        linear        nonlinear gradient           learned         noise
        operator      (Kuramoto + free energy      residual        (Langevin)
        (K-value      + drift repulsion)           (active at
         navigation)                                inference)
```

> **Transparency Note:** This is a *hybrid* formulation — linear operator + nonlinear gradient term + learned residual + noise — not a purely linear matrix equation. Composition follows verified mechanisms; no claim of full linearity is made.

---

## Transformer ↔ PMNN Correspondence

| Dimension | Transformer | PMNN |
|---|---|---|
| Basic Unit | Token / embedding vector | Complex-valued oscillator (amplitude + phase) |
| Interaction | Attention (`softmax QKᵀ`, statistical affinity) | Phase Synchronization (Kuramoto coupling, dynamic phase locking) |
| Positional Encoding | RoPE / ALiBi (heuristic) | Standing-Wave Modes (wave interference) |
| Feed-Forward | FFN (ReLU/GELU) | Resonance–Standing-Wave FNN (frequency-selective amplification) |
| Depth | Discrete layer stack | Physical-time ODE (continuous, reversible) |
| Distribution Modeling | Autoregressive / softmax CE | Flow matching + drift field |
| Output | Vocabulary distribution | Continuous denoising trajectory → terminal de-embedding |
| Training Signal | Token-level cross-entropy | Velocity field matching + synchronization/entropy constraints |
| Control | Sampling temperature / policy | K-value × entropy budget |
| Universality | Tokenization of any modality | Oscillatorization of any modality |

---

## Training & Inference

### Training

**Aggregate loss:**

```
L = L_fm + λ₁·L_sync + λ₂·L_drift + λ₃·L_F
```

| Term | Form | Role |
|------|------|------|
| `L_fm` | `E‖v_θ(x_t, t, c) − (x₁ − x₀)‖²` | Flow matching velocity (ELF primary) |
| `L_sync` | `‖r(t) − r_target(t)‖²` | Order parameter tracks K-curriculum |
| `L_drift` | `MSE(φ(x), stopgrad(φ(x)+V))` | Drift field consistency |
| `L_F` | `U(Z) − T·S(Z)` | Free energy regularization |

**K-Curriculum:** `κ(t) = κ_max · σ((t − t_c)/τ)` — synchronization phase transition at `t_c`.

**Two-stage protocol:**
- **Stage A** — Pretrain physical core on frozen encoder targets (synchronization + resonance basis + drift field)
- **Stage B** — End-to-end flow matching fine-tuning (T annealing + noise injection)

**Numerical methods:** Heun/RK4 integration, reversible layers → training memory independent of depth.

### Inference

- **Generation:** `x₀ ~ N(0, I)` → ODE forward integration `t: 0→1` → `x₁` (clean embedding) → de-embedding → tokens
- **Encoding/Understanding:** Forward integration until `r → 1`; representation = synchronized field
- **Modality Extension (Video):** Frame sequence spatiotemporal patches → spatiotemporal oscillator field; trajectories operate on latent embeddings (interfacing with V6's 480p/21:9 pipeline)

---

## Recommended Configuration

> *Illustrative — not empirically validated.*

| Parameter | Value | Rationale |
|---|---|---|
| Trajectory bottleneck `d` | 128 | ELF-consistent (T5 512 → 128) |
| Internal hidden dim | 768 | Comparable to equivalent Transformers |
| Oscillator count `N` | 512 – 4,096 | Sequence / token / patch count |
| Physical rounds `R` | 12 – 24 | Same order as layer count; reversible integration conserves memory |
| Coupling matrix `W` | Low-rank (`r = 32`) or Toeplitz | UN-0 finding: sparse coupling improves trainability |
| Compute | `O(N²·d)` full / `O(N·d·r)` structured | Same order as attention |
| Training memory | ≈ depth-independent | Reversible ODE technique |

---

## Source Verification

PMNN is built on three independently verified sources. Honest attribution and corrections are documented below:

| Source | Status | Key Correction |
|--------|--------|----------------|
| **ELF** ([arXiv:2605.10938](https://arxiv.org/pdf/2605.10938), He et al., 2026-05) | ✅ Authentic | Encoder used only for training targets; omitted at inference. Bottleneck: 512 → 128. |
| **Drifting** ([arXiv:2602.04770](https://arxiv.org/pdf/2602.04770), He et al., 2026-02) | ✅ Authentic | ⚠️ Drift field is **nonlinear** (pairwise L2 + temperature softmax + geometric-mean cross-weighting), not a linear formula. Must use "linear operator + nonlinear coupling" form. Per-round inference recomputation is an intentional departure — stability must be validated. |
| **Un-0** ([github.com/unconv-ai/un-0](https://github.com/unconv-ai/un-0), 2026-06) | ✅ Authentic | ⚠️ Attribution: Unconventional AI (Naveen Rao), **not** He et al. License: weights are **CC-BY-NC-4.0** (non-commercial); commercial use requires self-implementation. FID 6.74 on ImageNet-64. |

**Terminological Note:** "Oscillator" denotes a complex-valued field element (amplitude + phase), corresponding to the Kuramoto phase oscillator formalism. This is a technical term and does not constitute a literal physical claim.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| **"Non-statistical" boundary** — flow matching & drift field remain distribution modeling | Precise formulation: *the mechanism is physical; the distributional target is statistical.* Not contradictory. |
| **"Unprecedented" claim** — components all have ML precedent | Downgraded: only the *combination* (modality-agnostic physical matrix + K-value navigation + inference-time drift) is novel. |
| **Full-synchronization collapse** (`r → 1` homogenizes info) | Heterogeneous `ω_i` + amplitude channel preserves individual info + sparse/local coupling |
| **Continuous-space training instability** (ELF-documented) | 128-dim bottleneck + normalization + K-curriculum |
| **ODE numerical error accumulation** | Heun/RK4 + residual connections |
| **Inference-time drift activity** (intentional departure) | Stability and `×R` cost must be empirically validated |
| **UN-0 licensing** | CC-BY-NC-4.0 weights; commercial use requires self-implementation. All PMNN math is public. |
| **"Physics" boundary** | Operators are computable, differentiable, measurable (XY energy, Lorentzian resonance, ODE time reversal) — analogous, not literal. |

---

## Repository Structure

```
pmnn/
├── README.md                  # This document
├── docs/
│   ├── specification/         # Full design specification (v0.1)
│   ├── verification/          # Source verification records
│   └── roadmap/               # Evolution roadmap
├── src/
│   └── (planned)              # Reference implementation (TBD)
├── experiments/
│   └── (planned)              # Empirical validation benchmarks
└── LICENSE                    # TBD
```

> **Current state:** This repository hosts the design specification only. Reference implementation is pending.

---

## Roadmap

- [x] **v0.1** — Design specification (this document)
- [ ] **v0.2** — Notation table, glossary, formal proofs of key identities
- [ ] **v0.3** — Reference implementation: Kuramoto layer, drift field, ELF decoder
- [ ] **v0.4** — Toy benchmarks (MNIST, CIFAR-10 flow matching)
- [ ] **v0.5** — Modality extension: video spatiotemporal oscillator field (V6 pipeline)
- [ ] **v1.0** — First empirical validation against Transformer baselines

---

## Contributing

This is a design specification in active development. Contributions are welcome in the following forms:

- **Verification** — independent re-derivation of the identities in §3.4 and §5
- **Implementation** — reference code for any sublayer (especially Kuramoto coupling and drift field)
- **Critique** — identification of unstated assumptions or unstated risks
- **Empirical validation** — small-scale benchmarks testing stability claims

Please open an issue first to discuss substantial changes before submitting a PR.

---

## Citation

If you reference this architecture in academic work, please cite:

```bibtex
@misc{pmnn2026,
  title  = {PMNN: Physical Matrix Neural Network — Universal Foundational Architecture},
  note   = {Design Specification v0.1},
  year   = {2026},
  month  = {September},
  url    = {https://github.com/<your-org>/pmnn}
}
```

---

## References

1. He, K. *ELF: Embedded Language Flows*. arXiv:2605.10938, 2026. https://arxiv.org/pdf/2605.10938
2. He, K. et al. *Generative Modeling via Drifting*. arXiv:2602.04770, 2026. https://arxiv.org/pdf/2602.04770
3. Unconventional AI. *Un-0: Image Generation via Kuramoto Dynamics*. GitHub, 2026. https://github.com/unconv-ai/un-0

---

## License

**To be determined.** The PMNN design specification itself will be released under an open license. Note that any reference implementation reusing UN-0 weights must respect the CC-BY-NC-4.0 non-commercial restriction; commercial implementations must self-implement the Kuramoto layer using the public mathematical formulations documented herein.

---

<div align="center">

*PMNN is a research-stage architectural proposal. No empirical claims are made at v0.1.*

</div>
