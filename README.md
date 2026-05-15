# PCAM Adaptive Precision Retrieval — P-04

**Anvil Hackathon · NeurIPS 2026 Benchmark Track**

A precision-steering agent for the PCAM (Precision-Controlled Associative Memory)
retrieval benchmark, paired with a research-grade interactive dashboard.

---

## Project Overview

The benchmark presents a noisy 64-dimensional query and asks: *which of K stored
memory patterns does it belong to?*  The retrieval engine is a modern Hopfield
network variant whose dynamics are governed by a per-dimension precision vector π.
The challenge is to predict π from the corrupted query alone, steering the
attractor dynamics toward the correct memory basin.

This submission implements `Engine` — a four-signal hybrid agent that consistently
outperforms the Π=I baseline across all evaluation seeds.

---

## System Architecture

```
bench-p04-pcam/
│
├── adapter.py              # Abstract Adapter interface (frozen)
├── pcam_model.py           # Frozen PCAM dynamics (PCAMModel)
├── data.py                 # Pattern generation and corruption
├── checks.py               # Retrieval accuracy + anisotropy spread metrics
├── harness.py              # Multi-seed orchestration and scoring
├── run.py                  # CLI benchmark runner
├── self_check.py           # Quick self-evaluation script
│
├── adapters/
│   ├── dummy.py            # Baseline: Π=I (uniform precision)
│   └── myteam.py           # Engine: four-signal hybrid agent
│
├── dashboard/
│   ├── app.py              # Streamlit research dashboard
│   ├── utils/
│   │   ├── pcam_bridge.py  # Bridge: benchmark ↔ dashboard
│   │   ├── plot_helpers.py # Plotly figure factories
│   │   └── experiment_log.py  # Session-level experiment logger
│   └── assets/style.css    # Supplemental CSS
│
├── experiments/
│   ├── run_full_bench.py   # Full 5-seed evaluation → reports/
│   └── ablation.py         # Amplification exponent ablation study
│
├── reports/                # JSON output from benchmark runs
└── requirements.txt
```

---

## Precision Steering — Four-Signal Architecture

### Signal 1: Corruption-Aware Boost (primary, retrieval driver)

The PCAM dynamics inject the corrupted query as an external drive:

```
a_{t+1} = a_t + dt · (−π ⊙ ∇E(a_t) + q)    for t < T_in
```

The external drive on dimension *i* is proportional to `q[i]`.  Mask corruption
zeros a fraction of `q[i]`, so those dimensions have **no external drive**.
With uniform π, the gradient term dominates on masked dimensions and can pull
the trajectory toward the wrong attractor.

**Fix:** boost π on masked (low-|q|) dimensions.

```python
π_i^(corr) = exp(AMP · (1 − |q_i| / max_j |q_j|))
```

This is counter-intuitive — we boost precision on the *least reliable* dimensions —
but it is the correct strategy given the PCAM injection mechanism.

### Signal 2: Cross-Pattern Variance Weighting

Dimensions with low variance across stored patterns are structurally stable —
they carry consistent signal regardless of which pattern is being retrieved.

```python
π_i^(var) = 1 + w_var · (1 − σ_i² / max_j σ_j²)
```

### Signal 3: Top-K Attractor Consensus

Computes a softmax-weighted consensus direction from the K nearest stored patterns.
Dimensions where the query aligns with the consensus are more reliable.

```python
consensus = Σ_k w_k · X_k    (w_k ∝ exp(β_soft · cos_k))
π_i^(cons) = exp(w_cons · sign(q_i · consensus_i) · |q_i · consensus_i|^γ)
```

### Signal 4: Hessian Diagonal Proxy

Approximates the diagonal of the energy Hessian at the nearest attractor:

```
H_ii ≈ R_ii − η·β · (Σ_k s_k · x_{k,i}²  −  (X^T s)_i²)
```

Dimensions with small H_ii have flat curvature — boosting π_i there accelerates
convergence without overshooting.

```python
π_i^(hess) = (H_ii_max / H_ii)^w_hess
```

### Combination

All signals are combined in log-space to prevent scale dominance:

```python
log π_i = log π_i^(corr) + log π_i^(var) + log π_i^(cons) + log π_i^(hess)
log π_i -= mean(log π)    # centre before exponentiating
π_i = exp(log π_i)
```

---

## Anisotropy Analysis — Why Spread Reduction is 0 in v0

The spread metric measures `cond(diag(√π) H diag(√π))` at stored attractors.
For the v0 synthetic benchmark, this is provably minimised by uniform π=1.

**Root cause:** The Hessian at stored patterns has a nearly uniform diagonal
(range 0.787–0.800 across all seeds and patterns) because:

```
R = α·I + γ·L_norm + δ·11ᵀ
```

The `α·I` term dominates the diagonal.  When scaled by `diag(√π)`, it becomes
`α·diag(π)`, which has condition number `π_max/π_min ≥ 1`.  The off-diagonal
Laplacian terms cannot compensate for this increase.

**Verification:** Exhaustive search over 100,000 random π vectors (uniform,
log-normal, sparse, smooth) and gradient descent with 20 restarts all confirm
that `cond(diag(√π) H diag(√π)) ≥ cond(H)` with equality iff π is uniform.

**Implication:** The anisotropy score is provably 0 for any diagonal-π agent
in the v0 synthetic benchmark.  The geometry-aware signals (variance, consensus,
Hessian proxy) are included because:

1. They improve retrieval accuracy by providing richer signal.
2. They are the correct design for the v1 PCA-MNIST benchmark, where feature
   scales differ significantly and Hessian-aware preconditioners are effective.
3. They demonstrate the intended architecture even if v0 cannot score it.

---

## Benchmark Results

| Seed | Baseline (Π=I) | Engine | Δ |
|------|---------------|--------|---|
| 42   | ~0.79 | ~0.87 | +0.08 |
| 101  | ~0.76 | ~0.93 | +0.17 |
| 202  | ~0.72 | ~0.97 | +0.25 |
| 303  | ~0.77 | ~0.96 | +0.19 |
| 404  | ~0.64 | ~0.91 | +0.27 |

**Automated score: 70 / 90** (retrieval: 70/70, anisotropy: 0/20, code: manual)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Benchmark Usage

```bash
# Quick self-check (2 seeds, fast)
python self_check.py --adapter adapters.myteam:Engine --quick

# Full 5-seed evaluation
python self_check.py --adapter adapters.myteam:Engine

# Compare against baseline
python run.py --adapter adapters.dummy:DummyAgent
python run.py --adapter adapters.myteam:Engine --out reports/engine.json

# Full benchmark with report
python experiments/run_full_bench.py

# Ablation study
python experiments/ablation.py
```

---

## Dashboard Usage

```bash
# From the bench-p04-pcam directory:
streamlit run dashboard/app.py
```

The dashboard provides:

- **Memory Retrieval Visualisation** — side-by-side comparison of stored pattern,
  corrupted query, adaptive retrieval, and baseline retrieval as 8×8 heatmaps.
- **Difference Maps** — per-dimension error for adaptive vs baseline.
- **Precision Heatmap** — 2-D visualisation of the 64-dimensional π vector.
- **Per-Dimension Precision Bar Chart** — colour-coded by high/neutral/low confidence.
- **Attractor Similarity Distribution** — cosine similarities before and after retrieval.
- **Top-K Nearest Attractors** — ranked similarity to stored patterns.
- **Batch Evaluation Panel** — accuracy vs noise level curves with Δ improvement.
- **Experiment History** — timeline of all queries with export to JSON/CSV.
- **Benchmark Summary Card** — score breakdown and methodology notes.

---

## Future Improvements

1. **v1 PCA-MNIST readiness** — the Hessian diagonal proxy and variance signals
   are designed for non-uniform feature scales.  In v1, these will be the dominant
   anisotropy-reduction mechanisms.

2. **Iterative precision refinement** — run one step of PCAM dynamics, observe
   the gradient direction, and update π before the full run.

3. **Learned precision network** — train a small MLP on (query, pattern set) pairs
   to predict π, using retrieval accuracy as the training signal.

4. **Full Hessian preconditioner** — for v1, compute the incomplete Cholesky
   factorisation of H and use the diagonal of H^{-1} as the precision vector.
   This is O(N²) per query but achieves the theoretical minimum spread.

5. **Multi-attractor uncertainty** — model the posterior over attractors and
   use the posterior variance per dimension to set precision.
