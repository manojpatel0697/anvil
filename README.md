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

This submission implements `Engine` — a corruption-aware adaptive precision agent
that consistently outperforms the Π=I baseline across all evaluation seeds.

---

## System Architecture

```
bench-p04-pcam/
│
├── adapter.py              # Abstract Adapter interface (frozen, do not modify)
├── pcam_model.py           # Frozen PCAM dynamics (PCAMModel)
├── data.py                 # Pattern generation and corruption
├── checks.py               # Retrieval accuracy + anisotropy spread metrics
├── harness.py              # Multi-seed orchestration and scoring
├── run.py                  # CLI benchmark runner
├── self_check.py           # Quick self-evaluation script
│
├── adapters/
│   ├── dummy.py            # Baseline: Π=I (uniform precision)
│   └── myteam.py           # Engine: corruption-aware adaptive agent
│
├── dashboard/
│   ├── app.py              # Streamlit research dashboard
│   ├── utils/
│   │   ├── pcam_bridge.py  # Bridge: benchmark ↔ dashboard
│   │   ├── plot_helpers.py # Plotly figure factories
│   │   └── experiment_log.py  # Session-level experiment logger
│   ├── components/         # (reserved for future component extraction)
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

## Precision Steering — Core Insight

The PCAM dynamics inject the corrupted query as an external drive:

```
a_{t+1} = a_t + dt * (-π ⊙ ∇E(a_t) + q)    for t < T_in
```

The external drive on dimension *i* is proportional to `q[i]`.  Mask corruption
sets a fraction of `q[i]` to zero, so those dimensions have **no external drive**.
With uniform π, the gradient term dominates on masked dimensions and can pull the
trajectory toward the wrong attractor.

**The fix:** boost π on masked (low-|q|) dimensions.  This amplifies the attractor
pull exactly where the external signal is absent, steering the trajectory toward
the correct basin even under heavy corruption.

```python
mag_norm = |q| / max(|q|)                    # in [0, 1]
π_i = exp(amp × (1 - mag_norm_i))            # max on masked dims
```

This is counter-intuitive — we boost precision on the *least reliable* dimensions —
but it is the correct strategy given the PCAM injection mechanism.

---

## Retrieval Pipeline

```
Corrupted query q
       │
       ▼
Engine.predict_precision(q)
  ├─ Primary:   π_i = exp(3 × (1 - |q_i|/max|q|))
  └─ Secondary: π_i *= 1 + 0.15 × (1 - var_i/max_var)
       │
       ▼
PCAMModel.clip_and_normalise(π)
  ├─ clip to [π_min, π_max] = [0.1, 10.0]
  └─ divide by mean(π)
       │
       ▼
PCAMModel.run(q, π, u_const=q)
  └─ gradient descent with precision-scaled steps
       │
       ▼
PCAMModel.classify(a*)
  └─ cosine similarity to stored patterns → predicted index
```

---

## Benchmark Results

| Seed | Baseline (Π=I) | Engine | Δ |
|------|---------------|--------|---|
| 42   | 0.790 | 0.870 | +0.080 |
| 101  | 0.760 | 0.940 | +0.180 |
| 202  | ~0.60 | ~0.87 | +0.27  |
| 303  | ~0.67 | ~0.87 | +0.20  |
| 404  | ~0.63 | ~0.92 | +0.29  |

**Automated score: 70 / 90** (retrieval: 70/70, anisotropy: 0/20, code: manual)

The anisotropy score is 0 because the spread metric measures the condition number
of `diag(√π) H diag(√π)` at stored attractors.  The Hessian diagonal is nearly
uniform across dimensions (range 0.797–0.800), so no diagonal preconditioner can
reduce the condition number below the baseline.  The retrieval gain is the
dominant contribution to the score.

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

1. **Hessian-aware anisotropy reduction** — the current Hessian diagonal is nearly
   uniform due to the Erdős–Rényi graph structure.  A full-matrix preconditioner
   (e.g., incomplete Cholesky) could reduce the condition number, but requires
   O(N²) computation per query.

2. **Learned precision network** — train a small MLP on (query, pattern set) pairs
   to predict π, using retrieval accuracy as the training signal.

3. **Iterative refinement** — run one step of PCAM dynamics, observe the gradient
   direction, and update π before the full run.

4. **PCA-MNIST evaluation** — the benchmark will swap in PCA-MNIST + mask noise
   for the final evaluation.  The current strategy should transfer directly since
   it operates on the query magnitude, not the pattern structure.

5. **Multi-attractor confidence** — weight the precision signal by the softmax
   similarity to the top-K attractors, not just the nearest one.
