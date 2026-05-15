"""
Ablation study: decompose the four-signal Engine architecture.

Tests the marginal contribution of each signal component to retrieval
accuracy across all 5 evaluation seeds.

Usage:
    python experiments/ablation.py
"""
from __future__ import annotations

import sys
import os
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import make_patterns, make_test_queries
from pcam_model import PCAMModel, build_default_R
from checks import retrieval_accuracy
from adapters.dummy import DummyAgent
from adapter import Adapter


class _AblationEngine(Adapter):
    """Configurable Engine for ablation — each signal can be toggled."""

    def __init__(self, stored_patterns, model_params,
                 amp=3.0, use_var=True, use_cons=True, use_hess=True,
                 w_var=0.15, w_cons=0.25, w_hess=0.10, top_k=3):
        self.X = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape
        self.pi_min = model_params["pi_min"]
        self.pi_max = model_params["pi_max"]
        self.eta    = model_params["eta"]
        self.beta   = model_params["beta"]
        self.R      = model_params["R"].astype(np.float64)
        self.amp    = amp
        self.use_var  = use_var
        self.use_cons = use_cons
        self.use_hess = use_hess
        self.w_var  = w_var
        self.w_cons = w_cons
        self.w_hess = w_hess
        self.top_k  = top_k
        self._dim_var = self.X.var(axis=0) + 1e-8
        self._R_diag  = np.diag(self.R)

    def predict_precision(self, q):
        q = np.asarray(q, dtype=np.float64)
        mag = np.abs(q)
        mag_norm = mag / (mag.max() + 1e-8)
        log_pi = self.amp * (1.0 - mag_norm)

        if self.use_var:
            var_norm = self._dim_var / (self._dim_var.max() + 1e-8)
            log_pi += np.log1p(self.w_var * (1.0 - var_norm))

        if self.use_cons:
            cosines = self.X @ q
            top_idx = np.argsort(cosines)[-self.top_k:]
            top_cos = cosines[top_idx]
            w = np.exp(4.0 * (top_cos - top_cos.max()))
            w /= w.sum() + 1e-12
            consensus = (w[:, None] * self.X[top_idx]).sum(axis=0)
            agreement = q * consensus
            sign = np.sign(agreement)
            log_pi += self.w_cons * sign * (np.abs(agreement) ** 0.5)

        if self.use_hess:
            cosines = self.X @ q
            nearest = self.X[np.argmax(cosines)]
            z = self.beta * (self.X @ nearest)
            z -= z.max()
            s = np.exp(z); s /= s.sum() + 1e-12
            diag_XsX = (s[:, None] * self.X ** 2).sum(axis=0)
            Xts = self.X.T @ s
            hd = self._R_diag - self.eta * self.beta * (diag_XsX - Xts ** 2)
            hd = np.clip(hd, 1e-3, None)
            log_pi += -self.w_hess * np.log(hd / hd.mean())

        log_pi -= log_pi.mean()
        return np.clip(np.exp(log_pi), self.pi_min, self.pi_max)


def run_ablation():
    seeds = [42, 101, 202, 303, 404]
    noise_levels = [0.5, 0.7, 0.8]
    n_per_level = 80

    variants = [
        ("Baseline (Π=I)",                  None),
        ("S1: corruption only",             dict(use_var=False, use_cons=False, use_hess=False)),
        ("S1+S2: +variance",                dict(use_var=True,  use_cons=False, use_hess=False)),
        ("S1+S3: +consensus",               dict(use_var=False, use_cons=True,  use_hess=False)),
        ("S1+S4: +hessian",                 dict(use_var=False, use_cons=False, use_hess=True)),
        ("S1+S2+S3: +var+cons",             dict(use_var=True,  use_cons=True,  use_hess=False)),
        ("S1+S2+S3+S4: full Engine",        dict(use_var=True,  use_cons=True,  use_hess=True)),
    ]

    results = {name: [] for name, _ in variants}

    for seed in seeds:
        X = make_patterns(K=16, N=64, seed=seed)
        R = build_default_R(N=64, seed=seed)
        model = PCAMModel(X, R)
        params = {
            "R": R, "eta": 0.5, "beta": 8.0, "dt": 0.01,
            "T_max": 3000, "tol": 1e-6, "T_in": 100,
            "pi_min": 0.1, "pi_max": 10.0,
        }
        queries, truths, _ = make_test_queries(X, noise_levels, n_per_level, seed=seed)

        for name, kwargs in variants:
            if kwargs is None:
                agent = DummyAgent(X, params)
            else:
                agent = _AblationEngine(X, params, **kwargs)
            acc = retrieval_accuracy(model, agent, queries, truths)
            results[name].append(acc)
            print(f"  seed={seed}  {name:<35}  acc={acc:.3f}")

    base_mean = np.mean(results["Baseline (Π=I)"])

    print("\n── Ablation Summary ──────────────────────────────────────────────")
    print(f"{'Variant':<38} {'Mean Acc':>9} {'Mean Δ':>9}")
    print("-" * 60)
    for name, _ in variants:
        mean_acc = np.mean(results[name])
        delta = mean_acc - base_mean
        marker = " ← Engine" if "full Engine" in name else ""
        print(f"{name:<38} {mean_acc:>9.3f} {delta:>+9.3f}{marker}")

    out = Path("reports/ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {name: {"per_seed": accs, "mean": float(np.mean(accs)),
                "delta": float(np.mean(accs) - base_mean)}
         for name, accs in results.items()},
        indent=2,
    ))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_ablation()
