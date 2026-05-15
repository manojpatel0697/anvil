"""
Bridge between the dashboard and the frozen benchmark engine.

Provides a clean, stateless API for the dashboard components.
All heavy computation is isolated here so the UI layer stays thin.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Resolve benchmark root so imports work regardless of cwd.
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from data import make_patterns, make_test_queries, corrupt
from pcam_model import PCAMModel, build_default_R
from adapters.myteam import Engine
from adapters.dummy import DummyAgent
from checks import per_pattern_spread


@dataclass
class RetrievalResult:
    query: np.ndarray
    true_pattern: np.ndarray
    true_idx: int
    precision: np.ndarray
    retrieved: np.ndarray
    retrieved_idx: int
    baseline_retrieved: np.ndarray
    baseline_idx: int
    noise_level: float
    cosines_to_all: np.ndarray          # (K,) cosine similarities at query
    cosines_retrieved: np.ndarray       # (K,) cosine similarities at retrieved state
    correct: bool
    baseline_correct: bool


@dataclass
class SessionState:
    """Holds all objects for a given (seed, K, N) configuration."""
    seed: int
    K: int
    N: int
    patterns: np.ndarray                # (K, N)
    model: PCAMModel
    agent: Engine
    dummy: DummyAgent
    params: dict[str, Any]


def build_session(seed: int = 42, K: int = 16, N: int = 64) -> SessionState:
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)
    params = {
        "R": R, "eta": model.eta, "beta": model.beta,
        "dt": model.dt, "T_max": model.T_max, "tol": model.tol,
        "T_in": model.T_in, "pi_min": model.pi_min, "pi_max": model.pi_max,
    }
    agent = Engine(X, params)
    dummy = DummyAgent(X, params)
    return SessionState(seed=seed, K=K, N=N, patterns=X,
                        model=model, agent=agent, dummy=dummy, params=params)


def run_single_retrieval(
    session: SessionState,
    query_idx: int,
    noise_level: float,
    noise_seed: int = 0,
) -> RetrievalResult:
    """Run one retrieval with both the adaptive agent and the baseline."""
    X = session.patterns
    model = session.model
    true_idx = query_idx % session.K
    true_pattern = X[true_idx]

    rng = np.random.default_rng(noise_seed)
    q = corrupt(true_pattern, noise_level, rng)

    pi = session.agent.predict_precision(q)
    pi_dummy = session.dummy.predict_precision(q)

    retrieved = model.run(q, pi, u_const=q)
    baseline_retrieved = model.run(q, pi_dummy, u_const=q)

    retrieved_idx = model.classify(retrieved)
    baseline_idx = model.classify(baseline_retrieved)

    # Cosine similarities at query and at retrieved state
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    r_norm = retrieved / (np.linalg.norm(retrieved) + 1e-12)
    cosines_query = X @ q_norm
    cosines_retrieved = X @ r_norm

    return RetrievalResult(
        query=q,
        true_pattern=true_pattern,
        true_idx=true_idx,
        precision=model.clip_and_normalise(pi),
        retrieved=retrieved,
        retrieved_idx=retrieved_idx,
        baseline_retrieved=baseline_retrieved,
        baseline_idx=baseline_idx,
        noise_level=noise_level,
        cosines_to_all=cosines_query,
        cosines_retrieved=cosines_retrieved,
        correct=(retrieved_idx == true_idx),
        baseline_correct=(baseline_idx == true_idx),
    )


def run_batch_evaluation(
    session: SessionState,
    noise_levels: list[float],
    n_per_level: int,
    batch_seed: int = 0,
) -> dict[str, Any]:
    """Run a batch evaluation and return per-noise-level accuracy metrics."""
    X = session.patterns
    model = session.model
    queries, truths, levels = make_test_queries(
        X, noise_levels, n_per_level, seed=batch_seed
    )

    agent_correct = {lvl: [] for lvl in noise_levels}
    base_correct = {lvl: [] for lvl in noise_levels}

    for q, t, lvl in zip(queries, truths, levels):
        pi_agent = session.agent.predict_precision(q)
        pi_dummy = session.dummy.predict_precision(q)
        a_agent = model.run(q, pi_agent, u_const=q)
        a_dummy = model.run(q, pi_dummy, u_const=q)
        agent_correct[lvl].append(int(model.classify(a_agent) == int(t)))
        base_correct[lvl].append(int(model.classify(a_dummy) == int(t)))

    results = {}
    for lvl in noise_levels:
        ac = np.mean(agent_correct[lvl])
        bc = np.mean(base_correct[lvl])
        results[lvl] = {
            "agent_accuracy": float(ac),
            "baseline_accuracy": float(bc),
            "delta": float(ac - bc),
            "n_queries": len(agent_correct[lvl]),
        }
    return results


def compute_spread_profile(
    session: SessionState,
    n_patterns: int = 8,
) -> dict[str, Any]:
    """Compute anisotropy spread for agent vs baseline across stored patterns."""
    model = session.model
    rng = np.random.default_rng(session.seed)
    indices = list(range(min(n_patterns, session.K)))

    agent_spreads = []
    base_spreads = []

    for idx in indices:
        pattern = session.patterns[idx]
        probe = pattern + rng.standard_normal(session.N) * 0.05
        probe = probe / (np.linalg.norm(probe) + 1e-12)

        pi_agent = session.agent.predict_precision(probe)
        pi_dummy = session.dummy.predict_precision(probe)

        s_agent = per_pattern_spread(model, pi_agent, pattern)
        s_base = per_pattern_spread(model, pi_dummy, pattern)

        if s_agent is not None:
            agent_spreads.append(s_agent)
        if s_base is not None:
            base_spreads.append(s_base)

    return {
        "pattern_indices": indices,
        "agent_spreads": agent_spreads,
        "base_spreads": base_spreads,
        "mean_agent": float(np.mean(agent_spreads)) if agent_spreads else 0.0,
        "mean_base": float(np.mean(base_spreads)) if base_spreads else 0.0,
    }


def get_top_k_attractors(
    session: SessionState,
    query: np.ndarray,
    k: int = 5,
) -> dict[str, Any]:
    """Return the top-K nearest attractors and their similarity scores."""
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    cosines = session.patterns @ q_norm
    top_idx = np.argsort(cosines)[::-1][:k]
    return {
        "indices": top_idx.tolist(),
        "cosines": cosines[top_idx].tolist(),
        "patterns": session.patterns[top_idx],
    }
