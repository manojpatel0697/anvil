"""
Full benchmark runner for the Engine agent.

Runs the official 5-seed evaluation and writes a JSON report to reports/.

Usage:
    python experiments/run_full_bench.py
    python experiments/run_full_bench.py --seeds 42 101 202 --out reports/quick.json
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import run_multi
from adapters.myteam import Engine


def factory(X, params):
    return Engine(X, params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.5, 0.7, 0.8])
    ap.add_argument("--n-per-level", type=int, default=250)
    ap.add_argument("--out", default="reports/full_bench.json")
    args = ap.parse_args()

    print(f"Running {len(args.seeds)} seeds × {len(args.noise_levels)} noise levels "
          f"× {args.n_per_level} queries each ...")

    t0 = time.monotonic()
    report = run_multi(
        agent_factory=factory,
        seeds=args.seeds,
        K=args.K, N=args.N,
        noise_levels=args.noise_levels,
        n_per_level=args.n_per_level,
    )
    elapsed = time.monotonic() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    agg = report["aggregated"]
    sc = report["score"]
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Mean Δ accuracy : {agg['mean_delta']:+.3f}")
    print(f"Min  Δ accuracy : {agg['min_delta']:+.3f}")
    print(f"Mean spread     : {agg['mean_spread']:.3f}×")
    print(f"Retrieval pts   : {sc['retrieval_pts']:.1f} / 70")
    print(f"Anisotropy pts  : {sc['anisotropy_pts']:.1f} / 20")
    print(f"Total automated : {sc['total_automated']:.1f} / 90")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
