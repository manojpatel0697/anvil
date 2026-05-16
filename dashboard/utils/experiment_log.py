"""
Lightweight experiment logger for the dashboard.

Stores retrieval events in a pandas DataFrame and supports
JSON export and session-level summary statistics.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LogEntry:
    timestamp: float
    seed: int
    noise_level: float
    query_idx: int
    true_idx: int
    retrieved_idx: int
    baseline_idx: int
    correct: bool
    baseline_correct: bool
    precision_mean: float
    precision_std: float
    precision_min: float
    precision_max: float
    cosine_to_true: float


class ExperimentLog:
    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def record(self, result: Any, seed: int) -> None:
        """Append a RetrievalResult to the log."""
        pi = result.precision
        q_norm = result.query / (np.linalg.norm(result.query) + 1e-12)
        cosine_true = float(result.true_pattern @ q_norm)

        entry = LogEntry(
            timestamp=time.time(),
            seed=seed,
            noise_level=result.noise_level,
            query_idx=result.true_idx,
            true_idx=result.true_idx,
            retrieved_idx=result.retrieved_idx,
            baseline_idx=result.baseline_idx,
            correct=result.correct,
            baseline_correct=result.baseline_correct,
            precision_mean=float(pi.mean()),
            precision_std=float(pi.std()),
            precision_min=float(pi.min()),
            precision_max=float(pi.max()),
            cosine_to_true=cosine_true,
        )
        self._entries.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame()
        rows = [asdict(e) for e in self._entries]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def summary(self) -> dict[str, Any]:
        if not self._entries:
            return {}
        df = self.to_dataframe()
        return {
            "total_queries": len(df),
            "agent_accuracy": float(df["correct"].mean()),
            "baseline_accuracy": float(df["baseline_correct"].mean()),
            "delta": float(df["correct"].mean() - df["baseline_correct"].mean()),
            "mean_precision_std": float(df["precision_std"].mean()),
            "by_noise": df.groupby("noise_level")[["correct", "baseline_correct"]]
                          .mean()
                          .rename(columns={"correct": "agent", "baseline_correct": "baseline"})
                          .to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(
            [asdict(e) for e in self._entries],
            indent=2,
            default=str,
        )

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
