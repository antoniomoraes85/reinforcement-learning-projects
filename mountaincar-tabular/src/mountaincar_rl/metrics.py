from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable
import math

import numpy as np
from scipy import stats


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if window <= 0:
        raise ValueError("A janela deve ser positiva.")
    if len(array) < window:
        return np.full(len(array), np.nan)
    result = np.full(len(array), np.nan)
    result[window - 1 :] = np.convolve(array, np.ones(window) / window, mode="valid")
    return result


def normalized_auc(episode_returns: Iterable[float], max_steps: int = 200) -> float:
    returns = np.asarray(list(episode_returns), dtype=float)
    if returns.size == 0:
        raise ValueError("A curva de retornos está vazia.")
    performance = np.clip((returns + max_steps) / max_steps, 0.0, 1.0)
    if performance.size == 1:
        return float(performance[0])
    return float(np.trapezoid(performance, dx=1.0) / (performance.size - 1))


def first_stable_episode(
    successes: Iterable[bool],
    window: int,
    threshold: float,
    consecutive_windows: int,
) -> int | None:
    rolling = moving_average([float(value) for value in successes], window)
    streak = 0
    for index, value in enumerate(rolling):
        if np.isfinite(value) and value >= threshold:
            streak += 1
            if streak >= consecutive_windows:
                return index + 1
        else:
            streak = 0
    return None


def summarize_run(result: dict[str, Any], selection_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = result["config"]
    train = result["training_metrics"]
    evaluation = result["evaluation_metrics"]
    final_window = int(selection_cfg.get("final_window_episodes", 100))
    stable_episode = first_stable_episode(
        train["episode_successes"],
        int(selection_cfg.get("rolling_window", 20)),
        float(selection_cfg.get("stability_success_threshold", 0.8)),
        int(selection_cfg.get("stability_consecutive_windows", 5)),
    )
    stable_index = stable_episode - 1 if stable_episode is not None else None
    hp = cfg["hyperparameters"]
    return {
        "config_hash": result["config_hash"],
        "algorithm": cfg["algorithm"],
        "discretization": cfg["discretization"]["name"],
        "seed": int(cfg["seed"]),
        "alpha": float(hp["alpha"]),
        "lambda": float(hp["lambda"]),
        "epsilon_decay": float(hp["epsilon_decay"]),
        "N": int(hp["N"]),
        "auc_normalized": normalized_auc(train["episode_returns"], cfg["environment"]["max_episode_steps"]),
        "final_window_success": float(np.mean(train["episode_successes"][-final_window:])),
        "evaluation_success_rate": float(evaluation["success_rate"]),
        "mean_steps_success": evaluation["mean_steps_success"],
        "mean_steps_all": float(evaluation["mean_steps_all"]),
        "stable_episode": stable_episode,
        "interactions_to_stability": (
            int(train["cumulative_env_steps"][stable_index]) if stable_index is not None else None
        ),
        "q_updates_to_stability": (
            int(train["cumulative_q_updates"][stable_index]) if stable_index is not None else None
        ),
        "total_env_steps": int(train["cumulative_env_steps"][-1]),
        "total_q_updates": int(train["cumulative_q_updates"][-1]),
        "training_time_seconds": float(train["training_time_seconds"]),
    }


def student_t_curve_ci(matrix: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("A matriz deve ter formato sementes × episódios.")
    n = values.shape[0]
    counts = np.sum(np.isfinite(values), axis=0)
    means = np.divide(np.nansum(values, axis=0), counts, out=np.full(values.shape[1], np.nan), where=counts>0)
    if n < 2:
        nan = np.full(values.shape[1], np.nan)
        return means, nan, nan, nan
    std = np.sqrt(np.divide(np.nansum((values-means)**2, axis=0), counts-1,
                           out=np.full(values.shape[1],np.nan), where=counts>1))
    critical = stats.t.ppf((1.0 + confidence) / 2.0, np.maximum(counts - 1, 1))
    margin = critical * std / np.sqrt(np.maximum(counts, 1))
    margin[counts < 2] = np.nan
    return means, std, means - margin, means + margin


def paired_rank_biserial(x: Iterable[float], y: Iterable[float]) -> float:
    difference = np.asarray(list(x), dtype=float) - np.asarray(list(y), dtype=float)
    nonzero = difference[difference != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_bootstrap_ci(
    x: Iterable[float],
    y: Iterable[float],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    difference = np.asarray(list(x), dtype=float) - np.asarray(list(y), dtype=float)
    if difference.size == 0:
        raise ValueError("Não há pares para bootstrap.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(resamples, difference.size))
    boot = difference[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(difference.mean()), float(lower), float(upper)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
    }

