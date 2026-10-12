from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
from scipy import stats

from .config import ALGORITHMS, config_hash
from .metrics import (
    describe,
    holm_adjust,
    moving_average,
    paired_bootstrap_ci,
    paired_rank_biserial,
    student_t_curve_ci,
    summarize_run,
)
from .storage import inspect_result_file, load_json


PARAMETER_FIELDS = ("alpha", "lambda", "epsilon_decay", "N")


def load_valid_results(directory: Path | str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*__*__s*__*.json")):
        valid, errors = inspect_result_file(path)
        if not valid:
            problems.append({"path": str(path), "errors": errors})
            continue
        results.append(load_json(path))
    return results, problems


def triage_rows(results: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [summarize_run(result, protocol["selection"]) for result in results]


def _mean_if_all_reached(values: list[int | float | None]) -> float | None:
    if not values or any(value is None or not np.isfinite(value) for value in values):
        return None
    return float(np.mean([float(value) for value in values]))


def aggregate_triage_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], *(row[field] for field in PARAMETER_FIELDS))
        groups[key].append(row)

    candidates: list[dict[str, Any]] = []
    for key, group in groups.items():
        algorithm = key[0]
        seed_auc: dict[int, list[float]] = defaultdict(list)
        for row in group:
            seed_auc[int(row["seed"])].append(float(row["auc_normalized"]))
        auc_by_seed = [float(np.mean(values)) for values in seed_auc.values()]
        candidate = {
            "algorithm": algorithm,
            **dict(zip(PARAMETER_FIELDS, key[1:])),
            "run_count": len(group),
            "discretization_count": len({row["discretization"] for row in group}),
            "seed_count": len(seed_auc),
            "auc_normalized_mean": float(np.mean([row["auc_normalized"] for row in group])),
            "final_window_success_mean": float(np.mean([row["final_window_success"] for row in group])),
            "interactions_to_stability_mean": _mean_if_all_reached(
                [row["interactions_to_stability"] for row in group]
            ),
            "q_updates_to_stability_mean": _mean_if_all_reached(
                [row["q_updates_to_stability"] for row in group]
            ),
            "auc_between_seed_std": float(np.std(auc_by_seed, ddof=1)) if len(auc_by_seed) > 1 else 0.0,
            "total_q_updates_mean": float(np.mean([row["total_q_updates"] for row in group])),
            "total_env_steps_mean": float(np.mean([row["total_env_steps"] for row in group])),
            "stability_reached_fraction": float(
                np.mean([row["stable_episode"] is not None for row in group])
            ),
        }
        candidates.append(candidate)
    return candidates


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    interactions = candidate["interactions_to_stability_mean"]
    return (
        -float(candidate["auc_normalized_mean"]),
        -float(candidate["final_window_success_mean"]),
        math.inf if interactions is None else float(interactions),
        float(candidate["total_q_updates_mean"]),
        float(candidate["auc_between_seed_std"]),
    )


def select_hyperparameters(
    candidates: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for algorithm in ALGORITHMS:
        eligible = [candidate for candidate in candidates if candidate["algorithm"] == algorithm]
        if algorithm == "dyna_q" and protocol["selection"].get("exclude_dyna_n_zero", True):
            eligible = [candidate for candidate in eligible if int(candidate["N"]) != 0]
        if not eligible:
            raise ValueError(f"Nenhuma configuração elegível para {algorithm}.")
        winner = sorted(eligible, key=_selection_key)[0]
        selected[algorithm] = {field: winner[field] for field in PARAMETER_FIELDS}
        evidence[algorithm] = winner
    payload = {
        "schema_version": "1.0",
        "source_experiment_version": protocol["experiment_version"],
        "selection_rule": protocol["selection"],
        "selected_hyperparameters": selected,
        "selection_evidence": evidence,
    }
    payload["selection_hash"] = config_hash(payload["selected_hyperparameters"])
    return payload


def curve_rows(
    results: list[dict[str, Any]],
    metric: str,
    confidence: float,
    rolling_window: int | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        cfg = result["config"]
        groups[(cfg["discretization"]["name"], cfg["algorithm"])].append(result)
    rows: list[dict[str, Any]] = []
    for (discretization, algorithm), group in groups.items():
        matrix = []
        for result in sorted(group, key=lambda item: item["config"]["seed"]):
            values = result["training_metrics"][metric]
            if rolling_window is not None:
                values = moving_average(values, rolling_window)
            matrix.append(values)
        means, std, lower, upper = student_t_curve_ci(np.asarray(matrix, dtype=float), confidence)
        for index in range(len(means)):
            rows.append({
                "discretization": discretization,
                "algorithm": algorithm,
                "episode": index + 1,
                "mean": None if not np.isfinite(means[index]) else float(means[index]),
                "std": None if not np.isfinite(std[index]) else float(std[index]),
                "ci95_low": None if not np.isfinite(lower[index]) else float(lower[index]),
                "ci95_high": None if not np.isfinite(upper[index]) else float(upper[index]),
                "n_seeds": len(group),
            })
    return rows


def final_seed_rows(results: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    selection_cfg = {
        "final_window_episodes": 100,
        "rolling_window": protocol["analysis"]["rolling_window"],
        "stability_success_threshold": protocol["analysis"]["stability_success_threshold"],
        "stability_consecutive_windows": protocol["analysis"]["stability_consecutive_windows"],
    }
    return [summarize_run(result, selection_cfg) for result in results]


def descriptive_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["discretization"], row["algorithm"])].append(row)
    output = []
    for (discretization, algorithm), group in groups.items():
        record: dict[str, Any] = {
            "discretization": discretization,
            "algorithm": algorithm,
        }
        metrics = {
            "auc": "auc_normalized",
            "success": "evaluation_success_rate",
            "steps_all": "mean_steps_all",
            "steps_success": "mean_steps_success",
            "env_steps_total": "total_env_steps",
            "q_updates_total": "total_q_updates",
            "interactions_to_stability": "interactions_to_stability",
            "q_updates_to_stability": "q_updates_to_stability",
        }
        for prefix, field in metrics.items():
            values = [row[field] for row in group if row[field] is not None]
            if values:
                record.update({f"{prefix}_{key}": value for key, value in describe(values).items()})
            else:
                record.update(
                    {
                        f"{prefix}_n": 0,
                        f"{prefix}_mean": None,
                        f"{prefix}_std": None,
                        f"{prefix}_median": None,
                        f"{prefix}_q1": None,
                        f"{prefix}_q3": None,
                    }
                )
        output.append(record)
    return output


def inferential_tests(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    confidence = float(protocol["analysis"]["confidence_level"])
    resamples = int(protocol["analysis"]["bootstrap_resamples"])
    base_seed = int(protocol["analysis"]["bootstrap_seed"])
    output: dict[str, Any] = {"metric": "auc_normalized", "by_discretization": {}}
    for disc_index, discretization in enumerate(protocol["discretizations"]):
        disc_rows = [row for row in rows if row["discretization"] == discretization]
        by_algorithm = {
            algorithm: {int(row["seed"]): float(row["auc_normalized"]) for row in disc_rows if row["algorithm"] == algorithm}
            for algorithm in ALGORITHMS
        }
        common_seeds = sorted(set.intersection(*(set(values) for values in by_algorithm.values())))
        if len(common_seeds) < 2:
            raise ValueError(f"Sementes pareadas insuficientes em {discretization}.")
        arrays = {algorithm: np.asarray([by_algorithm[algorithm][seed] for seed in common_seeds]) for algorithm in ALGORITHMS}
        all_tied = all(np.array_equal(arrays[ALGORITHMS[0]], arrays[a]) for a in ALGORITHMS[1:])
        friedman = None if all_tied else stats.friedmanchisquare(*(arrays[algorithm] for algorithm in ALGORITHMS))
        pairs = []
        raw_p = []
        for pair_index, (left, right) in enumerate(combinations(ALGORITHMS, 2)):
            try:
                if np.array_equal(arrays[left], arrays[right]):
                    raise ValueError('All paired differences zero')
                wilcoxon = stats.wilcoxon(arrays[left], arrays[right], zero_method="wilcox", alternative="two-sided")
                statistic = float(wilcoxon.statistic)
                p_value = float(wilcoxon.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            mean_difference, ci_low, ci_high = paired_bootstrap_ci(
                arrays[left], arrays[right], resamples, confidence, base_seed + disc_index * 10 + pair_index
            )
            raw_p.append(p_value)
            pairs.append({
                "left": left,
                "right": right,
                "n_pairs": len(common_seeds),
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "rank_biserial_left_minus_right": paired_rank_biserial(arrays[left], arrays[right]),
                "mean_difference_left_minus_right": mean_difference,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
            })
        adjusted = holm_adjust(raw_p)
        for pair, p_adjusted in zip(pairs, adjusted):
            pair["p_holm"] = p_adjusted
        output["by_discretization"][discretization] = {
            "seeds": common_seeds,
            "friedman": {"statistic": None if all_tied else float(friedman.statistic),
                         "p_value": None if all_tied else float(friedman.pvalue),
                         "status": 'undefined_all_tied' if all_tied else 'computed'},
            "post_hoc": pairs,
        }
    return output

