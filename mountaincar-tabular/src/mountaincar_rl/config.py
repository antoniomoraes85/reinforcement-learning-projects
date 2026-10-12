from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Iterable
import json
import math


ALGORITHMS = ("q_learning", "sarsa_lambda", "dyna_q")


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(config: dict[str, Any]) -> str:
    return sha256(canonical_json(config).encode("utf-8")).hexdigest()


def run_filename(config: dict[str, Any]) -> str:
    digest = config_hash(config)
    return (
        f"{config['algorithm']}__{config['discretization']['name']}__"
        f"s{config['seed']}__{digest[:16]}.json"
    )


def validate_protocol(protocol: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_version",
        "phase",
        "environment",
        "training",
        "evaluation",
        "seeds",
        "discretizations",
    }
    missing = required.difference(protocol)
    if missing:
        raise ValueError(f"Protocolo sem campos obrigatórios: {sorted(missing)}")
    env = protocol["environment"]
    if env.get("id") != "MountainCar-v0":
        raise ValueError("Somente MountainCar-v0 é aceito.")
    if env.get("action_count") != 3:
        raise ValueError("MountainCar-v0 deve usar exatamente três ações.")
    if env.get("max_episode_steps") != 200:
        raise ValueError("O limite oficial deve permanecer em 200 passos.")
    if not protocol["seeds"] or len(set(protocol["seeds"])) != len(protocol["seeds"]):
        raise ValueError("As sementes devem ser não vazias e únicas.")
    for name, spec in protocol["discretizations"].items():
        expected_shape = [len(spec["position_edges"]) + 1, len(spec["velocity_edges"]) + 1]
        if list(spec["shape"]) != expected_shape:
            raise ValueError(f"Discretização {name} incompatível com seus limiares.")
        for field, low, high in [('position_edges', -1.2, .6), ('velocity_edges', -.07, .07)]:
            edges = spec[field]
            if any(not math.isfinite(x) or not low < x < high for x in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
                raise ValueError(f'Invalid discretization edges: {name}/{field}')
    if int(protocol['training']['episodes']) < 1 or int(protocol['evaluation']['episodes']) < 1:
        raise ValueError('Episode counts must be positive')
    evaluation_seeds = protocol['evaluation'].get('reset_seeds')
    if evaluation_seeds is not None and (len(set(evaluation_seeds)) != len(evaluation_seeds) or len(evaluation_seeds) != protocol['evaluation']['episodes']):
        raise ValueError('Invalid evaluation seed set')


def _base_run(protocol: dict[str, Any], discretization: str, seed: int) -> dict[str, Any]:
    result = {
        "schema_version": protocol["schema_version"],
        "experiment_version": protocol["experiment_version"],
        "phase": protocol["phase"],
        "environment": deepcopy(protocol["environment"]),
        "training": deepcopy(protocol["training"]),
        "evaluation": deepcopy(protocol["evaluation"]),
        "discretization": {
            "name": discretization,
            **deepcopy(protocol["discretizations"][discretization]),
        },
        "seed": int(seed),
    }
    for field in ('experiment_id', 'bootstrap_time_limit'):
        if field in protocol:
            result[field] = deepcopy(protocol[field])
    return result


def expand_triage(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    if protocol["phase"] != "triagem":
        raise ValueError("O protocolo informado não é de triagem.")
    runs: list[dict[str, Any]] = []
    for discretization, seed, algorithm in product(
        protocol["discretizations"], protocol["seeds"], protocol["algorithms"]
    ):
        grid = protocol["algorithms"][algorithm]["grid"]
        names = ("alpha", "lambda", "epsilon_decay", "N")
        for values in product(*(grid[name] for name in names)):
            cfg = _base_run(protocol, discretization, seed)
            cfg["algorithm"] = algorithm
            cfg["hyperparameters"] = dict(zip(names, values))
            if algorithm == "dyna_q":
                cfg["model"] = protocol["algorithms"][algorithm]["model"]
            runs.append(cfg)
    hashes = [config_hash(run) for run in runs]
    if len(hashes) != len(set(hashes)):
        raise ValueError("A expansão da triagem gerou configurações duplicadas.")
    return runs


def _validate_selection(selection: dict[str, Any]) -> None:
    selected = selection.get("selected_hyperparameters", {})
    if set(selected) != set(ALGORITHMS):
        raise ValueError("A seleção deve conter exatamente q_learning, sarsa_lambda e dyna_q.")
    for algorithm, params in selected.items():
        for name in ("alpha", "lambda", "epsilon_decay", "N"):
            if name not in params:
                raise ValueError(f"Seleção de {algorithm} sem {name}.")
    if int(selected["dyna_q"]["N"]) == 0:
        raise ValueError("Dyna-Q com N=0 não pode ser selecionado para o experimento final.")


def expand_final(protocol: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    if protocol["phase"] != "final":
        raise ValueError("O protocolo informado não é final.")
    _validate_selection(selection)
    runs: list[dict[str, Any]] = []
    for discretization, seed, algorithm in product(
        protocol["discretizations"], protocol["seeds"], protocol["algorithms"]
    ):
        cfg = _base_run(protocol, discretization, seed)
        cfg["algorithm"] = algorithm
        cfg["hyperparameters"] = deepcopy(selection["selected_hyperparameters"][algorithm])
        if algorithm == "dyna_q":
            cfg["model"] = "empirical_outcome_distribution"
        cfg["selection_hash"] = selection.get("selection_hash") or config_hash(
            selection["selected_hyperparameters"]
        )
        runs.append(cfg)
    return runs


def resolve_selection_path(protocol_path: Path, protocol: dict[str, Any]) -> Path:
    return (protocol_path.parent / protocol["selection_file"]).resolve()


def filter_runs(
    runs: Iterable[dict[str, Any]],
    algorithm: str | None = None,
    discretization: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for cfg in runs:
        if algorithm is not None and cfg["algorithm"] != algorithm:
            continue
        if discretization is not None and cfg["discretization"]["name"] != discretization:
            continue
        if seed is not None and cfg["seed"] != seed:
            continue
        selected.append(cfg)
    return selected

