from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import csv
import json
import os
import tempfile
import numpy as np

from .config import canonical_json, config_hash


RESULT_SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_replace_text(path: Path, text: str) -> None:
    if 'triagem' in path.resolve().parts:
        raise PermissionError('Historical triage is logically read-only')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path | str, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_replace_text(Path(path), text)


def atomic_write_csv(path: Path | str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    target = Path(path)
    if 'triagem' in target.resolve().parts:
        raise PermissionError('Historical triage is logically read-only')
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def result_hash(result: dict[str, Any]) -> str:
    payload = {key: value for key, value in result.items() if key != "result_hash"}
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def finalize_result(result: dict[str, Any]) -> dict[str, Any]:
    completed = dict(result)
    completed["result_hash"] = result_hash(completed)
    return completed


def load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_result_document(
    result: dict[str, Any], expected_config: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    required = {
        "result_schema_version",
        "status",
        "config_hash",
        "config",
        "versions",
        "timing",
        "training_metrics",
        "evaluation_metrics",
        "summary",
        "result_hash",
    }
    missing = required.difference(result)
    if missing:
        return [f"campos ausentes: {sorted(missing)}"]
    if result["result_schema_version"] != RESULT_SCHEMA_VERSION:
        errors.append("versão de schema de resultado incompatível")
    if result["status"] != "complete":
        errors.append("status diferente de complete")
    if result["config_hash"] != config_hash(result["config"]):
        errors.append("hash da configuração inválido")
    if expected_config is not None and result["config_hash"] != config_hash(expected_config):
        errors.append("resultado pertence a outra configuração")
    if result["result_hash"] != result_hash(result):
        errors.append("hash do resultado inválido")

    cfg = result["config"]
    train = result["training_metrics"]
    evaluation = result["evaluation_metrics"]
    episodes = int(cfg["training"]["episodes"])
    eval_episodes = int(cfg["evaluation"]["episodes"])
    train_fields = (
        "episode_returns",
        "episode_steps",
        "episode_successes",
        "cumulative_env_steps",
        "cumulative_q_updates",
    )
    for field in train_fields:
        if len(train.get(field, [])) != episodes:
            errors.append(f"{field} não tem {episodes} elementos")
    if len(evaluation.get("successes", [])) != eval_episodes:
        errors.append("quantidade incorreta de sucessos de avaliação")
    if len(evaluation.get("steps", [])) != eval_episodes:
        errors.append("quantidade incorreta de passos de avaliação")
    max_steps = int(cfg["environment"]["max_episode_steps"])
    if any(not 1 <= int(value) <= max_steps for value in train.get("episode_steps", [])):
        errors.append("passos de treino fora do limite")
    if any(not 1 <= int(value) <= max_steps for value in evaluation.get("steps", [])):
        errors.append("passos de avaliação fora do limite")
    for field in ("cumulative_env_steps", "cumulative_q_updates"):
        values = train.get(field, [])
        if any(b < a for a, b in zip(values, values[1:])):
            errors.append(f"{field} não é monotônica")
    if cfg["environment"].get("action_count") != 3:
        errors.append("configuração não usa três ações")
    if cfg["environment"].get("max_episode_steps") != 200:
        errors.append("configuração não usa o limite de 200 passos")
    if train.get('episode_steps'):
        if not np.array_equal(np.cumsum(train['episode_steps']), train['cumulative_env_steps']):
            errors.append('Interaction counts do not reconcile')
        if not np.array_equal(-np.asarray(train['episode_steps']), train['episode_returns']):
            errors.append('MountainCar returns must equal negative steps')
    if evaluation.get('steps'):
        if not np.isclose(np.mean(evaluation['successes']), evaluation['success_rate']):
            errors.append('Evaluation success summary inconsistent')
        if not np.isclose(np.mean(evaluation['steps']), evaluation['mean_steps_all']):
            errors.append('Evaluation duration summary inconsistent')
    if cfg.get('experiment_id'):
        for key in ['episode_epsilon','episode_terminated','episode_truncated','episode_reset_seeds',
                    'cumulative_real_td_updates','cumulative_planning_updates','cumulative_trace_components']:
            if len(train.get(key,[])) != episodes:
                errors.append(f'Missing new training metric: {key}')
        for key in ['returns','terminated','truncated','reset_seeds']:
            if len(evaluation.get(key,[])) != eval_episodes:
                errors.append(f'Missing new evaluation metric: {key}')
        if not errors:
            real=np.asarray(train['cumulative_env_steps'])
            if not np.array_equal(real,train['cumulative_real_td_updates']):
                errors.append('Real TD counts inconsistent')
            planning=real*int(cfg['hyperparameters']['N']) if cfg['algorithm']=='dyna_q' else np.zeros_like(real)
            if not np.array_equal(planning,train['cumulative_planning_updates']):
                errors.append('Planning counts inconsistent')
            if not np.array_equal(evaluation['returns'],-np.asarray(evaluation['steps'])):
                errors.append('Evaluation returns inconsistent')
    return errors


def inspect_result_file(path: Path | str, expected_config: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    try:
        result = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, [f"não foi possível ler JSON: {exc}"]
    try:
        errors = validate_result_document(result, expected_config)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors = [f'malformed result: {exc}']
    return not errors, errors


def quarantine_invalid(path: Path | str) -> Path:
    source = Path(path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = source.with_suffix(source.suffix + f".corrompido-{timestamp}")
    counter = 1
    while destination.exists():
        destination = source.with_suffix(source.suffix + f".corrompido-{timestamp}-{counter}")
        counter += 1
    source.replace(destination)
    return destination

