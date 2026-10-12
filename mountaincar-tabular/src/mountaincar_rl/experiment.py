from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import platform
import sys
import subprocess
import os
import traceback

import gymnasium
import matplotlib
import numpy
import pandas
import scipy
import seaborn

from .agents import evaluate_agent, train_agent
from .config import config_hash, run_filename
from .checkpoints import save_checkpoint, load_checkpoint, file_hash
from .storage import (
    RESULT_SCHEMA_VERSION,
    atomic_write_json,
    finalize_result,
    inspect_result_file,
    quarantine_invalid,
    utc_now,
    atomic_write_csv,
    load_json,
)


LOGGER = logging.getLogger(__name__)


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gymnasium": gymnasium.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
    }


def execute_config(config: dict[str, Any], checkpoint_path: Path | None = None) -> dict[str, Any]:
    started_at = utc_now()
    q, training = train_agent(config)
    evaluation = evaluate_agent(q, config)
    summary = {
        "success_rate": evaluation["success_rate"],
        "mean_steps_success": evaluation["mean_steps_success"],
        "mean_steps_all": evaluation["mean_steps_all"],
        "total_env_steps": training["cumulative_env_steps"][-1],
        "total_q_updates": training["cumulative_q_updates"][-1],
        "training_time_seconds": training["training_time_seconds"],
        "evaluation_time_seconds": evaluation["evaluation_time_seconds"],
    }
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "config_hash": config_hash(config),
        "config": config,
        "versions": library_versions(),
        "timing": {
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
        },
        "training_metrics": training,
        "evaluation_metrics": evaluation,
        "summary": summary,
    }
    result['provenance'] = {'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                          cwd=Path(__file__).resolve().parents[2], text=True).strip()),
        'experiment_id': config.get('experiment_id', config['experiment_version'])}
    if checkpoint_path is not None:
        result['checkpoint'] = {'file': checkpoint_path.name,
                                'sha256': save_checkpoint(checkpoint_path, q, config)}
    return finalize_result(result)


def execute_and_store(config: dict[str, Any], output_dir: Path | str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if config.get('phase') == 'triagem' or 'triagem' in output_dir.resolve().parts:
        raise PermissionError('Historical triage is read-only; use a new versioned campaign')
    if not config.get('experiment_id'):
        raise ValueError('New runs require an experiment_id')
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / run_filename(config)
    if target.exists():
        valid, errors = inspect_result_file(target, config)
        if valid:
            saved = load_json(target)
            cp = saved['checkpoint']
            load_checkpoint(output_dir / cp['file'], cp['sha256'], config)
            for item in saved['tables']:
                if file_hash(output_dir / item['file']) != item['sha256']:
                    raise ValueError('Cached CSV hash mismatch')
            return {"status": "cached", "path": str(target), "config_hash": config_hash(config)}
        raise ValueError(f'Invalid cached result preserved for inspection: {errors}')
    checkpoint = target.with_suffix('.npz')
    csv_train = target.with_suffix('.training.csv')
    csv_eval = target.with_suffix('.evaluation.csv')
    lock = target.with_suffix('.lock')
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        if any(p.exists() for p in (target, checkpoint, csv_train, csv_eval)):
            raise FileExistsError('Partial bundle preserved; inspect it or use a fresh campaign directory')
        result = execute_config(config, checkpoint)
        tables = []
        for path, metrics in [(csv_train, result['training_metrics']), (csv_eval, result['evaluation_metrics'])]:
            arrays = {k: v for k, v in metrics.items() if isinstance(v, list)}
            rows = [dict(episode=i + 1, **{k: v[i] for k, v in arrays.items()})
                    for i in range(len(next(iter(arrays.values()))))]
            atomic_write_csv(path, rows)
            tables.append({'file': path.name, 'sha256': file_hash(path)})
        result['tables'] = tables
        result = finalize_result(result)
        atomic_write_json(target, result)
    except Exception:
        atomic_write_json(target.with_suffix('.error.json'), {'config': config, 'traceback': traceback.format_exc(), 'utc': utc_now()})
        raise
    finally:
        lock.unlink()
    valid, errors = inspect_result_file(target, config)
    if not valid:
        raise RuntimeError(f"Resultado recém-gravado falhou na validação: {errors}")
    return {"status": "executed", "path": str(target), "config_hash": result["config_hash"]}

