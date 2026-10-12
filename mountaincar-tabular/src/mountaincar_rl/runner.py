from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import multiprocessing
import os

from .experiment import execute_and_store
from .storage import atomic_write_csv, atomic_write_json, utc_now


def default_workers() -> int:
    return max(1, min(2, os.cpu_count() or 1))


def setup_logging(output_dir: Path | str, phase: str) -> Path:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{phase}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(processName)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)
    return path


def run_batch(
    configs: list[dict[str, Any]],
    output_dir: Path | str,
    workers: int,
    phase: str,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("--workers deve ser pelo menos 1.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.info("Iniciando %s: %d configurações, %d worker(s).", phase, len(configs), workers)
    started = utc_now()
    receipts: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        future_map = {
            executor.submit(execute_and_store, config, output_dir): config for config in configs
        }
        for completed, future in enumerate(as_completed(future_map), start=1):
            config = future_map[future]
            try:
                receipt = future.result()
            except Exception as exc:
                logger.exception(
                    "Falha em %s/%s/seed=%s",
                    config["algorithm"],
                    config["discretization"]["name"],
                    config["seed"],
                )
                receipt = {
                    "status": "failed",
                    "path": None,
                    "config_hash": None,
                    "error": repr(exc),
                }
            receipts.append(receipt)
            logger.info("Progresso %d/%d: %s", completed, len(configs), receipt["status"])

    manifest = {
        "phase": phase,
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "workers": workers,
        "planned": len(configs),
        "executed": sum(item["status"] == "executed" for item in receipts),
        "cached": sum(item["status"] == "cached" for item in receipts),
        "failed": sum(item["status"] == "failed" for item in receipts),
        "receipts": receipts,
    }
    atomic_write_json(output_dir / "ultimo_manifesto_execucao.json", manifest)
    atomic_write_csv(
        output_dir / "ultimo_manifesto_execucao.csv",
        receipts,
        fieldnames=["status", "path", "config_hash", "error"],
    )
    if manifest["failed"]:
        raise RuntimeError(f"{manifest['failed']} execução(ões) falharam; consulte o log e o manifesto.")
    return receipts

