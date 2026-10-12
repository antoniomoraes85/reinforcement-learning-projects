from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.analysis import (
    aggregate_triage_candidates,
    load_valid_results,
    select_hyperparameters,
    triage_rows,
)
from mountaincar_rl.config import config_hash, expand_triage, read_json
from mountaincar_rl.plotting import plot_triage
from mountaincar_rl.storage import atomic_write_csv, atomic_write_json, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa a triagem e seleciona uma configuração por algoritmo.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "protocolo_triagem.json")
    parser.add_argument("--resultados", type=Path, default=ROOT / "resultados" / "triagem")
    parser.add_argument("--figuras", type=Path, default=ROOT / "figuras" / "triagem")
    parser.add_argument("--permitir-incompleto", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = read_json(args.config)
    expected_configs = expand_triage(protocol)
    expected_hashes = {config_hash(config) for config in expected_configs}
    expected = len(expected_configs)
    results, problems = load_valid_results(args.resultados)
    if problems:
        raise SystemExit(f"Há {len(problems)} arquivo(s) inválido(s). Rode validar_resultados.py.")
    actual_hashes = [result["config_hash"] for result in results]
    unexpected = set(actual_hashes) - expected_hashes
    missing = expected_hashes - set(actual_hashes)
    duplicates = len(actual_hashes) - len(set(actual_hashes))
    if unexpected or duplicates:
        raise SystemExit(
            f"Resultados incompatíveis: {len(unexpected)} configurações inesperadas e {duplicates} duplicatas."
        )
    if missing and not args.permitir_incompleto:
        raise SystemExit(f"Triagem incompleta: {len(results)}/{expected} resultados válidos.")
    if not results:
        raise SystemExit("Nenhum resultado de triagem válido foi encontrado.")

    rows = triage_rows(results, protocol)
    candidates = aggregate_triage_candidates(rows)
    atomic_write_csv(args.resultados / "resumo_execucoes.csv", rows)
    atomic_write_csv(args.resultados / "resumo_configuracoes.csv", candidates)
    summary = {
        "generated_at_utc": utc_now(),
        "experiment_version": protocol["experiment_version"],
        "expected_runs": expected,
        "valid_runs": len(results),
        "complete": not missing,
        "missing_run_count": len(missing),
        "candidate_configurations": candidates,
    }
    atomic_write_json(args.resultados / "resumo_triagem.json", summary)
    figures = plot_triage(rows, candidates, args.figuras)

    if not missing:
        selection = select_hyperparameters(candidates, protocol)
        atomic_write_json(args.resultados / "hiperparametros_selecionados.json", selection)
        print("Seleção gravada em:", args.resultados / "hiperparametros_selecionados.json")
    else:
        print("Análise parcial: nenhuma seleção final foi gravada.")
    print(f"Resultados válidos: {len(results)}/{expected}")
    print(f"Figuras geradas: {len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

