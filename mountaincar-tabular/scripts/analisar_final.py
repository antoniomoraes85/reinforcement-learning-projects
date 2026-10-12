from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.analysis import (
    curve_rows,
    descriptive_summary,
    final_seed_rows,
    inferential_tests,
    load_valid_results,
)
from mountaincar_rl.config import config_hash, expand_final, read_json, resolve_selection_path
from mountaincar_rl.plotting import plot_final
from mountaincar_rl.storage import atomic_write_csv, atomic_write_json, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa o experimento final e gera estatística e gráficos.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "protocolo_final.json")
    parser.add_argument("--selecao", type=Path)
    parser.add_argument("--resultados", type=Path, default=ROOT / "resultados" / "final")
    parser.add_argument("--figuras", type=Path, default=ROOT / "figuras" / "final")
    parser.add_argument("--relatorios", type=Path, default=ROOT / "relatorios")
    parser.add_argument("--permitir-incompleto", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = read_json(args.config)
    selection_path = args.selecao or resolve_selection_path(args.config, protocol)
    if not selection_path.exists():
        raise SystemExit(f"Seleção da triagem ausente: {selection_path}")
    selection = read_json(selection_path)
    expected_configs = expand_final(protocol, selection)
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
        raise SystemExit(f"Experimento final incompleto: {len(results)}/{expected} resultados válidos.")
    if not results:
        raise SystemExit("Nenhum resultado final válido foi encontrado.")
    if missing:
        raise SystemExit("A inferência pareada não será executada com o experimento incompleto.")

    seed_rows = final_seed_rows(results, protocol)
    descriptive = descriptive_summary(seed_rows)
    inferential = inferential_tests(seed_rows, protocol)
    confidence = float(protocol["analysis"]["confidence_level"])
    rolling = int(protocol["analysis"]["rolling_window"])
    steps_curve = curve_rows(results, "episode_steps", confidence, rolling)
    updates_curve = curve_rows(results, "cumulative_q_updates", confidence, None)
    success_curve = curve_rows(results, "episode_successes", confidence, rolling)

    args.relatorios.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(args.relatorios / "metricas_por_semente.csv", seed_rows)
    atomic_write_csv(args.relatorios / "resumo_descritivo_final.csv", descriptive)
    atomic_write_csv(args.relatorios / "curva_passos_ic95.csv", steps_curve)
    atomic_write_csv(args.relatorios / "curva_atualizacoes_q_ic95.csv", updates_curve)
    atomic_write_csv(args.relatorios / "curva_sucesso_movel_ic95.csv", success_curve)
    report = {
        "generated_at_utc": utc_now(),
        "experiment_version": protocol["experiment_version"],
        "selection_hash": selection["selection_hash"],
        "run_count": len(results),
        "unit_of_analysis": "seed",
        "descriptive": descriptive,
        "inferential": inferential,
    }
    atomic_write_json(args.relatorios / "analise_estatistica_final.json", report)
    figures = plot_final(results, seed_rows, protocol, args.figuras)
    print(f"Resultados válidos: {len(results)}/{expected}")
    print(f"Arquivos de figura gerados: {len(figures)}")
    print("Relatório estatístico:", args.relatorios / "analise_estatistica_final.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

