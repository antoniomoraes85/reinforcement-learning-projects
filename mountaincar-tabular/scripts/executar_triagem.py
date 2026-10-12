from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.config import expand_triage, filter_runs, read_json
from mountaincar_rl.plotting import plot_smoke
from mountaincar_rl.runner import default_workers, run_batch, setup_logging
from mountaincar_rl.storage import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa a triagem reprodutível do MountainCar-v0.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "protocolo_triagem.json")
    parser.add_argument("--saida", type=Path, default=ROOT / "resultados" / "triagem")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--algoritmo", choices=["q_learning", "sarsa_lambda", "dyna_q"])
    parser.add_argument("--discretizacao", choices=["4x2", "12x4"])
    parser.add_argument("--semente", type=int)
    parser.add_argument("--smoke", action="store_true", help="Executa exatamente 1 configuração e no máximo 5 episódios.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = read_json(args.config)
    configs = expand_triage(protocol)
    if args.smoke:
        configs = filter_runs(configs, "q_learning", "4x2", 100)[:1]
        config = configs[0]
        config["experiment_version"] += "-smoke"
        config["training"]["episodes"] = 5
        config["evaluation"]["episodes"] = 2
        output_dir = args.saida / "smoke"
        workers = 1
    else:
        configs = filter_runs(configs, args.algoritmo, args.discretizacao, args.semente)
        output_dir = args.saida
        workers = args.workers
    if not configs:
        raise SystemExit("Nenhuma configuração corresponde aos filtros.")
    log_path = setup_logging(output_dir, "triagem-smoke" if args.smoke else "triagem")
    receipts = run_batch(configs, output_dir, workers, "triagem-smoke" if args.smoke else "triagem")
    if args.smoke:
        result = load_json(receipts[0]["path"])
        figures = plot_smoke(result, ROOT / "figuras" / "triagem" / "smoke")
        print("Smoke test concluído.")
        print("Resultado:", receipts[0]["path"])
        print("Figuras:", ", ".join(str(path) for path in figures))
    print("Log:", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

