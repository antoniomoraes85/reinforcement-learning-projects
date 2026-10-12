from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.config import expand_final, read_json, resolve_selection_path
from mountaincar_rl.runner import default_workers, run_batch, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o experimento final do MountainCar-v0.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "protocolo_final_v2.json")
    parser.add_argument("--selecao", type=Path)
    parser.add_argument("--saida", type=Path, default=ROOT / "resultados" / "cmp629-final-v2-20261011")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument('--autorizado-apos-go', action='store_true', help='Only after explicit user approval')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.autorizado_apos_go:
        raise SystemExit('BLOCKED: 60 final trainings require explicit authorization after GO.')
    gate = ROOT / 'relatorios' / 'gate_fase_a.json'
    if not gate.exists() or read_json(gate).get('decision') != 'GO':
        raise SystemExit('BLOCKED: missing GO evidence from phase A')
    protocol = read_json(args.config)
    if not protocol.get('experiment_id') or not protocol['evaluation'].get('reset_seeds'):
        raise SystemExit('Use the audited v2 protocol with a new experiment_id and fixed evaluation seeds')
    selection_path = args.selecao or resolve_selection_path(args.config, protocol)
    if not selection_path.exists():
        raise SystemExit(
            f"Seleção ausente: {selection_path}. Execute e analise a triagem antes do experimento final."
        )
    configs = expand_final(protocol, read_json(selection_path))
    if len(configs) != 60:
        raise SystemExit(f"Protocolo final deveria gerar 60 treinamentos, mas gerou {len(configs)}.")
    log_path = setup_logging(args.saida, "final")
    run_batch(configs, args.saida, args.workers, "final")
    print("Log:", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

