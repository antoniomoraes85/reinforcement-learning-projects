from __future__ import annotations

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.config import read_json, resolve_selection_path
from mountaincar_rl.storage import atomic_write_json, utc_now
from mountaincar_rl.validation import validate_figure_pairs, validate_phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida integridade, cobertura e artefatos do mini-projeto.")
    parser.add_argument("--fase", choices=["triagem", "final", "all"], default="all")
    parser.add_argument("--permitir-incompleto", action="store_true")
    parser.add_argument("--relatorio", type=Path, default=ROOT / "relatorios" / "validacao_resultados.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    if args.fase in {"triagem", "all"}:
        reports.append(
            validate_phase(
                ROOT / "configs" / "protocolo_triagem.json",
                ROOT / "resultados" / "triagem",
            )
        )
    if args.fase in {"final", "all"}:
        final_protocol_path = ROOT / "configs" / "protocolo_final.json"
        final_protocol = read_json(final_protocol_path)
        reports.append(
            validate_phase(
                final_protocol_path,
                ROOT / "resultados" / "final",
                resolve_selection_path(final_protocol_path, final_protocol),
            )
        )
    figures = {
        "triagem": validate_figure_pairs(ROOT / "figuras" / "triagem"),
        "final": validate_figure_pairs(ROOT / "figuras" / "final"),
        "smoke": validate_figure_pairs(ROOT / "figuras" / "triagem" / "smoke"),
    }
    problems = []
    for report in reports:
        problems.extend(report.get("problems", []))
        if report.get("corrupt"):
            problems.append(f"{report['phase']}: arquivos corrompidos")
        if report.get("unexpected_files"):
            problems.append(f"{report['phase']}: resultados inesperados para o protocolo atual")
        if report.get("missing", 0) and not args.permitir_incompleto:
            problems.append(f"{report['phase']}: {report['missing']} resultados ausentes")
    for name, report in figures.items():
        if report["missing_pdf_for_png"] or report["missing_png_for_pdf"]:
            problems.append(f"figuras {name}: pares PNG/PDF incompletos")
        if report["low_resolution_png"]:
            problems.append(f"figuras {name}: PNG abaixo de 300 DPI")
        if report["unreadable_png"]:
            problems.append(f"figuras {name}: PNG ilegível")
    payload = {
        "generated_at_utc": utc_now(),
        "allow_incomplete": args.permitir_incompleto,
        "phases": reports,
        "figures": figures,
        "problems": problems,
        "valid": not problems,
    }
    atomic_write_json(args.relatorio, payload)
    for report in reports:
        print(
            f"{report['phase']}: {report['valid']}/{report['expected']} válidos; "
            f"{report['missing']} ausentes; {len(report['corrupt'])} corrompidos"
        )
    print("Relatório:", args.relatorio)
    if problems:
        print("Problemas:", "; ".join(problems))
        return 1
    print("Validação concluída sem problemas no escopo solicitado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

