from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .config import expand_final, expand_triage, read_json, run_filename
from .storage import inspect_result_file


def validate_phase(
    protocol_path: Path,
    results_dir: Path,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol["phase"] == "triagem":
        configs = expand_triage(protocol)
    else:
        if selection_path is None or not selection_path.exists():
            return {
                "phase": "final",
                "expected": 60,
                "valid": 0,
                "missing": 60,
                "corrupt": [],
                "selection_missing": True,
                "problems": [],
            }
        configs = expand_final(protocol, read_json(selection_path))

    valid_count = 0
    missing: list[str] = []
    corrupt: list[dict[str, Any]] = []
    expected_names = {run_filename(config) for config in configs}
    for config in configs:
        path = results_dir / run_filename(config)
        if not path.exists():
            missing.append(path.name)
            continue
        valid, errors = inspect_result_file(path, config)
        if valid:
            valid_count += 1
        else:
            corrupt.append({"path": str(path), "errors": errors})
    actual_names = {path.name for path in results_dir.glob("*__*__s*__*.json")}
    unexpected = sorted(actual_names - expected_names)
    return {
        "phase": protocol["phase"],
        "expected": len(configs),
        "valid": valid_count,
        "missing": len(missing),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "corrupt": corrupt,
        "problems": [],
    }


def validate_figure_pairs(directory: Path | str) -> dict[str, Any]:
    directory = Path(directory)
    png_stems = {path.stem for path in directory.glob("*.png") if path.stat().st_size > 0}
    pdf_stems = {path.stem for path in directory.glob("*.pdf") if path.stat().st_size > 0}
    low_resolution = []
    unreadable_png = []
    for path in directory.glob("*.png"):
        try:
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0, 0))
                if min(dpi) < 299:
                    low_resolution.append({"path": str(path), "dpi": list(dpi)})
        except OSError as exc:
            unreadable_png.append({"path": str(path), "error": str(exc)})
    return {
        "directory": str(directory),
        "png_count": len(png_stems),
        "pdf_count": len(pdf_stems),
        "missing_pdf_for_png": sorted(png_stems - pdf_stems),
        "missing_png_for_pdf": sorted(pdf_stems - png_stems),
        "low_resolution_png": low_resolution,
        "unreadable_png": unreadable_png,
    }

