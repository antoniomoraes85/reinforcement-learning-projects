from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.metrics import first_stable_episode, holm_adjust, normalized_auc, paired_rank_biserial
from mountaincar_rl.storage import atomic_write_json, inspect_result_file, load_json


def test_metrics_known_values():
    assert np.isclose(normalized_auc([-200, -100, 0], 200), 0.5)
    successes = [False, True, True, True, True]
    assert first_stable_episode(successes, window=2, threshold=1.0, consecutive_windows=2) == 4
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert paired_rank_biserial([3, 4, 5], [1, 2, 3]) == 1.0


def test_atomic_json_roundtrip(tmp_path):
    path = tmp_path / "document.json"
    atomic_write_json(path, {"texto": "ação", "valor": 1})
    assert load_json(path) == {"texto": "ação", "valor": 1}


def test_corrupt_result_is_detected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{incompleto", encoding="utf-8")
    valid, errors = inspect_result_file(path)
    assert not valid
    assert errors

