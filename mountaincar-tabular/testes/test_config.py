from copy import deepcopy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.config import config_hash, expand_final, expand_triage, read_json


def test_protocol_counts_and_seed_separation():
    triage = read_json(ROOT / "configs" / "protocolo_triagem.json")
    final = read_json(ROOT / "configs" / "protocolo_final.json")
    triage_runs = expand_triage(triage)
    assert len(triage_runs) == 108
    selection = {
        "selected_hyperparameters": {
            "q_learning": {"alpha": 0.1, "lambda": 0.0, "epsilon_decay": 0.995, "N": 0},
            "sarsa_lambda": {"alpha": 0.05, "lambda": 0.9, "epsilon_decay": 0.995, "N": 0},
            "dyna_q": {"alpha": 0.1, "lambda": 0.0, "epsilon_decay": 0.995, "N": 5},
        }
    }
    final_runs = expand_final(final, selection)
    assert len(final_runs) == 60
    assert set(triage["seeds"]).isdisjoint(final["seeds"])


def test_hash_changes_when_required_field_changes():
    protocol = read_json(ROOT / "configs" / "protocolo_triagem.json")
    base = expand_triage(protocol)[0]
    original = config_hash(base)
    paths = [
        ("algorithm", "sarsa_lambda"),
        ("seed", 999),
    ]
    for field, value in paths:
        changed = deepcopy(base)
        changed[field] = value
        assert config_hash(changed) != original
    nested_changes = [
        ("discretization", "name", "outro"),
        ("hyperparameters", "alpha", 0.123),
        ("hyperparameters", "lambda", 0.456),
        ("hyperparameters", "epsilon_decay", 0.9),
        ("hyperparameters", "N", 7),
        ("training", "episodes", 999),
        ("training", "gamma", 0.5),
        ("training", "epsilon_start", 0.8),
        ("training", "epsilon_min", 0.01),
    ]
    for section, field, value in nested_changes:
        changed = deepcopy(base)
        changed[section][field] = value
        assert config_hash(changed) != original

