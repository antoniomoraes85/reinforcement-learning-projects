from copy import deepcopy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mountaincar_rl.agents import EmpiricalTransitionModel, planning_update, scalar_q_update, train_agent
from mountaincar_rl.config import expand_triage, read_json


def small_config():
    cfg = deepcopy(expand_triage(read_json(ROOT / "configs" / "protocolo_triagem.json"))[0])
    cfg["training"]["episodes"] = 3
    cfg["evaluation"]["episodes"] = 1
    return cfg


def test_manual_q_update_and_empirical_model():
    q = np.zeros((4, 2, 3))
    q[0, 0, 1] = 1.0
    q[1, 1, 2] = 2.0
    target = -1.0 + 0.99 * 2.0
    _, new_value = scalar_q_update(q, (0, 0), 1, target, 0.1)
    assert np.isclose(new_value, 0.998)

    model = EmpiricalTransitionModel()
    model.observe((0, 0), 1, -1, (0, 1), False)
    model.observe((0, 0), 1, -1, (0, 1), False)
    model.observe((0, 0), 1, -1, (1, 1), True)
    assert len(model.outcomes[((0, 0), 1)]) == 2
    assert sum(model.outcomes[((0, 0), 1)].values()) == 3
    q2 = np.zeros((4, 2, 3))
    detail = planning_update(q2, model, 0.1, 0.99, np.random.default_rng(7))
    assert np.isclose(detail["new_value"], 0.1 * detail["target"])


def test_dyna_n_zero_equivalent_to_q_learning():
    q_config = small_config()
    dyna_config = deepcopy(q_config)
    dyna_config["algorithm"] = "dyna_q"
    dyna_config["hyperparameters"]["N"] = 0
    dyna_config["model"] = "empirical_outcome_distribution"
    q_table, q_metrics = train_agent(q_config)
    dyna_table, dyna_metrics = train_agent(dyna_config)
    np.testing.assert_array_equal(q_table, dyna_table)
    for field in (
        "episode_returns",
        "episode_steps",
        "episode_successes",
        "cumulative_env_steps",
        "cumulative_q_updates",
    ):
        assert q_metrics[field] == dyna_metrics[field]


def test_training_is_deterministic_excluding_time():
    config = small_config()
    q1, metrics1 = train_agent(config)
    q2, metrics2 = train_agent(config)
    np.testing.assert_array_equal(q1, q2)
    for field in (
        "episode_returns",
        "episode_steps",
        "episode_successes",
        "cumulative_env_steps",
        "cumulative_q_updates",
    ):
        assert metrics1[field] == metrics2[field]
    assert max(metrics1["episode_steps"]) <= 200

