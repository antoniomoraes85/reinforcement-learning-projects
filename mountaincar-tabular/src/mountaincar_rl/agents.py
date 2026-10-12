from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
import math
import time

import gymnasium as gym
import numpy as np


ACTIONS = (0, 1, 2)


def episode_seed(base_seed: int, stage: int, episode: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(stage), int(episode)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class Discretizer:
    position_edges: tuple[float, ...]
    velocity_edges: tuple[float, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Discretizer":
        spec = config["discretization"]
        return cls(tuple(spec["position_edges"]), tuple(spec["velocity_edges"]))

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.position_edges) + 1, len(self.velocity_edges) + 1

    def transform(self, observation: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        position, velocity = observation
        return (
            int(np.digitize(position, self.position_edges)),
            int(np.digitize(velocity, self.velocity_edges)),
        )


class EmpiricalTransitionModel:
    """Distribuição empírica de resultados por par (estado discreto, ação)."""

    def __init__(self) -> None:
        self.outcomes: dict[tuple[tuple[int, int], int], Counter] = defaultdict(Counter)
        self.seen_pairs: list[tuple[tuple[int, int], int]] = []

    def observe(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        terminated: bool,
    ) -> None:
        key = (tuple(state), int(action))
        if key not in self.outcomes:
            self.seen_pairs.append(key)
        outcome = (float(reward), tuple(next_state), bool(terminated))
        self.outcomes[key][outcome] += 1

    def sample(self, rng: np.random.Generator):
        if not self.seen_pairs:
            raise RuntimeError("Não é possível amostrar um modelo vazio.")
        key = self.seen_pairs[int(rng.integers(len(self.seen_pairs)))]
        counter = self.outcomes[key]
        outcomes = list(counter)
        counts = np.asarray([counter[outcome] for outcome in outcomes], dtype=float)
        outcome = outcomes[int(rng.choice(len(outcomes), p=counts / counts.sum()))]
        return key, outcome


def make_env(config: dict[str, Any]):
    env_cfg = config["environment"]
    env = gym.make(env_cfg["id"])
    actual_limit = getattr(env.spec, "max_episode_steps", None)
    if actual_limit != env_cfg["max_episode_steps"] or actual_limit != 200:
        env.close()
        raise RuntimeError(f"TimeLimit inesperado: {actual_limit}; esperado: 200")
    if env.action_space.n != env_cfg["action_count"] or env.action_space.n != 3:
        env.close()
        raise RuntimeError(f"Espaço de ações inesperado: {env.action_space}")
    return env


def epsilon_greedy(
    q: np.ndarray,
    state: tuple[int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(ACTIONS))
    values = q[state]
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def scalar_q_update(
    q: np.ndarray,
    state: tuple[int, int],
    action: int,
    target: float,
    alpha: float,
) -> tuple[float, float]:
    index = state + (action,)
    old_value = float(q[index])
    q[index] += alpha * (target - old_value)
    return old_value, float(q[index])


def planning_update(
    q: np.ndarray,
    model: EmpiricalTransitionModel,
    alpha: float,
    gamma: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    (state, action), (reward, next_state, terminated) = model.sample(rng)
    target = reward if terminated else reward + gamma * float(np.max(q[next_state]))
    old_value, new_value = scalar_q_update(q, state, action, target, alpha)
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "terminated": terminated,
        "target": target,
        "old_value": old_value,
        "new_value": new_value,
    }


def sarsa_trace_update(q, traces, state, action, reward, next_state, next_action,
                       terminated, alpha, gamma, lambda_):
    """Accumulating traces; bootstrap on time limits, never on true terminal."""
    bootstrap = 0.0 if terminated else gamma * q[next_state + (next_action,)]
    delta = float(reward) + bootstrap - q[state + (action,)]
    traces[state + (action,)] += 1.0
    active = int(np.count_nonzero(traces))
    q += alpha * delta * traces
    traces *= gamma * lambda_
    return float(delta), active


def train_agent(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    algorithm = config["algorithm"]
    if algorithm not in {"q_learning", "sarsa_lambda", "dyna_q"}:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")
    hp = config["hyperparameters"]
    training = config["training"]
    rng = np.random.default_rng(config["seed"])
    discretizer = Discretizer.from_config(config)
    q = np.zeros(discretizer.shape + (3,), dtype=np.float64)
    model = EmpiricalTransitionModel() if algorithm == "dyna_q" else None
    env = make_env(config)
    env.action_space.seed(config["seed"])

    returns: list[float] = []
    steps_per_episode: list[int] = []
    successes: list[bool] = []
    cumulative_env_steps: list[int] = []
    cumulative_q_updates: list[int] = []
    cumulative_planning = []
    cumulative_trace_components = []
    epsilons, terminations, truncations, reset_seeds = [], [], [], []
    planning_total = trace_total = 0
    env_steps_total = 0
    q_updates_total = 0
    epsilon = float(training["epsilon_start"])

    started = time.perf_counter()
    try:
        for episode in range(int(training["episodes"])):
            reset_seed = episode_seed(config["seed"], 1, episode)
            reset_seeds.append(reset_seed)
            epsilons.append(float(epsilon))
            observation, _ = env.reset(seed=reset_seed)
            state = discretizer.transform(observation)
            episode_return = 0.0
            steps = 0
            terminated = False

            if algorithm == "sarsa_lambda":
                traces = np.zeros_like(q)
                action = epsilon_greedy(q, state, epsilon, rng)

            while True:
                if algorithm != "sarsa_lambda":
                    action = epsilon_greedy(q, state, epsilon, rng)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = discretizer.transform(next_observation)
                episode_return += float(reward)
                steps += 1
                env_steps_total += 1

                if algorithm == "sarsa_lambda":
                    next_action = None if terminated else epsilon_greedy(q, next_state, epsilon, rng)
                    _, active_updates = sarsa_trace_update(
                        q, traces, state, action, reward, next_state, next_action,
                        terminated, float(hp['alpha']), float(training['gamma']), float(hp['lambda']))
                    q_updates_total += active_updates
                    trace_total += active_updates
                else:
                    target = (
                        float(reward)
                        if terminated
                        else float(reward) + float(training["gamma"]) * float(np.max(q[next_state]))
                    )
                    scalar_q_update(q, state, action, target, float(hp["alpha"]))
                    q_updates_total += 1
                    if algorithm == "dyna_q":
                        model.observe(state, action, reward, next_state, terminated)
                        for _ in range(int(hp["N"])):
                            planning_update(q, model, float(hp["alpha"]), float(training["gamma"]), rng)
                            q_updates_total += 1
                            planning_total += 1

                if terminated or truncated:
                    break
                state = next_state
                if algorithm == "sarsa_lambda":
                    action = next_action

            returns.append(float(episode_return))
            steps_per_episode.append(int(steps))
            successes.append(bool(terminated))
            cumulative_env_steps.append(int(env_steps_total))
            cumulative_q_updates.append(int(q_updates_total))
            cumulative_planning.append(planning_total)
            cumulative_trace_components.append(trace_total)
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))
            epsilon = max(float(training["epsilon_min"]), epsilon * float(hp["epsilon_decay"]))
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    return q, {
        "episode_returns": returns,
        "episode_steps": steps_per_episode,
        "episode_successes": successes,
        "cumulative_env_steps": cumulative_env_steps,
        "cumulative_q_updates": cumulative_q_updates,
        "cumulative_real_td_updates": cumulative_env_steps.copy(),
        "cumulative_planning_updates": cumulative_planning,
        "cumulative_trace_components": cumulative_trace_components,
        "episode_epsilon": epsilons,
        "episode_terminated": terminations,
        "episode_truncated": truncations,
        "episode_reset_seeds": reset_seeds,
        "training_time_seconds": float(elapsed),
        "final_epsilon": float(epsilon),
        "empirical_model_distinct_pairs": len(model.seen_pairs) if model is not None else 0,
        "empirical_model_distinct_outcomes": (
            sum(len(counter) for counter in model.outcomes.values()) if model is not None else 0
        ),
    }


def evaluate_agent(q: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = config['evaluation']
    seeds = eval_cfg.get('reset_seeds', [episode_seed(config['seed'], 2, i)
                                       for i in range(int(eval_cfg['episodes']))])
    if len(seeds) != int(eval_cfg['episodes']) or len(set(seeds)) != len(seeds):
        raise ValueError('Evaluation seeds must be unique and match episode count')
    discretizer = Discretizer.from_config(config)
    env = make_env(config)
    successes: list[bool] = []
    steps_list: list[int] = []
    returns, terminals, truncations, latencies = [], [], [], []
    started = time.perf_counter()
    try:
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 3]))
            observation, _ = env.reset(seed=int(seed))
            state = discretizer.transform(observation)
            total_return = 0.0
            for step in range(1, int(config["environment"]["max_episode_steps"]) + 1):
                inference_start = time.perf_counter_ns()
                values = q[state]
                best = np.flatnonzero(values == values.max())
                action = int(rng.choice(best))
                latencies.append(time.perf_counter_ns() - inference_start)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                state = discretizer.transform(next_observation)
                if terminated or truncated:
                    successes.append(bool(terminated))
                    steps_list.append(int(step))
                    returns.append(total_return)
                    terminals.append(bool(terminated))
                    truncations.append(bool(truncated))
                    break
    finally:
        env.close()
    successful_steps = [steps for steps, success in zip(steps_list, successes) if success]
    elapsed = time.perf_counter() - started
    return {
        "successes": successes,
        "steps": steps_list,
        "returns": returns,
        "terminated": terminals,
        "truncated": truncations,
        "reset_seeds": list(seeds),
        "inference_calls": len(latencies),
        "inference_time_ns": int(sum(latencies)),
        "inference_mean_ns": float(np.mean(latencies)),
        "inference_p95_ns": float(np.percentile(latencies, 95)),
        "inference_scope": "Q lookup, argmax and random tie-break; excludes discretization, env.step and rendering",
        "success_rate": float(np.mean(successes)),
        "mean_steps_success": float(np.mean(successful_steps)) if successful_steps else None,
        "mean_steps_all": float(np.mean(steps_list)),
        "evaluation_time_seconds": float(elapsed),
    }

