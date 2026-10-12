from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .analysis import curve_rows


PALETTE = {
    "q_learning": "#2F6B9A",
    "sarsa_lambda": "#D28E2A",
    "dyna_q": "#6B8E23",
}
DISC_PALETTE = {"4x2": "#6F96B5", "12x4": "#D6A04A"}
LINESTYLES = {"q_learning": "-", "sarsa_lambda": "--", "dyna_q": "-."}
LABELS = {
    "q_learning": "Q-learning",
    "sarsa_lambda": "SARSA(λ)",
    "dyna_q": "Dyna-Q",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": "#D9DEE5",
            "text.color": "#252A31",
            "axes.labelcolor": "#252A31",
        }
    )


def save_figure(fig: plt.Figure, base_path: Path | str) -> list[Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]


def plot_smoke(result: dict[str, Any], output_dir: Path | str) -> list[Path]:
    configure_style()
    metrics = result["training_metrics"]
    episodes = np.arange(1, len(metrics["episode_steps"]) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(episodes, metrics["episode_steps"], color=PALETTE[result["config"]["algorithm"]], marker="o")
    ax.set_title("Smoke test: passos por episódio")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Passos no episódio")
    ax.set_xlim(1, max(episodes))
    ax.set_ylim(0, 205)
    ax.text(
        0.99,
        0.02,
        "Execução técnica de até 5 episódios; não é resultado experimental.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#5D6673",
    )
    return save_figure(fig, Path(output_dir) / "smoke_exemplo")


def _heatmap_axes(algorithm: str) -> tuple[str, str]:
    if algorithm == "q_learning":
        return "epsilon_decay", "alpha"
    if algorithm == "sarsa_lambda":
        return "lambda", "alpha"
    return "N", "alpha"


def _plot_triage_heatmaps(
    candidates: list[dict[str, Any]],
    value: str,
    title: str,
    output_base: Path,
    percent: bool = False,
) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    frame = pd.DataFrame(candidates)
    for ax, algorithm in zip(axes, PALETTE):
        x_field, y_field = _heatmap_axes(algorithm)
        subset = frame[frame["algorithm"] == algorithm]
        pivot = subset.pivot(index=y_field, columns=x_field, values=value).sort_index(ascending=False)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="Blues",
            annot=True,
            fmt=".1%" if percent else ".3f",
            cbar=ax is axes[-1],
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(LABELS[algorithm])
        ax.set_xlabel(x_field)
        ax.set_ylabel(y_field)
    fig.suptitle(title, fontsize=14)
    return save_figure(fig, output_base)


def plot_triage(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    output_dir = Path(output_dir)
    files: list[Path] = []
    files += _plot_triage_heatmaps(
        candidates,
        "auc_normalized_mean",
        "Triagem: AUC normalizada média por hiperparâmetros",
        output_dir / "heatmap_auc_hiperparametros",
    )
    files += _plot_triage_heatmaps(
        candidates,
        "final_window_success_mean",
        "Triagem: sucesso na janela final por hiperparâmetros",
        output_dir / "heatmap_sucesso_final",
        percent=True,
    )

    configure_style()
    dyna = pd.DataFrame([candidate for candidate in candidates if candidate["algorithm"] == "dyna_q"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for alpha, group in dyna.groupby("alpha"):
        ordered = group.sort_values("N")
        axes[0].plot(ordered["N"], ordered["auc_normalized_mean"], marker="o", label=f"α={alpha:g}")
        axes[1].plot(ordered["N"], ordered["final_window_success_mean"], marker="o", label=f"α={alpha:g}")
    axes[0].set(title="AUC normalizada", xlabel="Passos de planejamento N", ylabel="AUC normalizada")
    axes[1].set(title="Sucesso na janela final", xlabel="Passos de planejamento N", ylabel="Taxa de sucesso")
    axes[1].set_ylim(0, 1)
    axes[0].legend(title="Taxa de aprendizado")
    axes[1].legend(title="Taxa de aprendizado")
    fig.suptitle("Triagem: sensibilidade do Dyna-Q ao planejamento")
    files += save_figure(fig, output_dir / "sensibilidade_dyna_n")

    configure_style()
    cost = pd.DataFrame(rows)
    cost["config"] = cost.apply(
        lambda row: (
            f"{LABELS[row['algorithm']]} | α={row['alpha']:g} | "
            f"λ={row['lambda']:g} | dec={row['epsilon_decay']:g} | N={int(row['N'])}"
        ),
        axis=1,
    )
    order = cost.groupby("config")["total_q_updates"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    sns.stripplot(
        data=cost,
        x="total_q_updates",
        y="config",
        hue="algorithm",
        palette=PALETTE,
        order=order,
        dodge=False,
        jitter=0.15,
        alpha=0.65,
        ax=ax,
    )
    means = cost.groupby("config")["total_q_updates"].mean().reindex(order)
    ax.scatter(means.values, np.arange(len(means)), color="#252A31", marker="|", s=140, label="média")
    ax.set_title("Triagem: custo em atualizações Q por configuração")
    ax.set_xlabel("Atualizações Q totais por execução")
    ax.set_ylabel("Configuração")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], [LABELS.get(label, label) for label in labels[:4]], loc="lower right")
    files += save_figure(fig, output_dir / "custo_atualizacoes_q")
    return files


def _plot_curve_group(
    rows: list[dict[str, Any]],
    discretization: str,
    title: str,
    ylabel: str,
    output_base: Path,
) -> list[Path]:
    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    frame = pd.DataFrame([row for row in rows if row["discretization"] == discretization])
    for algorithm in PALETTE:
        group = frame[frame["algorithm"] == algorithm].sort_values("episode")
        if group.empty:
            continue
        x = group["episode"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        low = group["ci95_low"].to_numpy(dtype=float)
        high = group["ci95_high"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=PALETTE[algorithm],
            linestyle=LINESTYLES[algorithm],
            label=LABELS[algorithm],
            linewidth=1.8,
        )
        ax.fill_between(x, low, high, color=PALETTE[algorithm], alpha=0.18)
    ax.set_title(f"{title} — discretização {discretization}")
    ax.set_xlabel("Episódio de treinamento")
    ax.set_ylabel(ylabel)
    ax.legend(title="Algoritmo")
    return save_figure(fig, output_base)


def _t_interval(values: np.ndarray, confidence: float) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan
    margin = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def plot_final(
    results: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    output_dir: Path | str,
) -> list[Path]:
    output_dir = Path(output_dir)
    analysis_cfg = protocol["analysis"]
    confidence = float(analysis_cfg["confidence_level"])
    rolling = int(analysis_cfg["rolling_window"])
    files: list[Path] = []

    steps_curve = curve_rows(results, "episode_steps", confidence, rolling)
    updates_curve = curve_rows(results, "cumulative_q_updates", confidence, None)
    success_curve = curve_rows(results, "episode_successes", confidence, rolling)
    for discretization in protocol["discretizations"]:
        files += _plot_curve_group(
            steps_curve,
            discretization,
            f"Passos por episódio (média móvel de {rolling})",
            "Passos por episódio",
            output_dir / f"passos_por_episodio_{discretization}",
        )
        files += _plot_curve_group(
            updates_curve,
            discretization,
            "Atualizações Q acumuladas",
            "Atualizações Q acumuladas",
            output_dir / f"atualizacoes_q_acumuladas_{discretization}",
        )
        files += _plot_curve_group(
            success_curve,
            discretization,
            f"Taxa móvel de sucesso (janela {rolling})",
            "Taxa de sucesso",
            output_dir / f"taxa_movel_sucesso_{discretization}",
        )

    configure_style()
    frame = pd.DataFrame(seed_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for ax, discretization in zip(axes, protocol["discretizations"]):
        subset = frame[frame["discretization"] == discretization]
        order = list(PALETTE)
        sns.boxplot(data=subset, x="algorithm", y="auc_normalized", order=order, hue="algorithm", hue_order=order, palette=PALETTE, ax=ax, legend=False)
        sns.stripplot(data=subset, x="algorithm", y="auc_normalized", order=order, color="#252A31", jitter=0.12, size=4, ax=ax)
        ax.set_title(f"Discretização {discretization}")
        ax.set_xlabel("Algoritmo")
        ax.set_ylabel("AUC normalizada")
        ax.set_xticks(range(3), [LABELS[name] for name in PALETTE])
    fig.suptitle("Distribuição da AUC por semente")
    files += save_figure(fig, output_dir / "boxplot_auc_sementes")

    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for ax, discretization in zip(axes, protocol["discretizations"]):
        subset = frame[frame["discretization"] == discretization]
        for index, algorithm in enumerate(PALETTE):
            values = subset.loc[subset["algorithm"] == algorithm, "evaluation_success_rate"].to_numpy(float)
            mean, low, high = _t_interval(values, confidence)
            ax.scatter(np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)), values, color=PALETTE[algorithm], alpha=0.55)
            ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="#252A31", capsize=4)
        ax.set(title=f"Discretização {discretization}", xlabel="Algoritmo", ylabel="Taxa de sucesso na avaliação")
        ax.set_xticks(range(3), [LABELS[name] for name in PALETTE])
        ax.set_ylim(0, 1.05)
    fig.suptitle("Taxa de sucesso final por semente e IC 95% t")
    files += save_figure(fig, output_dir / "sucesso_final_ic95")

    configure_style()
    success_steps = frame.dropna(subset=["mean_steps_success"])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    order = list(PALETTE)
    sns.boxplot(data=success_steps, x="algorithm", y="mean_steps_success", order=order, hue="discretization", palette=DISC_PALETTE, ax=ax)
    sns.stripplot(data=success_steps, x="algorithm", y="mean_steps_success", order=order, hue="discretization", palette=DISC_PALETTE, dodge=True, edgecolor="#252A31", linewidth=0.5, alpha=0.65, ax=ax, legend=False)
    ax.set_title("Passos médios condicionados ao sucesso")
    ax.set_xlabel("Algoritmo")
    ax.set_ylabel("Passos médios nos episódios bem-sucedidos")
    ax.set_xticks(range(3), [LABELS[name] for name in PALETTE])
    files += save_figure(fig, output_dir / "passos_condicionados_sucesso")

    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for algorithm in PALETTE:
        subset = frame[frame["algorithm"] == algorithm]
        for discretization, marker in (("4x2", "o"), ("12x4", "s")):
            group = subset[subset["discretization"] == discretization]
            ax.scatter(group["total_q_updates"], group["auc_normalized"], color=PALETTE[algorithm], marker=marker, alpha=0.65, label=f"{LABELS[algorithm]} — {discretization}")
    ax.set_title("Desempenho versus custo de atualização")
    ax.set_xlabel("Atualizações Q totais")
    ax.set_ylabel("AUC normalizada")
    ax.legend(ncol=2, fontsize=8)
    files += save_figure(fig, output_dir / "desempenho_vs_atualizacoes_q")

    grouped = frame.groupby(["discretization", "algorithm"], as_index=False).agg(
        auc_normalized=("auc_normalized", "mean"), total_q_updates=("total_q_updates", "mean")
    )
    pareto_mask = []
    for _, row in grouped.iterrows():
        dominated = ((grouped["total_q_updates"] <= row["total_q_updates"]) & (grouped["auc_normalized"] >= row["auc_normalized"]) & ((grouped["total_q_updates"] < row["total_q_updates"]) | (grouped["auc_normalized"] > row["auc_normalized"]))).any()
        pareto_mask.append(not dominated)
    grouped["pareto"] = pareto_mask
    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for _, row in grouped.iterrows():
        marker = "o" if row["discretization"] == "4x2" else "s"
        ax.scatter(row["total_q_updates"], row["auc_normalized"], color=PALETTE[row["algorithm"]], marker=marker, s=90 if row["pareto"] else 55, edgecolor="#252A31" if row["pareto"] else "none")
        ax.annotate(f"{LABELS[row['algorithm']]} {row['discretization']}", (row["total_q_updates"], row["auc_normalized"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    frontier = grouped[grouped["pareto"]].sort_values("total_q_updates")
    ax.plot(frontier["total_q_updates"], frontier["auc_normalized"], color="#252A31", linestyle="--", linewidth=1)
    ax.set_title("Fronteira de Pareto: AUC e atualizações Q")
    ax.set_xlabel("Atualizações Q médias (menor é melhor)")
    ax.set_ylabel("AUC normalizada média (maior é melhor)")
    files += save_figure(fig, output_dir / "fronteira_pareto")

    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True, sharey=True)
    for ax, algorithm in zip(axes, PALETTE):
        subset = frame[frame["algorithm"] == algorithm].pivot(index="seed", columns="discretization", values="auc_normalized").dropna()
        for _, row in subset.iterrows():
            ax.plot([0, 1], [row["4x2"], row["12x4"]], color=PALETTE[algorithm], alpha=0.35)
            ax.scatter([0, 1], [row["4x2"], row["12x4"]], color=PALETTE[algorithm], s=18)
        ax.set_title(LABELS[algorithm])
        ax.set_xticks([0, 1], ["4x2", "12x4"])
        ax.set_xlabel("Discretização")
    axes[0].set_ylabel("AUC normalizada por semente")
    fig.suptitle("Comparação pareada das discretizações")
    files += save_figure(fig, output_dir / "comparacao_discretizacoes")
    return files

