"""asycaus.plots — publication-quality matplotlib plots.

All functions accept an optional pre-existing `ax` and a `save` path; they
return the matplotlib Figure for further customisation.

Author : Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# Consistent professional palette
_COL = {"navy": "#1f3a93", "green": "#27ae60", "orange": "#e67e22",
        "red":  "#c0392b", "grey":  "#7f8c8d", "lblue": "#5dade2"}


def _new(ax=None, figsize=(7, 4.5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _save(fig, save):
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=200)


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
def plot_static(res, *, ax=None, save=None):
    fig, ax = _new(ax)
    df = res.table.reset_index()
    xs = np.arange(len(df))
    ax.bar(xs, df["Wald"], color=_COL["navy"], alpha=0.85,
           edgecolor=_COL["navy"], label="Wald")
    ax.scatter(xs, df["cv10"], marker="D", color=_COL["green"],   label="10% CV", zorder=3)
    ax.scatter(xs, df["cv5"],  marker="^", color=_COL["orange"],  label="5% CV",  zorder=3)
    ax.scatter(xs, df["cv1"],  marker="X", color=_COL["red"],     label="1% CV",  zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(df["shock"])
    ax.set_ylabel("Test statistic"); ax.set_xlabel("")
    ax.set_title(f"Asymmetric Causality: {res.causvar} -> {res.depvar}",
                 fontsize=12, fontweight="bold")
    ax.text(0.5, 1.02, "Hatemi-J (2012) -- leverage bootstrap CVs",
            transform=ax.transAxes, ha="center", fontsize=9, color="grey")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Dynamic -- time-varying causality
# ---------------------------------------------------------------------------
def plot_dynamic(res, *, ax=None, save=None):
    fig, ax = _new(ax, figsize=(9, 5))
    df = res.table
    ax.plot(df["sub_end"], df["Wald"], color=_COL["navy"], lw=2,
            label="Wald")
    ax.plot(df["sub_end"], df["cv10"], color=_COL["green"],   ls=":",  lw=1.3, label="10% CV")
    ax.plot(df["sub_end"], df["cv5"],  color=_COL["orange"],  ls="--", lw=1.3, label="5% CV")
    ax.plot(df["sub_end"], df["cv1"],  color=_COL["red"],     ls="-.", lw=1.3, label="1% CV")
    ax.set_xlabel("Subsample end (time index)")
    ax.set_ylabel("Wald statistic")
    ax.set_title(f"Dynamic Asymmetric Causality: {res.causvar} -> {res.depvar}",
                 fontsize=12, fontweight="bold")
    ax.text(0.5, 1.02, f"{res.mode} -- {res.shock} components (Hatemi-J 2021)",
            transform=ax.transAxes, ha="center", fontsize=9, color="grey")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------------
def plot_fourier(res, *, ax=None, save=None):
    fig, ax = _new(ax)
    df = res.table.reset_index()
    xs = np.arange(len(df))
    bars = ax.bar(xs, df["Wald"], color=_COL["navy"], alpha=0.85,
                  edgecolor=_COL["navy"])
    for bar, pv in zip(bars, df["asy_p"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"p={pv:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(df["shock"])
    ax.set_ylabel("Fourier-Wald statistic")
    ax.set_title(f"Fourier Asymmetric TY: {res.causvar} -> {res.depvar}",
                 fontsize=12, fontweight="bold")
    ax.text(0.5, 1.02, "Nazlioglu, Gormus & Soytas (2016)",
            transform=ax.transAxes, ha="center", fontsize=9, color="grey")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------
def plot_spectral(res, *, save=None):
    shocks = res.summary.index.tolist()
    fig, axes = plt.subplots(len(shocks), 1, figsize=(8, 4 * len(shocks)),
                             sharex=True, squeeze=False)
    for k, s in enumerate(shocks):
        ax = axes[k, 0]
        sub = res.table[res.table["shock"] == s]
        ax.plot(sub["omega"], sub["Wald"], color=_COL["navy"], lw=2, label="Wald")
        ax.plot(sub["omega"], sub["cv10"], color=_COL["green"],   ls=":",  lw=1.3, label="10% CV")
        ax.plot(sub["omega"], sub["cv5"],  color=_COL["orange"],  ls="--", lw=1.3, label="5% CV")
        ax.plot(sub["omega"], sub["cv1"],  color=_COL["red"],     ls="-.", lw=1.3, label="1% CV")
        ax.set_ylabel("Wald statistic")
        ax.set_title(f"{s} shocks: {res.causvar} -> {res.depvar}", fontsize=11, fontweight="bold")
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Frequency w  (0=long run, pi=short run)")
    fig.suptitle("Asymmetric Spectral Causality -- Bahmani-Oskooee et al. (2016)",
                 y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Quantile
# ---------------------------------------------------------------------------
def plot_quantile(res, *, save=None):
    shocks = sorted(res.table["shock"].unique())
    fig, axes = plt.subplots(len(shocks), 1, figsize=(7, 4 * len(shocks)),
                             sharex=True, squeeze=False)
    for k, s in enumerate(shocks):
        ax = axes[k, 0]
        sub = res.table[res.table["shock"] == s]
        ax.plot(sub["tau"], sub["Wald"], "o-", color=_COL["navy"], lw=2,
                markersize=7, label="Wald")
        ax.set_ylabel("Wald statistic")
        ax.set_title(f"{s} shocks: {res.causvar} -> {res.depvar}",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(r"Quantile $\tau$")
    fig.suptitle("Quantile Asymmetric Causality -- Fang et al. (2026)",
                 y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Efficient
# ---------------------------------------------------------------------------
def plot_efficient(res, *, ax=None, save=None):
    fig, ax = _new(ax)
    df = res.table.reset_index()
    xs = np.arange(len(df))
    bars = ax.bar(xs, df["Wald"], color=_COL["navy"], alpha=0.85)
    for bar, pv in zip(bars, df["asy_p"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"p={pv:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([h.replace("causality ", "").replace("causality", "").strip()
                        for h in df["hypothesis"]],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Wald statistic")
    ax.set_title(f"Efficient Asymmetric Tests: {res.causvar} -> {res.depvar}",
                 fontsize=12, fontweight="bold")
    ax.text(0.5, 1.02, "Hatemi-J (2024) SUR -- Pos / Neg / Joint / Diff",
            transform=ax.transAxes, ha="center", fontsize=9, color="grey")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save)
    return fig


# ---------------------------------------------------------------------------
# Unified dashboard
# ---------------------------------------------------------------------------
def plot_dashboard(res, *, save=None):
    """6-panel dashboard combining every test (subplots stacked)."""
    panels = []
    titles = []
    if res.static_ is not None:    panels.append(("static",    res.static_));     titles.append("Static")
    if res.fourier_ is not None:   panels.append(("fourier",   res.fourier_));    titles.append("Fourier")
    if res.efficient_ is not None: panels.append(("efficient", res.efficient_));  titles.append("Efficient (SUR)")
    if res.spectral_ is not None:  panels.append(("spectral",  res.spectral_));   titles.append("Spectral")
    if res.quantile_ is not None:  panels.append(("quantile",  res.quantile_));   titles.append("Quantile")
    if res.dynamic_  is not None:  panels.append(("dynamic",   res.dynamic_));    titles.append("Dynamic")

    n = len(panels)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
    axes = axes.flatten()
    for i, ((kind, r), title) in enumerate(zip(panels, titles)):
        ax = axes[i]
        if kind == "static":
            plot_static(r, ax=ax)
        elif kind == "fourier":
            plot_fourier(r, ax=ax)
        elif kind == "efficient":
            plot_efficient(r, ax=ax)
        elif kind == "dynamic":
            plot_dynamic(r, ax=ax)
        elif kind == "spectral":
            # take the first shock's curve for compactness
            sub = r.table[r.table["shock_id"] == 1]
            ax.plot(sub["omega"], sub["Wald"], color=_COL["navy"], lw=2)
            ax.plot(sub["omega"], sub["cv5"], "--", color=_COL["orange"], lw=1)
            ax.set_xlabel("Frequency w"); ax.set_ylabel("Wald")
            ax.set_title(f"Spectral (Pos): {r.causvar} -> {r.depvar}",
                         fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)
        elif kind == "quantile":
            sub = r.table[r.table["shock"] == "Positive"]
            if len(sub):
                ax.plot(sub["tau"], sub["Wald"], "o-", color=_COL["navy"], lw=2)
            ax.set_xlabel(r"$\tau$"); ax.set_ylabel("Wald")
            ax.set_title(f"Quantile (Pos): {r.causvar} -> {r.depvar}",
                         fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)
    for j in range(len(panels), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Asymmetric Causality Dashboard: {res.causvar} -> {res.depvar}",
                 y=1.01, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save)
    return fig
