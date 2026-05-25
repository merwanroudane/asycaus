"""asycaus.quantile — Fang, Wang, Shieh & Chung (2026) quantile asymmetric causality."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.stats import chi2

from .engine import (pos_neg_components, select_lag, wald_quantile,
                     fourier_terms, ic_label)


@dataclass
class QuantileResult:
    table: pd.DataFrame
    depvar: str = "y"
    causvar: str = "x"
    ic: str = "HJC"
    intorder: int = 1
    fourier_used: bool = False
    kmax: int = 0

    def print(self):
        from .tables import print_quantile_table
        print_quantile_table(self)
        return self

    def plot(self, *, save=None):
        from .plots import plot_quantile
        return plot_quantile(self, save=save)


def quantile(
    y, x,
    shock: str = "both",
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    max_lag: int = 4,
    ic: str = "hjc",
    intorder: int = 1,
    fourier: bool = False,
    kmax: int = 3,
    lnform: bool = False,
    show: bool = True,
    plot: bool = True,
) -> QuantileResult:
    """Quantile asymmetric Granger-causality (Fang et al. 2026).

    If `fourier=True`, the cumulative components are first detrended against
    a cumulative Fourier basis of size kmax (sin/cos with k = 1..kmax).
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if lnform:
        y = np.log(y); x = np.log(x)
    Y = np.column_stack([y, x])

    shock = shock.lower()
    shocks = []
    if shock in {"pos", "positive", "both"}:
        shocks.append("Positive")
    if shock in {"neg", "negative", "both"}:
        shocks.append("Negative")

    rows = []
    for sid, s in enumerate(shocks, start=1):
        Z = pos_neg_components(Y, positive=(s == "Positive"))
        p = select_lag(Z, 1, max_lag, ic)
        if fourier:
            T = Z.shape[0]
            F = fourier_terms(T, kmax, "cumulative")
            X = np.column_stack([np.ones(T), F])
            B, *_ = lstsq(X, Z, rcond=None)
            Z = Z - X @ B
        for tau in quantiles:
            W, dof = wald_quantile(Z, p, intorder, 0, 1, tau)
            pv = float(chi2.sf(W, df=dof))
            rows.append({
                "shock_id": sid, "shock": s, "tau": tau,
                "Wald": W, "lag": p, "asy_p": pv,
                "decision_5pct": "Reject" if pv < 0.05 else "Fail to reject",
            })

    table = pd.DataFrame(rows)
    res = QuantileResult(table=table, ic=ic_label(ic), intorder=intorder,
                         fourier_used=fourier, kmax=kmax if fourier else 0)
    if show:
        res.print()
    if plot:
        try:
            res.plot()
        except Exception:
            pass
    return res
