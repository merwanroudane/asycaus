"""asycaus.spectral — Bahmani-Oskooee et al. (2016) asymmetric Breitung-Candelon (2006)."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .engine import pos_neg_components, select_lag, bc_at_omega, ic_label


@dataclass
class SpectralResult:
    table: pd.DataFrame
    summary: pd.DataFrame
    depvar: str = "y"
    causvar: str = "x"
    nfreq: int = 50
    ic: str = "HJC"

    def print(self):
        from .tables import print_spectral_table
        print_spectral_table(self)
        return self

    def plot(self, *, save=None):
        from .plots import plot_spectral
        return plot_spectral(self, save=save)


def spectral(
    y, x,
    shock: str = "both",
    nfreq: int = 50,
    max_lag: int = 8,
    ic: str = "hjc",
    lnform: bool = False,
    show: bool = True,
    plot: bool = True,
) -> SpectralResult:
    """Asymmetric frequency-domain causality (BC 2006 on Pos/Neg components).

    At each frequency omega in (0, pi], the BC Wald statistic is computed and
    compared with chi-square(2) critical values.  Returns the full grid and
    a per-shock summary of rejection counts at 1%, 5%, 10%.
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

    cv10, cv5, cv1 = chi2.isf([0.10, 0.05, 0.01], df=2)

    rows = []
    summary_rows = []
    for sid, s in enumerate(shocks, start=1):
        Z = pos_neg_components(Y, positive=(s == "Positive"))
        p = select_lag(Z, 1, max_lag, ic)
        nrej = {1: 0, 5: 0, 10: 0}
        for j in range(1, nfreq + 1):
            omega = j * np.pi / nfreq
            W = bc_at_omega(Z, p, omega, 0, 1)
            rows.append({
                "shock_id": sid, "shock": s, "omega": omega,
                "Wald": W, "cv10": cv10, "cv5": cv5, "cv1": cv1, "lag": p,
            })
            if W > cv1:  nrej[1]  += 1
            if W > cv5:  nrej[5]  += 1
            if W > cv10: nrej[10] += 1
        summary_rows.append({
            "shock": s, "lag": p, "nfreq": nfreq,
            "n_reject_10pct": nrej[10],
            "n_reject_5pct":  nrej[5],
            "n_reject_1pct":  nrej[1],
            "pct_reject_5pct": nrej[5] / nfreq,
        })

    table = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows).set_index("shock")
    res = SpectralResult(table=table, summary=summary, nfreq=nfreq, ic=ic_label(ic))
    if show:
        res.print()
    if plot:
        try:
            res.plot()
        except Exception:
            pass
    return res
