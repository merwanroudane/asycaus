"""asycaus.fourier — Nazlioglu et al. (2016) / Pata (2020) Fourier asymmetric TY."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .engine import (pos_neg_components, select_lag, wald_fourier, ic_label)


@dataclass
class FourierResult:
    table: pd.DataFrame
    depvar: str = "y"
    causvar: str = "x"
    form: str = "single"
    kmax: int = 5
    ic: str = "HJC"
    intorder: int = 1

    def print(self):
        from .tables import print_fourier_table
        print_fourier_table(self)
        return self

    def plot(self, *, ax=None, save=None):
        from .plots import plot_fourier
        return plot_fourier(self, ax=ax, save=save)


def fourier(
    y, x,
    shock: str = "both",
    kmax: int = 5,
    form: str = "single",
    max_lag: int = 8,
    ic: str = "hjc",
    intorder: int = 1,
    lnform: bool = False,
    show: bool = True,
    plot: bool = True,
) -> FourierResult:
    """Fourier-augmented asymmetric Toda-Yamamoto causality.

    Selects the optimal Fourier frequency k* in [1, kmax] by maximising the
    Wald statistic for non-causality (most informative basis).
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if lnform:
        y = np.log(y); x = np.log(x)
    Y = np.column_stack([y, x])

    shock = shock.lower()
    if shock not in {"pos", "positive", "neg", "negative", "both"}:
        raise ValueError("shock must be 'pos', 'neg' or 'both'.")
    form = form.lower()
    if form not in {"single", "cumulative"}:
        raise ValueError("form must be 'single' or 'cumulative'.")
    shocks = []
    if shock in {"pos", "positive", "both"}:
        shocks.append("Positive")
    if shock in {"neg", "negative", "both"}:
        shocks.append("Negative")

    rows = []
    for s in shocks:
        Z = pos_neg_components(Y, positive=(s == "Positive"))
        p = select_lag(Z, 1, max_lag, ic)
        best = (None, None, None)
        for k in range(1, kmax + 1):
            W, dof = wald_fourier(Z, p, intorder, 0, 1, k, form)
            pv = float(chi2.sf(W, df=dof))
            if best[0] is None or W > best[0]:
                best = (W, pv, k)
        W, pv, k_opt = best
        rows.append({
            "shock": s, "Wald": W, "lag": p, "k_opt": k_opt,
            "asy_p": pv, "sample": Z.shape[0],
            "decision_5pct": "Reject" if pv < 0.05 else "Fail to reject",
        })

    table = pd.DataFrame(rows).set_index("shock")
    res = FourierResult(table=table, form=form, kmax=kmax, ic=ic_label(ic),
                        intorder=intorder)
    if show:
        res.print()
    if plot:
        try:
            res.plot()
        except Exception:
            pass
    return res
