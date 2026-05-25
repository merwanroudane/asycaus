"""asycaus.static — Hatemi-J (2012) static asymmetric causality test."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .engine import (pos_neg_components, select_lag, wald_test,
                     bootstrap_critical_values, ic_label)


@dataclass
class StaticResult:
    """Hatemi-J (2012) asymmetric causality result.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per shock type ('Positive' / 'Negative') with columns
        Wald, lag, dof, asy_p, cv10, cv5, cv1, decision_5pct.
    depvar : str
    causvar : str
    ic : str
        Selected information criterion (label).
    boot : int
    sample_size : int
    """
    table: pd.DataFrame
    depvar: str = "y"
    causvar: str = "x"
    ic: str = "HJC"
    boot: int = 1000
    sample_size: int = 0
    intorder: int = 1

    def print(self, *, console=None):
        from .tables import print_static_table
        print_static_table(self, console=console)
        return self

    def plot(self, *, ax=None, save=None):
        from .plots import plot_static
        return plot_static(self, ax=ax, save=save)


def static(
    y, x,
    shock: str = "both",
    max_lag: int = 8,
    ic: str = "hjc",
    intorder: int = 1,
    boot: int = 1000,
    seed: int | None = 12345,
    lnform: bool = False,
    show: bool = True,
    plot: bool = True,
) -> StaticResult:
    """Hatemi-J (2012) static asymmetric Granger-causality test.

    Parameters
    ----------
    y, x : array_like
        Two time-series (length T).  H0: x does NOT Granger-cause y.
    shock : {'pos','neg','both'}, default 'both'
        Which cumulative shocks to test.
    max_lag : int, default 8
        Maximum VAR lag to search over.
    ic : {'aic','aicc','sbc','hqc','hjc'}, default 'hjc'
        Lag-selection criterion.
    intorder : int, default 1
        Toda-Yamamoto augmentation lags (max order of integration).
    boot : int, default 1000
        Bootstrap replications for critical values.
    seed : int or None, default 12345
    lnform : bool, default False
        Take natural log of inputs before decomposition.
    show : bool, default True
        Print the table to stdout.
    plot : bool, default True
        Render the bar plot.

    Returns
    -------
    StaticResult
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if lnform:
        y = np.log(y)
        x = np.log(x)
    Y = np.column_stack([y, x])

    shock = shock.lower()
    if shock not in {"pos", "positive", "neg", "negative", "both"}:
        raise ValueError("shock must be 'pos', 'neg', or 'both'.")
    shocks = []
    if shock in {"pos", "positive", "both"}:
        shocks.append("Positive")
    if shock in {"neg", "negative", "both"}:
        shocks.append("Negative")

    rows = []
    nn = 0
    for s in shocks:
        Z = pos_neg_components(Y, positive=(s == "Positive"))
        nn = Z.shape[0]
        p = select_lag(Z, 1, max_lag, ic)
        W, dof = wald_test(Z, p, intorder, dep_idx=0, cause_idx=1)
        asy_p = float(chi2.sf(W, df=dof))
        cv = bootstrap_critical_values(Z, p, intorder, 0, 1, B=boot, seed=seed)
        dec = "Reject" if W > cv["cv5"] else "Fail to reject"
        rows.append({
            "shock": s, "Wald": W, "lag": p, "dof": dof, "asy_p": asy_p,
            "cv10": cv["cv10"], "cv5": cv["cv5"], "cv1": cv["cv1"],
            "decision_5pct": dec,
        })

    table = pd.DataFrame(rows).set_index("shock")
    res = StaticResult(
        table=table, depvar="y", causvar="x", ic=ic_label(ic),
        boot=boot, sample_size=nn, intorder=intorder,
    )
    if show:
        res.print()
    if plot:
        try:
            res.plot()
        except Exception:
            pass
    return res
