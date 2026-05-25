"""asycaus.dynamic — Hatemi-J (2021) rolling / recursive asymmetric causality."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from .engine import (pos_neg_components, select_lag, wald_test,
                     bootstrap_critical_values, ic_label)


@dataclass
class DynamicResult:
    table: pd.DataFrame
    depvar: str = "y"
    causvar: str = "x"
    shock: str = "pos"
    mode: str = "Rolling window"
    window: int = 0
    smin: int = 0
    nsub: int = 0
    ic: str = "HJC"
    boot: int = 200
    intorder: int = 1

    def print(self):
        from .tables import print_dynamic_table
        print_dynamic_table(self)
        return self

    def plot(self, *, ax=None, save=None):
        from .plots import plot_dynamic
        return plot_dynamic(self, ax=ax, save=save)


def dynamic(
    y, x,
    shock: str = "pos",
    mode: str = "rolling",
    window: int | None = None,
    max_lag: int = 4,
    ic: str = "hjc",
    intorder: int = 1,
    boot: int = 200,
    seed: int | None = 12345,
    lnform: bool = False,
    show: bool = True,
    plot: bool = True,
    progress: bool = True,
) -> DynamicResult:
    """Hatemi-J (2021) dynamic asymmetric Granger-causality.

    Parameters
    ----------
    mode : {'rolling','recursive'}, default 'rolling'.
    window : int or None
        Window length S.  Defaults to Phillips-Shi-Yu (2015) lower bound
        S = ceil(T*(0.01 + 1.8/sqrt(T))).
    progress : bool
        Print "subsample k/N" every 10 windows.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if lnform:
        y = np.log(y); x = np.log(x)
    Y = np.column_stack([y, x])

    if shock.lower() in {"pos", "positive"}:
        Zfull = pos_neg_components(Y, positive=True); s_lbl = "Positive components"
        s_short = "pos"
    elif shock.lower() in {"neg", "negative"}:
        Zfull = pos_neg_components(Y, positive=False); s_lbl = "Negative components"
        s_short = "neg"
    else:
        raise ValueError("shock must be 'pos' or 'neg'")

    Tcomp = Zfull.shape[0]
    if Tcomp < 10:
        raise ValueError("Too few observations after differencing.")

    smin = math.ceil(Tcomp * (0.01 + 1.8 / math.sqrt(Tcomp)))
    if window is None:
        window = smin
    min_window = max_lag + intorder + 3
    if window < min_window:
        raise ValueError(f"window must be at least {min_window}.")

    nsub = Tcomp - window + 1
    if nsub < 1:
        raise ValueError("window too large for the sample.")

    mode = mode.lower()
    if mode not in {"rolling", "recursive"}:
        raise ValueError("mode must be 'rolling' or 'recursive'.")
    mode_lbl = "Rolling window" if mode == "rolling" else "Recursive"

    rows = []
    for k in range(1, nsub + 1):
        if progress and (k % 10 == 0 or k == nsub):
            print(f"  subsample {k}/{nsub}")
        if mode == "rolling":
            s_idx, e_idx = k - 1, k - 1 + window
        else:
            s_idx, e_idx = 0, window + k - 1
        Zsub = Zfull[s_idx:e_idx, :]
        p = select_lag(Zsub, 1, max_lag, ic)
        W, _ = wald_test(Zsub, p, intorder, dep_idx=0, cause_idx=1)
        cv = bootstrap_critical_values(Zsub, p, intorder, 0, 1, B=boot,
                                       seed=(seed or 0) + k)
        rows.append({
            "sub_start": s_idx + 1, "sub_end": e_idx, "lag": p,
            "Wald": W, "cv10": cv["cv10"], "cv5": cv["cv5"], "cv1": cv["cv1"],
            "ratio_5pct": W / cv["cv5"] if cv["cv5"] else np.nan,
        })

    table = pd.DataFrame(rows)
    res = DynamicResult(
        table=table, depvar="y", causvar="x", shock=s_short,
        mode=mode_lbl, window=window, smin=smin, nsub=nsub,
        ic=ic_label(ic), boot=boot, intorder=intorder,
    )
    if show:
        res.print()
    if plot:
        try:
            res.plot()
        except Exception:
            pass
    return res
