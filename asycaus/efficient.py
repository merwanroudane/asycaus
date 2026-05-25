"""asycaus.efficient — Hatemi-J (2024) efficient SUR asymmetric causality."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .engine import pos_neg_components, select_lag, efficient_sur, ic_label


@dataclass
class EfficientResult:
    table: pd.DataFrame
    raw: dict
    depvar: str = "y"
    causvar: str = "x"
    ic: str = "HJC"
    intorder: int = 1
    lag: int = 1

    def print(self):
        from .tables import print_efficient_table
        print_efficient_table(self)
        return self

    def plot(self, *, ax=None, save=None):
        from .plots import plot_efficient
        return plot_efficient(self, ax=ax, save=save)


def efficient(
    y, x,
    max_lag: int = 8,
    ic: str = "hjc",
    intorder: int = 1,
    lnform: bool = False,
    show: bool = True,
    plot: bool = True,
) -> EfficientResult:
    """Hatemi-J (2024) efficient asymmetric causality test (SUR).

    Tests four hypotheses jointly within one SUR system:
      H1: no causality via positive shocks
      H2: no causality via negative shocks
      H3: joint no causality (H1 AND H2)
      H4: Pos = Neg causal coefficients  --  the formal asymmetry test.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if lnform:
        y = np.log(y); x = np.log(x)
    Y = np.column_stack([y, x])

    Zpos = pos_neg_components(Y, positive=True)
    Zneg = pos_neg_components(Y, positive=False)
    p_pos = select_lag(Zpos, 1, max_lag, ic)
    p_neg = select_lag(Zneg, 1, max_lag, ic)
    p = max(p_pos, p_neg)

    out = efficient_sur(Zpos, Zneg, p, intorder, 0, 1)

    rows = [
        {"hypothesis": "No causality via POS shocks", "Wald": out["W_pos"],
         "df": p,     "asy_p": out["p_pos"],
         "decision_5pct": "Reject" if out["p_pos"]   < 0.05 else "Fail to reject"},
        {"hypothesis": "No causality via NEG shocks", "Wald": out["W_neg"],
         "df": p,     "asy_p": out["p_neg"],
         "decision_5pct": "Reject" if out["p_neg"]   < 0.05 else "Fail to reject"},
        {"hypothesis": "Joint no causality",          "Wald": out["W_joint"],
         "df": 2 * p, "asy_p": out["p_joint"],
         "decision_5pct": "Reject" if out["p_joint"] < 0.05 else "Fail to reject"},
        {"hypothesis": "POS = NEG causal effects",    "Wald": out["W_diff"],
         "df": p,     "asy_p": out["p_diff"],
         "decision_5pct": "Reject" if out["p_diff"]  < 0.05 else "Fail to reject"},
    ]
    table = pd.DataFrame(rows).set_index("hypothesis")
    res = EfficientResult(table=table, raw=out, ic=ic_label(ic),
                          intorder=intorder, lag=p)
    if show:
        res.print()
    if plot:
        try:
            res.plot()
        except Exception:
            pass
    return res
