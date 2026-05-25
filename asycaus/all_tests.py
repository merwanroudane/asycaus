"""asycaus.all_tests — run the full battery and print a unified summary."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from .static import static, StaticResult
from .dynamic import dynamic, DynamicResult
from .fourier import fourier, FourierResult
from .spectral import spectral, SpectralResult
from .quantile import quantile, QuantileResult
from .efficient import efficient, EfficientResult


@dataclass
class AllResult:
    static_:    StaticResult     | None = None
    fourier_:   FourierResult    | None = None
    efficient_: EfficientResult  | None = None
    spectral_:  SpectralResult   | None = None
    quantile_:  QuantileResult   | None = None
    dynamic_:   DynamicResult    | None = None
    summary:    pd.DataFrame     = field(default_factory=pd.DataFrame)
    depvar: str = "y"
    causvar: str = "x"

    def print(self):
        from .tables import print_all_summary
        print_all_summary(self)
        return self

    def plot(self, *, save=None):
        from .plots import plot_dashboard
        return plot_dashboard(self, save=save)


def all_tests(
    y, x,
    max_lag: int = 4,
    ic: str = "hjc",
    intorder: int = 1,
    boot: int = 500,
    seed: int | None = 12345,
    kmax: int = 5,
    nfreq: int = 50,
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    window: int | None = None,
    lnform: bool = False,
    skip_dynamic: bool = False,
    skip_spectral: bool = False,
    skip_quantile: bool = False,
    show: bool = True,
    plot: bool = False,
) -> AllResult:
    """Run every asymmetric-causality test on the same (y, x) pair and print a
    unified summary table at the end.  Dashboard plot optional via `plot=True`.
    """
    print("\n" + "=" * 78)
    print(f"{'ASYMMETRIC CAUSALITY BATTERY':^78}")
    print(f"{'Author: Dr Merwan Roudane':^78}")
    print("=" * 78)
    print(f" Direction tested:  causvar -> depvar")

    common = dict(max_lag=max_lag, ic=ic, intorder=intorder, lnform=lnform,
                  show=False, plot=False)

    print("\n[1/6] Static Asymmetric Causality (Hatemi-J 2012)...")
    sr = static(y, x, shock="both", boot=boot, seed=seed, **common)

    print("\n[2/6] Fourier Asymmetric TY (Nazlioglu et al. 2016)...")
    fr = fourier(y, x, shock="both", kmax=kmax, form="single", **common)

    print("\n[3/6] Efficient Asymmetric (Hatemi-J 2024)...")
    er = efficient(y, x, max_lag=max_lag, ic=ic, intorder=intorder,
                   lnform=lnform, show=False, plot=False)

    sp = None
    if not skip_spectral:
        print("\n[4/6] Spectral Asymmetric (Bahmani-Oskooee et al. 2016)...")
        sp = spectral(y, x, shock="both", nfreq=nfreq, max_lag=max_lag, ic=ic,
                      lnform=lnform, show=False, plot=False)

    qr = None
    if not skip_quantile:
        print("\n[5/6] Quantile Asymmetric (Fang et al. 2026)...")
        qr = quantile(y, x, shock="both", quantiles=quantiles,
                      max_lag=max_lag, ic=ic, intorder=intorder,
                      lnform=lnform, show=False, plot=False)

    dy = None
    if not skip_dynamic:
        print("\n[*]   Dynamic Asymmetric (Hatemi-J 2021, Pos shocks)...")
        try:
            dy = dynamic(y, x, shock="pos", mode="rolling",
                         window=window, max_lag=max_lag, ic=ic,
                         intorder=intorder, boot=min(boot, 200), seed=seed,
                         lnform=lnform, show=False, plot=False, progress=False)
        except Exception as ex:
            print(f"   (dynamic skipped: {ex})")
            dy = None

    # Build unified summary -------------------------------------------------
    rows = []
    for s in ["Positive", "Negative"]:
        if s in sr.table.index:
            row = sr.table.loc[s]
            rows.append({"Test": "Static (Hatemi-J 2012)",
                         "Shock": s[:3], "Statistic": row["Wald"],
                         "p-value": row["asy_p"],
                         "Decision": row["decision_5pct"]})
    for s in ["Positive", "Negative"]:
        if s in fr.table.index:
            row = fr.table.loc[s]
            rows.append({"Test": "Fourier (Nazlioglu 2016)",
                         "Shock": s[:3], "Statistic": row["Wald"],
                         "p-value": row["asy_p"],
                         "Decision": row["decision_5pct"]})
    eff = er.raw
    rows.extend([
        {"Test": "Efficient Pos only (HJ 2024)", "Shock": "Pos",
         "Statistic": eff["W_pos"], "p-value": eff["p_pos"],
         "Decision": "Reject" if eff["p_pos"] < 0.05 else "Fail to reject"},
        {"Test": "Efficient Neg only (HJ 2024)", "Shock": "Neg",
         "Statistic": eff["W_neg"], "p-value": eff["p_neg"],
         "Decision": "Reject" if eff["p_neg"] < 0.05 else "Fail to reject"},
        {"Test": "Efficient Joint (HJ 2024)", "Shock": "both",
         "Statistic": eff["W_joint"], "p-value": eff["p_joint"],
         "Decision": "Reject" if eff["p_joint"] < 0.05 else "Fail to reject"},
        {"Test": "Efficient Pos=Neg (HJ 2024)", "Shock": "diff",
         "Statistic": eff["W_diff"], "p-value": eff["p_diff"],
         "Decision": "Reject" if eff["p_diff"] < 0.05 else "Fail to reject"},
    ])
    if sp is not None:
        for s in sp.summary.index:
            pct = sp.summary.loc[s, "pct_reject_5pct"]
            rows.append({"Test": "Spectral (BCRanjbar 2016)",
                         "Shock": s[:3], "Statistic": f"{pct:.2%} freq.",
                         "p-value": pct,
                         "Decision": "Reject at some" if pct > 0 else "Fail to reject"})
    if qr is not None:
        for s in ["Positive", "Negative"]:
            sub = qr.table[qr.table["shock"] == s]
            if len(sub):
                nrej = (sub["asy_p"] < 0.05).sum()
                pct = nrej / len(sub)
                rows.append({"Test": "Quantile (Fang et al. 2026)",
                             "Shock": s[:3], "Statistic": f"{pct:.2%} quant.",
                             "p-value": pct,
                             "Decision": "Reject at some" if nrej > 0 else "Fail to reject"})

    summary = pd.DataFrame(rows)
    out = AllResult(static_=sr, fourier_=fr, efficient_=er, spectral_=sp,
                    quantile_=qr, dynamic_=dy, summary=summary)
    if show:
        out.print()
    if plot:
        try: out.plot()
        except Exception: pass
    return out
