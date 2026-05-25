"""asycaus.tables — pretty boxed tables that mirror the Stata output.

Prefers `rich.table.Table` for colour terminals.  Falls back to a clean ASCII
boxed style if rich is not installed.

Author : Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations
from typing import Iterable

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
except Exception:
    _RICH = False


def _console(c=None):
    if not _RICH:
        return None
    return c or Console()


def _star(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ---------------------------------------------------------------------------
# Plain fallbacks
# ---------------------------------------------------------------------------
def _hline(width: int = 78, ch: str = "-") -> None:
    print(ch * width)


def _hdr(title: str, sub: str | None = None) -> None:
    _hline()
    print(f" {title}")
    if sub:
        print(f" {sub}")
    _hline()


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
def print_static_table(res, *, console=None) -> None:
    c = _console(console)
    if c is None:
        _hdr("Asymmetric Causality Test  --  Hatemi-J (2012)",
             f"H0: {res.causvar} does not Granger-cause {res.depvar}")
        print(f" Lag selection:                 {res.ic}")
        print(f" Augmentation lags:             {res.intorder}")
        print(f" Bootstrap reps:                {res.boot}")
        print(f" Sample (after differencing):   {res.sample_size}")
        _hline()
        hdr = f" {'Shock':<10}{'Wald':>11}{'Lag':>6}{'Asy p':>11}{'CV 10%':>11}{'CV 5%':>11}{'CV 1%':>11}"
        print(hdr)
        _hline()
        for shock, row in res.table.iterrows():
            star = _star(row["asy_p"])
            print(f" {shock:<10}{row['Wald']:>11.4f}{int(row['lag']):>6d}"
                  f"{row['asy_p']:>11.4f}{row['cv10']:>11.4f}"
                  f"{row['cv5']:>11.4f}{row['cv1']:>11.4f}  {star}")
        _hline()
        print(" Significance: * 10%  ** 5%  *** 1%  (leverage-adjusted bootstrap CVs)")
        return

    t = Table(title="Asymmetric Causality Test  --  Hatemi-J (2012)",
              box=box.HEAVY_HEAD, header_style="bold cyan", title_style="bold")
    for col in ["Shock", "Wald", "Lag", "Asy p", "CV 10%", "CV 5%", "CV 1%", "Sig"]:
        t.add_column(col, justify="right" if col != "Shock" else "left")
    for shock, row in res.table.iterrows():
        t.add_row(shock,
                  f"{row['Wald']:.4f}", str(int(row['lag'])),
                  f"{row['asy_p']:.4f}",
                  f"{row['cv10']:.4f}", f"{row['cv5']:.4f}", f"{row['cv1']:.4f}",
                  _star(row['asy_p']) or "")
    c.rule("[bold]Asymmetric Causality Test -- Hatemi-J (2012)")
    c.print(f"[dim]H0: {res.causvar} does not Granger-cause {res.depvar}")
    c.print(f"[dim]Lag selection: [bold]{res.ic}[/]    "
            f"Augmentation lags: [bold]{res.intorder}[/]    "
            f"Bootstrap reps: [bold]{res.boot}[/]    "
            f"Sample: [bold]{res.sample_size}[/]")
    c.print(t)
    c.print("[dim]Significance: * 10%  ** 5%  *** 1%  (leverage-adjusted bootstrap CVs)")


# ---------------------------------------------------------------------------
# Dynamic
# ---------------------------------------------------------------------------
def print_dynamic_table(res) -> None:
    c = _console()
    df = res.table
    if c is None:
        _hdr("Dynamic Asymmetric Causality  --  Hatemi-J (2021)",
             f"H0: {res.causvar} does not Granger-cause {res.depvar}")
        print(f" Mode:               {res.mode}")
        print(f" Window length S:    {res.window}    (PSY min = {res.smin})")
        print(f" Lag selection:      {res.ic}    Aug. lags: {res.intorder}    Boot: {res.boot}")
        _hline()
        print(f" {'SS End':>8}{'Lag':>6}{'Wald':>11}{'CV 10%':>11}{'CV 5%':>11}{'CV 1%':>11}{'W/CV5%':>11}")
        _hline()
        for i, row in df.iterrows():
            if i % 10 == 0 or i == len(df) - 1:
                mark = "*" if row["ratio_5pct"] > 1 else ""
                print(f" {int(row['sub_end']):>8d}{int(row['lag']):>6d}"
                      f"{row['Wald']:>11.4f}{row['cv10']:>11.4f}"
                      f"{row['cv5']:>11.4f}{row['cv1']:>11.4f}"
                      f"{row['ratio_5pct']:>11.4f}  {mark}")
        _hline()
        r1  = int((df["Wald"] > df["cv1"]).sum())
        r5  = int((df["Wald"] > df["cv5"]).sum())
        r10 = int((df["Wald"] > df["cv10"]).sum())
        print(f" Subsamples rejecting H0: {r1} at 1%,  {r5} at 5%,  {r10} at 10%   "
              f"(of {res.nsub} windows)")
        return

    t = Table(title=f"Dynamic Asymmetric Causality -- Hatemi-J (2021)",
              box=box.HEAVY_HEAD, header_style="bold cyan", title_style="bold")
    for col, just in [("SS End", "right"), ("Lag", "right"), ("Wald", "right"),
                      ("CV 10%", "right"), ("CV 5%", "right"), ("CV 1%", "right"),
                      ("W/CV5%", "right")]:
        t.add_column(col, justify=just)
    for i, row in df.iterrows():
        if i % 10 == 0 or i == len(df) - 1:
            style = "yellow" if row["ratio_5pct"] > 1 else None
            t.add_row(str(int(row['sub_end'])), str(int(row['lag'])),
                      f"{row['Wald']:.4f}", f"{row['cv10']:.4f}",
                      f"{row['cv5']:.4f}", f"{row['cv1']:.4f}",
                      f"{row['ratio_5pct']:.4f}", style=style)
    c.rule("[bold]Dynamic Asymmetric Causality -- Hatemi-J (2021)")
    c.print(f"[dim]H0: {res.causvar} does not Granger-cause {res.depvar}  ({res.shock} shocks)")
    c.print(f"[dim]Mode: [bold]{res.mode}[/]    Window: [bold]{res.window}[/]  "
            f"(PSY min={res.smin})    Lag IC: [bold]{res.ic}[/]    "
            f"Boot: [bold]{res.boot}[/]")
    c.print(t)
    r1  = int((df["Wald"] > df["cv1"]).sum())
    r5  = int((df["Wald"] > df["cv5"]).sum())
    r10 = int((df["Wald"] > df["cv10"]).sum())
    c.print(f"[bold green]Subsamples rejecting H0: "
            f"{r1} at 1%,  {r5} at 5%,  {r10} at 10%   (of {res.nsub} windows)")


# ---------------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------------
def print_fourier_table(res) -> None:
    c = _console()
    if c is None:
        _hdr("Fourier Asymmetric TY Causality",
             f"H0: {res.causvar} does not Granger-cause {res.depvar}")
        print(f" Form: {res.form}    kmax: {res.kmax}    IC: {res.ic}    Aug.lags: {res.intorder}")
        _hline()
        print(f" {'Shock':<10}{'Wald':>11}{'Lag':>6}{'k*':>5}{'Asy p':>11}{'Obs':>7}  Sig")
        _hline()
        for shock, row in res.table.iterrows():
            print(f" {shock:<10}{row['Wald']:>11.4f}{int(row['lag']):>6d}"
                  f"{int(row['k_opt']):>5d}{row['asy_p']:>11.4f}{int(row['sample']):>7d}  "
                  f"{_star(row['asy_p'])}")
        _hline()
        return

    t = Table(title="Fourier Asymmetric TY Causality (Nazlioglu et al. 2016)",
              box=box.HEAVY_HEAD, header_style="bold cyan", title_style="bold")
    for col, just in [("Shock","left"),("Wald","right"),("Lag","right"),
                      ("k*","right"),("Asy p","right"),("Obs","right"),("Sig","center")]:
        t.add_column(col, justify=just)
    for shock, row in res.table.iterrows():
        t.add_row(shock, f"{row['Wald']:.4f}", str(int(row['lag'])),
                  str(int(row['k_opt'])), f"{row['asy_p']:.4f}",
                  str(int(row['sample'])), _star(row['asy_p']))
    c.print(t)


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------
def print_spectral_table(res) -> None:
    c = _console()
    if c is None:
        _hdr("Asymmetric Frequency-Domain Causality  (Bahmani-Oskooee et al. 2016)",
             f"H0: {res.causvar} does not Granger-cause {res.depvar} at frequency w")
        print(f" Grid size (0,pi]:  {res.nfreq}    IC: {res.ic}")
        _hline()
        for shock, row in res.summary.iterrows():
            print(f" {shock:<10}: freq. rejecting H0 at 1%/5%/10%:  "
                  f"{int(row['n_reject_1pct']):>3}/{int(row['n_reject_5pct']):>3}/"
                  f"{int(row['n_reject_10pct']):>3}    (of {int(row['nfreq'])})  "
                  f"lag={int(row['lag'])}")
        _hline()
        return

    t = Table(title="Asymmetric Frequency-Domain Causality (BCRanjbar 2016)",
              box=box.HEAVY_HEAD, header_style="bold cyan", title_style="bold")
    for col in ["Shock", "Lag", "n_reject_1%", "n_reject_5%", "n_reject_10%",
                "% reject @5%"]:
        t.add_column(col, justify="right" if col != "Shock" else "left")
    for shock, row in res.summary.iterrows():
        t.add_row(shock, str(int(row['lag'])),
                  str(int(row['n_reject_1pct'])),
                  str(int(row['n_reject_5pct'])),
                  str(int(row['n_reject_10pct'])),
                  f"{row['pct_reject_5pct']:.2%}")
    c.print(t)


# ---------------------------------------------------------------------------
# Quantile
# ---------------------------------------------------------------------------
def print_quantile_table(res) -> None:
    c = _console()
    df = res.table
    if c is None:
        _hdr("Quantile Asymmetric Causality  (Fang et al. 2026)",
             f"H0: {res.causvar} does not Granger-cause {res.depvar}")
        if res.fourier_used:
            print(f" Fourier detrending: yes (kmax={res.kmax})")
        print(f" IC: {res.ic}    Aug.lags: {res.intorder}")
        _hline()
        print(f" {'Shock':<10}{'tau':>6}{'Wald':>11}{'Lag':>6}{'Asy p':>11}  Sig")
        _hline()
        for _, row in df.iterrows():
            print(f" {row['shock']:<10}{row['tau']:>6.2f}{row['Wald']:>11.4f}"
                  f"{int(row['lag']):>6d}{row['asy_p']:>11.4f}  {_star(row['asy_p'])}")
        _hline()
        return

    t = Table(title="Quantile Asymmetric Causality (Fang et al. 2026)",
              box=box.HEAVY_HEAD, header_style="bold cyan", title_style="bold")
    for col in ["Shock", "tau", "Wald", "Lag", "Asy p", "Sig"]:
        t.add_column(col, justify="right" if col != "Shock" else "left")
    for _, row in df.iterrows():
        t.add_row(row['shock'], f"{row['tau']:.2f}", f"{row['Wald']:.4f}",
                  str(int(row['lag'])), f"{row['asy_p']:.4f}",
                  _star(row['asy_p']))
    c.print(t)


# ---------------------------------------------------------------------------
# Efficient
# ---------------------------------------------------------------------------
def print_efficient_table(res) -> None:
    c = _console()
    if c is None:
        _hdr("Efficient Asymmetric Causality Tests  --  Hatemi-J (2024)",
             f"H0: {res.causvar} does not Granger-cause {res.depvar}")
        print(f" Lag selection: {res.ic}    Common lag p: {res.lag}    Aug.lags: {res.intorder}")
        _hline()
        print(f" {'Null hypothesis':<32}{'Wald':>11}{'df':>5}{'Asy p':>11}  Decision")
        _hline()
        for hyp, row in res.table.iterrows():
            print(f" {hyp:<32}{row['Wald']:>11.4f}{int(row['df']):>5d}"
                  f"{row['asy_p']:>11.4f}  {row['decision_5pct']}")
        _hline()
        print(" Hypothesis 4 (POS = NEG) is the key asymmetry test.")
        return

    t = Table(title="Efficient Asymmetric Causality (SUR)  --  Hatemi-J (2024)",
              box=box.HEAVY_HEAD, header_style="bold cyan", title_style="bold")
    for col, j in [("Null hypothesis", "left"), ("Wald", "right"),
                   ("df", "right"), ("Asy p", "right"), ("Decision", "left")]:
        t.add_column(col, justify=j)
    for hyp, row in res.table.iterrows():
        t.add_row(hyp, f"{row['Wald']:.4f}", str(int(row['df'])),
                  f"{row['asy_p']:.4f}", row['decision_5pct'])
    c.print(t)


# ---------------------------------------------------------------------------
# All -- unified summary
# ---------------------------------------------------------------------------
def print_all_summary(res) -> None:
    c = _console()
    summary = res.summary
    if c is None:
        print()
        _hdr("UNIFIED ASYMMETRIC CAUSALITY SUMMARY",
             "H0: causvar does not Granger-cause depvar")
        print(f" {'Test':<34}{'Shock':>8}{'Statistic':>14}{'p-value':>12}  Decision")
        _hline()
        for _, row in summary.iterrows():
            stat = row["Statistic"]
            stat_s = f"{stat:.4f}" if isinstance(stat, (int, float)) else str(stat)
            pv = row["p-value"]
            pv_s = f"{pv:.4f}" if isinstance(pv, (int, float)) else str(pv)
            print(f" {row['Test']:<34}{row['Shock']:>8}{stat_s:>14}{pv_s:>12}  {row['Decision']}")
        _hline()
        return

    t = Table(title="UNIFIED ASYMMETRIC CAUSALITY SUMMARY",
              caption="Decision at 5% level.  Spectral/quantile rows report fraction of frequencies/quantiles rejecting.",
              box=box.DOUBLE_EDGE, header_style="bold cyan", title_style="bold")
    for col, j in [("Test", "left"), ("Shock", "center"),
                   ("Statistic", "right"), ("p-value", "right"),
                   ("Decision", "left")]:
        t.add_column(col, justify=j)
    for _, row in summary.iterrows():
        stat = row["Statistic"]
        pv = row["p-value"]
        stat_s = f"{stat:.4f}" if isinstance(stat, (int, float)) else str(stat)
        pv_s = f"{pv:.4f}" if isinstance(pv, (int, float)) else str(pv)
        style = "bold red" if row["Decision"].startswith("Reject") else None
        t.add_row(row['Test'], row['Shock'], stat_s, pv_s, row['Decision'],
                  style=style)
    c.print(t)
