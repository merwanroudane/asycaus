"""
asycaus
=======

Asymmetric Granger-causality suite for Python.  Python mirror of the Stata
package `asycaus` by the same author.

Quick start
-----------

>>> import numpy as np, asycaus
>>> rng = np.random.default_rng(0)
>>> x = np.cumsum(rng.standard_normal(300))
>>> y = np.r_[0, 0.5*x[:-1] + rng.standard_normal(299)]
>>> r = asycaus.static(y, x, shock="both", boot=200, plot=False)

Available tests
---------------

asycaus.static        Hatemi-J (2012) static asymmetric (leverage bootstrap)
asycaus.dynamic       Hatemi-J (2021) rolling / recursive
asycaus.fourier       Nazlioglu et al. (2016) Fourier-augmented TY
asycaus.spectral      Bahmani-Oskooee et al. (2016) BC frequency-domain
asycaus.quantile      Fang et al. (2026) quantile asymmetric
asycaus.efficient     Hatemi-J (2024) SUR (Pos / Neg / Joint / Pos=Neg)
asycaus.all_tests     Run every test, return unified summary

Author : Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub : https://github.com/merwanroudane/asycaus
License: MIT
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("asycaus")
except PackageNotFoundError:
    __version__ = "1.0.0"

__author__  = "Dr Merwan Roudane"
__email__   = "merwanroudane920@gmail.com"
__license__ = "MIT"
__url__     = "https://github.com/merwanroudane/asycaus"

from . import engine
from . import tables
from . import plots
from .static    import static,    StaticResult
from .dynamic   import dynamic,   DynamicResult
from .fourier   import fourier,   FourierResult
from .spectral  import spectral,  SpectralResult
from .quantile  import quantile,  QuantileResult
from .efficient import efficient, EfficientResult
from .all_tests import all_tests, AllResult
from .engine    import pos_neg_components

__all__ = [
    "static", "dynamic", "fourier", "spectral", "quantile",
    "efficient", "all_tests", "pos_neg_components",
    "StaticResult", "DynamicResult", "FourierResult", "SpectralResult",
    "QuantileResult", "EfficientResult", "AllResult",
    "engine", "tables", "plots",
    "__version__", "__author__", "__email__", "__license__", "__url__",
]
